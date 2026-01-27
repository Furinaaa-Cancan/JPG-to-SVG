#!/usr/bin/env python3
"""
可微分矢量化器 - 基于LIVE和Bézier Splatting思想
核心：直接优化贝塞尔曲线参数，使渲染结果逼近原图

技术来源：
- LIVE (CVPR 2022): Layer-wise Image Vectorization
- Bézier Splatting (2024): 贝塞尔曲线的高斯表示
- DiffVG: 可微分矢量渲染

简化实现：
1. 初始化随机贝塞尔路径
2. 渲染路径为图像
3. 计算与原图的损失
4. 梯度下降优化路径参数
5. 自适应添加新路径
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import svgwrite
import time


class BezierPath(nn.Module):
    """可优化的贝塞尔路径"""
    
    def __init__(self, n_points: int = 4, canvas_size: tuple = (512, 512)):
        super().__init__()
        h, w = canvas_size
        
        # 控制点 (可优化)
        self.points = nn.Parameter(torch.rand(n_points, 2) * torch.tensor([w, h]))
        
        # 颜色 RGB (可优化)
        self.color = nn.Parameter(torch.rand(3))
        
        # 透明度 (可优化)
        self.alpha = nn.Parameter(torch.tensor(0.8))
    
    def get_bezier_points(self, n_samples: int = 100) -> torch.Tensor:
        """计算贝塞尔曲线上的采样点"""
        t = torch.linspace(0, 1, n_samples, device=self.points.device)
        n = len(self.points) - 1
        
        # 计算Bernstein多项式
        points = torch.zeros(n_samples, 2, device=self.points.device)
        for i, p in enumerate(self.points):
            # Bernstein系数
            coef = self._bernstein(n, i, t)
            points += coef.unsqueeze(1) * p
        
        return points
    
    def _bernstein(self, n: int, i: int, t: torch.Tensor) -> torch.Tensor:
        """Bernstein基函数"""
        from math import comb
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


class DifferentiableRenderer(nn.Module):
    """可微分渲染器 - 将路径渲染为图像"""
    
    def __init__(self, canvas_size: tuple = (512, 512)):
        super().__init__()
        self.h, self.w = canvas_size
        
        # 创建像素坐标网格
        y, x = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32),
            torch.arange(self.w, dtype=torch.float32),
            indexing='ij'
        )
        self.register_buffer('grid_x', x)
        self.register_buffer('grid_y', y)
    
    def render_path(self, path: BezierPath, sigma: float = 3.0) -> torch.Tensor:
        """渲染单个路径为软掩码 - 优化版"""
        
        bezier_points = path.get_bezier_points(20)  # 减少采样点
        
        # 批量计算距离
        points = bezier_points.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
        grid = torch.stack([self.grid_x, self.grid_y], dim=-1).unsqueeze(2)  # [H, W, 1, 2]
        
        dists = torch.norm(grid - points, dim=-1)  # [H, W, N]
        min_dist = dists.min(dim=-1)[0]  # [H, W]
        
        # 高斯软边界
        mask = torch.exp(-min_dist ** 2 / (2 * sigma ** 2))
        
        # 应用颜色和透明度
        color = torch.sigmoid(path.color)
        alpha = torch.sigmoid(path.alpha)
        
        # 返回 RGBA - 内存高效
        rgba = torch.stack([
            mask * color[0],
            mask * color[1],
            mask * color[2],
            mask * alpha
        ])
        
        return rgba
    
    def composite(self, paths: list, background: torch.Tensor = None) -> torch.Tensor:
        """合成多个路径"""
        
        if background is None:
            canvas = torch.zeros(3, self.h, self.w, device=paths[0].points.device)
        else:
            canvas = background.clone()
        
        for path in paths:
            rgba = self.render_path(path)
            alpha = rgba[3:4]
            rgb = rgba[:3]
            
            # Alpha混合
            canvas = canvas * (1 - alpha) + rgb * alpha
        
        return canvas


class DifferentiableVectorizer:
    """可微分矢量化器"""
    
    def __init__(self, n_paths: int = 128, canvas_size: tuple = (512, 512)):
        self.n_paths = n_paths
        self.canvas_size = canvas_size
        self.device = "cpu"  # CPU更稳定，避免MPS内存问题
        
        print(f"\n🚀 Differentiable Vectorizer")
        print(f"   Device: {self.device}")
        print(f"   Paths: {n_paths}")
    
    def vectorize(self, image_path: str, output_dir: str = "02_输出结果/diff_svg",
                  n_iterations: int = 500, lr: float = 0.1):
        """可微分矢量化"""
        
        print("\n" + "="*70)
        print("💎 DIFFERENTIABLE VECTORIZATION")
        print("="*70)
        
        start_time = time.time()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载目标图像
        img_pil = Image.open(image_path).convert("RGB")
        img_pil = img_pil.resize(self.canvas_size, Image.LANCZOS)
        target = torch.from_numpy(np.array(img_pil)).float() / 255.0
        target = target.permute(2, 0, 1).to(self.device)  # [3, H, W]
        
        print(f"\n📷 Target: {self.canvas_size}")
        
        # 初始化路径
        print("\n🎨 Initializing paths...")
        paths = self._initialize_paths(target)
        
        # 渲染器
        renderer = DifferentiableRenderer(self.canvas_size).to(self.device)
        
        # 优化器
        params = []
        for path in paths:
            params.extend([path.points, path.color, path.alpha])
        optimizer = optim.Adam(params, lr=lr)
        
        # 优化循环
        print(f"\n🔄 Optimizing ({n_iterations} iterations)...")
        
        losses = []
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            # 渲染当前路径
            rendered = renderer.composite(paths)
            
            # 计算损失 (L2 + 感知损失)
            l2_loss = ((rendered - target) ** 2).mean()
            
            # 简单的边缘损失
            edge_loss = self._edge_loss(rendered, target)
            
            loss = l2_loss + 0.1 * edge_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if (i + 1) % 100 == 0:
                print(f"   Iter {i+1}/{n_iterations}, Loss: {loss.item():.4f}")
        
        # 生成SVG
        print("\n✨ Generating SVG...")
        svg_path = output_path / "diff_vector.svg"
        self._save_svg(paths, str(svg_path))
        
        # 保存渲染结果
        final_render = renderer.composite(paths).detach().cpu()
        final_img = (final_render.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(final_img).save(output_path / "rendered.png")
        
        # 创建对比HTML
        self._create_html(image_path, str(svg_path), output_path)
        
        process_time = time.time() - start_time
        
        print(f"\n" + "="*70)
        print(f"✅ DIFFERENTIABLE VECTORIZATION COMPLETE!")
        print(f"   Paths: {len(paths)}")
        print(f"   Final Loss: {losses[-1]:.4f}")
        print(f"   Time: {process_time:.1f}s")
        print("="*70)
        
        import subprocess
        subprocess.run(["open", str(output_path / "result.html")])
        
        return {'paths': len(paths), 'loss': losses[-1]}
    
    def _initialize_paths(self, target: torch.Tensor) -> list:
        """智能初始化路径 - 基于图像特征"""
        
        paths = []
        
        # 从目标图像采样颜色和位置
        target_np = target.permute(1, 2, 0).cpu().numpy()
        h, w = self.canvas_size
        
        for _ in range(self.n_paths):
            path = BezierPath(n_points=4, canvas_size=self.canvas_size)
            
            # 随机采样一个位置
            cx, cy = np.random.randint(0, w), np.random.randint(0, h)
            
            # 从该位置获取颜色
            color = target_np[cy, cx]
            path.color.data = torch.tensor(color, dtype=torch.float32)
            
            # 围绕该点初始化控制点
            radius = np.random.randint(10, 50)
            angles = np.linspace(0, 2 * np.pi, 5)[:-1]
            points = []
            for angle in angles:
                px = cx + radius * np.cos(angle)
                py = cy + radius * np.sin(angle)
                points.append([np.clip(px, 0, w-1), np.clip(py, 0, h-1)])
            path.points.data = torch.tensor(points, dtype=torch.float32)
            
            path = path.to(self.device)
            paths.append(path)
        
        return paths
    
    def _edge_loss(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """边缘损失 - 鼓励边缘对齐"""
        
        # 简单的Sobel边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=rendered.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        
        # 转为灰度
        rendered_gray = rendered.mean(dim=0, keepdim=True).unsqueeze(0)
        target_gray = target.mean(dim=0, keepdim=True).unsqueeze(0)
        
        # 边缘
        rendered_edge_x = torch.nn.functional.conv2d(rendered_gray, sobel_x, padding=1)
        rendered_edge_y = torch.nn.functional.conv2d(rendered_gray, sobel_y, padding=1)
        rendered_edge = torch.sqrt(rendered_edge_x**2 + rendered_edge_y**2 + 1e-6)
        
        target_edge_x = torch.nn.functional.conv2d(target_gray, sobel_x, padding=1)
        target_edge_y = torch.nn.functional.conv2d(target_gray, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-6)
        
        return ((rendered_edge - target_edge) ** 2).mean()
    
    def _save_svg(self, paths: list, output_path: str):
        """保存为SVG"""
        
        h, w = self.canvas_size
        dwg = svgwrite.Drawing(output_path, size=(w, h))
        dwg.viewbox(0, 0, w, h)
        
        for path in paths:
            points = path.points.detach().cpu().numpy()
            color = torch.sigmoid(path.color).detach().cpu().numpy()
            alpha = torch.sigmoid(path.alpha).detach().cpu().item()
            
            # 创建路径
            if len(points) >= 4:
                path_d = f"M{points[0][0]},{points[0][1]}"
                path_d += f" C{points[1][0]},{points[1][1]} {points[2][0]},{points[2][1]} {points[3][0]},{points[3][1]}"
                path_d += " Z"
                
                color_hex = f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"
                
                dwg.add(dwg.path(d=path_d, fill=color_hex, fill_opacity=alpha, stroke="none"))
        
        dwg.save()
    
    def _create_html(self, original: str, svg: str, output_path: Path):
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Differentiable Vectorization</title>
            <style>
                body {{ margin:0; background:#0a0a0a; color:#fff; font-family:sans-serif; }}
                .header {{ text-align:center; padding:50px; background:linear-gradient(135deg,#f093fb,#f5576c); }}
                h1 {{ font-size:3em; margin:0; }}
                .grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; padding:40px; max-width:1800px; margin:0 auto; }}
                .card {{ background:#1a1a1a; border-radius:15px; overflow:hidden; }}
                .card-header {{ padding:15px; background:#2a2a2a; font-weight:bold; text-align:center; }}
                img, object {{ width:100%; display:block; }}
                .tech {{ text-align:center; padding:30px; background:#1a1a1a; margin:20px 40px; border-radius:15px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 可微分矢量化</h1>
                <p>基于LIVE和Bézier Splatting思想</p>
            </div>
            <div class="grid">
                <div class="card">
                    <div class="card-header">📷 原图</div>
                    <img src="../../{original}">
                </div>
                <div class="card">
                    <div class="card-header">🎨 渲染结果</div>
                    <img src="rendered.png">
                </div>
                <div class="card">
                    <div class="card-header">✨ SVG</div>
                    <object data="{Path(svg).name}" type="image/svg+xml"></object>
                </div>
            </div>
            <div class="tech">
                <h2>💡 核心思想</h2>
                <p>直接优化贝塞尔曲线参数，使渲染结果逼近原图</p>
                <p>损失函数 = L2像素损失 + 边缘对齐损失</p>
            </div>
        </body>
        </html>
        """
        with open(output_path / "result.html", 'w') as f:
            f.write(html)


def main():
    vectorizer = DifferentiableVectorizer(n_paths=64, canvas_size=(256, 256))
    return vectorizer.vectorize("01_输入图片/Ladygaga_2.jpg", n_iterations=200)


if __name__ == "__main__":
    main()
