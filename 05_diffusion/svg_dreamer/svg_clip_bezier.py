"""
SVG + CLIP 贝塞尔曲线版本
使用闭合贝塞尔曲线作为图元，而不是圆形
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import svgwrite
import cairosvg
from io import BytesIO
from pathlib import Path
from datetime import datetime
import clip
import warnings
warnings.filterwarnings('ignore')


class BezierRenderer(nn.Module):
    """可微分的贝塞尔曲线渲染器"""
    
    def __init__(self, canvas_size=224, n_segments=4):
        super().__init__()
        self.canvas_size = canvas_size
        self.n_segments = n_segments  # 每条路径的贝塞尔段数
        
        # 创建坐标网格
        y, x = torch.meshgrid(
            torch.linspace(0, 1, canvas_size),
            torch.linspace(0, 1, canvas_size),
            indexing='ij'
        )
        self.register_buffer('grid_x', x)
        self.register_buffer('grid_y', y)
    
    def bezier_point(self, t, p0, p1, p2, p3):
        """计算三次贝塞尔曲线上的点"""
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        return mt3 * p0 + 3 * mt2 * t * p1 + 3 * mt * t2 * p2 + t3 * p3
    
    def render_filled_path(self, control_points, color, alpha):
        """
        渲染一个填充的闭合路径
        control_points: (n_segments, 4, 2) - 每段4个控制点
        """
        device = control_points.device
        H, W = self.canvas_size, self.canvas_size
        
        # 采样贝塞尔曲线上的点
        n_samples = 50
        t_vals = torch.linspace(0, 1, n_samples, device=device)
        
        path_points = []
        for seg_idx in range(self.n_segments):
            p0 = control_points[seg_idx, 0]
            p1 = control_points[seg_idx, 1]
            p2 = control_points[seg_idx, 2]
            p3 = control_points[seg_idx, 3]
            
            for t in t_vals[:-1]:  # 避免重复最后一个点
                pt = self.bezier_point(t, p0, p1, p2, p3)
                path_points.append(pt)
        
        path_points = torch.stack(path_points)  # (N, 2)
        
        # 使用软光栅化：计算每个像素到路径的"内部性"
        # 简化方法：使用路径中心和到边界的距离
        center = path_points.mean(dim=0)
        
        # 计算路径的近似半径
        radii = torch.norm(path_points - center, dim=1)
        avg_radius = radii.mean()
        
        # 计算每个像素到中心的距离
        dx = self.grid_x - center[0]
        dy = self.grid_y - center[1]
        dist_to_center = torch.sqrt(dx**2 + dy**2)
        
        # 软边缘mask
        softness = 0.02
        mask = torch.sigmoid((avg_radius - dist_to_center) / softness)
        
        return mask * alpha, color
    
    def forward(self, paths_control_points, colors):
        """
        渲染多个路径
        paths_control_points: (n_paths, n_segments, 4, 2)
        colors: (n_paths, 4) RGBA
        """
        device = paths_control_points.device
        H, W = self.canvas_size, self.canvas_size
        
        # 白色背景
        canvas = torch.ones(3, H, W, device=device)
        
        n_paths = paths_control_points.shape[0]
        
        for i in range(n_paths):
            ctrl_pts = paths_control_points[i]  # (n_segments, 4, 2)
            color = colors[i, :3]
            alpha = colors[i, 3]
            
            mask, _ = self.render_filled_path(ctrl_pts, color, alpha)
            
            # Alpha混合
            mask_3d = mask.unsqueeze(0)  # (1, H, W)
            new_canvas = canvas * (1 - mask_3d) + color.view(3, 1, 1) * mask_3d
            canvas = new_canvas
        
        return canvas


class SimpleBezierRenderer:
    """非可微分的SVG保存器"""
    
    def __init__(self, canvas_size=224, n_segments=4):
        self.canvas_size = canvas_size
        self.n_segments = n_segments
    
    def save_svg(self, paths_control_points, colors, path, canvas_size=None):
        """保存为SVG"""
        size = canvas_size or self.canvas_size
        dwg = svgwrite.Drawing(path, size=(size, size))
        dwg.viewbox(0, 0, size, size)
        
        # 白色背景
        dwg.add(dwg.rect(insert=(0, 0), size=(size, size), fill='white'))
        
        n_paths = len(paths_control_points)
        
        for i in range(n_paths):
            ctrl_pts = paths_control_points[i]  # (n_segments, 4, 2)
            color = colors[i]
            
            # 构建SVG path
            path_d = ""
            for seg_idx in range(self.n_segments):
                p0 = ctrl_pts[seg_idx, 0] * size
                p1 = ctrl_pts[seg_idx, 1] * size
                p2 = ctrl_pts[seg_idx, 2] * size
                p3 = ctrl_pts[seg_idx, 3] * size
                
                if seg_idx == 0:
                    path_d += f"M {p0[0]:.2f},{p0[1]:.2f} "
                
                path_d += f"C {p1[0]:.2f},{p1[1]:.2f} {p2[0]:.2f},{p2[1]:.2f} {p3[0]:.2f},{p3[1]:.2f} "
            
            path_d += "Z"  # 闭合路径
            
            rgb = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'
            alpha = color[3] if len(color) > 3 else 1.0
            
            dwg.add(dwg.path(d=path_d, fill=rgb, fill_opacity=float(alpha), stroke='none'))
        
        dwg.save()
    
    def render_to_image(self, paths_control_points, colors, canvas_size=None):
        """渲染为图像"""
        size = canvas_size or self.canvas_size
        
        # 创建临时SVG
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
            temp_path = f.name
        
        self.save_svg(paths_control_points, colors, temp_path, size)
        
        # SVG -> PNG
        png_data = cairosvg.svg2png(url=temp_path, output_width=size, output_height=size)
        img = Image.open(BytesIO(png_data)).convert('RGB')
        
        import os
        os.unlink(temp_path)
        
        return np.array(img)


class CLIPLoss(nn.Module):
    """CLIP相似度损失"""
    
    def __init__(self, device="mps"):
        super().__init__()
        self.device = device
        
        print("Loading CLIP...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()
        
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        
        print("✅ CLIP loaded!")
    
    @torch.no_grad()
    def encode_text(self, text):
        tokens = clip.tokenize([text]).to(self.device)
        return self.model.encode_text(tokens)
    
    def forward(self, image, text_features):
        if torch.isnan(image).any():
            return torch.tensor(float('nan'), device=self.device)
        
        if image.shape[-1] != 224:
            image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        
        image = (image - self.mean) / (self.std + 1e-8)
        image_features = self.model.encode_image(image)
        
        image_norm = image_features.norm(dim=-1, keepdim=True) + 1e-8
        text_norm = text_features.norm(dim=-1, keepdim=True) + 1e-8
        
        image_features = image_features / image_norm
        text_features = text_features / text_norm
        
        similarity = (image_features * text_features).sum(dim=-1)
        return -similarity.mean()


class TextToSVGBezier:
    """文本到SVG生成器（贝塞尔曲线版）"""
    
    def __init__(
        self,
        n_paths=32,
        n_segments=4,
        canvas_size=224,
        device="mps"
    ):
        self.n_paths = n_paths
        self.n_segments = n_segments
        self.canvas_size = canvas_size
        self.device = device
        
        self.renderer = BezierRenderer(canvas_size, n_segments).to(device)
        self.svg_renderer = SimpleBezierRenderer(canvas_size, n_segments)
        self.clip_loss = CLIPLoss(device=device)
    
    def init_params(self):
        """初始化贝塞尔曲线控制点"""
        # 每条路径：n_segments段，每段4个控制点，每点2D坐标
        # 初始化为随机的小形状
        
        paths = []
        for _ in range(self.n_paths):
            # 随机中心
            cx = torch.rand(1, device=self.device) * 0.8 + 0.1
            cy = torch.rand(1, device=self.device) * 0.8 + 0.1
            
            # 随机大小
            size = torch.rand(1, device=self.device) * 0.15 + 0.05
            
            # 创建一个近似圆形的贝塞尔曲线（4段）
            segments = []
            angles = torch.linspace(0, 2 * np.pi, self.n_segments + 1, device=self.device)[:-1]
            
            for i in range(self.n_segments):
                a0 = angles[i]
                a1 = angles[(i + 1) % self.n_segments]
                
                # 控制点偏移
                kappa = 0.5522847498  # 近似圆的贝塞尔系数
                
                p0 = torch.stack([cx + size * torch.cos(a0), cy + size * torch.sin(a0)]).squeeze()
                p3 = torch.stack([cx + size * torch.cos(a1), cy + size * torch.sin(a1)]).squeeze()
                
                # 控制点
                t0 = a0 + np.pi / 2
                t1 = a1 - np.pi / 2
                
                p1 = p0 + size * kappa * torch.stack([torch.cos(torch.tensor(t0)), torch.sin(torch.tensor(t0))]).to(self.device) * 0.3
                p2 = p3 + size * kappa * torch.stack([torch.cos(torch.tensor(t1)), torch.sin(torch.tensor(t1))]).to(self.device) * 0.3
                
                # 添加随机扰动
                noise = torch.randn(4, 2, device=self.device) * 0.02
                segment = torch.stack([p0, p1, p2, p3]) + noise
                segments.append(segment)
            
            path = torch.stack(segments)  # (n_segments, 4, 2)
            paths.append(path)
        
        paths_tensor = torch.stack(paths)  # (n_paths, n_segments, 4, 2)
        paths_tensor.requires_grad_(True)
        
        # 颜色
        colors = torch.rand(self.n_paths, 4, device=self.device)
        colors[:, 3] = 0.5 + torch.rand(self.n_paths, device=self.device) * 0.5
        colors.requires_grad_(True)
        
        return paths_tensor, colors
    
    def generate(
        self,
        prompt: str,
        n_iterations: int = 500,
        lr: float = 0.01,
        save_interval: int = 100,
        output_dir: str = "output"
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🎨 Text-to-SVG (Bezier Curves)")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Paths: {self.n_paths}, Segments: {self.n_segments}")
        print(f"Iterations: {n_iterations}")
        print(f"{'='*60}\n")
        
        text_features = self.clip_loss.encode_text(prompt)
        paths, colors = self.init_params()
        
        optimizer = torch.optim.Adam([
            {'params': paths, 'lr': lr},
            {'params': colors, 'lr': lr * 0.5}
        ])
        
        best_loss = float('inf')
        best_params = None
        
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            img = self.renderer(paths, colors)
            img_input = img.unsqueeze(0)
            
            loss = self.clip_loss(img_input, text_features)
            
            if torch.isnan(loss) or torch.isnan(paths).any():
                print(f"   NaN at {i+1}, resetting...")
                paths, colors = self.init_params()
                optimizer = torch.optim.Adam([
                    {'params': paths, 'lr': lr},
                    {'params': colors, 'lr': lr * 0.5}
                ])
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_([paths, colors], 0.5)
            optimizer.step()
            
            with torch.no_grad():
                paths.clamp_(0.01, 0.99)
                colors.clamp_(0.01, 0.99)
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = (paths.clone(), colors.clone())
            
            if (i + 1) % 10 == 0:
                print(f"Iter {i+1}/{n_iterations}, Similarity: {-loss.item():.4f}")
            
            if (i + 1) % save_interval == 0:
                self._save(paths, colors, output_path, f"iter_{i+1:04d}")
        
        if best_params:
            paths, colors = best_params
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = self._save(paths, colors, output_path, f"final_{timestamp}")
        
        print(f"\n✅ Done! Best similarity: {-best_loss:.4f}")
        
        import subprocess
        subprocess.run(["open", str(final_path)])
        
        return paths.detach(), colors.detach()
    
    def _save(self, paths, colors, output_path, name):
        p = paths.detach().cpu().numpy()
        c = colors.detach().cpu().numpy()
        
        svg_path = output_path / f"{name}.svg"
        self.svg_renderer.save_svg(p, c, str(svg_path), 512)
        
        img = self.svg_renderer.render_to_image(p, c, 512)
        png_path = output_path / f"{name}.png"
        Image.fromarray(img).save(png_path)
        
        print(f"   Saved: {svg_path}")
        return png_path


def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    generator = TextToSVGBezier(
        n_paths=48,
        n_segments=4,
        canvas_size=224,
        device=device
    )
    
    prompt = "a red apple, simple flat design"
    
    generator.generate(
        prompt=prompt,
        n_iterations=400,
        lr=0.008,
        save_interval=100,
        output_dir="/Volumes/Seagate/SAM3/13_SVG_Diffusion/output_bezier"
    )


if __name__ == "__main__":
    main()
