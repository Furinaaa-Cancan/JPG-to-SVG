"""
SVG + CLIP 简化实现
使用 CLIP 作为指导信号，比 SDS 更稳定

原理：
1. 初始化 SVG 参数（圆形的位置、大小、颜色）
2. 可微分渲染 SVG 为图像
3. 用 CLIP 计算图像与文本的相似度
4. 最大化相似度来优化 SVG 参数
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


class SimpleSVGRenderer:
    """简单的 SVG 渲染器（非可微分，用于保存）"""
    
    def __init__(self, canvas_size=224):
        self.canvas_size = canvas_size
    
    def render_circles(self, centers, radii, colors, canvas_size=None):
        """渲染圆形到图像"""
        size = canvas_size or self.canvas_size
        dwg = svgwrite.Drawing(size=(size, size))
        dwg.viewbox(0, 0, size, size)
        
        dwg.add(dwg.rect(insert=(0, 0), size=(size, size), fill='white'))
        
        n_shapes = len(centers)
        for i in range(n_shapes):
            cx, cy = centers[i]
            r = radii[i]
            color = colors[i]
            
            rgb = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'
            alpha = color[3] if len(color) > 3 else 1.0
            
            dwg.add(dwg.circle(
                center=(cx * size, cy * size),
                r=r * size,
                fill=rgb,
                fill_opacity=alpha
            ))
        
        svg_str = dwg.tostring()
        png_data = cairosvg.svg2png(bytestring=svg_str.encode(), output_width=size, output_height=size)
        img = Image.open(BytesIO(png_data)).convert('RGB')
        
        return np.array(img)
    
    def save_svg(self, centers, radii, colors, path, canvas_size=None):
        """保存 SVG 文件"""
        size = canvas_size or self.canvas_size
        dwg = svgwrite.Drawing(path, size=(size, size))
        dwg.viewbox(0, 0, size, size)
        
        dwg.add(dwg.rect(insert=(0, 0), size=(size, size), fill='white'))
        
        n_shapes = len(centers)
        for i in range(n_shapes):
            cx, cy = centers[i]
            r = radii[i]
            color = colors[i]
            
            rgb = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'
            alpha = color[3] if len(color) > 3 else 1.0
            
            dwg.add(dwg.circle(
                center=(cx * size, cy * size),
                r=r * size,
                fill=rgb,
                fill_opacity=alpha
            ))
        
        dwg.save()


class DifferentiableCircleRenderer(nn.Module):
    """可微分的圆形渲染器"""
    
    def __init__(self, canvas_size=224):
        super().__init__()
        self.canvas_size = canvas_size
        
        y, x = torch.meshgrid(
            torch.linspace(0, 1, canvas_size),
            torch.linspace(0, 1, canvas_size),
            indexing='ij'
        )
        self.register_buffer('grid_x', x)
        self.register_buffer('grid_y', y)
    
    def forward(self, centers, radii, colors):
        """可微分渲染"""
        device = centers.device
        H, W = self.canvas_size, self.canvas_size
        
        canvas = torch.ones(3, H, W, device=device)
        
        n_shapes = centers.shape[0]
        
        cx = centers[:, 0].view(-1, 1, 1)
        cy = centers[:, 1].view(-1, 1, 1)
        r = radii.view(-1, 1, 1)
        
        grid_x = self.grid_x.unsqueeze(0)
        grid_y = self.grid_y.unsqueeze(0)
        
        dist = torch.sqrt((grid_x - cx)**2 + (grid_y - cy)**2)
        
        softness = 0.01
        masks = torch.sigmoid((r - dist) / softness)
        
        rgb = colors[:, :3]
        alpha = colors[:, 3] if colors.shape[1] > 3 else torch.ones(n_shapes, device=device)
        
        for i in range(n_shapes):
            mask = masks[i] * alpha[i]
            color = rgb[i]
            new_canvas = canvas * (1 - mask.unsqueeze(0)) + color.view(3, 1, 1) * mask.unsqueeze(0)
            canvas = new_canvas
        
        return canvas


class CLIPLoss(nn.Module):
    """CLIP 相似度损失"""
    
    def __init__(self, device="mps"):
        super().__init__()
        self.device = device
        
        print("Loading CLIP...")
        import clip
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()
        
        # CLIP 图像预处理参数
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        
        print("✅ CLIP loaded!")
    
    @torch.no_grad()
    def encode_text(self, text):
        """编码文本"""
        tokens = clip.tokenize([text]).to(self.device)
        return self.model.encode_text(tokens)
    
    def forward(self, image, text_features):
        """
        计算 CLIP 损失
        image: (1, 3, H, W) 范围 [0, 1]
        """
        # 检查输入
        if torch.isnan(image).any():
            return torch.tensor(float('nan'), device=self.device)
        
        # 调整大小到 224x224
        if image.shape[-1] != 224:
            image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        
        # CLIP 预处理
        image = (image - self.mean) / (self.std + 1e-8)
        
        # 编码图像
        image_features = self.model.encode_image(image)
        
        # 归一化 (添加epsilon防止除0)
        image_norm = image_features.norm(dim=-1, keepdim=True) + 1e-8
        text_norm = text_features.norm(dim=-1, keepdim=True) + 1e-8
        
        image_features = image_features / image_norm
        text_features = text_features / text_norm
        
        # 余弦相似度 (越大越好，所以取负)
        similarity = (image_features * text_features).sum(dim=-1)
        
        # 返回负相似度作为损失
        return -similarity.mean()


class TextToSVG:
    """文本到 SVG 生成器 (CLIP引导)"""
    
    def __init__(
        self,
        n_circles=64,
        canvas_size=224,
        device="mps"
    ):
        self.n_circles = n_circles
        self.canvas_size = canvas_size
        self.device = device
        
        self.renderer = DifferentiableCircleRenderer(canvas_size).to(device)
        self.svg_renderer = SimpleSVGRenderer(canvas_size)
        self.clip_loss = CLIPLoss(device=device)
    
    def init_params(self):
        """初始化 SVG 参数"""
        centers = torch.rand(self.n_circles, 2, device=self.device)
        radii = torch.rand(self.n_circles, device=self.device) * 0.15 + 0.02
        colors = torch.rand(self.n_circles, 4, device=self.device)
        colors[:, 3] = 0.6 + torch.rand(self.n_circles, device=self.device) * 0.4
        
        centers.requires_grad_(True)
        radii.requires_grad_(True)
        colors.requires_grad_(True)
        
        return centers, radii, colors
    
    def generate(
        self,
        prompt: str,
        n_iterations: int = 500,
        lr: float = 0.03,
        save_interval: int = 50,
        output_dir: str = "output"
    ):
        """生成 SVG"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🎨 Text-to-SVG Generation (CLIP Guided)")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Circles: {self.n_circles}")
        print(f"Iterations: {n_iterations}")
        print(f"{'='*60}\n")
        
        # 编码文本
        text_features = self.clip_loss.encode_text(prompt)
        
        # 初始化参数
        centers, radii, colors = self.init_params()
        
        # 优化器
        optimizer = torch.optim.Adam([
            {'params': centers, 'lr': lr},
            {'params': radii, 'lr': lr * 0.5},
            {'params': colors, 'lr': lr * 0.5}
        ])
        
        best_loss = float('inf')
        best_params = None
        
        # 训练循环
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            # 渲染图像
            img = self.renderer(centers, radii, colors)
            
            # 转换格式 [0, 1]
            img_input = img.unsqueeze(0)
            
            # 计算 CLIP loss
            loss = self.clip_loss(img_input, text_features)
            
            # 检查loss和参数
            if torch.isnan(loss) or torch.isnan(centers).any() or torch.isnan(colors).any():
                print(f"   Warning: NaN detected at iteration {i+1}, resetting params...")
                # 重置参数
                with torch.no_grad():
                    centers.copy_(torch.rand_like(centers))
                    radii.copy_(torch.rand(self.n_circles, device=self.device) * 0.15 + 0.02)
                    colors.copy_(torch.rand_like(colors))
                    colors[:, 3] = 0.6 + torch.rand(self.n_circles, device=self.device) * 0.4
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_([centers, radii, colors], 0.5)
            
            # 更新参数
            optimizer.step()
            
            # 约束参数范围
            with torch.no_grad():
                centers.clamp_(0.01, 0.99)
                radii.clamp_(0.02, 0.25)
                colors.clamp_(0.01, 0.99)
            
            # 记录最佳
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = (centers.clone(), radii.clone(), colors.clone())
            
            # 打印进度
            if (i + 1) % 10 == 0:
                similarity = -loss.item()
                print(f"Iteration {i+1}/{n_iterations}, Similarity: {similarity:.4f}")
            
            # 保存中间结果
            if (i + 1) % save_interval == 0:
                self._save_result(centers, radii, colors, output_path, f"iter_{i+1:04d}")
        
        # 使用最佳结果
        if best_params:
            centers, radii, colors = best_params
        
        # 保存最终结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = self._save_result(centers, radii, colors, output_path, f"final_{timestamp}")
        
        print(f"\n✅ Done! Best similarity: {-best_loss:.4f}")
        print(f"Results saved to {output_path}")
        
        # 打开结果
        import subprocess
        subprocess.run(["open", str(final_path)])
        
        return centers.detach(), radii.detach(), colors.detach()
    
    def _save_result(self, centers, radii, colors, output_path, name):
        """保存结果"""
        c = centers.detach().cpu().numpy()
        r = radii.detach().cpu().numpy()
        col = colors.detach().cpu().numpy()
        
        svg_path = output_path / f"{name}.svg"
        self.svg_renderer.save_svg(c, r, col, str(svg_path))
        
        img = self.svg_renderer.render_circles(c, r, col)
        png_path = output_path / f"{name}.png"
        Image.fromarray(img).save(png_path)
        
        print(f"   Saved: {svg_path}")
        return png_path


def main():
    """主函数"""
    
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # 创建生成器
    generator = TextToSVG(
        n_circles=100,
        canvas_size=224,
        device=device
    )
    
    # 生成 SVG
    prompt = "a red apple"
    
    generator.generate(
        prompt=prompt,
        n_iterations=300,
        lr=0.01,  # 降低学习率
        save_interval=50,
        output_dir="/Volumes/Seagate/SAM3/13_SVG_Diffusion/output"
    )


if __name__ == "__main__":
    main()
