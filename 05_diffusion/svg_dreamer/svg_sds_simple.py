"""
SVG + SDS 简化实现
不依赖 diffvg，使用迭代优化方式

原理：
1. 初始化 SVG 参数（路径点、颜色）
2. 渲染 SVG 为图像
3. 用 Stable Diffusion 计算 SDS loss
4. 优化 SVG 参数
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
import warnings
warnings.filterwarnings('ignore')


class SimpleSVGRenderer:
    """简单的 SVG 渲染器（非可微分，用于可视化）"""
    
    def __init__(self, canvas_size=512):
        self.canvas_size = canvas_size
    
    def render_circles(self, centers, radii, colors, canvas_size=None):
        """渲染圆形到图像"""
        size = canvas_size or self.canvas_size
        dwg = svgwrite.Drawing(size=(size, size))
        dwg.viewbox(0, 0, size, size)
        
        # 白色背景
        dwg.add(dwg.rect(insert=(0, 0), size=(size, size), fill='white'))
        
        n_shapes = len(centers)
        for i in range(n_shapes):
            cx, cy = centers[i]
            r = radii[i]
            color = colors[i]
            
            # 转换颜色
            rgb = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'
            alpha = color[3] if len(color) > 3 else 1.0
            
            dwg.add(dwg.circle(
                center=(cx * size, cy * size),
                r=r * size,
                fill=rgb,
                fill_opacity=alpha
            ))
        
        # SVG -> PNG
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
    """可微分的圆形渲染器（使用软光栅化）"""
    
    def __init__(self, canvas_size=512):
        super().__init__()
        self.canvas_size = canvas_size
        
        # 创建坐标网格
        y, x = torch.meshgrid(
            torch.linspace(0, 1, canvas_size),
            torch.linspace(0, 1, canvas_size),
            indexing='ij'
        )
        self.register_buffer('grid_x', x)
        self.register_buffer('grid_y', y)
    
    def forward(self, centers, radii, colors):
        """
        可微分渲染
        centers: (N, 2) 归一化坐标 [0, 1]
        radii: (N,) 归一化半径
        colors: (N, 4) RGBA
        """
        device = centers.device
        H, W = self.canvas_size, self.canvas_size
        
        # 初始化白色画布
        canvas = torch.ones(3, H, W, device=device)
        
        n_shapes = centers.shape[0]
        
        # 计算所有圆的mask (N, H, W)
        cx = centers[:, 0].view(-1, 1, 1)  # (N, 1, 1)
        cy = centers[:, 1].view(-1, 1, 1)
        r = radii.view(-1, 1, 1)
        
        # 距离场
        grid_x = self.grid_x.unsqueeze(0)  # (1, H, W)
        grid_y = self.grid_y.unsqueeze(0)
        
        dist = torch.sqrt((grid_x - cx)**2 + (grid_y - cy)**2)  # (N, H, W)
        
        # 软边缘
        softness = 0.01
        masks = torch.sigmoid((r - dist) / softness)  # (N, H, W)
        
        # 获取颜色和alpha
        rgb = colors[:, :3]  # (N, 3)
        alpha = colors[:, 3] if colors.shape[1] > 3 else torch.ones(n_shapes, device=device)
        
        # 逐层混合 (避免inplace)
        for i in range(n_shapes):
            mask = masks[i] * alpha[i]  # (H, W)
            color = rgb[i]  # (3,)
            
            # 非inplace的alpha混合
            new_canvas = canvas * (1 - mask.unsqueeze(0)) + color.view(3, 1, 1) * mask.unsqueeze(0)
            canvas = new_canvas
        
        return canvas


class SDSLoss(nn.Module):
    """Score Distillation Sampling Loss"""
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="mps"):
        super().__init__()
        self.device = device
        
        print("Loading Stable Diffusion for SDS...")
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # MPS 需要 float32
            safety_checker=None
        ).to(device)
        
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler
        self.scheduler.set_timesteps(1000)
        
        # 冻结参数
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        print("✅ SD loaded!")
    
    @torch.no_grad()
    def encode_text(self, prompt):
        """编码文本提示"""
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        return self.text_encoder(tokens)[0]
    
    def forward(self, image, text_embedding, guidance_scale=7.5, t_range=(0.2, 0.8)):
        """
        计算 SDS loss (简化稳定版)
        image: (1, 3, H, W) 归一化到 [-1, 1]
        """
        # 编码图像到 latent
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.mean * 0.18215
        latents = latents.requires_grad_(True)
        
        # 随机时间步 (避免极端值)
        t = torch.randint(
            int(t_range[0] * 1000),
            int(t_range[1] * 1000),
            (1,),
            device=self.device
        ).long()
        
        # 添加噪声
        noise = torch.randn_like(latents)
        
        # 手动计算 noisy latent
        alpha_t = self.scheduler.alphas_cumprod[t].view(-1, 1, 1, 1).to(latents.device)
        sigma_t = (1 - alpha_t).sqrt()
        noisy_latents = alpha_t.sqrt() * latents + sigma_t * noise
        
        # 预测噪声
        with torch.no_grad():
            # Conditional
            noise_pred_cond = self.unet(noisy_latents.detach(), t, text_embedding).sample
            
            # Unconditional
            uncond_embedding = self.encode_text("")
            noise_pred_uncond = self.unet(noisy_latents.detach(), t, uncond_embedding).sample
        
        # Classifier-free guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # SDS loss: MSE between predicted noise and actual noise, weighted
        # 简化版：直接用预测差异
        target = noise  # 原始噪声作为目标
        
        # 使用MSE loss
        loss = F.mse_loss(latents, (latents - 0.1 * (noise_pred - noise)).detach())
        
        return loss


class TextToSVG:
    """文本到 SVG 生成器"""
    
    def __init__(
        self,
        n_circles=64,
        canvas_size=512,
        device="mps"
    ):
        self.n_circles = n_circles
        self.canvas_size = canvas_size
        self.device = device
        
        # 可微分渲染器
        self.renderer = DifferentiableCircleRenderer(canvas_size).to(device)
        
        # SVG 渲染器（用于保存）
        self.svg_renderer = SimpleSVGRenderer(canvas_size)
        
        # SDS loss
        self.sds = SDSLoss(device=device)
    
    def init_params(self):
        """初始化 SVG 参数"""
        # 随机初始化圆心
        centers = torch.rand(self.n_circles, 2, device=self.device)
        
        # 随机初始化半径
        radii = torch.rand(self.n_circles, device=self.device) * 0.1 + 0.02
        
        # 随机初始化颜色 (RGBA)
        colors = torch.rand(self.n_circles, 4, device=self.device)
        colors[:, 3] = 0.5 + torch.rand(self.n_circles, device=self.device) * 0.5  # alpha 0.5-1.0
        
        # 设置为可优化参数
        centers.requires_grad_(True)
        radii.requires_grad_(True)
        colors.requires_grad_(True)
        
        return centers, radii, colors
    
    def generate(
        self,
        prompt: str,
        n_iterations: int = 500,
        lr: float = 0.01,
        guidance_scale: float = 100,
        save_interval: int = 50,
        output_dir: str = "output"
    ):
        """生成 SVG"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🎨 Text-to-SVG Generation")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Circles: {self.n_circles}")
        print(f"Iterations: {n_iterations}")
        print(f"{'='*60}\n")
        
        # 编码文本
        text_embedding = self.sds.encode_text(prompt)
        
        # 初始化参数
        centers, radii, colors = self.init_params()
        
        # 优化器
        optimizer = torch.optim.Adam([
            {'params': centers, 'lr': lr},
            {'params': radii, 'lr': lr * 0.5},
            {'params': colors, 'lr': lr * 0.5}
        ])
        
        # 训练循环
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            # 渲染图像
            img = self.renderer(centers, radii, colors)
            
            # 转换为 SD 输入格式 [-1, 1]
            img_input = img.unsqueeze(0) * 2 - 1
            
            # 调整大小到 512x512
            if img_input.shape[-1] != 512:
                img_input = F.interpolate(img_input, size=(512, 512), mode='bilinear')
            
            # 计算 SDS loss
            loss = self.sds(img_input, text_embedding, guidance_scale)
            
            # 检查loss是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"   Warning: Invalid loss at iteration {i+1}, skipping...")
                continue
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 约束参数范围
            with torch.no_grad():
                centers.clamp_(0, 1)
                radii.clamp_(0.01, 0.3)
                colors.clamp_(0, 1)
            
            # 打印进度
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{n_iterations}, Loss: {loss.item():.4f}")
            
            # 保存中间结果
            if (i + 1) % save_interval == 0:
                self._save_result(centers, radii, colors, output_path, f"iter_{i+1:04d}")
        
        # 保存最终结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_result(centers, radii, colors, output_path, f"final_{timestamp}")
        
        print(f"\n✅ Done! Results saved to {output_path}")
        
        return centers.detach(), radii.detach(), colors.detach()
    
    def _save_result(self, centers, radii, colors, output_path, name):
        """保存结果"""
        # 转换为 numpy
        c = centers.detach().cpu().numpy()
        r = radii.detach().cpu().numpy()
        col = colors.detach().cpu().numpy()
        
        # 保存 SVG
        svg_path = output_path / f"{name}.svg"
        self.svg_renderer.save_svg(c, r, col, str(svg_path))
        
        # 保存 PNG
        img = self.svg_renderer.render_circles(c, r, col)
        png_path = output_path / f"{name}.png"
        Image.fromarray(img).save(png_path)
        
        print(f"   Saved: {svg_path}")


def main():
    """主函数"""
    
    # 检查设备
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # 创建生成器
    generator = TextToSVG(
        n_circles=128,
        canvas_size=512,
        device=device
    )
    
    # 生成 SVG
    prompt = "a cute cat, simple illustration, flat design"
    
    generator.generate(
        prompt=prompt,
        n_iterations=200,
        lr=0.02,
        guidance_scale=100,
        save_interval=50,
        output_dir="/Volumes/Seagate/SAM3/13_SVG_Diffusion/output"
    )


if __name__ == "__main__":
    main()
