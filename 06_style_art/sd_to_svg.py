"""
SD风格化 → SVG矢量化

流程：
1. 用SDXL将图片转换为杜尚/立体主义风格
2. 用Potrace将结果矢量化为SVG
"""

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
import svgwrite


class SDtoSVG:
    """SD风格化 + SVG矢量化"""
    
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pipe = None
    
    def load_sd(self, use_controlnet: bool = True):
        """加载SDXL + ControlNet"""
        print("📦 加载模型...")
        
        if use_controlnet:
            from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
            
            print("   加载ControlNet Canny...")
            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0",
                torch_dtype=torch.float16,
            )
            
            self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet,
                torch_dtype=torch.float16,
            ).to(self.device)
            self.use_controlnet = True
        else:
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            ).to(self.device)
            self.use_controlnet = False
            
        print("✅ 模型加载完成")
    
    def extract_canny(self, image: Image.Image, low: int = 50, high: int = 150) -> Image.Image:
        """提取Canny边缘"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low, high)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)
    
    def stylize(
        self,
        image: Image.Image,
        prompt: str,
        strength: float = 0.55,
        guidance_scale: float = 8.0,
        steps: int = 30,
        controlnet_scale: float = 0.5,
    ) -> Image.Image:
        """用SD进行风格化"""
        
        # 调整尺寸
        w, h = image.size
        new_w = min(1024, (w // 64) * 64)
        scale = new_w / w
        new_h = int(h * scale // 64) * 64
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        negative_prompt = (
            "blurry, ugly, bad anatomy, realistic photograph, "
            "3d render, photorealistic, bad face"
        )
        
        if self.use_controlnet:
            # 提取边缘
            canny_image = self.extract_canny(image, low=30, high=100)
            
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                control_image=canny_image,
                strength=strength,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
                num_inference_steps=steps,
                generator=torch.Generator(device=self.device).manual_seed(42),
            ).images[0]
        else:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=torch.Generator(device=self.device).manual_seed(42),
            ).images[0]
        
        return result
    
    def image_to_svg_potrace(self, image: Image.Image, num_colors: int = 16) -> str:
        """用颜色量化 + Potrace矢量化"""
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # 颜色量化
        pixels = img_array.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = centers.astype(np.uint8)
        quantized = centers[labels.flatten()].reshape(img_array.shape)
        
        # 创建SVG
        dwg = svgwrite.Drawing(size=(w, h))
        
        # 添加背景
        bg_color = centers[np.argmax(np.bincount(labels.flatten()))]
        dwg.add(dwg.rect(insert=(0, 0), size=(w, h), 
                        fill=f'rgb({bg_color[0]},{bg_color[1]},{bg_color[2]})'))
        
        # 为每个颜色创建一个图层
        for i, color in enumerate(centers):
            # 创建该颜色的mask
            mask = np.all(quantized == color, axis=2).astype(np.uint8) * 255
            
            # 找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 转换为SVG path
            for contour in contours:
                if len(contour) < 3:
                    continue
                
                # 简化轮廓
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) < 3:
                    continue
                
                points = [(int(p[0][0]), int(p[0][1])) for p in approx]
                fill_color = f'rgb({color[0]},{color[1]},{color[2]})'
                
                dwg.add(dwg.polygon(points=points, fill=fill_color, stroke='none'))
        
        return dwg.tostring()
    
    def generate(
        self,
        image_path: str,
        output_dir: str,
        style: str = "duchamp",
        strength: float = 0.55,
        num_colors: int = 24,
        controlnet_scale: float = 0.5,
    ) -> dict:
        """完整流程：风格化 + 矢量化"""
        
        # 风格prompt - 更精准
        prompts = {
            "duchamp": (
                "abstract geometric painting, fragmented overlapping shapes, "
                "motion study, chronophotography style, multiple exposures of a figure, "
                "muted brown ochre sepia palette, angular faceted forms, "
                "Nude Descending a Staircase style, Marcel Duchamp, "
                "futurism cubism, mechanical rhythm, high contrast"
            ),
            "cubism": (
                "analytical cubism painting, geometric fragmentation, "
                "multiple viewpoints simultaneously, Pablo Picasso Georges Braque style, "
                "muted earth tones grays browns, angular planes, "
                "deconstructed form, abstract portrait, faceted surface"
            ),
            "pop_art": (
                "pop art screen print, Andy Warhol style, bold flat colors, "
                "high contrast, limited color palette, graphic posterized, "
                "celebrity portrait, halftone effect, vibrant saturated"
            ),
        }
        
        prompt = prompts.get(style, prompts["duchamp"])
        
        print(f"🎨 SD → SVG 风格化矢量生成")
        print(f"{'=' * 50}")
        print(f"📷 输入: {image_path}")
        print(f"   风格: {style}")
        print(f"   强度: {strength}")
        print(f"   颜色数: {num_colors}")
        
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        
        # 1. SD风格化
        print("\n🎨 SD风格化...")
        styled = self.stylize(image, prompt, strength=strength, controlnet_scale=controlnet_scale)
        
        # 2. SVG矢量化
        print("📐 SVG矢量化...")
        svg_content = self.image_to_svg_potrace(styled, num_colors=num_colors)
        
        # 保存
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取下一个版本号
        existing = list(output_dir.glob(f"{style}_v*.svg"))
        if existing:
            versions = [int(f.stem.split('_v')[1]) for f in existing if '_v' in f.stem]
            version = max(versions) + 1 if versions else 2
        else:
            version = 2  # v1已存在
        
        # 只保存SVG
        svg_path = output_dir / f"{style}_v{version}.svg"
        with open(svg_path, 'w') as f:
            f.write(svg_content)
        svg_size = svg_path.stat().st_size / 1024
        print(f"✅ SVG: {svg_path} ({svg_size:.1f} KB)")
        
        return {
            'svg_path': str(svg_path),
            'version': version,
        }


def main():
    input_image = "/Volumes/Seagate/SAM3/01_input/Ladygaga_2.jpg"
    output_dir = "/Volumes/Seagate/SAM3/06_style_art/output/sd_svg"
    
    if not Path(input_image).exists():
        print(f"❌ 找不到: {input_image}")
        return
    
    print("=" * 60)
    print("🎨 SD风格化 → SVG矢量图")
    print("=" * 60)
    
    converter = SDtoSVG()
    converter.load_sd(use_controlnet=False)  # v1没用ControlNet
    
    # 杜尚风格 - 和v1相同参数
    result = converter.generate(
        input_image,
        output_dir,
        style="duchamp",
        strength=0.55,      # 和v1一样
        num_colors=24,
    )
    
    print("\n" + "=" * 60)
    print(f"✅ 完成！杜尚 v{result['version']}")
    print("=" * 60)
    
    import subprocess
    subprocess.run(["open", result['svg_path']])


if __name__ == "__main__":
    main()
