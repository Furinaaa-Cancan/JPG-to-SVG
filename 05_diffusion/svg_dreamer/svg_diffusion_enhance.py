"""
SVG Diffusion 增强器
流程：已有图像 → SD处理 → 重新矢量化
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, "/Volumes/Seagate/SAM3/12_语义矢量化")

from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
import cairosvg
from io import BytesIO


class SVGDiffusionEnhancer:
    """SVG Diffusion 增强器"""
    
    def __init__(self, device="mps"):
        self.device = device
        self.pipe = None
        
    def load_sd(self, model_type="sdxl"):
        """加载SD模型"""
        print(f"Loading {model_type.upper()}...")
        
        if model_type == "sdxl":
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float32,
                use_safetensors=True
            ).to(self.device)
        else:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                safety_checker=None
            ).to(self.device)
        
        print("✅ SD loaded!")
    
    def load_image(self, path: str) -> Image.Image:
        """加载图像（支持SVG、PNG、JPG等）"""
        path = Path(path)
        
        if path.suffix.lower() in ['.svg', '.svgz']:
            # SVG转PNG
            print(f"Converting SVG to PNG...")
            png_data = cairosvg.svg2png(url=str(path), output_width=1024, output_height=1024)
            img = Image.open(BytesIO(png_data)).convert("RGB")
        else:
            img = Image.open(path).convert("RGB")
        
        return img
    
    def process(
        self,
        input_path: str,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted",
        strength: float = 0.5,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        output_dir: str = "output"
    ):
        """
        处理图像
        
        Args:
            input_path: 输入图像路径（SVG/PNG/JPG）
            prompt: 风格/修改提示词
            strength: 变化强度 (0-1)，越大变化越大
            guidance_scale: 提示词引导强度
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🎨 SVG Diffusion Enhancement")
        print(f"{'='*60}")
        print(f"Input: {input_path}")
        print(f"Prompt: {prompt}")
        print(f"Strength: {strength}")
        print(f"{'='*60}\n")
        
        # 加载输入图像
        input_img = self.load_image(input_path)
        print(f"📷 Input size: {input_img.size}")
        
        # 调整大小（SDXL需要1024，SD1.5需要512）
        target_size = 1024 if "xl" in str(type(self.pipe)).lower() else 512
        input_img = input_img.resize((target_size, target_size), Image.LANCZOS)
        
        # SD处理
        print("🔄 Running Diffusion...")
        
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_img,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存处理后的PNG
        png_path = output_path / f"sd_enhanced_{timestamp}.png"
        result.save(png_path)
        print(f"✅ PNG saved: {png_path}")
        
        # 矢量化
        print("\n🔄 Vectorizing...")
        svg_path = self.vectorize(result, output_path, timestamp)
        
        print(f"\n✅ All done!")
        print(f"   PNG: {png_path}")
        print(f"   SVG: {svg_path}")
        
        # 打开结果
        import subprocess
        subprocess.run(["open", str(png_path)])
        
        return png_path, svg_path
    
    def vectorize(self, img: Image.Image, output_path: Path, timestamp: str):
        """矢量化图像"""
        try:
            # 使用已有的矢量化器
            from sam3_color_vectorizer_fast import SAM3ColorVectorizerFast
            
            # 保存临时文件
            temp_path = output_path / f"temp_{timestamp}.png"
            img.save(temp_path)
            
            # 矢量化
            vectorizer = SAM3ColorVectorizerFast(n_workers=8)
            result = vectorizer.vectorize(str(temp_path))
            
            # 获取SVG路径（返回值可能是dict或str）
            if isinstance(result, dict):
                svg_path = result.get('svg_path', result.get('path', ''))
            else:
                svg_path = str(result) if result else ''
            
            # 移动到输出目录
            import shutil
            final_svg = output_path / f"vectorized_{timestamp}.svg"
            
            if svg_path and Path(svg_path).exists():
                shutil.copy(svg_path, final_svg)
                print(f"✅ SVG saved: {final_svg}")
            else:
                # 找最新生成的svg
                svg_dir = Path("/Volumes/Seagate/SAM3/12_语义矢量化/02_输出结果/sam3_color_svg")
                svgs = list(svg_dir.glob("*.svg"))
                if svgs:
                    latest = max(svgs, key=lambda p: p.stat().st_mtime)
                    shutil.copy(latest, final_svg)
                    print(f"✅ SVG saved: {final_svg}")
            
            # 清理临时文件
            temp_path.unlink(missing_ok=True)
            
            return final_svg
            
        except Exception as e:
            print(f"   ⚠️ Vectorization failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """示例用法"""
    
    # 检查设备
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 创建增强器
    enhancer = SVGDiffusionEnhancer(device=device)
    
    # 加载SD（选择sd15或sdxl）
    enhancer.load_sd(model_type="sd15")  # sd15更快，sdxl更好
    
    # 处理图像
    # 可以输入SVG或普通图片
    input_path = "/Volumes/Seagate/SAM3/12_语义矢量化/02_输出结果/sam3_color_svg/sam3_color_vector.svg"
    
    # 不同的处理效果示例：
    
    # 1. 艺术风格化
    # prompt = "oil painting style, artistic, vibrant colors"
    
    # 2. 写实增强
    # prompt = "photorealistic, detailed, high quality, 4k"
    
    # 3. 卡通风格
    # prompt = "cartoon style, flat colors, vector art, simple shapes"
    
    # 4. 保持原样但增强细节
    prompt = "enhanced details, high quality, sharp, professional"
    
    enhancer.process(
        input_path=input_path,
        prompt=prompt,
        strength=0.4,  # 0.3-0.5 保留原图较多，0.6-0.8 变化较大
        guidance_scale=7.5,
        num_inference_steps=30,
        output_dir="/Volumes/Seagate/SAM3/13_SVG_Diffusion/output_enhanced"
    )


if __name__ == "__main__":
    main()
