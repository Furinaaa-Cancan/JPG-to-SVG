"""
SVG增强修复工具 - 针对有艺术感但质量不高的版本
解决问题：膨胀不足、空隙残缺
"""

import torch
import gc
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import svgwrite
import json
import re


class SVGEnhancer:
    """SVG质量增强器"""

    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pipe = None

    def load_sd(self):
        print("📦 加载SDXL模型...")
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
        ).to(self.device)
        print("✅ 模型就绪")

    def clear_memory(self):
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def to_svg_high_quality(self, image, num_colors=64, dilate_iter=2, simplify=0.0005):
        """
        高质量SVG转换 - 消除空隙
        
        参数:
            num_colors: 颜色数量，越多细节越丰富
            dilate_iter: 膨胀迭代次数，越大空隙越少
            simplify: 简化系数，越小轮廓越精细
        """
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # 更精确的K-means颜色量化
        pixels = img_array.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.05)
        _, labels, centers = cv2.kmeans(
            pixels, num_colors, None, criteria, 20, cv2.KMEANS_PP_CENTERS
        )
        centers = centers.astype(np.uint8)
        quantized = centers[labels.flatten()].reshape(img_array.shape)

        dwg = svgwrite.Drawing(size=(w, h))
        
        # 背景色
        bg_color = centers[np.argmax(np.bincount(labels.flatten()))]
        dwg.add(dwg.rect(insert=(0, 0), size=(w, h), 
                        fill=f'rgb({bg_color[0]},{bg_color[1]},{bg_color[2]})'))

        # 按面积排序（大的先画，避免小块被覆盖）
        color_areas = []
        for i, color in enumerate(centers):
            mask = np.all(quantized == color, axis=2)
            area = np.sum(mask)
            color_areas.append((area, color, mask))
        color_areas.sort(reverse=True)

        # 膨胀核 - 用于填补空隙
        kernel = np.ones((3, 3), np.uint8)

        for _, color, mask in color_areas:
            mask_uint8 = mask.astype(np.uint8) * 255
            
            # 形态学膨胀 - 填补空隙
            if dilate_iter > 0:
                mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=dilate_iter)
            
            # 形态学闭运算 - 填补内部孔洞
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

            for contour in contours:
                if len(contour) < 3:
                    continue
                # 更精细的简化
                epsilon = simplify * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) < 3:
                    continue
                points = [(int(p[0][0]), int(p[0][1])) for p in approx]
                fill = f'rgb({color[0]},{color[1]},{color[2]})'
                dwg.add(dwg.polygon(points=points, fill=fill, stroke='none'))

        return dwg.tostring()

    def enhance_from_params(self, params_path, output_suffix="_enhanced"):
        """从参数文件重新生成高质量版本"""
        params_path = Path(params_path)
        
        # 读取原始参数
        with open(params_path) as f:
            params = json.load(f)
        
        print(f"🔧 增强处理: {params_path.stem}")
        print(f"   原始参数: strength={params.get('strength')}, colors={params.get('num_colors')}")
        
        # 加载原图
        input_image = params.get('input_image', '/Volumes/Seagate/SAM3/01_input/Ladygaga_2.jpg')
        image = Image.open(input_image).convert("RGB")
        
        # 使用原始prompt重新生成，但用更高质量参数
        prompt = params.get('prompt', '')
        negative = params.get('negative_prompt', '')
        seed = params.get('seed', 42)
        strength = params.get('strength', 0.55)
        
        # 风格化
        styled = self.stylize(image, prompt, negative, strength, seed, steps=35)
        
        # 高质量SVG转换
        svg_content = self.to_svg_high_quality(
            styled, 
            num_colors=80,      # 更多颜色
            dilate_iter=2,      # 膨胀填补空隙
            simplify=0.0003     # 更精细轮廓
        )
        
        # 保存
        svg_path = params_path.parent / f"{params_path.stem.replace('_params', '')}{output_suffix}.svg"
        with open(svg_path, 'w') as f:
            f.write(svg_content)
        
        size_mb = svg_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ 保存: {svg_path.name} ({size_mb:.2f} MB)")
        
        return svg_path

    def stylize(self, image, prompt, negative, strength, seed, steps=35):
        """风格化"""
        w, h = image.size
        new_w = min(896, (w // 64) * 64)
        scale = new_w / w
        new_h = int(h * scale // 64) * 64
        image = image.resize((new_w, new_h), Image.LANCZOS)

        try:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=image,
                strength=strength,
                guidance_scale=8.5,
                num_inference_steps=steps,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            ).images[0]
        finally:
            self.clear_memory()

        return result

    def enhance_svg_directly(self, svg_path, num_colors=80, dilate_iter=2):
        """
        直接增强现有SVG（通过重新渲染和矢量化）
        适用于没有参数文件的情况
        """
        from cairosvg import svg2png
        import io
        
        svg_path = Path(svg_path)
        print(f"🔧 直接增强: {svg_path.name}")
        
        # SVG转PNG
        png_data = svg2png(url=str(svg_path), output_width=1024)
        image = Image.open(io.BytesIO(png_data)).convert("RGB")
        
        # 重新矢量化，使用更高质量参数
        svg_content = self.to_svg_high_quality(
            image,
            num_colors=num_colors,
            dilate_iter=dilate_iter,
            simplify=0.0003
        )
        
        # 保存
        output_path = svg_path.parent / f"{svg_path.stem}_enhanced.svg"
        with open(output_path, 'w') as f:
            f.write(svg_content)
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ 保存: {output_path.name} ({size_mb:.2f} MB)")
        
        return output_path

    def enhance_image_to_svg(self, image_path, output_path, num_colors=80, dilate_iter=2):
        """
        从任意图片生成高质量SVG
        """
        image_path = Path(image_path)
        print(f"🔧 图片转高质量SVG: {image_path.name}")
        
        image = Image.open(image_path).convert("RGB")
        
        svg_content = self.to_svg_high_quality(
            image,
            num_colors=num_colors,
            dilate_iter=dilate_iter,
            simplify=0.0003
        )
        
        with open(output_path, 'w') as f:
            f.write(svg_content)
        
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"   ✅ 保存: {output_path} ({size_mb:.2f} MB)")
        
        return output_path


def main():
    """
    使用示例 - 增强指定版本
    """
    # 要增强的版本列表（填入你觉得艺术感好但质量不高的版本号）
    VERSIONS_TO_ENHANCE = [
        # 在这里填入版本号，例如:
        # 45, 67, 89, 102
    ]
    
    # 或者直接指定SVG文件路径
    SVG_FILES_TO_ENHANCE = [
        # "/Volumes/Seagate/SAM3/06_style_art/output/massive_art/art_v045_expressionism_german.svg",
    ]
    
    input_dir = Path("/Volumes/Seagate/SAM3/06_style_art/output/massive_art")
    
    enhancer = SVGEnhancer()
    
    if VERSIONS_TO_ENHANCE or SVG_FILES_TO_ENHANCE:
        # 如果有指定文件，需要加载SD模型
        # enhancer.load_sd()  # 如果需要重新风格化
        
        # 直接增强SVG（不需要SD模型）
        try:
            for version in VERSIONS_TO_ENHANCE:
                # 找到对应的SVG文件
                svg_files = list(input_dir.glob(f"art_v{version:03d}_*.svg"))
                for svg_file in svg_files:
                    enhancer.enhance_svg_directly(svg_file, num_colors=80, dilate_iter=2)
            
            for svg_file in SVG_FILES_TO_ENHANCE:
                enhancer.enhance_svg_directly(svg_file, num_colors=80, dilate_iter=2)
                
        except ImportError:
            print("⚠️ 需要安装cairosvg: pip install cairosvg")
            print("   或者使用enhance_from_params方法（需要参数文件）")
    else:
        print("📝 使用方法:")
        print("   1. 编辑脚本，在 VERSIONS_TO_ENHANCE 列表中填入要增强的版本号")
        print("   2. 或在 SVG_FILES_TO_ENHANCE 列表中填入SVG文件路径")
        print("   3. 运行脚本")
        print("\n   示例:")
        print("   VERSIONS_TO_ENHANCE = [45, 67, 89, 102]")


if __name__ == "__main__":
    main()
