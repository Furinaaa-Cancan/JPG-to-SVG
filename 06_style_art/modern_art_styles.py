"""
现代艺术风格系统

支持风格：
1. 立体主义 (Cubism) - 毕加索、布拉克
2. 未来主义 (Futurism) - 杜尚、波丘尼  
3. 波普艺术 (Pop Art) - 沃霍尔、利希滕斯坦
4. 野兽派 (Fauvism) - 马蒂斯
5. 表现主义 (Expressionism) - 蒙克
6. 抽象表现主义 (Abstract Expressionism) - 波洛克

当前专注：立体主义

python modern_art_styles.py
"""

import torch
import gc
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from scipy.spatial import Delaunay
from pathlib import Path
from datetime import datetime
import svgwrite
import json
from abc import ABC, abstractmethod


class ArtStyle(ABC):
    """艺术风格基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    @abstractmethod
    def get_prompt(self) -> tuple:
        """返回 (positive_prompt, negative_prompt)"""
        pass
    
    @abstractmethod
    def get_palette(self) -> list:
        """返回该风格的调色板"""
        pass
    
    @abstractmethod
    def post_process(self, image: np.ndarray) -> np.ndarray:
        """风格特定的后处理"""
        pass


class CubismStyle(ArtStyle):
    """
    立体主义风格
    
    核心技法：
    1. 多视角碎片 - 将物体分解，从多角度重组
    2. 几何简化 - 所有形体归纳为几何体
    3. 平面交织 - 前景背景穿透
    4. 分析期色调 - 灰褐单色
    5. 锐利边缘 - 明确的平面分界
    """
    
    # 分析立体主义色板（毕加索/布拉克 1908-1912）
    ANALYTICAL_PALETTE = [
        (89, 78, 65),     # 深褐灰
        (112, 98, 82),    # 中褐灰
        (135, 120, 100),  # 浅褐灰
        (156, 142, 122),  # 暖灰
        (68, 60, 50),     # 深影
        (178, 165, 145),  # 高光灰
        (100, 88, 72),    # 橄榄褐
        (145, 132, 115),  # 米灰
        (55, 48, 40),     # 最深
        (190, 178, 160),  # 最亮
    ]
    
    # 综合立体主义色板（1912-1920）
    SYNTHETIC_PALETTE = [
        (45, 65, 95),     # 深蓝
        (180, 75, 55),    # 赭红
        (85, 120, 80),    # 橄榄绿
        (200, 170, 120),  # 米黄
        (60, 50, 45),     # 深褐
        (150, 140, 125),  # 中性灰
        (120, 90, 60),    # 土黄
        (170, 160, 150),  # 浅灰
        (100, 50, 40),    # 深红褐
        (210, 195, 170),  # 象牙白
    ]
    
    def __init__(self, sub_style: str = "analytical"):
        """
        sub_style: "analytical" 分析立体主义 / "synthetic" 综合立体主义
        """
        super().__init__("Cubism")
        self.sub_style = sub_style
    
    def get_prompt(self) -> tuple:
        if self.sub_style == "analytical":
            positive = (
                "analytical cubism masterpiece by Pablo Picasso and Georges Braque, "
                "portrait fragmented into geometric planes, "
                "multiple perspectives shown simultaneously, "
                "face deconstructed with profile and frontal view combined, "
                "overlapping angular planes intersecting shapes, "
                "monochromatic earth tones brown gray ochre palette, "
                "broken planes with spatial ambiguity, "
                "subtle geometric analysis, intellectually deconstructed forms, "
                "oil on canvas texture, museum quality fine art, "
                "revolutionary composition, 1910 Paris avant-garde"
            )
        else:  # synthetic
            positive = (
                "synthetic cubism masterpiece by Pablo Picasso, "
                "bold geometric color blocks, bright vibrant accents, "
                "collage papier colle aesthetic with newspaper elements, "
                "simplified playful shapes, flat overlapping planes, "
                "figure reconstructed through geometric forms, "
                "dynamic balance of abstract composition, "
                "earth tones with bold color harmony, "
                "decorative patterns, modern art revolution, "
                "oil and mixed media on canvas, museum exhibition quality"
            )
        
        negative = (
            "blurry, soft focus, realistic photograph, photorealistic, "
            "3d render, smooth gradients, soft edges, "
            "anime, cartoon, digital art, low quality, "
            "normal perspective, traditional portrait, realistic face"
        )
        
        return positive, negative
    
    def get_palette(self) -> list:
        if self.sub_style == "analytical":
            return self.ANALYTICAL_PALETTE
        return self.SYNTHETIC_PALETTE
    
    def create_cubist_fragmentation(
        self, 
        image: np.ndarray,
        num_fragments: int = 800,
        edge_weight: float = 0.7,
    ) -> np.ndarray:
        """
        立体主义碎片化
        
        核心：在边缘和结构线上加密几何分割
        """
        h, w = image.shape[:2]
        
        # 多层次边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges1 = cv2.Canny(gray, 20, 60)   # 弱边缘
        edges2 = cv2.Canny(gray, 60, 150)  # 强边缘
        edges = cv2.bitwise_or(edges1, edges2)
        
        # 添加结构线（基于梯度方向）
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_mask = (gradient_mag > np.percentile(gradient_mag, 70)).astype(np.uint8) * 255
        edges = cv2.bitwise_or(edges, gradient_mask)
        
        # 采样点
        points = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]  # 四角
        
        # 边缘采样
        edge_coords = np.column_stack(np.where(edges > 0))
        n_edge = int(num_fragments * edge_weight)
        if len(edge_coords) > 0:
            indices = np.random.choice(len(edge_coords), min(n_edge, len(edge_coords)), replace=False)
            for idx in indices:
                y, x = edge_coords[idx]
                points.append((x, y))
        
        # 随机填充
        n_random = num_fragments - len(points)
        for _ in range(max(0, n_random)):
            points.append((np.random.randint(0, w), np.random.randint(0, h)))
        
        points = np.array(points)
        
        # Delaunay三角剖分
        tri = Delaunay(points)
        
        # 绘制碎片
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        for simplex in tri.simplices:
            pts = points[simplex].astype(np.int32)
            
            # 获取颜色
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cx, cy = max(0, min(w-1, cx)), max(0, min(h-1, cy))
            
            r = 2
            region = image[max(0,cy-r):min(h,cy+r+1), max(0,cx-r):min(w,cx+r+1)]
            if region.size > 0:
                color = np.median(region.reshape(-1, 3), axis=0).astype(int)
            else:
                color = image[cy, cx]
            
            cv2.fillPoly(canvas, [pts], tuple(int(c) for c in color))
        
        return canvas
    
    def add_plane_edges(self, image: np.ndarray, thickness: int = 1) -> np.ndarray:
        """添加平面分界线（立体主义特征）"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 膨胀边缘
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 叠加深色边缘线
        result = image.copy()
        edge_color = np.array([40, 35, 30])  # 深褐色线条
        result[edges > 0] = result[edges > 0] * 0.3 + edge_color * 0.7
        
        return result.astype(np.uint8)
    
    def map_to_palette(self, image: np.ndarray) -> np.ndarray:
        """映射到立体主义色板"""
        palette = np.array(self.get_palette())
        h, w = image.shape[:2]
        
        pixels = image.reshape(-1, 3)
        result = np.zeros_like(pixels)
        
        for i, pixel in enumerate(pixels):
            distances = np.sqrt(np.sum((palette - pixel) ** 2, axis=1))
            result[i] = palette[np.argmin(distances)]
        
        return result.reshape(h, w, 3).astype(np.uint8)
    
    def post_process(self, image: np.ndarray) -> np.ndarray:
        """立体主义后处理"""
        # 1. 几何碎片化
        fragmented = self.create_cubist_fragmentation(image)
        
        # 2. 色板映射（可选）
        # fragmented = self.map_to_palette(fragmented)
        
        # 3. 添加平面边缘
        result = self.add_plane_edges(fragmented)
        
        # 4. 增强对比度
        pil_img = Image.fromarray(result)
        enhancer = ImageEnhance.Contrast(pil_img)
        result = np.array(enhancer.enhance(1.15))
        
        return result


class ModernArtGenerator:
    """现代艺术生成器"""
    
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pipe = None
        self.controlnet = None
        self.use_controlnet = False
        self.output_dir = Path("/Volumes/Seagate/SAM3/06_style_art/output")
    
    def _get_next_version(self, style_name: str) -> int:
        """获取下一个版本号"""
        style_dir = self.output_dir / style_name.lower()
        style_dir.mkdir(parents=True, exist_ok=True)
        existing = list(style_dir.glob(f"{style_name.lower()}_v*.svg"))
        if not existing:
            return 1
        versions = []
        for f in existing:
            try:
                v = int(f.stem.split('_v')[1])
                versions.append(v)
            except:
                pass
        return max(versions) + 1 if versions else 1
    
    def load_sd(self, use_controlnet: bool = True):
        """加载SDXL"""
        if self.pipe is None or (use_controlnet != self.use_controlnet):
            print("📦 加载SDXL模型...")
            dtype = torch.float16 if self.device != "cpu" else torch.float32

            local_sdxl_path = "/Volumes/Seagate/SAM3/models/stable_diffusion/base_models/sdxl-base"
            
            # 强制使用本地模型
            if Path(local_sdxl_path).exists():
                sdxl_id = local_sdxl_path
                print(f"   使用本地SDXL: {sdxl_id}")
            else:
                sdxl_id = "stabilityai/stable-diffusion-xl-base-1.0"
                print(f"   下载SDXL: {sdxl_id}")

            # 暂时禁用ControlNet（本地没有）
            self.controlnet = None
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                sdxl_id,
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=Path(local_sdxl_path).exists(),
            ).to(self.device)
            self.use_controlnet = False

            print("✅ 模型加载完成")

    def clear_memory(self):
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def extract_canny(self, image: Image.Image, low: int = 30, high: int = 100) -> Image.Image:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low, high)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)

    def match_colors(self, image: np.ndarray, reference: np.ndarray, strength: float = 1.0) -> np.ndarray:
        if strength <= 0:
            return image

        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ref_bgr = cv2.cvtColor(reference, cv2.COLOR_RGB2BGR)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        img_mean, img_std = cv2.meanStdDev(img_lab)
        ref_mean, ref_std = cv2.meanStdDev(ref_lab)

        img_mean = img_mean.reshape((1, 1, 3))
        img_std = img_std.reshape((1, 1, 3))
        ref_mean = ref_mean.reshape((1, 1, 3))
        ref_std = ref_std.reshape((1, 1, 3))

        eps = 1e-6
        result_lab = (img_lab - img_mean) * (ref_std / (img_std + eps)) + ref_mean
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)

        result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        if strength >= 1:
            return result_rgb

        blended = image.astype(np.float32) * (1 - strength) + result_rgb.astype(np.float32) * strength
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def stylize_with_sd(
        self,
        image: Image.Image,
        style: ArtStyle,
        strength: float = 0.55,
        seed: int = 42,
        controlnet_scale: float = 0.5,
        guidance_scale: float = 10.0,
        num_inference_steps: int = 40,
        preserve_colors: bool = False,
        color_match_strength: float = 1.0,
    ) -> Image.Image:
        """用SD进行风格化"""
        
        # 调整尺寸
        w, h = image.size
        new_w = min(1024, (w // 64) * 64)
        scale = new_w / w
        new_h = int(h * scale // 64) * 64
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        positive, negative = style.get_prompt()
        if preserve_colors:
            positive = f"{positive}, preserve original colors, keep original color palette"
            negative = f"{negative}, sepia, monochrome"

        generator = torch.Generator(device="cpu").manual_seed(seed)

        try:
            if self.use_controlnet:
                canny_image = self.extract_canny(image, low=30, high=100)
                result = self.pipe(
                    prompt=positive,
                    negative_prompt=negative,
                    image=image,
                    control_image=canny_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).images[0]
            else:
                result = self.pipe(
                    prompt=positive,
                    negative_prompt=negative,
                    image=image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).images[0]

            if preserve_colors:
                result_np = np.array(result)
                ref_np = np.array(image)
                result_np = self.match_colors(result_np, ref_np, strength=color_match_strength)
                result = Image.fromarray(result_np)

            return result
        finally:
            self.clear_memory()
    
    def to_svg(self, image: np.ndarray, num_colors: int = 20) -> str:
        """转换为SVG"""
        h, w = image.shape[:2]
        
        # 颜色量化
        pixels = image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = centers.astype(np.uint8)
        quantized = centers[labels.flatten()].reshape(image.shape)
        
        dwg = svgwrite.Drawing(size=(w, h))
        
        # 背景
        bg_color = centers[np.argmax(np.bincount(labels.flatten()))]
        dwg.add(dwg.rect(insert=(0, 0), size=(w, h), 
                        fill=f'rgb({bg_color[0]},{bg_color[1]},{bg_color[2]})'))
        
        for color in centers:
            mask = np.all(quantized == color, axis=2).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) < 3:
                    continue
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) < 3:
                    continue
                points = [(int(p[0][0]), int(p[0][1])) for p in approx]
                fill = f'rgb({color[0]},{color[1]},{color[2]})'
                dwg.add(dwg.polygon(points=points, fill=fill, stroke='none'))
        
        return dwg.tostring()
    
    def generate(
        self,
        image_path: str,
        style: ArtStyle,
        strength: float = 0.55,
        num_colors: int = 24,
        use_sd: bool = True,
        use_controlnet: bool = True,
        controlnet_scale: float = 0.5,
        guidance_scale: float = 10.0,
        num_inference_steps: int = 40,
        preserve_colors: bool = False,
        color_match_strength: float = 1.0,
        seed: int = None,
        use_post_process: bool = True,
    ) -> dict:
        """生成艺术化SVG"""
        
        version = self._get_next_version(style.name)
        style_dir = self.output_dir / style.name.lower()
        
        print(f"🎨 {style.name} v{version}")
        print(f"{'=' * 50}")
        print(f"📷 输入: {image_path}")
        print(f"   子风格: {getattr(style, 'sub_style', 'default')}")
        print(f"   SD强度: {strength}")
        print(f"   SVG颜色: {num_colors}")
        print(f"   preserve_colors: {preserve_colors}")
        
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        
        # 1. SD风格化
        if use_sd:
            if seed is None:
                seed = int(np.random.randint(1, 2147483647))
            self.load_sd(use_controlnet=use_controlnet)
            print("\n🎨 SD风格化...")
            styled = self.stylize_with_sd(
                image,
                style,
                strength,
                seed=seed,
                controlnet_scale=controlnet_scale,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                preserve_colors=preserve_colors,
                color_match_strength=color_match_strength,
            )
            result_array = np.array(styled)
        else:
            result_array = np.array(image)
        
        # 2. 风格后处理
        if use_post_process:
            print("📐 风格后处理...")
            result_array = style.post_process(result_array)

        if preserve_colors:
            ref = np.array(image.resize((result_array.shape[1], result_array.shape[0]), Image.LANCZOS))
            result_array = self.match_colors(result_array, ref, strength=color_match_strength)
        
        # 3. SVG矢量化
        print("📐 SVG矢量化...")
        svg_content = self.to_svg(result_array, num_colors)
        
        # 保存
        svg_path = style_dir / f"{style.name.lower()}_v{version}.svg"
        with open(svg_path, 'w') as f:
            f.write(svg_content)
        svg_size = svg_path.stat().st_size / 1024
        print(f"✅ SVG: {svg_path} ({svg_size:.1f} KB)")
        
        # 保存预览PNG
        png_path = style_dir / f"{style.name.lower()}_v{version}_preview.png"
        Image.fromarray(result_array).save(str(png_path))
        
        # 保存参数
        params = {
            "version": version,
            "style": style.name,
            "sub_style": getattr(style, 'sub_style', 'default'),
            "strength": strength,
            "num_colors": num_colors,
            "use_sd": use_sd,
            "use_controlnet": use_controlnet,
            "controlnet_scale": controlnet_scale,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "preserve_colors": preserve_colors,
            "color_match_strength": color_match_strength,
            "seed": seed,
            "use_post_process": use_post_process,
        }
        params_path = style_dir / f"{style.name.lower()}_v{version}_params.json"
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
        
        return {
            'svg_path': str(svg_path),
            'png_path': str(png_path),
            'version': version,
        }


def main():
    input_image = "/Volumes/Seagate/SAM3/01_input/Ladygaga_2.jpg"
    
    if not Path(input_image).exists():
        print(f"❌ 找不到: {input_image}")
        return
    
    print("=" * 60)
    print("🎨 现代艺术风格系统 - 立体主义")
    print("=" * 60)
    
    # 创建立体主义风格
    cubism = CubismStyle(sub_style="synthetic")  # analytical / synthetic
    
    # 生成器
    generator = ModernArtGenerator()
    
    result = generator.generate(
        input_image,
        style=cubism,
        strength=0.55,
        num_colors=96,
        use_sd=True,
        use_controlnet=True,
        controlnet_scale=0.6,
        guidance_scale=8.0,
        num_inference_steps=30,
        preserve_colors=True,
        color_match_strength=1.0,
        use_post_process=True,
    )
    
    print("\n" + "=" * 60)
    print(f"✅ 完成！立体主义 v{result['version']}")
    print("=" * 60)
    
    import subprocess
    subprocess.run(["open", result['svg_path']])


if __name__ == "__main__":
    main()
