"""
芙莉莲艺术风格生成器
基于modern_art_styles.py的成熟架构
"""

import torch
import gc
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import svgwrite

# 路径配置
INPUT_IMAGE = "/Volumes/Seagate/SAM3/01_input/Picture1.jpg"
OUTPUT_DIR = Path("/Volumes/Seagate/SAM3/06_style_art/output/frieren")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设备
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ==================== 混合艺术家风格 ====================
# 保持仰视构图，改变人物外观

STYLES = {
    "klimt_mucha": {
        "name": "克里姆特+穆夏",
        "prompt": (
            "portrait of a mysterious woman looking up, low angle view, "
            "fusion of Gustav Klimt and Alphonse Mucha styles, "
            "golden spirals, mosaic patterns, Byzantine gold leaf, "
            "art nouveau flowing lines, decorative floral borders, "
            "ornate jewelry, intricate patterns, "
            "rich gold and jewel tones, ethereal beauty, "
            "masterpiece painting, museum quality"
        ),
        "negative": "anime, cartoon, 3d render, photo, modern, simple, blurry",
    },
    
    "picasso_matisse": {
        "name": "毕加索+马蒂斯",
        "prompt": (
            "portrait of a woman from below angle, looking upward, "
            "fusion of Pablo Picasso cubism and Henri Matisse fauvism, "
            "fragmented geometric planes, bold vivid colors, "
            "multiple perspectives combined, expressive brushwork, "
            "abstract figurative style, emotional intensity, "
            "blue period meets dance of color, "
            "avant-garde masterpiece, oil on canvas"
        ),
        "negative": "realistic, photo, anime, 3d, smooth, traditional portrait, blurry",
    },
    
    "vangogh_munch": {
        "name": "梵高+蒙克",
        "prompt": (
            "expressive portrait from low angle perspective, face looking up, "
            "fusion of Van Gogh and Edvard Munch styles, "
            "swirling starry night brushstrokes, emotional turbulence, "
            "vivid yellows blues and oranges, thick impasto texture, "
            "psychological depth, existential mood, "
            "post-impressionist expressionism, "
            "dramatic sky, passionate brushwork, fine art masterpiece"
        ),
        "negative": "flat, digital, anime, cartoon, calm, peaceful, blurry",
    },
    
    "hokusai_hiroshige": {
        "name": "北斋+广重",
        "prompt": (
            "ukiyo-e style portrait from below, figure gazing upward, "
            "fusion of Hokusai wave dynamics and Hiroshige landscapes, "
            "woodblock print aesthetic, bold outlines, flat color areas, "
            "dramatic composition, nature elements, "
            "traditional Japanese patterns, elegant simplicity, "
            "Edo period masterwork, decorative beauty"
        ),
        "negative": "3d, photorealistic, western, modern, gradient, blurry",
    },
    
    "monet_renoir": {
        "name": "莫奈+雷诺阿",
        "prompt": (
            "impressionist portrait from low viewpoint, subject looking up, "
            "fusion of Claude Monet light effects and Renoir soft beauty, "
            "dappled sunlight, visible brushstrokes, "
            "soft feminine features, warm skin tones, "
            "garden atmosphere, outdoor light, "
            "romantic mood, pastel harmonies, "
            "French impressionism masterpiece, oil on canvas"
        ),
        "negative": "sharp edges, digital, anime, dark, harsh, flat, blurry",
    },
    
    "dali_magritte": {
        "name": "达利+马格里特",
        "prompt": (
            "surrealist portrait from unusual low angle, upward gaze, "
            "fusion of Salvador Dali melting reality and Rene Magritte mystery, "
            "dreamlike impossible imagery, symbolic elements, "
            "hyper-detailed surreal landscape, floating objects, "
            "metaphysical atmosphere, thought-provoking composition, "
            "subconscious imagery, precise surrealism, "
            "museum quality surrealist masterpiece"
        ),
        "negative": "normal, ordinary, anime, cartoon, abstract, messy, blurry",
    },
}


class FrierenArtGenerator:
    """芙莉莲艺术风格生成器"""
    
    def __init__(self):
        self.device = DEVICE
        self.pipe = None
        
    def load_sd(self):
        """加载SDXL（与Lady Gaga相同配置）"""
        if self.pipe is not None:
            return
            
        print("📦 加载SDXL模型...")
        print(f"   设备: {self.device}")
        
        # float16 for MPS (与Lady Gaga相同)
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        # 从HuggingFace加载（本地缓存）
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
        ).to(self.device)
        
        print("✅ SDXL加载完成")
    
    def clear_memory(self):
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def stylize(
        self, 
        image: Image.Image, 
        style_key: str,
        strength: float = 0.65,
        guidance_scale: float = 10.0,
        num_inference_steps: int = 40,
        seed: int = 42,
    ) -> Image.Image:
        """风格化图像（与Lady Gaga相同参数）"""
        style = STYLES[style_key]
        
        # 调整尺寸
        w, h = image.size
        new_w = min(1024, (w // 64) * 64)
        scale = new_w / w
        new_h = int(h * scale // 64) * 64
        resized = image.resize((new_w, new_h), Image.LANCZOS)
        
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        result = self.pipe(
            prompt=style["prompt"],
            negative_prompt=style["negative"],
            image=resized,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        
        self.clear_memory()
        return result
    
    def to_svg(self, image: np.ndarray, num_colors: int = 64) -> str:
        """转换为SVG"""
        h, w = image.shape[:2]
        
        # 颜色量化
        pixels = image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        centers = centers.astype(np.uint8)
        quantized = centers[labels.flatten()].reshape(image.shape)
        
        dwg = svgwrite.Drawing(size=(w, h))
        
        # 背景
        bg_color = centers[np.argmax(np.bincount(labels.flatten()))]
        dwg.add(dwg.rect(insert=(0, 0), size=(w, h), 
                        fill=f'rgb({bg_color[0]},{bg_color[1]},{bg_color[2]})'))
        
        # 按面积排序
        color_areas = []
        for i, color in enumerate(centers):
            area = np.sum(labels == i)
            color_areas.append((area, color))
        color_areas.sort(reverse=True, key=lambda x: x[0])
        
        for _, color in color_areas:
            mask = np.all(quantized == color, axis=2).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
            
            for contour in contours:
                if len(contour) < 3:
                    continue
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) < 3:
                    continue
                points = [(int(p[0][0]), int(p[0][1])) for p in approx]
                fill = f'rgb({color[0]},{color[1]},{color[2]})'
                dwg.add(dwg.polygon(points=points, fill=fill, stroke='none'))
        
        return dwg.tostring()


def main():
    print("=" * 60)
    print("🎨 芙莉莲艺术风格生成器")
    print("=" * 60)
    
    # 加载原图
    if not Path(INPUT_IMAGE).exists():
        print(f"❌ 找不到图片: {INPUT_IMAGE}")
        return
    
    original = Image.open(INPUT_IMAGE).convert("RGB")
    print(f"📷 原图: {original.size}")
    
    # 创建生成器
    gen = FrierenArtGenerator()
    gen.load_sd()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, (style_key, style_info) in enumerate(STYLES.items(), 1):
        print(f"\n[{i}/{len(STYLES)}] 🎨 生成 {style_info['name']}...")
        
        # 风格化（与Lady Gaga相同参数）
        result = gen.stylize(
            original, 
            style_key,
            strength=0.65,  # 较高strength改变外观
            guidance_scale=10.0,
            num_inference_steps=40,
            seed=42,
        )
        
        # 保存SVG
        svg_content = gen.to_svg(np.array(result), num_colors=96)
        svg_path = OUTPUT_DIR / f"frieren_{style_key}_{timestamp}.svg"
        with open(svg_path, 'w') as f:
            f.write(svg_content)
        svg_size = svg_path.stat().st_size / 1024
        print(f"   ✅ SVG: {svg_path.name} ({svg_size:.1f} KB)")
    
    print("\n" + "=" * 60)
    print(f"✅ 完成！生成了 {len(STYLES)} 种SVG")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    
    import subprocess
    subprocess.run(["open", str(OUTPUT_DIR)])


if __name__ == "__main__":
    main()
