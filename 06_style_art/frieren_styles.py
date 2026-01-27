"""
芙莉莲艺术风格生成器
生成多种艺术风格的芙莉莲图片
"""

import torch
import gc
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import svgwrite

# 输入输出路径
INPUT_IMAGE = "/Volumes/Seagate/SAM3/01_input/Picture1.jpg"
OUTPUT_DIR = Path("/Volumes/Seagate/SAM3/06_style_art/output/frieren")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设备
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ==================== 混合艺术家风格定义 ====================
# 保持仰视构图，改变人物外观，结合多种艺术家风格

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
        "negative": "anime, cartoon, 3d render, photo, modern, simple",
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
        "negative": "realistic, photo, anime, 3d, smooth, traditional portrait",
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
        "negative": "flat, digital, anime, cartoon, calm, peaceful",
    },
    
    "hokusai_hiroshige": {
        "name": "北斉+广重",
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
        "negative": "sharp edges, digital, anime, dark, harsh, flat",
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
        "negative": "normal, ordinary, anime, cartoon, abstract, messy",
    },
}


def load_pipeline():
    """加载本地SDXL-Turbo模型"""
    print("📦 加载本地SDXL-Turbo...")
    print(f"   设备: {DEVICE}")
    
    # fp16 + MPS
    dtype = torch.float16
    local_path = "/Volumes/Seagate/SAM3/models/stable_diffusion/base_models/sdxl-turbo"
    
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        local_path,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
        variant="fp16",
    ).to(DEVICE)
    
    print("✅ SDXL-Turbo就绪")
    return pipe


def generate_style(pipe, image: Image.Image, style_key: str, seed: int = None) -> Image.Image:
    """生成单个风格"""
    style = STYLES[style_key]
    
    if seed is None:
        seed = np.random.randint(1, 2147483647)
    
    # 调整尺寸
    w, h = image.size
    new_w = min(1024, (w // 64) * 64)
    scale = new_w / w
    new_h = int(h * scale // 64) * 64
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # SDXL-Turbo参数（官方推荐）
    result = pipe(
        prompt=style["prompt"],
        negative_prompt=style["negative"],
        image=resized,
        strength=0.5,
        guidance_scale=0.0,
        num_inference_steps=2,
        generator=generator,
    ).images[0]
    
    # 清理显存
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return result


def to_svg(image: np.ndarray, num_colors: int = 64) -> str:
    """
    高质量SVG矢量化
    - 更多颜色层次
    - 保留内部细节
    - 更精细的轮廓
    """
    h, w = image.shape[:2]
    
    # 颜色量化（更多颜色 = 更多细节）
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
    
    # 按颜色面积排序（大的先画，小的后画覆盖）
    color_areas = []
    for i, color in enumerate(centers):
        area = np.sum(labels == i)
        color_areas.append((area, color))
    color_areas.sort(reverse=True, key=lambda x: x[0])
    
    for _, color in color_areas:
        mask = np.all(quantized == color, axis=2).astype(np.uint8) * 255
        # RETR_TREE保留层次结构，包含内部轮廓
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        
        for contour in contours:
            if len(contour) < 3:
                continue
            # 更精细的轮廓简化
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                continue
            
            # 构建SVG路径（更平滑）
            points = [(int(p[0][0]), int(p[0][1])) for p in approx]
            fill = f'rgb({color[0]},{color[1]},{color[2]})'
            dwg.add(dwg.polygon(points=points, fill=fill, stroke='none'))
    
    return dwg.tostring()


def create_comparison(original: Image.Image, results: dict) -> Image.Image:
    """创建对比图"""
    # 计算布局
    n = len(results) + 1  # 原图 + 各风格
    cols = 3
    rows = (n + cols - 1) // cols
    
    # 缩略图尺寸
    thumb_w, thumb_h = 400, 400
    padding = 20
    label_height = 40
    
    # 画布
    canvas_w = cols * (thumb_w + padding) + padding
    canvas_h = rows * (thumb_h + label_height + padding) + padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    
    # 绘制原图
    orig_thumb = original.copy()
    orig_thumb.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
    x = padding + (thumb_w - orig_thumb.width) // 2
    y = padding + label_height + (thumb_h - orig_thumb.height) // 2
    canvas.paste(orig_thumb, (x, y))
    
    # 添加标签（使用PIL绘制）
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((padding + thumb_w//2, padding + 10), "原图", fill=(0, 0, 0), font=font, anchor="mm")
    
    # 绘制各风格
    idx = 1
    for style_key, result_img in results.items():
        row = idx // cols
        col = idx % cols
        
        thumb = result_img.copy()
        thumb.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
        
        x = padding + col * (thumb_w + padding) + (thumb_w - thumb.width) // 2
        y = padding + row * (thumb_h + label_height + padding) + label_height + (thumb_h - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        
        # 标签
        label_x = padding + col * (thumb_w + padding) + thumb_w // 2
        label_y = padding + row * (thumb_h + label_height + padding) + 10
        draw.text((label_x, label_y), STYLES[style_key]["name"], fill=(0, 0, 0), font=font, anchor="mm")
        
        idx += 1
    
    return canvas


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
    
    # 加载模型
    pipe = load_pipeline()
    
    # 混合艺术家风格
    selected_styles = list(STYLES.keys())
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, style_key in enumerate(selected_styles, 1):
        style_name = STYLES[style_key]["name"]
        print(f"\n[{i}/{len(selected_styles)}] 🎨 生成 {style_name}...")
        
        result = generate_style(pipe, original, style_key, seed=42)
        results[style_key] = result
        
        # 保存SVG
        svg_content = to_svg(np.array(result), num_colors=96)
        svg_path = OUTPUT_DIR / f"frieren_{style_key}_{timestamp}.svg"
        with open(svg_path, 'w') as f:
            f.write(svg_content)
        svg_size = svg_path.stat().st_size / 1024
        print(f"   ✅ SVG: {svg_path.name} ({svg_size:.1f} KB)")
    
    print("\n" + "=" * 60)
    print(f"✅ 完成！生成了 {len(results)} 种SVG")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    
    # 打开输出目录
    import subprocess
    subprocess.run(["open", str(OUTPUT_DIR)])


if __name__ == "__main__":
    main()
