"""
毕加索名作风格生成器 - 基于真实名作的艺术化处理

基于毕加索10大名作的真实风格特征：
1. Les Demoiselles d'Avignon (1907) - 非洲面具、棱角变形
2. Guernica (1937) - 灰黑白单色、碎片化恐惧
3. La Femme qui pleure (1937) - 哭泣女人、鲜艳碎片
4. Le Rêve (1932) - 分裂面孔、柔和曲线
5. Dora Maar au Chat (1941) - 几何肖像、猫
6. Girl Before a Mirror (1932) - 镜像双重、大胆色彩
7. The Old Guitarist (1903) - 蓝色时期、瘦长忧郁
8. Three Musicians (1921) - 综合立体主义、平面几何
9. La Vie (1903) - 蓝色时期、生死主题
10. Portrait of Dora Maar (1937) - 多视角面孔

python picasso_masterpiece.py
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
from datetime import datetime
import random


class PicassoMasterpieceGenerator:
    """基于毕加索真实名作的风格生成器"""
    
    # 基于真实名作的风格定义
    MASTERPIECE_STYLES = {
        
        # ===== 1. Les Demoiselles d'Avignon (1907) =====
        "demoiselles_avignon": {
            "name": "亚维农少女",
            "year": 1907,
            "description": "原始立体主义开山之作，非洲面具影响",
            "prompt": (
                "Les Demoiselles d'Avignon style by Pablo Picasso 1907, "
                "African mask influence on face, Iberian sculpture features, "
                "angular distorted face with sharp geometric planes, "
                "primitive art aesthetic, bold black outlines, "
                "fractured perspective showing multiple angles simultaneously, "
                "earthy pink ochre terracotta palette with blue accents, "
                "revolutionary proto-cubism, raw primitive power, "
                "museum masterpiece oil painting, thick bold brushstrokes"
            ),
            "negative": "realistic, smooth, soft, photographic, anime style, cute, pretty",
            "strength": 0.88,
            "guidance": 12.0,
        },
        
        # ===== 2. Guernica (1937) =====
        "guernica": {
            "name": "格尔尼卡",
            "year": 1937,
            "description": "反战巨作，灰黑白单色，碎片化恐惧",
            "prompt": (
                "Guernica style by Pablo Picasso 1937, "
                "monochromatic grey black white palette only, "
                "fragmented anguished figures, screaming faces, "
                "sharp angular geometric distortion, "
                "horror and chaos of war, broken bodies, "
                "newspaper print texture, harsh lighting contrasts, "
                "cubist fragmentation, emotional devastation, "
                "political protest art, museum scale masterpiece"
            ),
            "negative": "colorful, happy, peaceful, realistic, soft, anime",
            "strength": 0.90,
            "guidance": 14.0,
        },
        
        # ===== 3. La Femme qui pleure (1937) =====
        "weeping_woman": {
            "name": "哭泣的女人",
            "year": 1937,
            "description": "Dora Maar肖像，极度情感表达",
            "prompt": (
                "La Femme qui pleure Weeping Woman by Pablo Picasso 1937, "
                "face fragmented into sharp angular colorful shards, "
                "tears streaming down geometric fractured cheeks, "
                "vivid intense colors yellow green red blue purple, "
                "anguished expression, handkerchief pressed to face, "
                "psychological intensity, emotional devastation, "
                "cubist portrait with recognizable grief, "
                "bold black outlines, thick impasto brushwork, "
                "Dora Maar inspired, museum quality masterpiece"
            ),
            "negative": "calm, happy, realistic, soft, muted colors, anime",
            "strength": 0.85,
            "guidance": 11.0,
        },
        
        # ===== 4. Le Rêve (1932) =====
        "le_reve": {
            "name": "梦",
            "year": 1932,
            "description": "Marie-Thérèse肖像，分裂面孔，柔和曲线",
            "prompt": (
                "Le Reve The Dream by Pablo Picasso 1932, "
                "face split into two halves profile and frontal view, "
                "soft sensual curved lines, voluptuous rounded forms, "
                "warm pink red yellow palette, peaceful sleeping expression, "
                "Marie-Therese Walter style blonde beauty, "
                "surrealist dreamlike quality, erotic undertones, "
                "simplified bold shapes, thick black outlines, "
                "intimate portrait, museum masterpiece oil painting"
            ),
            "negative": "angular, harsh, realistic photo, dark, sad, anime",
            "strength": 0.82,
            "guidance": 10.0,
        },
        
        # ===== 5. Dora Maar au Chat (1941) =====
        "dora_maar_cat": {
            "name": "多拉·玛尔与猫",
            "year": 1941,
            "description": "几何肖像，强烈色彩对比，心理深度",
            "prompt": (
                "Dora Maar au Chat by Pablo Picasso 1941, "
                "seated woman portrait with small cat on shoulder, "
                "face shown from multiple angles simultaneously, "
                "angular geometric cubist fragmentation, "
                "vibrant contrasting colors red green blue yellow, "
                "penetrating intense gaze, psychological complexity, "
                "decorative patterned clothing and chair, "
                "bold black outlines, thick brushstrokes, "
                "powerful emotional portrait, museum masterpiece"
            ),
            "negative": "realistic, soft, photographic, simple, anime, cute",
            "strength": 0.85,
            "guidance": 11.0,
        },
        
        # ===== 6. Girl Before a Mirror (1932) =====
        "girl_mirror": {
            "name": "镜前少女",
            "year": 1932,
            "description": "镜像双重形象，圆形曲线，大胆色彩",
            "prompt": (
                "Girl Before a Mirror by Pablo Picasso 1932, "
                "woman and her mirror reflection showing dual nature, "
                "circular curved organic shapes, "
                "bold vivid colors purple green yellow red black, "
                "striped diamond wallpaper pattern background, "
                "face split showing youth and age simultaneously, "
                "Marie-Therese Walter inspired beauty, "
                "psychological depth, vanity theme, "
                "thick black outlines, decorative patterns, "
                "surrealist cubist masterpiece"
            ),
            "negative": "realistic, photographic, single view, muted, anime",
            "strength": 0.85,
            "guidance": 11.0,
        },
        
        # ===== 7. The Old Guitarist (1903) =====
        "old_guitarist": {
            "name": "老吉他手",
            "year": 1903,
            "description": "蓝色时期代表作，瘦长人物，深沉忧郁",
            "prompt": (
                "The Old Guitarist Blue Period by Pablo Picasso 1903-1904, "
                "monochromatic blue palette with subtle green undertones, "
                "elongated emaciated figure, El Greco influence, "
                "deep melancholy and poverty, blind musician, "
                "angular bony limbs, hunched posture, "
                "somber introspective mood, social outcasts theme, "
                "thin delicate brushwork, Barcelona period, "
                "emotional depth, museum masterpiece oil painting"
            ),
            "negative": "colorful, happy, healthy, realistic photo, anime, bright",
            "strength": 0.88,
            "guidance": 12.0,
        },
        
        # ===== 8. Three Musicians (1921) =====
        "three_musicians": {
            "name": "三个音乐家",
            "year": 1921,
            "description": "综合立体主义巅峰，平面几何拼贴",
            "prompt": (
                "Three Musicians Synthetic Cubism by Pablo Picasso 1921, "
                "flat geometric color planes like paper collage, "
                "bold primary colors red yellow blue brown black white, "
                "Harlequin Pierrot and Monk figures, "
                "playful decorative patterns, musical instruments, "
                "overlapping flat shapes, papier colle aesthetic, "
                "simplified abstracted forms, jigsaw puzzle composition, "
                "large scale monumental, museum masterpiece"
            ),
            "negative": "realistic, 3d depth, photographic, soft gradients, anime",
            "strength": 0.90,
            "guidance": 13.0,
        },
        
        # ===== 9. La Vie (1903) =====
        "la_vie": {
            "name": "生命",
            "year": 1903,
            "description": "蓝色时期巨作，生死主题，深沉象征",
            "prompt": (
                "La Vie Life by Pablo Picasso 1903 Blue Period, "
                "monochromatic blue palette, melancholic atmosphere, "
                "symbolic composition about life death destiny, "
                "elongated figures, tender embrace, "
                "mother and child, nude couple, "
                "existential themes, Barcelona poverty, "
                "somber contemplative mood, "
                "thin delicate brushwork, museum masterpiece"
            ),
            "negative": "colorful, happy, bright, realistic photo, anime",
            "strength": 0.88,
            "guidance": 12.0,
        },
        
        # ===== 10. Portrait of Dora Maar (多视角) =====
        "dora_maar_portrait": {
            "name": "多拉·玛尔肖像",
            "year": 1937,
            "description": "多视角面孔，立体主义肖像巅峰",
            "prompt": (
                "Portrait of Dora Maar by Pablo Picasso 1937, "
                "face shown from multiple angles front and profile combined, "
                "angular geometric planes fragmenting the face, "
                "intense penetrating eyes from different viewpoints, "
                "vibrant colors green red yellow blue, "
                "psychological intensity and complexity, "
                "bold black outlines defining shapes, "
                "cubist deconstruction of portrait, "
                "thick expressive brushwork, museum masterpiece"
            ),
            "negative": "single viewpoint, realistic, soft, photographic, anime",
            "strength": 0.85,
            "guidance": 11.0,
        },
        
        # ===== 额外：分析立体主义 =====
        "analytical_cubism": {
            "name": "分析立体主义",
            "year": 1910,
            "description": "与布拉克共创，单色几何分解",
            "prompt": (
                "Analytical Cubism by Pablo Picasso and Georges Braque 1910-1912, "
                "monochromatic earth tones brown grey ochre beige, "
                "object fragmented into geometric faceted planes, "
                "multiple viewpoints shown simultaneously, "
                "overlapping transparent angular shapes, "
                "intellectual deconstruction of form, "
                "subtle tonal gradations, complex spatial ambiguity, "
                "Ma Jolie period, revolutionary avant-garde, "
                "museum quality masterpiece oil painting"
            ),
            "negative": "colorful, simple, realistic, clear forms, anime",
            "strength": 0.90,
            "guidance": 13.0,
        },
        
        # ===== 额外：玫瑰时期 =====
        "rose_period": {
            "name": "玫瑰时期",
            "year": 1905,
            "description": "温暖粉橙色调，马戏团主题",
            "prompt": (
                "Rose Period by Pablo Picasso 1904-1906, "
                "warm pink orange terracotta earth tones, "
                "circus performers harlequins acrobats, "
                "tender romantic melancholic atmosphere, "
                "elongated graceful figures, "
                "soft gentle brushwork, intimate scenes, "
                "Garcon a la pipe style, youth and innocence, "
                "transitional period, museum masterpiece"
            ),
            "negative": "blue, cold, harsh, geometric, cubist, anime",
            "strength": 0.82,
            "guidance": 10.0,
        },
    }
    
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pipe = None
        # 芙莉莲特征前缀 - 保留核心识别特征
        self.character_prefix = (
            "elf maiden with white silver hair, green emerald eyes, pointed elf ears, "
            "red teardrop earring, elegant fantasy character, "
        )
        self.negative_base = (
            "blurry, low quality, bad anatomy, extra limbs, "
            "text, watermark, signature, frame, border"
        )
    
    def load_model(self):
        """加载SDXL模型"""
        print("📦 加载SDXL模型...")
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
        ).to(self.device)
        print(f"✅ 模型加载完成 (device: {self.device})")
    
    def clear_memory(self):
        """清理内存"""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def generate_style(self, image, style_key, seed=None):
        """生成单个风格"""
        style = self.MASTERPIECE_STYLES[style_key]
        
        if seed is None:
            seed = random.randint(1, 2147483647)
        
        # 调整图像尺寸
        w, h = image.size
        new_w = min(1024, (w // 64) * 64)
        scale = new_w / w
        new_h = int(h * scale // 64) * 64
        new_h = min(new_h, 1024)
        resized = image.resize((new_w, new_h), Image.LANCZOS)
        
        # 构建prompt
        full_prompt = f"{self.character_prefix}{style['prompt']}"
        full_negative = f"{self.negative_base}, {style['negative']}"
        
        try:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            result = self.pipe(
                prompt=full_prompt,
                negative_prompt=full_negative,
                image=resized,
                strength=style["strength"],
                guidance_scale=style["guidance"],
                num_inference_steps=40,
                generator=generator,
            ).images[0]
        finally:
            self.clear_memory()
        
        return result, seed
    
    def to_svg(self, image, num_colors=64, simplify=0.003):
        """转换为SVG"""
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
        bg_color = centers[np.argmax(np.bincount(labels.flatten()))]
        dwg.add(dwg.rect(insert=(0, 0), size=(w, h), 
                        fill=f'rgb({bg_color[0]},{bg_color[1]},{bg_color[2]})'))
        
        polygon_count = 0
        for color in centers:
            mask = np.all(quantized == color, axis=2).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) < 3:
                    continue
                epsilon = simplify * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) < 3:
                    continue
                points = [(int(p[0][0]), int(p[0][1])) for p in approx]
                fill = f'rgb({color[0]},{color[1]},{color[2]})'
                dwg.add(dwg.polygon(points=points, fill=fill, stroke='none'))
                polygon_count += 1
        
        return dwg.tostring(), polygon_count
    
    def generate_gallery(self, image_path, output_dir, styles=None, count_per_style=3, save_png=True):
        """生成画廊"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image = Image.open(image_path).convert("RGB")
        
        if styles is None:
            styles = list(self.MASTERPIECE_STYLES.keys())
        
        print("=" * 70)
        print("🎨 毕加索名作风格生成器")
        print("=" * 70)
        print(f"📊 选择风格: {len(styles)} 种")
        print(f"📁 输出目录: {output_dir}")
        print(f"🔢 每种风格: {count_per_style} 张")
        print("=" * 70)
        
        for i, key in enumerate(styles):
            style = self.MASTERPIECE_STYLES[key]
            print(f"\n  {i+1}. {style['name']} ({style['year']}) - {style['description'][:30]}...")
        
        print("\n" + "=" * 70)
        
        results = []
        total = len(styles) * count_per_style
        current = 0
        
        for style_key in styles:
            style = self.MASTERPIECE_STYLES[style_key]
            print(f"\n{'='*60}")
            print(f"🖼️  {style['name']} ({style['year']})")
            print(f"   {style['description']}")
            print(f"{'='*60}")
            
            for var_idx in range(count_per_style):
                current += 1
                print(f"\n  [{current}/{total}] 生成中...")
                
                try:
                    styled_img, seed = self.generate_style(image, style_key)
                    
                    # 保存PNG
                    if save_png:
                        png_path = output_dir / f"{style_key}_v{var_idx+1:02d}.png"
                        styled_img.save(png_path, quality=95)
                        png_size = png_path.stat().st_size / 1024
                        print(f"     PNG: {png_path.name} ({png_size:.1f} KB)")
                    
                    # 转换SVG
                    svg_content, poly_count = self.to_svg(styled_img)
                    svg_path = output_dir / f"{style_key}_v{var_idx+1:02d}.svg"
                    with open(svg_path, 'w') as f:
                        f.write(svg_content)
                    svg_size = svg_path.stat().st_size / (1024 * 1024)
                    print(f"     SVG: {svg_path.name} ({svg_size:.2f} MB, {poly_count} polygons)")
                    
                    results.append({
                        "style_key": style_key,
                        "style_name": style["name"],
                        "year": style["year"],
                        "variant": var_idx + 1,
                        "seed": seed,
                        "strength": style["strength"],
                        "guidance": style["guidance"],
                        "svg_file": svg_path.name,
                        "png_file": png_path.name if save_png else None,
                        "polygons": poly_count,
                    })
                    
                except Exception as e:
                    print(f"     ❌ 错误: {e}")
                    import traceback
                    traceback.print_exc()
                
                if current % 3 == 0:
                    self.clear_memory()
        
        # 保存日志
        log_path = output_dir / "generation_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "created_at": datetime.now().isoformat(),
                "total_count": len(results),
                "styles_used": styles,
                "results": results,
            }, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 70)
        print(f"✅ 完成！生成了 {len(results)} 张作品")
        print(f"📁 位置: {output_dir}")
        print("=" * 70)
        
        return results


def main():
    input_image = "/Volumes/Seagate/SAM3/01_input/Picture1.jpg"
    output_dir = "/Volumes/Seagate/SAM3/06_style_art/output/picasso_masterpiece"
    
    if not Path(input_image).exists():
        print(f"❌ 找不到输入图像: {input_image}")
        return
    
    generator = PicassoMasterpieceGenerator()
    generator.load_model()
    
    # 选择最具代表性的名作风格
    selected_styles = [
        "demoiselles_avignon",   # 原始立体主义
        "guernica",              # 反战巨作
        "weeping_woman",         # 哭泣女人
        "le_reve",               # 梦
        "dora_maar_cat",         # 多拉与猫
        "girl_mirror",           # 镜前少女
        "old_guitarist",         # 老吉他手
        "three_musicians",       # 三个音乐家
        "dora_maar_portrait",    # 多拉肖像
        "analytical_cubism",     # 分析立体主义
    ]
    
    generator.generate_gallery(
        input_image, 
        output_dir, 
        styles=selected_styles,
        count_per_style=3,
        save_png=True
    )
    
    import subprocess
    subprocess.run(["open", output_dir])


if __name__ == "__main__":
    main()
