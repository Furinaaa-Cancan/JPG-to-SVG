"""
毕加索具象化风格生成器 - 仰视角度 + 芙莉莲面部特征

特点：
1. 保留人物具象特征（不过度抽象）
2. 仰视角度（from below, low angle）
3. 面部风格贴近芙莉莲（精灵耳朵、白发、绿眼）
4. 立体主义几何化但保持可识别性

python picasso_figurative.py
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


class PicassoFigurativeGenerator:
    """毕加索具象化风格生成器 - 仰视 + 芙莉莲特征"""
    
    # 具象化毕加索风格定义
    FIGURATIVE_STYLES = {
        "picasso_portrait_figurative": {
            "name": "毕加索具象肖像",
            "period": "1920s-1930s",
            "prompt": (
                "Pablo Picasso figurative portrait painting, recognizable human features, "
                "low angle view from below looking up, dramatic upward perspective, "
                "Frieren elf character style, white silver hair, green emerald eyes, pointed elf ears, "
                "geometric cubist structure but maintaining facial recognition, "
                "bold angular planes with clear facial features, "
                "synthetic cubism color blocks, vibrant colors red yellow blue green, "
                "expressive eyes and face, elegant composition, "
                "1920s neoclassical Picasso period, museum quality masterpiece, "
                "detailed brushwork, high resolution fine art"
            ),
            "negative": "abstract, unrecognizable, blurry, photorealistic, smooth, top view, bird's eye view"
        },
        "picasso_rose_period_figurative": {
            "name": "玫瑰时期具象风格",
            "period": "1904-1906",
            "prompt": (
                "Pablo Picasso Rose Period portrait, warm pink orange earth tones, "
                "low angle upward view from below, looking up at subject, "
                "Frieren elf features, white hair, green eyes, elf ears visible, "
                "tender romantic atmosphere with geometric structure, "
                "soft cubist planes, recognizable beautiful face, "
                "circus performer elegance, graceful composition, "
                "detailed facial features, expressive green eyes, "
                "masterpiece painting, intricate details"
            ),
            "negative": "abstract, distorted beyond recognition, dark, monochrome, top view, flat angle"
        },
        "picasso_neoclassical": {
            "name": "新古典主义毕加索",
            "period": "1918-1925",
            "prompt": (
                "Pablo Picasso neoclassical period portrait, monumental classical beauty, "
                "dramatic low angle perspective from below, heroic upward view, "
                "Frieren character, white silver hair flowing, piercing green eyes, elegant elf ears, "
                "sculptural volumetric forms, clear recognizable features, "
                "Greek Roman statue influence with cubist geometry, "
                "powerful presence, majestic composition, "
                "detailed realistic face with geometric structure, "
                "museum quality fine art masterpiece"
            ),
            "negative": "abstract, unrecognizable, blurry, modern, top view, downward angle"
        },
        "picasso_blue_figurative": {
            "name": "蓝色时期具象",
            "period": "1901-1904",
            "prompt": (
                "Pablo Picasso Blue Period portrait, melancholic blue palette, "
                "low angle view looking up, upward perspective from below, "
                "Frieren elf maiden, white hair, sad green eyes, pointed ears, "
                "elongated elegant forms, recognizable sorrowful face, "
                "geometric cubist structure with emotional depth, "
                "deep blues grays with green eye accents, "
                "tragic beauty, clear facial features, "
                "Barcelona early period masterpiece"
            ),
            "negative": "bright, colorful, happy, abstract beyond recognition, top view"
        },
        "picasso_synthetic_cubism_figurative": {
            "name": "综合立体主义具象",
            "period": "1912-1919",
            "prompt": (
                "Pablo Picasso synthetic cubism portrait, bold flat geometric color shapes, "
                "dramatic upward angle from below, low perspective looking up, "
                "Frieren elf character, white hair geometric blocks, green eyes prominent, elf ears angular, "
                "collage aesthetic with recognizable face, decorative patterns, "
                "bright vibrant colors, clear facial structure, "
                "playful reconstructed but identifiable features, "
                "modern art masterpiece, detailed composition"
            ),
            "negative": "completely abstract, unrecognizable, realistic photo, top view, bird's eye"
        },
        "picasso_weeping_woman_figurative": {
            "name": "哭泣女人具象风格",
            "period": "1937",
            "prompt": (
                "Pablo Picasso Weeping Woman style portrait, intense emotional expression, "
                "low angle upward view from below, dramatic perspective, "
                "Frieren elf crying, white hair fragmented, green tears streaming from eyes, pointed ears visible, "
                "geometric angular features but clearly recognizable face, "
                "vibrant colors green yellow red blue, psychological intensity, "
                "expressive eyes and mouth, clear emotional features, "
                "cubist distortion maintaining identity, masterpiece painting"
            ),
            "negative": "calm, abstract beyond recognition, realistic, smooth, top view"
        },
        "picasso_dora_maar_style": {
            "name": "朵拉·玛尔肖像风格",
            "period": "1937-1944",
            "prompt": (
                "Pablo Picasso Dora Maar portrait style, multiple viewpoints of face, "
                "low angle perspective from below looking up, upward dramatic view, "
                "Frieren elf beauty, white silver hair, intense green eyes both front and profile, elf ears from multiple angles, "
                "simultaneous perspectives but recognizable identity, "
                "bold colors, geometric facial planes, clear features, "
                "psychological depth, elegant composition, "
                "museum quality masterpiece, detailed brushwork"
            ),
            "negative": "single viewpoint, completely abstract, photorealistic, top view, downward angle"
        },
        "picasso_three_musicians_figurative": {
            "name": "三个音乐家具象风格",
            "period": "1921",
            "prompt": (
                "Pablo Picasso Three Musicians style portrait, flat geometric color planes, "
                "upward angle from below, low perspective looking up, "
                "Frieren elf musician, white hair bold shapes, green eyes striking, elf ears geometric, "
                "primary colors red yellow blue black white, recognizable face, "
                "decorative patterns, collage aesthetic with clear identity, "
                "large scale composition, simplified but identifiable features, "
                "masterpiece fine art, intricate details"
            ),
            "negative": "realistic, 3d render, abstract unrecognizable, top view, flat angle"
        },
    }
    
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pipe = None
        self.guidance_scale = 10.0
        self.prompt_prefix = "stylized, painterly, cubist, figurative, recognizable"
        self.negative_prefix = "photorealistic, realistic, photo, 3d render, smooth, completely abstract, unrecognizable"
    
    def load_sd(self):
        """加载SDXL高质量模型"""
        print("📦 加载SDXL高质量模型...")
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        # 使用本地SDXL-Turbo模型（有完整VAE）
        local_path = "/Volumes/Seagate/SAM3/models/stable_diffusion/base_models/sdxl-turbo"
        
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            local_path,
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=True,
        ).to(self.device)
        print(f"✅ 模型就绪 (本地: {local_path})")
    
    def clear_memory(self):
        """清理GPU内存"""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def stylize(self, image, prompt, negative, strength, seed, steps=30):
        """具象化风格化"""
        w, h = image.size
        new_w = min(1280, (w // 64) * 64)
        scale = new_w / w
        new_h = int(h * scale // 64) * 64
        new_h = min(new_h, 1280)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        try:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            result = self.pipe(
                prompt=f"{self.prompt_prefix}, {prompt}",
                negative_prompt=f"{self.negative_prefix}, {negative}",
                image=image,
                strength=strength,
                guidance_scale=self.guidance_scale,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]
        finally:
            self.clear_memory()
        
        return result
    
    def to_svg_high_quality(self, image, num_colors=80, simplify=0.002):
        """高质量SVG转换"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        pixels = img_array.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = centers.astype(np.uint8)
        quantized = centers[labels.flatten()].reshape(img_array.shape)
        
        dwg = svgwrite.Drawing(size=(w, h))
        bg_color = centers[np.argmax(np.bincount(labels.flatten()))]
        dwg.add(dwg.rect(insert=(0, 0), size=(w, h), 
                        fill=f'rgb({bg_color[0]},{bg_color[1]},{bg_color[2]})'))
        
        total_polygons = 0
        
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
                total_polygons += 1
        
        print(f"   生成 {total_polygons} 个多边形")
        return dwg.tostring()
    
    def generate_figurative(self, image_path, output_dir, count_per_style=5):
        """生成具象化毕加索风格"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image = Image.open(image_path).convert("RGB")
        
        # 固定最佳参数 - 允许风格大幅改变
        strength = 0.85  # 提高强度，允许面部风格化变形
        num_colors = 80
        simplify = 0.002
        
        print("=" * 70)
        print("🎨 毕加索风格生成器 - 仰视芙莉莲 + 风格化变形")
        print("=" * 70)
        print(f"📊 艺术流派: {len(self.FIGURATIVE_STYLES)} 种")
        print(f"📁 输出目录: {output_dir}")
        print(f"🎯 目标: 仰视角度 + 芙莉莲特征 + 毕加索风格化")
        print(f"🎨 固定参数: strength={strength}, colors={num_colors}, simplify={simplify}")
        print(f"💡 允许面部风格大幅改变，保留白发绿眼精灵耳")
        print("=" * 70)
        print("\n包含的具象化流派：")
        for i, (key, style) in enumerate(self.FIGURATIVE_STYLES.items(), 1):
            print(f"  {i}. {style['name']} ({style['period']})")
        print("=" * 70)
        
        log_data = []
        generated = 0
        
        for style_idx, (style_key, style) in enumerate(self.FIGURATIVE_STYLES.items()):
            print(f"\n{'='*60}")
            print(f"🎨 [{style_idx+1}/{len(self.FIGURATIVE_STYLES)}] {style['name']}")
            print(f"   时期: {style['period']}")
            print(f"{'='*60}")
            
            for var_idx in range(count_per_style):
                version = generated + 1
                seed = random.randint(1, 2147483647)
                
                print(f"\n  [{var_idx+1}/{count_per_style}] picasso_fig_v{version:03d}")
                print(f"     风格: {style['name']}")
                print(f"     参数: strength={strength}, colors={num_colors}, simplify={simplify}")
                print(f"     seed: {seed}")
                
                try:
                    styled = self.stylize(
                        image,
                        style["prompt"],
                        style["negative"],
                        strength,
                        seed,
                        steps=30
                    )
                    
                    svg_content = self.to_svg_high_quality(styled, num_colors, simplify)
                    
                    svg_path = output_dir / f"picasso_fig_v{version:03d}_{style_key}.svg"
                    with open(svg_path, 'w') as f:
                        f.write(svg_content)
                    
                    size_mb = svg_path.stat().st_size / (1024 * 1024)
                    print(f"     ✅ {svg_path.name} ({size_mb:.2f} MB)")
                    
                    log_entry = {
                        "version": version,
                        "style_key": style_key,
                        "style_name": style["name"],
                        "period": style["period"],
                        "seed": seed,
                        "strength": strength,
                        "num_colors": num_colors,
                        "simplify": simplify,
                        "file_size_mb": round(size_mb, 2),
                        "svg_file": svg_path.name,
                        "generated_at": datetime.now().isoformat(),
                    }
                    log_data.append(log_entry)
                    
                except Exception as e:
                    print(f"     ❌ 错误: {e}")
                    import traceback
                    traceback.print_exc()
                    self.clear_memory()
                
                generated += 1
                
                if generated % 3 == 0:
                    self.clear_memory()
                    print(f"     🧹 内存已清理 (已生成{generated}张)")
        
        log_path = output_dir / "picasso_figurative_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "created_at": datetime.now().isoformat(),
                "total_count": generated,
                "styles_count": len(self.FIGURATIVE_STYLES),
                "settings": {
                    "strength": strength,
                    "num_colors": num_colors,
                    "simplify": simplify,
                    "guidance_scale": self.guidance_scale,
                    "features": "仰视角度 + 芙莉莲面部特征 + 具象化"
                },
                "files": log_data
            }, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 70)
        print(f"✅ 完成！生成了 {generated} 个具象化毕加索SVG")
        print(f"📁 位置: {output_dir}")
        print(f"📋 日志: {log_path}")
        print("=" * 70)
        
        return generated


def main():
    input_image = "/Volumes/Seagate/SAM3/01_input/Picture1.jpg"
    output_dir = "/Volumes/Seagate/SAM3/06_style_art/output/picasso_figurative"
    
    if not Path(input_image).exists():
        print(f"❌ 找不到: {input_image}")
        return
    
    generator = PicassoFigurativeGenerator()
    generator.load_sd()
    generator.generate_figurative(input_image, output_dir, count_per_style=1)  # 先每种1张测试
    
    import subprocess
    subprocess.run(["open", output_dir])


if __name__ == "__main__":
    main()
