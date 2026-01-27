"""
批量生成立体主义风格SVG - 25个变体
包含多种立体主义流派：分析立体主义、综合立体主义、奥菲斯主义、立体未来主义等
"""

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import svgwrite


class CubismBatch:
    """批量立体主义风格生成"""

    # 不同的立体主义风格prompt变体
    PROMPTS = [
        # 1. 分析立体主义 - 毕加索/布拉克早期
        (
            "analytical cubism masterpiece, portrait fragmented into geometric planes, "
            "multiple perspectives shown simultaneously, Pablo Picasso Georges Braque style, "
            "monochromatic earth tones brown gray ochre palette, "
            "overlapping angular planes intersecting shapes, broken spatial forms"
        ),
        # 2. 综合立体主义 - 更明亮、拼贴感
        (
            "synthetic cubism artwork, bold geometric color blocks, bright vibrant accents, "
            "collage papier colle aesthetic, simplified playful shapes, "
            "flat overlapping planes, Pablo Picasso later period, "
            "decorative patterns mixed media texture"
        ),
        # 3. 奥菲斯主义 - 德劳内的彩色圆形
        (
            "orphism abstract art, Robert Delaunay Sonia Delaunay style, "
            "colorful concentric circles, prismatic color wheels, "
            "rhythmic circular forms, vibrant rainbow palette, "
            "simultaneous contrast, lyrical abstraction, dynamic color movement"
        ),
        # 4. 立体未来主义 - 俄罗斯先锋派
        (
            "cubo-futurism Russian avant-garde, Kazimir Malevich Natalia Goncharova style, "
            "dynamic angular forms, bold primary colors, "
            "mechanical energy movement, suprematist elements, "
            "geometric abstraction, revolutionary composition"
        ),
        # 5. 费尔南·莱热风格 - 机械立体主义
        (
            "tubism Fernand Leger style, cylindrical mechanical forms, "
            "bold black outlines, primary colors red blue yellow, "
            "industrial modern aesthetic, robotic figures, "
            "smooth tubular shapes, machine age modernism"
        ),
        # 6. 胡安·格里斯风格 - 精确立体主义
        (
            "Juan Gris crystal cubism, precise geometric composition, "
            "interlocking colored planes, still life abstraction, "
            "harmonious color relationships, mathematical precision, "
            "transparent overlapping forms, refined elegant cubism"
        ),
        # 7. 阿尔贝特·格莱兹风格
        (
            "Albert Gleizes cubism, dynamic rhythmic composition, "
            "swirling geometric forms, spiritual abstraction, "
            "muted sophisticated palette, monumental scale feeling, "
            "interlocking curved and angular planes"
        ),
    ]

    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pipe = None

    def load_sd(self):
        print("📦 加载SDXL...")
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
        ).to(self.device)
        print("✅ 模型就绪")

    def stylize(self, image, prompt, strength, seed):
        w, h = image.size
        new_w = min(1024, (w // 64) * 64)
        scale = new_w / w
        new_h = int(h * scale // 64) * 64
        image = image.resize((new_w, new_h), Image.LANCZOS)

        negative = (
            "blurry, ugly, realistic photo, 3d render, photorealistic, "
            "smooth gradients, soft focus, anime cartoon"
        )

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=image,
            strength=strength,
            guidance_scale=8.5,
            num_inference_steps=30,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        ).images[0]

        return result

    def to_svg(self, image, num_colors):
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

    def generate_batch(self, image_path, output_dir, count=25):
        """生成25个变体"""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert("RGB")

        # 参数组合 - 生成25个不同变体
        variations = []

        # 7种prompt，每种3-4个参数组合
        # 使用更大的参数差异来创造更迥异的风格
        param_sets = [
            # 低strength保留更多原图
            {'strength': 0.45, 'num_colors': 12, 'seed_offset': 0},
            {'strength': 0.55, 'num_colors': 18, 'seed_offset': 100},
            {'strength': 0.65, 'num_colors': 24, 'seed_offset': 200},
            {'strength': 0.75, 'num_colors': 32, 'seed_offset': 300},
        ]

        base_seeds = [42, 777, 1234, 2048, 3333, 4096, 5555]

        idx = 0
        for i, prompt in enumerate(self.PROMPTS):
            # 每种风格生成3-4个变体
            num_variants = 4 if i < 4 else 3  # 前4种风格各4个，后3种各3个 = 25个
            for j in range(num_variants):
                params = param_sets[j % len(param_sets)]
                variations.append({
                    'prompt_idx': i,
                    'prompt': prompt,
                    'style_name': self._get_style_name(i),
                    'strength': params['strength'],
                    'num_colors': params['num_colors'],
                    'seed': base_seeds[i] + params['seed_offset'],
                })
                idx += 1
                if idx >= count:
                    break
            if idx >= count:
                break

        print(f"🎨 批量生成 {len(variations)} 个立体主义风格变体")
        print("=" * 60)
        print("风格包含：")
        print("  1. 分析立体主义 (Picasso/Braque)")
        print("  2. 综合立体主义 (Synthetic)")
        print("  3. 奥菲斯主义 (Delaunay)")
        print("  4. 立体未来主义 (Russian)")
        print("  5. 机械立体主义 (Léger)")
        print("  6. 精确立体主义 (Juan Gris)")
        print("  7. 格莱兹风格 (Gleizes)")
        print("=" * 60)

        for idx, var in enumerate(variations):
            version = idx + 1

            print(f"\n[{idx+1}/{len(variations)}] cubism_v{version}")
            print(f"   风格: {var['style_name']}")
            print(f"   strength: {var['strength']}")
            print(f"   colors: {var['num_colors']}")
            print(f"   seed: {var['seed']}")

            # 风格化
            styled = self.stylize(image, var['prompt'], var['strength'], var['seed'])

            # SVG
            svg_content = self.to_svg(styled, var['num_colors'])

            # 保存
            svg_path = output_dir / f"cubism_v{version}.svg"
            with open(svg_path, 'w') as f:
                f.write(svg_content)

            size_kb = svg_path.stat().st_size / 1024
            print(f"   ✅ {svg_path.name} ({size_kb:.0f} KB)")

        print("\n" + "=" * 60)
        print(f"✅ 完成！生成了 {len(variations)} 个立体主义SVG")
        print(f"   位置: {output_dir}")
        print("=" * 60)

    def _get_style_name(self, idx):
        names = [
            "分析立体主义",
            "综合立体主义", 
            "奥菲斯主义",
            "立体未来主义",
            "机械立体主义",
            "精确立体主义",
            "格莱兹风格",
        ]
        return names[idx] if idx < len(names) else f"变体{idx+1}"


def main():
    input_image = "/Volumes/Seagate/SAM3/01_input/Ladygaga_2.jpg"
    output_dir = "/Volumes/Seagate/SAM3/06_style_art/output/cubism_batch"

    if not Path(input_image).exists():
        print(f"❌ 找不到: {input_image}")
        return

    generator = CubismBatch()
    generator.load_sd()
    generator.generate_batch(input_image, output_dir, count=25)

    # 打开文件夹
    import subprocess
    subprocess.run(["open", output_dir])


if __name__ == "__main__":
    main()
