"""
批量生成杜尚风格SVG - 25个变体
"""

import torch
import gc
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import svgwrite
import json
import random
from datetime import datetime


class DuchampBatch:
    """批量杜尚风格生成"""

    STYLE_PRESETS = [
        {
            "name": "duchamp_motion",
            "prompt": "Marcel Duchamp Nude Descending a Staircase style, fragmented angular geometric forms, chronophotography motion study, multiple overlapping movement phases, cubist futurist masterpiece, dynamic diagonal rhythm, oil painting texture",
            "strength_range": (0.65, 0.85),
            "guidance_range": (7.0, 10.0),
            "steps_range": (22, 32),
            "controlnet_scale_range": (0.35, 0.75),
            "preserve_colors": True,
            "color_match_strength_range": (0.55, 0.9),
        },
        {
            "name": "cubism_picasso",
            "prompt": "analytical cubism portrait, Pablo Picasso and Georges Braque style, face fragmented into angular planes, multiple perspectives, faceted geometry, museum quality fine art, oil on canvas texture",
            "strength_range": (0.6, 0.82),
            "guidance_range": (7.5, 11.0),
            "steps_range": (22, 34),
            "controlnet_scale_range": (0.3, 0.7),
            "preserve_colors": True,
            "color_match_strength_range": (0.45, 0.8),
        },
        {
            "name": "futurism_italian",
            "prompt": "Italian futurism painting, speed lines, dynamic motion, fractured forms, luminous energy, rhythmic repetition, Boccioni style, bold diagonals, brush strokes",
            "strength_range": (0.68, 0.88),
            "guidance_range": (7.0, 10.0),
            "steps_range": (20, 30),
            "controlnet_scale_range": (0.25, 0.65),
            "preserve_colors": True,
            "color_match_strength_range": (0.4, 0.75),
        },
        {
            "name": "art_nouveau",
            "prompt": "art nouveau poster illustration, Alphonse Mucha style, elegant flowing curves, ornamental decorative patterns, delicate linework, stylized portrait, flat graphic design",
            "strength_range": (0.62, 0.82),
            "guidance_range": (7.0, 10.5),
            "steps_range": (22, 34),
            "controlnet_scale_range": (0.35, 0.8),
            "preserve_colors": True,
            "color_match_strength_range": (0.5, 0.85),
        },
        {
            "name": "pop_art",
            "prompt": "pop art silkscreen portrait, Andy Warhol style, bold graphic shapes, high contrast, halftone dots, screen print texture, posterized color blocks",
            "strength_range": (0.65, 0.88),
            "guidance_range": (7.5, 11.5),
            "steps_range": (20, 28),
            "controlnet_scale_range": (0.2, 0.6),
            "preserve_colors": True,
            "color_match_strength_range": (0.25, 0.6),
        },
        {
            "name": "expressionism",
            "prompt": "expressionist painting portrait, bold emotional brushwork, distorted forms, dramatic lighting, textured impasto strokes, Edvard Munch inspired",
            "strength_range": (0.7, 0.9),
            "guidance_range": (7.0, 10.5),
            "steps_range": (22, 32),
            "controlnet_scale_range": (0.25, 0.65),
            "preserve_colors": True,
            "color_match_strength_range": (0.35, 0.75),
        },
        {
            "name": "ink_wash",
            "prompt": "Chinese ink wash painting portrait, sumi-e brush, expressive ink strokes, paper texture, minimal shading, elegant abstraction",
            "strength_range": (0.7, 0.92),
            "guidance_range": (7.0, 10.0),
            "steps_range": (22, 34),
            "controlnet_scale_range": (0.45, 0.9),
            "preserve_colors": True,
            "color_match_strength_range": (0.65, 1.0),
        },
        {
            "name": "stained_glass",
            "prompt": "stained glass window artwork, leaded outlines, mosaic pieces, bright luminous glass texture, geometric segmentation, cathedral style",
            "strength_range": (0.65, 0.9),
            "guidance_range": (7.0, 10.5),
            "steps_range": (20, 30),
            "controlnet_scale_range": (0.45, 0.95),
            "preserve_colors": True,
            "color_match_strength_range": (0.35, 0.7),
        },
        {
            "name": "low_poly",
            "prompt": "low poly geometric portrait, triangulated facets, sharp polygon planes, clean edges, modern vector aesthetic",
            "strength_range": (0.6, 0.85),
            "guidance_range": (7.0, 10.0),
            "steps_range": (20, 28),
            "controlnet_scale_range": (0.55, 1.0),
            "preserve_colors": True,
            "color_match_strength_range": (0.5, 0.9),
        },
        {
            "name": "bauhaus_poster",
            "prompt": "Bauhaus poster design, geometric shapes, modernist composition, flat graphic forms, clean typography space, abstract poster layout",
            "strength_range": (0.65, 0.9),
            "guidance_range": (7.5, 11.5),
            "steps_range": (20, 28),
            "controlnet_scale_range": (0.25, 0.7),
            "preserve_colors": True,
            "color_match_strength_range": (0.25, 0.65),
        },
        {
            "name": "abstract_expressionism",
            "prompt": "abstract expressionist painting, energetic brush splatter, gestural strokes, layered paint, bold textures, Jackson Pollock inspired",
            "strength_range": (0.72, 0.92),
            "guidance_range": (6.5, 9.0),
            "steps_range": (20, 28),
            "controlnet_scale_range": (0.1, 0.5),
            "preserve_colors": True,
            "color_match_strength_range": (0.25, 0.65),
        },
    ]

    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pipe = None
        self.controlnet = None
        self.use_controlnet = False

    def load_sd(self, use_controlnet: bool = True):
        print("📦 加载SDXL...")
        dtype = torch.float16 if self.device != "cpu" else torch.float32

        local_sdxl_path = "/Volumes/Seagate/SAM3/models/stable_diffusion/base_models/sdxl-base"
        if Path(local_sdxl_path).exists():
            sdxl_id = local_sdxl_path
        else:
            sdxl_id = "stabilityai/stable-diffusion-xl-base-1.0"

        vae = None
        if sdxl_id == local_sdxl_path:
            vae_dir = Path(local_sdxl_path) / "vae"
            vae_has_pytorch = any(
                (vae_dir / fname).exists()
                for fname in [
                    "diffusion_pytorch_model.safetensors",
                    "diffusion_pytorch_model.fp16.safetensors",
                    "diffusion_pytorch_model.bin",
                ]
            )
            if not vae_has_pytorch and (Path(local_sdxl_path) / "vae_1_0" / "diffusion_pytorch_model.safetensors").exists():
                vae = AutoencoderKL.from_pretrained(
                    local_sdxl_path,
                    subfolder="vae_1_0",
                    torch_dtype=dtype,
                )

        if use_controlnet:
            local_controlnet_path = "/Volumes/Seagate/SAM3/models/stable_diffusion/base_models/controlnet-canny-sdxl-1.0"
            if Path(local_controlnet_path).exists():
                controlnet_id = local_controlnet_path
            else:
                controlnet_id = "diffusers/controlnet-canny-sdxl-1.0"

            try:
                self.controlnet = ControlNetModel.from_pretrained(
                    controlnet_id,
                    torch_dtype=dtype,
                )
                pipe_kwargs = {"torch_dtype": dtype}
                if vae is not None:
                    pipe_kwargs["vae"] = vae
                self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    sdxl_id,
                    controlnet=self.controlnet,
                    **pipe_kwargs,
                ).to(self.device)
                self.use_controlnet = True
            except Exception as e:
                print(f"⚠️ ControlNet加载失败，将退回纯SDXL img2img: {e}")
                self.controlnet = None
                pipe_kwargs = {"torch_dtype": dtype}
                if vae is not None:
                    pipe_kwargs["vae"] = vae
                self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    sdxl_id,
                    **pipe_kwargs,
                ).to(self.device)
                self.use_controlnet = False
        else:
            pipe_kwargs = {"torch_dtype": dtype}
            if vae is not None:
                pipe_kwargs["vae"] = vae
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                sdxl_id,
                **pipe_kwargs,
            ).to(self.device)
            self.use_controlnet = False

        print("✅ 模型就绪")

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

    def stylize(
        self,
        image,
        prompt,
        strength,
        seed,
        controlnet_scale: float = 0.5,
        guidance_scale: float = 8.0,
        num_inference_steps: int = 30,
        preserve_colors: bool = True,
        color_match_strength: float = 1.0,
        negative: str = None,
    ):
        w, h = image.size
        new_w = min(1024, (w // 64) * 64)
        scale = new_w / w
        new_h = int(h * scale // 64) * 64
        image = image.resize((new_w, new_h), Image.LANCZOS)

        negative_prompt = (
            "blurry, ugly, photorealistic, realistic photo, 3d render, smooth, soft focus, sepia, monochrome"
            if negative is None
            else negative
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)

        try:
            if self.use_controlnet:
                canny_image = self.extract_canny(image, low=30, high=100)
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
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
                    prompt=prompt,
                    negative_prompt=negative_prompt,
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

    def _save_jpeg_size_capped(self, image: Image.Image, out_path: Path, max_kb: int = 2000) -> dict:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = image.convert("RGB")
        quality = 95
        step = 3
        min_quality = 70

        while True:
            img.save(str(out_path), format="JPEG", quality=quality, optimize=True)
            size_kb = out_path.stat().st_size / 1024
            if size_kb <= max_kb or quality <= min_quality:
                return {"quality": quality, "size_kb": size_kb}
            quality -= step

    def generate_gallery(
        self,
        image_path: str,
        output_dir: str,
        count: int = 100,
        use_controlnet: bool = True,
        max_file_kb: int = 2000,
    ) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"gallery_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if self.pipe is None:
            self.load_sd(use_controlnet=use_controlnet)

        records = []

        print(f"🎨 生成候选图库: {count} 张")
        print(f"   输出: {run_dir}")
        print("=" * 60)

        for idx in range(count):
            preset = random.choice(self.STYLE_PRESETS)
            seed = int(np.random.randint(1, 2147483647))

            strength = float(random.uniform(*preset["strength_range"]))
            guidance_scale = float(random.uniform(*preset["guidance_range"]))
            num_inference_steps = int(random.randint(*preset["steps_range"]))
            controlnet_scale = float(random.uniform(*preset["controlnet_scale_range"]))
            preserve_colors = bool(preset.get("preserve_colors", True))
            cms_range = preset.get("color_match_strength_range", (0.6, 1.0))
            color_match_strength = float(random.uniform(*cms_range))

            prompt = preset["prompt"]
            if preserve_colors:
                prompt = f"{prompt}, preserve original colors, keep original color palette"
                negative = "sepia, monochrome, grayscale"
            else:
                negative = None

            print(f"\n[{idx+1}/{count}] {preset['name']}")
            print(f"   strength={strength:.2f}  guidance={guidance_scale:.1f}  steps={num_inference_steps}  controlnet={controlnet_scale:.2f}")
            print(f"   seed={seed}  preserve_colors={preserve_colors}  color_match_strength={color_match_strength:.2f}")

            styled = self.stylize(
                image,
                prompt,
                strength,
                seed,
                controlnet_scale=controlnet_scale,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                preserve_colors=preserve_colors,
                color_match_strength=color_match_strength,
                negative=negative,
            )

            out_name = f"{idx+1:03d}_{preset['name']}_{seed}.jpg"
            out_path = run_dir / out_name
            save_meta = self._save_jpeg_size_capped(styled, out_path, max_kb=max_file_kb)
            print(f"   ✅ {out_name} ({save_meta['size_kb']:.0f} KB, q={save_meta['quality']})")

            records.append(
                {
                    "file": out_name,
                    "style": preset["name"],
                    "seed": seed,
                    "strength": strength,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps,
                    "controlnet_scale": controlnet_scale,
                    "preserve_colors": preserve_colors,
                    "color_match_strength": color_match_strength,
                    "prompt": prompt,
                }
            )

        index_path = run_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print(f"✅ 完成！输出: {run_dir}")
        print(f"   index: {index_path}")
        print("=" * 60)

        return {"dir": str(run_dir), "index": str(index_path), "count": count}

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
        dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill=f'rgb({bg_color[0]},{bg_color[1]},{bg_color[2]})'))

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

        # 5种prompt × 5种参数组合 = 25个
        strengths = [0.45, 0.50, 0.55, 0.60, 0.65]
        colors = [64, 80, 96, 112, 128]
        seeds = [42, 123, 456, 789, 1024]

        for i, prompt in enumerate(self.PROMPTS):
            for j in range(5):
                variations.append({
                    'prompt_idx': i,
                    'prompt': prompt,
                    'strength': strengths[j],
                    'num_colors': colors[j],
                    'seed': seeds[j] + i * 100,
                })

        print(f"🎨 批量生成 {len(variations)} 个杜尚风格变体")
        print("=" * 60)

        for idx, var in enumerate(variations):
            version = idx + 3  # 从v3开始（v1, v2已存在）

            print(f"\n[{idx+1}/{len(variations)}] v{version}")
            print(f"   prompt: #{var['prompt_idx']+1}")
            print(f"   strength: {var['strength']}")
            print(f"   colors: {var['num_colors']}")
            print(f"   seed: {var['seed']}")

            # 风格化
            styled = self.stylize(
                image,
                var['prompt'],
                var['strength'],
                var['seed'],
                controlnet_scale=0.5,
                guidance_scale=8.0,
                num_inference_steps=30,
                preserve_colors=True,
                color_match_strength=1.0,
            )

            # SVG
            svg_content = self.to_svg(styled, var['num_colors'])

            # 保存
            svg_path = output_dir / f"duchamp_v{version}.svg"
            with open(svg_path, 'w') as f:
                f.write(svg_content)
            
            size_kb = svg_path.stat().st_size / 1024
            print(f"   ✅ {svg_path.name} ({size_kb:.0f} KB)")
        
        print("\n" + "=" * 60)
        print(f"✅ 完成！生成了 {len(variations)} 个SVG")
        print(f"   位置: {output_dir}")
        print("=" * 60)


def main():
    input_image = "/Volumes/Seagate/SAM3/01_input/Ladygaga_2.jpg"
    output_dir = "/Volumes/Seagate/SAM3/06_style_art/output/sd_gallery"
    
    if not Path(input_image).exists():
        print(f"❌ 找不到: {input_image}")
        return
    
    generator = DuchampBatch()
    generator.generate_gallery(
        input_image,
        output_dir,
        count=100,
        use_controlnet=True,
        max_file_kb=2000,
    )
    
    # 打开文件夹
    import subprocess
    subprocess.run(["open", output_dir])


if __name__ == "__main__":
    main()
