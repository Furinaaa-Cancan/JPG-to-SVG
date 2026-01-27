#!/usr/bin/env python3
"""
SAM3直接矢量化
直接使用SAM3分割 -> SVG（一步完成）
"""

import sys
import cv2
import numpy as np
from PIL import Image
import svgwrite
from pathlib import Path
import time

sys.path.insert(0, "/Volumes/Seagate/SAM3/模型库/01_SAM3核心模型")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3DirectSVG:
    """SAM3直接转SVG"""
    
    def __init__(self):
        print("\n🚀 Initializing SAM3...")
        self.model = build_sam3_image_model(device="cpu")
        self.processor = Sam3Processor(self.model, device="cpu", confidence_threshold=0.1)
        print("✅ Ready!")
    
    def convert(self, image_path: str, output_dir: str = "02_输出结果/sam3_svg"):
        """一步完成：SAM3分割 + SVG生成"""
        
        print("\n" + "="*70)
        print("💎 SAM3 DIRECT TO SVG")
        print("="*70)
        
        start_time = time.time()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载图像
        img = Image.open(image_path)
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        print(f"\n📷 Image: {image_path}")
        print(f"   Size: {w}x{h}")
        
        # SAM3分割
        print("\n🔍 SAM3 Segmentation...")
        state = self.processor.set_image(img)
        
        # 收集所有masks
        all_regions = []
        prompts = self.get_prompts()
        
        for i, prompt in enumerate(prompts):
            try:
                self.processor.reset_all_prompts(state)
                prompt_state = self.processor.set_text_prompt(prompt, state)
                
                if prompt_state and 'masks' in prompt_state:
                    masks = prompt_state['masks']
                    if masks is not None:
                        masks_np = masks.cpu().numpy() if hasattr(masks, 'cpu') else np.array(masks)
                        
                        for j in range(masks_np.shape[0]):
                            mask = masks_np[j]
                            if len(mask.shape) > 2:
                                mask = mask.squeeze()
                            
                            # 确保尺寸正确
                            if mask.shape != (h, w):
                                mask = cv2.resize(mask.astype(np.float32), (w, h))
                            
                            # 二值化
                            binary_mask = (mask > 0.5).astype(np.uint8) * 255
                            
                            # 提取颜色
                            pixels = img_array[binary_mask > 127]
                            if len(pixels) > 10:
                                mean_color = np.mean(pixels, axis=0).astype(int)
                                color = f"#{mean_color[0]:02x}{mean_color[1]:02x}{mean_color[2]:02x}"
                                
                                area = np.sum(binary_mask > 0)
                                if area > 100:
                                    all_regions.append({
                                        'mask': binary_mask,
                                        'color': color,
                                        'area': area,
                                        'prompt': prompt
                                    })
                
                if (i + 1) % 20 == 0:
                    print(f"   Progress: {i+1}/{len(prompts)}, regions: {len(all_regions)}")
                    
            except:
                pass
        
        print(f"\n   Total regions: {len(all_regions)}")
        
        # 生成SVG
        print("\n✨ Generating SVG...")
        svg_path = output_path / "sam3_vector.svg"
        stats = self.create_svg(all_regions, w, h, str(svg_path))
        
        # 对比HTML
        self.create_html(image_path, str(svg_path), output_path, stats)
        
        process_time = time.time() - start_time
        
        print(f"\n" + "="*70)
        print(f"✅ COMPLETE!")
        print(f"   Regions: {len(all_regions)}")
        print(f"   Paths: {stats['paths']}")
        print(f"   Size: {stats['size_kb']:.1f} KB")
        print(f"   Time: {process_time:.1f}s")
        print("="*70)
        
        import subprocess
        subprocess.run(["open", str(output_path / "result.html")])
        
        return stats
    
    def get_prompts(self):
        """简化的高效提示词"""
        return [
            # 主要对象
            "person", "woman", "singer",
            "blue dress", "blue costume", "dress", "costume",
            "skeleton", "skeleton prop", "bones", "skull",
            "background", "stage",
            
            # 服装细节
            "gold decoration", "gold trim", "embroidery",
            "button", "buckle", "belt",
            "sleeve", "collar", "shoulder",
            "dress skirt", "hem", "fold",
            
            # 人物
            "face", "head", "hair", "blonde hair",
            "eye", "lips", "mouth", "nose",
            "hand", "arm", "finger", "skin",
            
            # 骷髅细节
            "rib cage", "spine", "skull head",
            "arm bone", "leg bone", "hand bones",
            
            # 其他
            "white", "black", "gold", "blue area",
            "shadow", "highlight", "texture",
        ]
    
    def create_svg(self, regions: list, width: int, height: int, output_path: str) -> dict:
        """创建SVG"""
        
        dwg = svgwrite.Drawing(output_path, size=(width, height))
        dwg.viewbox(0, 0, width, height)
        
        # 按面积排序
        regions.sort(key=lambda x: x['area'], reverse=True)
        
        paths = 0
        
        for region in regions:
            mask = region['mask']
            color = region['color']
            
            # 找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 50:
                    continue
                
                # 简化
                approx = cv2.approxPolyDP(contour, 1.5, True)
                
                if len(approx) >= 3:
                    points = approx.squeeze()
                    if points.ndim == 1:
                        points = points.reshape(-1, 2)
                    
                    if len(points) < 3:
                        continue
                    
                    # 路径
                    path_d = f"M{points[0][0]},{points[0][1]}"
                    for p in points[1:]:
                        path_d += f" L{p[0]},{p[1]}"
                    path_d += " Z"
                    
                    dwg.add(dwg.path(d=path_d, fill=color, stroke="none", opacity=0.95))
                    paths += 1
        
        dwg.save()
        
        return {
            'paths': paths,
            'size_kb': Path(output_path).stat().st_size / 1024
        }
    
    def create_html(self, original: str, svg: str, output_path: Path, stats: dict):
        """创建对比HTML"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SAM3 SVG Result</title>
            <style>
                body {{ margin:0; background:#111; color:#fff; font-family:sans-serif; }}
                .header {{ text-align:center; padding:40px; background:linear-gradient(135deg,#667eea,#764ba2); }}
                h1 {{ font-size:3em; margin:0; }}
                .stats {{ font-size:1.5em; margin-top:15px; color:#ffd700; }}
                .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; padding:40px; max-width:1600px; margin:0 auto; }}
                .card {{ background:#222; border-radius:15px; overflow:hidden; }}
                .card-header {{ padding:15px; background:#333; font-weight:bold; text-align:center; }}
                img, object {{ width:100%; display:block; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎨 SAM3 矢量化</h1>
                <div class="stats">{stats['paths']} 路径 | {stats['size_kb']:.0f} KB</div>
            </div>
            <div class="grid">
                <div class="card">
                    <div class="card-header">📷 原图</div>
                    <img src="../../{original}">
                </div>
                <div class="card">
                    <div class="card-header">✨ SVG</div>
                    <object data="{Path(svg).name}" type="image/svg+xml"></object>
                </div>
            </div>
        </body>
        </html>
        """
        with open(output_path / "result.html", 'w') as f:
            f.write(html)


def main():
    converter = SAM3DirectSVG()
    return converter.convert("01_输入图片/Ladygaga_2.jpg")


if __name__ == "__main__":
    main()
