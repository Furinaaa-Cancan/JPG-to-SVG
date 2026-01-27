#!/usr/bin/env python3
"""
分层矢量化
- 背景层：渐变处理
- 主体层：SAM3语义分割
- 细节层：精细颜色量化
"""

import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import svgwrite
import time
from sklearn.cluster import KMeans

sys.path.insert(0, "/Volumes/Seagate/SAM3/模型库/01_SAM3核心模型")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class LayeredVectorizer:
    """分层矢量化"""
    
    def __init__(self):
        print("\n🚀 Initializing Layered Vectorizer...")
        self.sam3_model = build_sam3_image_model(device="cpu")
        self.sam3_processor = Sam3Processor(self.sam3_model, device="cpu", confidence_threshold=0.1)
        print("✅ SAM3 loaded!")
    
    def vectorize(self, image_path: str, output_dir: str = "02_输出结果/layered_svg"):
        """分层矢量化"""
        
        print("\n" + "="*70)
        print("💎 LAYERED VECTORIZATION")
        print("="*70)
        
        start_time = time.time()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载图像
        img_pil = Image.open(image_path).convert("RGB")
        img = np.array(img_pil)
        h, w = img.shape[:2]
        
        print(f"\n📷 Image: {w}x{h}")
        
        all_elements = []
        
        # Layer 1: 背景渐变
        print("\n🌅 Layer 1: Background gradient...")
        bg_elements = self.create_background_gradient(img, h, w)
        all_elements.extend(bg_elements)
        print(f"   Background elements: {len(bg_elements)}")
        
        # Layer 2: SAM3主体分割
        print("\n🎯 Layer 2: SAM3 semantic segmentation...")
        sam3_elements = self.sam3_segment(img_pil, img, h, w)
        all_elements.extend(sam3_elements)
        print(f"   SAM3 elements: {len(sam3_elements)}")
        
        # Layer 3: 精细颜色量化
        print("\n🎨 Layer 3: Fine color quantization...")
        color_elements = self.fine_color_quantize(img, h, w)
        all_elements.extend(color_elements)
        print(f"   Color elements: {len(color_elements)}")
        
        # Layer 4: 边缘细节
        print("\n📐 Layer 4: Edge details...")
        edge_elements = self.edge_details(img, h, w)
        all_elements.extend(edge_elements)
        print(f"   Edge elements: {len(edge_elements)}")
        
        print(f"\n📊 Total elements: {len(all_elements)}")
        
        # 生成SVG
        print("\n✨ Generating SVG...")
        svg_path = output_path / "layered_vector.svg"
        stats = self.create_svg(all_elements, w, h, str(svg_path))
        
        self.create_html(image_path, str(svg_path), output_path, stats)
        
        process_time = time.time() - start_time
        
        print(f"\n" + "="*70)
        print(f"✅ LAYERED VECTORIZATION COMPLETE!")
        print(f"   Paths: {stats['paths']}")
        print(f"   Size: {stats['size_kb']:.1f} KB")
        print(f"   Time: {process_time:.1f}s")
        print("="*70)
        
        import subprocess
        subprocess.run(["open", str(output_path / "result.html")])
        
        return stats
    
    def create_background_gradient(self, img: np.ndarray, h: int, w: int) -> list:
        """创建背景 - 简单的整体背景色"""
        
        elements = []
        
        # 只创建一个简单的背景色（从边缘采样）
        edge_pixels = np.concatenate([
            img[0:20, :].reshape(-1, 3),
            img[-20:, :].reshape(-1, 3),
            img[:, 0:20].reshape(-1, 3),
            img[:, -20:].reshape(-1, 3),
        ])
        bg_color = np.mean(edge_pixels, axis=0).astype(int)
        
        # 整个画布作为背景
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        elements.append({
            'mask': mask,
            'color': f"#{bg_color[0]:02x}{bg_color[1]:02x}{bg_color[2]:02x}",
            'area': h * w,
            'layer': 'background',
            'order': -1  # 最底层
        })
        
        return elements
    
    def sam3_segment(self, img_pil: Image.Image, img: np.ndarray, h: int, w: int) -> list:
        """SAM3语义分割"""
        
        elements = []
        state = self.sam3_processor.set_image(img_pil)
        
        # 主要对象
        prompts = [
            ("person", 1),
            ("woman", 1),
            ("blue dress", 2),
            ("blue costume", 2),
            ("skeleton", 3),
            ("skull", 3),
            ("bones", 3),
            ("blonde hair", 4),
            ("wavy hair", 4),
            ("face", 5),
            ("hand", 6),
            ("arm", 6),
        ]
        
        for prompt, order in prompts:
            try:
                self.sam3_processor.reset_all_prompts(state)
                result = self.sam3_processor.set_text_prompt(prompt, state)
                
                if result and 'masks' in result and result['masks'] is not None:
                    masks = result['masks'].cpu().numpy()
                    
                    for mask in masks:
                        if len(mask.shape) > 2:
                            mask = mask.squeeze()
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask.astype(np.float32), (w, h))
                        
                        binary = (mask > 0.5).astype(np.uint8) * 255
                        area = np.sum(binary > 0)
                        
                        if area > 200:
                            pixels = img[binary > 127]
                            if len(pixels) > 0:
                                color = np.mean(pixels, axis=0).astype(int)
                                elements.append({
                                    'mask': binary,
                                    'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                                    'area': area,
                                    'layer': 'sam3',
                                    'order': order
                                })
            except:
                pass
        
        return elements
    
    def fine_color_quantize(self, img: np.ndarray, h: int, w: int) -> list:
        """精细颜色量化"""
        
        elements = []
        
        # 多级量化
        for n_colors in [128, 256]:
            # 使用LAB颜色空间
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            
            pixels = lab.reshape(-1, 3).astype(np.float32)
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=3, max_iter=30)
            labels = kmeans.fit_predict(pixels)
            
            label_img = labels.reshape(h, w)
            
            for cid in range(n_colors):
                mask = (label_img == cid).astype(np.uint8) * 255
                
                # 分解连通组件
                n_labels, labeled = cv2.connectedComponents(mask)
                
                for lid in range(1, min(n_labels, 30)):
                    component = (labeled == lid).astype(np.uint8) * 255
                    area = np.sum(component > 0)
                    
                    if 100 < area < h * w * 0.05:
                        pixels_rgb = img[component > 127]
                        if len(pixels_rgb) > 0:
                            color = np.mean(pixels_rgb, axis=0).astype(int)
                            elements.append({
                                'mask': component,
                                'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                                'area': area,
                                'layer': 'color',
                                'order': 10
                            })
        
        return elements
    
    def edge_details(self, img: np.ndarray, h: int, w: int) -> list:
        """边缘细节"""
        
        elements = []
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # 金色装饰
        gold_mask = cv2.inRange(hsv, np.array([15, 80, 100]), np.array([35, 255, 255]))
        gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        
        n_labels, labeled = cv2.connectedComponents(gold_mask)
        for lid in range(1, min(n_labels, 300)):
            m = (labeled == lid).astype(np.uint8) * 255
            area = np.sum(m > 0)
            if area > 15:
                pixels = img[m > 127]
                if len(pixels) > 0:
                    color = pixels[np.argmax(np.sum(pixels, axis=1))]
                    elements.append({
                        'mask': m,
                        'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                        'area': area,
                        'layer': 'detail',
                        'order': 20
                    })
        
        # 高亮
        _, highlight = cv2.threshold(hsv[:, :, 2], 230, 255, cv2.THRESH_BINARY)
        n_labels, labeled = cv2.connectedComponents(highlight)
        for lid in range(1, min(n_labels, 200)):
            m = (labeled == lid).astype(np.uint8) * 255
            area = np.sum(m > 0)
            if 10 < area < h * w * 0.005:
                pixels = img[m > 127]
                if len(pixels) > 0:
                    color = np.mean(pixels, axis=0).astype(int)
                    elements.append({
                        'mask': m,
                        'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                        'area': area,
                        'layer': 'detail',
                        'order': 21
                    })
        
        return elements
    
    def create_svg(self, elements: list, width: int, height: int, output_path: str) -> dict:
        """创建SVG"""
        
        dwg = svgwrite.Drawing(output_path, size=(width, height))
        dwg.viewbox(0, 0, width, height)
        
        # 按层级和面积排序
        elements.sort(key=lambda x: (x.get('order', 0), -x['area']))
        
        paths = 0
        
        for elem in elements:
            mask = elem['mask']
            color = elem['color']
            layer = elem.get('layer', 'unknown')
            
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 20:
                    continue
                
                # 根据层级设置简化程度
                if layer == 'background':
                    epsilon = 5.0
                elif layer == 'sam3':
                    epsilon = 2.0
                elif layer == 'detail':
                    epsilon = 1.0
                else:
                    epsilon = 1.5
                
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 3:
                    points = approx.squeeze()
                    if points.ndim == 1:
                        points = points.reshape(-1, 2)
                    if len(points) < 3:
                        continue
                    
                    path_d = self.smooth_path(points)
                    dwg.add(dwg.path(d=path_d, fill=color, stroke="none"))
                    paths += 1
        
        dwg.save()
        
        return {
            'paths': paths,
            'size_kb': Path(output_path).stat().st_size / 1024
        }
    
    def smooth_path(self, points: np.ndarray) -> str:
        """平滑路径"""
        
        if len(points) < 3:
            return ""
        
        path_d = f"M{points[0][0]},{points[0][1]}"
        
        for i in range(1, len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            cx, cy = p1[0], p1[1]
            ex, ey = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            path_d += f" Q{cx},{cy} {ex},{ey}"
        
        path_d += f" L{points[-1][0]},{points[-1][1]} Z"
        return path_d
    
    def create_html(self, original: str, svg: str, output_path: Path, stats: dict):
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Layered Vectorization</title>
            <style>
                body {{ margin:0; background:#0a0a0a; color:#fff; font-family:sans-serif; }}
                .header {{ text-align:center; padding:50px; background:linear-gradient(135deg,#667eea,#764ba2); }}
                h1 {{ font-size:3em; margin:0; }}
                .stats {{ display:flex; justify-content:center; gap:40px; margin-top:20px; }}
                .stat {{ background:rgba(0,0,0,0.3); padding:15px 30px; border-radius:25px; }}
                .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; padding:40px; max-width:1600px; margin:0 auto; }}
                .card {{ background:#1a1a1a; border-radius:15px; overflow:hidden; }}
                .card-header {{ padding:15px; background:#2a2a2a; font-weight:bold; text-align:center; }}
                img, object {{ width:100%; display:block; }}
                .layers {{ text-align:center; padding:30px; }}
                .layer-list {{ display:flex; flex-wrap:wrap; justify-content:center; gap:15px; margin-top:20px; }}
                .layer-item {{ padding:10px 20px; background:#2a2a2a; border-radius:20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎭 分层矢量化</h1>
                <div class="stats">
                    <span class="stat">📊 {stats['paths']} 路径</span>
                    <span class="stat">📦 {stats['size_kb']:.0f} KB</span>
                </div>
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
            <div class="layers">
                <h2>4层分层处理</h2>
                <div class="layer-list">
                    <span class="layer-item">🌅 背景渐变</span>
                    <span class="layer-item">🎯 SAM3语义</span>
                    <span class="layer-item">🎨 颜色量化</span>
                    <span class="layer-item">✨ 边缘细节</span>
                </div>
            </div>
        </body>
        </html>
        """
        with open(output_path / "result.html", 'w') as f:
            f.write(html)


def main():
    vectorizer = LayeredVectorizer()
    return vectorizer.vectorize("01_输入图片/Ladygaga_2.jpg")


if __name__ == "__main__":
    main()
