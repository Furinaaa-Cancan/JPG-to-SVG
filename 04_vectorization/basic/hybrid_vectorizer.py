#!/usr/bin/env python3
"""
混合矢量化器
SAM3语义分割 + 边缘检测 + 颜色量化 = 最大细节
"""

import sys
import cv2
import numpy as np
from PIL import Image
import svgwrite
from pathlib import Path
import time
from sklearn.cluster import KMeans
from skimage import segmentation

sys.path.insert(0, "/Volumes/Seagate/SAM3/模型库/01_SAM3核心模型")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class HybridVectorizer:
    """混合矢量化器 - 最大细节"""
    
    def __init__(self):
        print("\n🚀 Initializing Hybrid Vectorizer...")
        self.model = build_sam3_image_model(device="cpu")
        self.processor = Sam3Processor(self.model, device="cpu", confidence_threshold=0.1)
        print("✅ Ready!")
    
    def vectorize(self, image_path: str, output_dir: str = "02_输出结果/hybrid_svg"):
        """混合矢量化"""
        
        print("\n" + "="*70)
        print("💎 HYBRID VECTORIZATION - MAXIMUM DETAIL")
        print("="*70)
        
        start_time = time.time()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载图像
        img_pil = Image.open(image_path)
        img = np.array(img_pil)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        
        print(f"\n📷 Image: {w}x{h}")
        
        all_regions = []
        
        # Layer 1: SAM3语义分割（大区域）
        print("\n🔷 Layer 1: SAM3 Semantic Segmentation")
        sam3_regions = self.sam3_segment(img_pil, img, h, w)
        all_regions.extend(sam3_regions)
        print(f"   SAM3 regions: {len(sam3_regions)}")
        
        # Layer 2: 超像素分割（中等区域）
        print("\n🔷 Layer 2: Superpixel Segmentation")
        superpixel_regions = self.superpixel_segment(img, h, w)
        all_regions.extend(superpixel_regions)
        print(f"   Superpixel regions: {len(superpixel_regions)}")
        
        # Layer 3: 颜色量化（细节区域）
        print("\n🔷 Layer 3: Color Quantization")
        color_regions = self.color_quantize(img, h, w)
        all_regions.extend(color_regions)
        print(f"   Color regions: {len(color_regions)}")
        
        # Layer 4: 边缘检测（最细节）
        print("\n🔷 Layer 4: Edge Detection")
        edge_regions = self.edge_detect(img_bgr, img, h, w)
        all_regions.extend(edge_regions)
        print(f"   Edge regions: {len(edge_regions)}")
        
        # Layer 5: 装饰检测（金色纹路等）
        print("\n🔷 Layer 5: Decoration Detection")
        deco_regions = self.detect_decorations(img, h, w)
        all_regions.extend(deco_regions)
        print(f"   Decoration regions: {len(deco_regions)}")
        
        print(f"\n📊 Total raw regions: {len(all_regions)}")
        
        # 去重（快速）
        print("\n🔄 Fast deduplication...")
        unique_regions = self.fast_dedupe(all_regions)
        print(f"   Unique regions: {len(unique_regions)}")
        
        # 生成SVG
        print("\n✨ Generating SVG...")
        svg_path = output_path / "hybrid_vector.svg"
        stats = self.create_svg(unique_regions, w, h, str(svg_path))
        
        # 对比
        self.create_html(image_path, str(svg_path), output_path, stats, len(all_regions))
        
        process_time = time.time() - start_time
        
        print(f"\n" + "="*70)
        print(f"✅ HYBRID VECTORIZATION COMPLETE!")
        print(f"   Regions: {len(unique_regions)}")
        print(f"   Paths: {stats['paths']}")
        print(f"   Size: {stats['size_kb']:.1f} KB")
        print(f"   Time: {process_time:.1f}s")
        print("="*70)
        
        import subprocess
        subprocess.run(["open", str(output_path / "result.html")])
        
        return stats
    
    def sam3_segment(self, img_pil, img_array, h, w) -> list:
        """SAM3语义分割"""
        
        regions = []
        state = self.processor.set_image(img_pil)
        
        prompts = [
            "blue dress", "costume", "skeleton", "skull", "bones",
            "gold decoration", "gold trim", "embroidery", "button",
            "face", "hair", "hand", "arm", "skin",
            "background", "shadow", "highlight"
        ]
        
        for prompt in prompts:
            try:
                self.processor.reset_all_prompts(state)
                result = self.processor.set_text_prompt(prompt, state)
                
                if result and 'masks' in result and result['masks'] is not None:
                    masks = result['masks'].cpu().numpy()
                    
                    for mask in masks:
                        if len(mask.shape) > 2:
                            mask = mask.squeeze()
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask.astype(np.float32), (w, h))
                        
                        binary = (mask > 0.5).astype(np.uint8) * 255
                        area = np.sum(binary > 0)
                        
                        if area > 500:
                            pixels = img_array[binary > 127]
                            if len(pixels) > 0:
                                color = np.mean(pixels, axis=0).astype(int)
                                regions.append({
                                    'mask': binary,
                                    'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                                    'area': area,
                                    'layer': 'sam3'
                                })
            except:
                pass
        
        return regions
    
    def superpixel_segment(self, img, h, w) -> list:
        """超像素分割 - 多尺度获取最大细节"""
        
        regions = []
        
        # 多尺度超像素
        for n_seg in [500, 1000, 2000, 3000]:
            segments = segmentation.slic(img, n_segments=n_seg, compactness=10, start_label=1)
            
            for seg_id in np.unique(segments):
                mask = (segments == seg_id).astype(np.uint8) * 255
                area = np.sum(mask > 0)
                
                if area > 100:
                    pixels = img[mask > 127]
                    if len(pixels) > 0:
                        color = np.mean(pixels, axis=0).astype(int)
                        regions.append({
                            'mask': mask,
                            'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                            'area': area,
                            'layer': 'superpixel'
                        })
        
        return regions
    
    def color_quantize(self, img, h, w) -> list:
        """颜色量化分割"""
        
        regions = []
        
        # 缩小加速
        scale = min(1.0, 600 / max(h, w))
        small = cv2.resize(img, None, fx=scale, fy=scale)
        
        for n_colors in [32, 64, 128, 256, 512]:
            pixels = small.reshape(-1, 3).astype(np.float32)
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=3, max_iter=30)
            labels = kmeans.fit_predict(pixels)
            
            label_img = labels.reshape(small.shape[:2])
            label_full = cv2.resize(label_img.astype(np.float32), (w, h), 
                                   interpolation=cv2.INTER_NEAREST).astype(int)
            
            for cid in range(n_colors):
                color_mask = (label_full == cid).astype(np.uint8) * 255
                
                # 只处理较小的连通组件
                n_labels, labeled = cv2.connectedComponents(color_mask)
                
                for lid in range(1, min(n_labels, 10)):
                    mask = (labeled == lid).astype(np.uint8) * 255
                    area = np.sum(mask > 0)
                    
                    if 50 < area < h * w * 0.1:  # 限制大小
                        color = kmeans.cluster_centers_[cid].astype(int)
                        regions.append({
                            'mask': mask,
                            'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                            'area': area,
                            'layer': 'color'
                        })
        
        return regions
    
    def edge_detect(self, img_bgr, img_rgb, h, w) -> list:
        """边缘检测分割"""
        
        regions = []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 多尺度边缘
        for (low, high) in [(30, 100), (50, 150), (80, 200)]:
            edges = cv2.Canny(gray, low, high)
            
            # 闭操作
            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 反转得到区域
            inv = 255 - closed
            
            n_labels, labeled = cv2.connectedComponents(inv)
            
            for lid in range(1, min(n_labels, 100)):
                mask = (labeled == lid).astype(np.uint8) * 255
                area = np.sum(mask > 0)
                
                if 30 < area < h * w * 0.05:
                    pixels = img_rgb[mask > 127]
                    if len(pixels) > 0:
                        color = np.mean(pixels, axis=0).astype(int)
                        regions.append({
                            'mask': mask,
                            'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                            'area': area,
                            'layer': 'edge'
                        })
        
        return regions
    
    def detect_decorations(self, img, h, w) -> list:
        """检测装饰元素（金色纹路、高光、细节）"""
        
        regions = []
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # 1. 金色检测 - 多个范围
        gold_ranges = [
            ([15, 80, 80], [40, 255, 255]),   # 标准金色
            ([10, 50, 150], [25, 200, 255]),  # 浅金色
            ([20, 100, 100], [35, 255, 200]), # 深金色
        ]
        
        for lower, upper in gold_ranges:
            gold_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # 形态学处理
            kernel = np.ones((2, 2), np.uint8)
            gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel)
            
            n_labels, labeled = cv2.connectedComponents(gold_mask)
            
            for lid in range(1, min(n_labels, 300)):
                mask = (labeled == lid).astype(np.uint8) * 255
                area = np.sum(mask > 0)
                
                if area > 10:
                    pixels = img[mask > 127]
                    if len(pixels) > 0:
                        brightness = np.sum(pixels, axis=1)
                        color = pixels[np.argmax(brightness)]
                        regions.append({
                            'mask': mask,
                            'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                            'area': area,
                            'layer': 'decoration'
                        })
        
        # 2. 高亮检测
        for thresh in [200, 220, 240]:
            _, highlight = cv2.threshold(hsv[:, :, 2], thresh, 255, cv2.THRESH_BINARY)
            
            n_labels, labeled = cv2.connectedComponents(highlight)
            
            for lid in range(1, min(n_labels, 100)):
                mask = (labeled == lid).astype(np.uint8) * 255
                area = np.sum(mask > 0)
                
                if 10 < area < h * w * 0.02:
                    pixels = img[mask > 127]
                    if len(pixels) > 0:
                        color = np.mean(pixels, axis=0).astype(int)
                        regions.append({
                            'mask': mask,
                            'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                            'area': area,
                            'layer': 'decoration'
                        })
        
        # 3. 白色检测（骨骼）
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        n_labels, labeled = cv2.connectedComponents(white_mask)
        
        for lid in range(1, min(n_labels, 200)):
            mask = (labeled == lid).astype(np.uint8) * 255
            area = np.sum(mask > 0)
            
            if area > 30:
                pixels = img[mask > 127]
                if len(pixels) > 0:
                    color = np.mean(pixels, axis=0).astype(int)
                    regions.append({
                        'mask': mask,
                        'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                        'area': area,
                        'layer': 'decoration'
                    })
        
        # 4. 红色检测（嘴唇）
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        
        n_labels, labeled = cv2.connectedComponents(red_mask)
        
        for lid in range(1, min(n_labels, 50)):
            mask = (labeled == lid).astype(np.uint8) * 255
            area = np.sum(mask > 0)
            
            if area > 20:
                pixels = img[mask > 127]
                if len(pixels) > 0:
                    color = np.mean(pixels, axis=0).astype(int)
                    regions.append({
                        'mask': mask,
                        'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                        'area': area,
                        'layer': 'decoration'
                    })
        
        return regions
    
    def fast_dedupe(self, regions: list) -> list:
        """快速去重"""
        
        if not regions:
            return []
        
        # 按层优先级排序：decoration > edge > color > superpixel > sam3
        priority = {'decoration': 0, 'edge': 1, 'color': 2, 'superpixel': 3, 'sam3': 4}
        regions.sort(key=lambda x: (priority.get(x['layer'], 5), -x['area']))
        
        # 只保留前5000个
        return regions[:5000]
    
    def create_svg(self, regions: list, width: int, height: int, output_path: str) -> dict:
        """创建SVG"""
        
        dwg = svgwrite.Drawing(output_path, size=(width, height))
        dwg.viewbox(0, 0, width, height)
        
        # 按面积排序（大的在底层）
        regions.sort(key=lambda x: x['area'], reverse=True)
        
        paths = 0
        
        for region in regions:
            mask = region['mask']
            color = region['color']
            
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 20:
                    continue
                
                # 简化
                epsilon = 1.0 if region['layer'] in ['decoration', 'edge'] else 1.5
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
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
                    
                    opacity = 1.0 if region['layer'] == 'sam3' else 0.9
                    dwg.add(dwg.path(d=path_d, fill=color, stroke="none", opacity=opacity))
                    paths += 1
        
        dwg.save()
        
        return {
            'paths': paths,
            'size_kb': Path(output_path).stat().st_size / 1024
        }
    
    def create_html(self, original: str, svg: str, output_path: Path, stats: dict, raw_count: int):
        """创建对比HTML"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hybrid Vectorization</title>
            <style>
                body {{ margin:0; background:#0a0a0a; color:#fff; font-family:sans-serif; }}
                .header {{ text-align:center; padding:50px; background:linear-gradient(135deg,#f093fb,#f5576c); }}
                h1 {{ font-size:3em; margin:0; }}
                .stats {{ display:flex; justify-content:center; gap:40px; margin-top:20px; font-size:1.3em; }}
                .stat {{ background:rgba(0,0,0,0.3); padding:10px 25px; border-radius:20px; }}
                .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; padding:40px; max-width:1600px; margin:0 auto; }}
                .card {{ background:#1a1a1a; border-radius:15px; overflow:hidden; }}
                .card-header {{ padding:15px; background:#2a2a2a; font-weight:bold; text-align:center; }}
                img, object {{ width:100%; display:block; }}
                .layers {{ text-align:center; padding:30px; }}
                .layer-list {{ display:flex; flex-wrap:wrap; justify-content:center; gap:10px; margin-top:15px; }}
                .layer {{ padding:8px 20px; background:#2a2a2a; border-radius:15px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎨 混合矢量化</h1>
                <div class="stats">
                    <span class="stat">📊 {stats['paths']} 路径</span>
                    <span class="stat">📦 {stats['size_kb']:.0f} KB</span>
                    <span class="stat">🔍 {raw_count} 原始区域</span>
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
                <h2>5层混合分割</h2>
                <div class="layer-list">
                    <span class="layer">🎯 SAM3语义</span>
                    <span class="layer">🔷 超像素</span>
                    <span class="layer">🎨 颜色量化</span>
                    <span class="layer">📐 边缘检测</span>
                    <span class="layer">✨ 装饰检测</span>
                </div>
            </div>
        </body>
        </html>
        """
        with open(output_path / "result.html", 'w') as f:
            f.write(html)


def main():
    vectorizer = HybridVectorizer()
    return vectorizer.vectorize("01_输入图片/Ladygaga_2.jpg")


if __name__ == "__main__":
    main()
