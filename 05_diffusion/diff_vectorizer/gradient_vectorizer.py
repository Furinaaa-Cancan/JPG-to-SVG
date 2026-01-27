#!/usr/bin/env python3
"""
梯度优化矢量化
核心思路：用梯度信息指导分割，而不是暴力超像素
- 图像梯度 = 边界信息
- 梯度方向 = 区域划分依据
- 自适应分割 = 细节多的地方分得细，平滑区域分得粗
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import svgwrite
import time
from scipy import ndimage
from sklearn.cluster import MeanShift


class GradientVectorizer:
    """基于梯度的智能矢量化"""
    
    def __init__(self):
        print("\n🚀 Gradient-based Vectorizer")
    
    def vectorize(self, image_path: str, output_dir: str = "02_输出结果/gradient_svg"):
        """梯度优化矢量化"""
        
        print("\n" + "="*70)
        print("💎 GRADIENT-BASED SMART VECTORIZATION")
        print("="*70)
        
        start_time = time.time()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载图像
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        print(f"\n📷 Image: {w}x{h}")
        
        # Step 1: 计算图像梯度（检测边界）
        print("\n🔧 Step 1: Computing gradients...")
        gradient_mag, gradient_dir = self.compute_gradients(img)
        
        # Step 2: 自适应采样（细节多的地方采样密，平滑区域采样稀）
        print("\n🎯 Step 2: Adaptive sampling based on gradient...")
        sample_points = self.adaptive_sample(gradient_mag, h, w)
        print(f"   Sample points: {len(sample_points)}")
        
        # Step 3: 基于颜色和位置的智能聚类
        print("\n🎨 Step 3: Smart clustering...")
        regions = self.smart_cluster(img_rgb, sample_points, h, w)
        print(f"   Regions: {len(regions)}")
        
        # Step 4: 检测重要细节区域（金色、皮肤、高光）
        print("\n🎨 Step 4: Detecting important details...")
        detail_regions = self.detect_important_details(img_rgb, h, w)
        print(f"   Detail regions: {len(detail_regions)}")
        
        # Step 5: 优化边界
        print("\n📐 Step 5: Refining boundaries with gradients...")
        refined_regions = self.refine_with_gradients(regions, gradient_mag, img_rgb, h, w)
        refined_regions.extend(detail_regions)
        print(f"   Total refined regions: {len(refined_regions)}")
        
        # Step 5: 生成SVG
        print("\n✨ Step 5: Generating SVG...")
        svg_path = output_path / "gradient_vector.svg"
        stats = self.create_svg(refined_regions, w, h, str(svg_path))
        
        # 对比HTML
        self.create_html(image_path, str(svg_path), output_path, stats)
        
        process_time = time.time() - start_time
        
        print(f"\n" + "="*70)
        print(f"✅ GRADIENT VECTORIZATION COMPLETE!")
        print(f"   Paths: {stats['paths']}")
        print(f"   Size: {stats['size_kb']:.1f} KB")
        print(f"   Time: {process_time:.1f}s")
        print("="*70)
        
        import subprocess
        subprocess.run(["open", str(output_path / "result.html")])
        
        return stats
    
    def compute_gradients(self, img: np.ndarray) -> tuple:
        """计算图像梯度"""
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Sobel梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 梯度幅值和方向
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # 归一化
        magnitude = magnitude / magnitude.max()
        
        return magnitude, direction
    
    def adaptive_sample(self, gradient_mag: np.ndarray, h: int, w: int) -> list:
        """根据梯度自适应采样"""
        
        points = []
        
        # 多尺度采样 - 增加密度
        scales = [
            (64, 0.02),   # 粗网格，几乎全部采样
            (32, 0.05),   # 中网格
            (16, 0.1),    # 细网格
            (8, 0.15),    # 更细
            (4, 0.25),    # 最细，高梯度区域
        ]
        
        for grid_size, grad_thresh in scales:
            for y in range(0, h, grid_size):
                for x in range(0, w, grid_size):
                    # 计算该区域的平均梯度
                    y_end = min(y + grid_size, h)
                    x_end = min(x + grid_size, w)
                    
                    region_grad = gradient_mag[y:y_end, x:x_end].mean()
                    
                    # 梯度高于阈值才采样
                    if region_grad > grad_thresh:
                        cx = (x + x_end) // 2
                        cy = (y + y_end) // 2
                        points.append((cx, cy))
        
        # 去重
        points = list(set(points))
        
        return points
    
    def smart_cluster(self, img: np.ndarray, sample_points: list, h: int, w: int) -> list:
        """基于颜色和位置的智能聚类"""
        
        if not sample_points:
            # 如果没有采样点，使用均匀网格
            sample_points = [(x, y) for y in range(0, h, 32) for x in range(0, w, 32)]
        
        # 构建特征向量 [R, G, B, x/w, y/h]
        features = []
        for x, y in sample_points:
            color = img[y, x]
            features.append([
                color[0] / 255.0,
                color[1] / 255.0,
                color[2] / 255.0,
                x / w * 0.3,  # 位置权重较低
                y / h * 0.3
            ])
        
        features = np.array(features)
        
        # MeanShift聚类（自动确定聚类数）- 减小带宽获得更细分割
        ms = MeanShift(bandwidth=0.08, bin_seeding=True)
        labels = ms.fit_predict(features)
        
        # 构建区域
        n_clusters = len(set(labels))
        
        # 为每个像素分配最近的聚类
        all_pixels = np.array([[
            img[y, x, 0] / 255.0,
            img[y, x, 1] / 255.0,
            img[y, x, 2] / 255.0,
            x / w * 0.3,
            y / h * 0.3
        ] for y in range(h) for x in range(w)])
        
        # 用最近邻分配
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(features)
        _, indices = nn.kneighbors(all_pixels)
        
        pixel_labels = labels[indices.flatten()].reshape(h, w)
        
        # 提取每个聚类的mask和颜色
        regions = []
        for cid in range(n_clusters):
            mask = (pixel_labels == cid).astype(np.uint8) * 255
            area = np.sum(mask > 0)
            
            if area > 100:
                pixels = img[mask > 127]
                if len(pixels) > 0:
                    color = np.mean(pixels, axis=0).astype(int)
                    regions.append({
                        'mask': mask,
                        'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                        'area': area
                    })
        
        return regions
    
    def detect_important_details(self, img: np.ndarray, h: int, w: int) -> list:
        """检测重要细节区域"""
        
        regions = []
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # 1. 金色装饰检测
        gold_ranges = [
            ([15, 80, 100], [35, 255, 255]),
            ([10, 50, 150], [25, 200, 255]),
        ]
        
        for lower, upper in gold_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            
            n_labels, labeled = cv2.connectedComponents(mask)
            for lid in range(1, min(n_labels, 200)):
                m = (labeled == lid).astype(np.uint8) * 255
                area = np.sum(m > 0)
                if area > 20:
                    pixels = img[m > 127]
                    if len(pixels) > 0:
                        color = pixels[np.argmax(np.sum(pixels, axis=1))]
                        regions.append({'mask': m, 'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}", 'area': area})
        
        # 2. 皮肤色检测
        lower_skin = np.array([0, 20, 100])
        upper_skin = np.array([25, 150, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        n_labels, labeled = cv2.connectedComponents(skin_mask)
        for lid in range(1, min(n_labels, 50)):
            m = (labeled == lid).astype(np.uint8) * 255
            area = np.sum(m > 0)
            if area > 100:
                pixels = img[m > 127]
                if len(pixels) > 0:
                    color = np.mean(pixels, axis=0).astype(int)
                    regions.append({'mask': m, 'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}", 'area': area})
        
        # 3. 高亮检测
        for thresh in [220, 240]:
            _, highlight = cv2.threshold(hsv[:, :, 2], thresh, 255, cv2.THRESH_BINARY)
            n_labels, labeled = cv2.connectedComponents(highlight)
            for lid in range(1, min(n_labels, 100)):
                m = (labeled == lid).astype(np.uint8) * 255
                area = np.sum(m > 0)
                if 20 < area < h * w * 0.01:
                    pixels = img[m > 127]
                    if len(pixels) > 0:
                        color = np.mean(pixels, axis=0).astype(int)
                        regions.append({'mask': m, 'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}", 'area': area})
        
        # 4. 白色（骨骼）
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        n_labels, labeled = cv2.connectedComponents(white_mask)
        for lid in range(1, min(n_labels, 100)):
            m = (labeled == lid).astype(np.uint8) * 255
            area = np.sum(m > 0)
            if area > 30:
                pixels = img[m > 127]
                if len(pixels) > 0:
                    color = np.mean(pixels, axis=0).astype(int)
                    regions.append({'mask': m, 'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}", 'area': area})
        
        # 5. 红色（嘴唇）
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        
        n_labels, labeled = cv2.connectedComponents(red_mask)
        for lid in range(1, min(n_labels, 20)):
            m = (labeled == lid).astype(np.uint8) * 255
            area = np.sum(m > 0)
            if area > 30:
                pixels = img[m > 127]
                if len(pixels) > 0:
                    color = np.mean(pixels, axis=0).astype(int)
                    regions.append({'mask': m, 'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}", 'area': area})
        
        return regions
    
    def refine_with_gradients(self, regions: list, gradient_mag: np.ndarray, 
                              img: np.ndarray, h: int, w: int) -> list:
        """用梯度信息优化边界"""
        
        refined = []
        
        for region in regions:
            mask = region['mask']
            
            # 找到mask边界
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            eroded = cv2.erode(mask, kernel, iterations=1)
            boundary = dilated - eroded
            
            # 在边界处，根据梯度调整
            # 梯度高的地方是真正的边界，保持锐利
            # 梯度低的地方可以平滑
            
            boundary_grad = gradient_mag * (boundary / 255.0)
            
            # 在高梯度边界处使用原始mask
            # 在低梯度边界处略微平滑
            smooth_kernel = np.ones((5, 5), np.float32) / 25
            smoothed_mask = cv2.filter2D(mask.astype(np.float32), -1, smooth_kernel)
            
            # 混合：高梯度用原始，低梯度用平滑
            gradient_weight = boundary_grad / (boundary_grad.max() + 1e-6)
            final_mask = mask * gradient_weight + smoothed_mask * (1 - gradient_weight)
            final_mask = (final_mask > 127).astype(np.uint8) * 255
            
            # 提取连通组件
            n_labels, labeled = cv2.connectedComponents(final_mask)
            
            for lid in range(1, n_labels):
                component_mask = (labeled == lid).astype(np.uint8) * 255
                component_area = np.sum(component_mask > 0)
                
                if component_area > 50:
                    component_pixels = img[component_mask > 127]
                    if len(component_pixels) > 0:
                        color = np.mean(component_pixels, axis=0).astype(int)
                        refined.append({
                            'mask': component_mask,
                            'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                            'area': component_area
                        })
        
        return refined
    
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
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 30:
                    continue
                
                # 智能简化：根据轮廓复杂度决定epsilon
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                complexity = perimeter / (np.sqrt(area) + 1)
                
                epsilon = max(1.0, min(3.0, complexity / 10))
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 3:
                    points = approx.squeeze()
                    if points.ndim == 1:
                        points = points.reshape(-1, 2)
                    if len(points) < 3:
                        continue
                    
                    # 贝塞尔曲线
                    path_d = self.bezier_path(points)
                    
                    dwg.add(dwg.path(d=path_d, fill=color, stroke="none"))
                    paths += 1
        
        dwg.save()
        
        return {
            'paths': paths,
            'size_kb': Path(output_path).stat().st_size / 1024
        }
    
    def bezier_path(self, points: np.ndarray) -> str:
        """生成平滑的贝塞尔曲线路径"""
        
        if len(points) < 3:
            return ""
        
        path_d = f"M{points[0][0]},{points[0][1]}"
        
        for i in range(1, len(points) - 1):
            p0 = points[i - 1]
            p1 = points[i]
            p2 = points[i + 1]
            
            # 控制点
            cx = p1[0]
            cy = p1[1]
            
            # 终点（中点）
            ex = (p1[0] + p2[0]) / 2
            ey = (p1[1] + p2[1]) / 2
            
            path_d += f" Q{cx},{cy} {ex},{ey}"
        
        # 最后一个点
        path_d += f" L{points[-1][0]},{points[-1][1]}"
        path_d += " Z"
        
        return path_d
    
    def create_html(self, original: str, svg: str, output_path: Path, stats: dict):
        """创建对比HTML"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Gradient Vectorization</title>
            <style>
                body {{ margin:0; background:#0a0a0a; color:#fff; font-family:sans-serif; }}
                .header {{ text-align:center; padding:50px; background:linear-gradient(135deg,#11998e,#38ef7d); }}
                h1 {{ font-size:3em; margin:0; }}
                .subtitle {{ margin-top:10px; opacity:0.9; }}
                .stats {{ display:flex; justify-content:center; gap:40px; margin-top:20px; }}
                .stat {{ background:rgba(0,0,0,0.3); padding:15px 30px; border-radius:25px; }}
                .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; padding:40px; max-width:1600px; margin:0 auto; }}
                .card {{ background:#1a1a1a; border-radius:15px; overflow:hidden; }}
                .card-header {{ padding:15px; background:#2a2a2a; font-weight:bold; text-align:center; }}
                img, object {{ width:100%; display:block; }}
                .tech {{ text-align:center; padding:30px; background:#1a1a1a; margin:20px 40px; border-radius:15px; }}
                .tech-list {{ display:flex; flex-wrap:wrap; justify-content:center; gap:15px; margin-top:20px; }}
                .tech-item {{ padding:10px 20px; background:#2a2a2a; border-radius:20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 梯度优化矢量化</h1>
                <div class="subtitle">自适应分割：细节多的地方分得细</div>
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
            <div class="tech">
                <h2>💡 算法创新</h2>
                <div class="tech-list">
                    <span class="tech-item">📈 梯度检测边界</span>
                    <span class="tech-item">🎯 自适应采样</span>
                    <span class="tech-item">🧠 MeanShift智能聚类</span>
                    <span class="tech-item">📐 梯度优化边界</span>
                </div>
            </div>
        </body>
        </html>
        """
        with open(output_path / "result.html", 'w') as f:
            f.write(html)


def main():
    vectorizer = GradientVectorizer()
    return vectorizer.vectorize("01_输入图片/Ladygaga_2.jpg")


if __name__ == "__main__":
    main()
