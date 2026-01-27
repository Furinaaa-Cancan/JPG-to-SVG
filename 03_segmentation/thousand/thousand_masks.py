#!/usr/bin/env python3
"""
千级Mask分割系统
使用高效CV方法在几秒内生成1000+个masks
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
import time
from skimage import segmentation, color, feature
from sklearn.cluster import KMeans
from scipy import ndimage
import multiprocessing as mp
from functools import partial


class ThousandMasks:
    """千级Mask生成器"""
    
    def __init__(self, target_masks: int = 1500):
        self.target_masks = target_masks
        print(f"\n🎯 Target: {target_masks} masks")
    
    def generate(self, image_path: str, output_dir: str = "02_输出结果/thousand_masks"):
        """生成上千个masks"""
        
        print("\n" + "="*70)
        print("💎 THOUSAND MASKS GENERATOR")
        print("="*70)
        
        start_time = time.time()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载图像
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        print(f"\n📷 Input: {image_path}")
        print(f"   Size: {w}x{h}")
        
        all_masks = []
        
        # 方法1: SLIC超像素 - 多层级
        print("\n🔷 Method 1: Multi-level SLIC Superpixels")
        t1 = time.time()
        slic_masks = self.generate_slic_masks(img_rgb)
        print(f"   Generated {len(slic_masks)} masks in {time.time()-t1:.1f}s")
        all_masks.extend(slic_masks)
        
        # 方法2: 颜色量化分割
        print("\n🔷 Method 2: Color Quantization")
        t2 = time.time()
        color_masks = self.generate_color_masks(img_rgb)
        print(f"   Generated {len(color_masks)} masks in {time.time()-t2:.1f}s")
        all_masks.extend(color_masks)
        
        # 方法3: 边缘检测分割
        print("\n🔷 Method 3: Edge-based Segmentation")
        t3 = time.time()
        edge_masks = self.generate_edge_masks(img)
        print(f"   Generated {len(edge_masks)} masks in {time.time()-t3:.1f}s")
        all_masks.extend(edge_masks)
        
        # 方法4: 均值漂移分割
        print("\n🔷 Method 4: Mean Shift Segmentation")
        t4 = time.time()
        shift_masks = self.generate_meanshift_masks(img)
        print(f"   Generated {len(shift_masks)} masks in {time.time()-t4:.1f}s")
        all_masks.extend(shift_masks)
        
        # 方法5: 分水岭分割
        print("\n🔷 Method 5: Watershed Segmentation")
        t5 = time.time()
        watershed_masks = self.generate_watershed_masks(img)
        print(f"   Generated {len(watershed_masks)} masks in {time.time()-t5:.1f}s")
        all_masks.extend(watershed_masks)
        
        # 方法6: Felzenszwalb分割
        print("\n🔷 Method 6: Felzenszwalb Segmentation")
        t6 = time.time()
        felz_masks = self.generate_felzenszwalb_masks(img_rgb)
        print(f"   Generated {len(felz_masks)} masks in {time.time()-t6:.1f}s")
        all_masks.extend(felz_masks)
        
        print(f"\n📊 Total raw masks: {len(all_masks)}")
        
        # 去重和过滤
        print("\n🔄 Deduplicating and filtering...")
        unique_masks = self.deduplicate_fast(all_masks)
        print(f"   Unique masks: {len(unique_masks)}")
        
        # 为每个mask提取颜色
        print("\n🎨 Extracting colors...")
        colored_masks = self.extract_colors(img_rgb, unique_masks)
        
        # 生成可视化
        print("\n🖼️  Generating visualizations...")
        self.save_visualizations(img_rgb, colored_masks, output_path)
        
        process_time = time.time() - start_time
        
        print(f"\n" + "="*70)
        print(f"✅ THOUSAND MASKS COMPLETE!")
        print(f"   Total masks: {len(colored_masks)}")
        print(f"   Processing time: {process_time:.1f}s")
        print(f"   Speed: {len(colored_masks)/process_time:.1f} masks/sec")
        print("="*70)
        
        # 打开展示
        import subprocess
        subprocess.run(["open", str(output_path / "thousand_showcase.html")])
        
        return {
            'masks': colored_masks,
            'count': len(colored_masks),
            'time': process_time
        }
    
    def generate_slic_masks(self, img: np.ndarray) -> list:
        """SLIC超像素分割 - 多层级"""
        
        masks = []
        
        # 多种超像素数量
        n_segments_list = [100, 200, 500, 1000, 2000]
        
        for n_seg in n_segments_list:
            segments = segmentation.slic(
                img, 
                n_segments=n_seg,
                compactness=10,
                start_label=1,
                channel_axis=2
            )
            
            for seg_id in np.unique(segments):
                mask = (segments == seg_id).astype(np.uint8) * 255
                area = np.sum(mask > 0)
                
                if area > 50:  # 最小区域
                    masks.append({
                        'mask': mask,
                        'area': area,
                        'method': f'slic_{n_seg}'
                    })
        
        return masks
    
    def generate_color_masks(self, img: np.ndarray) -> list:
        """颜色量化分割"""
        
        masks = []
        h, w = img.shape[:2]
        
        # 多种颜色聚类级别
        for n_colors in [16, 32, 64, 128]:
            # 缩小图像加速
            scale = min(1.0, 500 / max(h, w))
            small_img = cv2.resize(img, None, fx=scale, fy=scale)
            
            # K-means聚类
            pixels = small_img.reshape(-1, 3).astype(np.float32)
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=3, max_iter=50)
            labels = kmeans.fit_predict(pixels)
            
            # 放大回原始尺寸
            label_img = labels.reshape(small_img.shape[:2])
            label_img_full = cv2.resize(
                label_img.astype(np.float32), 
                (w, h), 
                interpolation=cv2.INTER_NEAREST
            ).astype(int)
            
            # 对每个颜色创建mask
            for color_id in range(n_colors):
                color_mask = (label_img_full == color_id).astype(np.uint8) * 255
                
                # 找连通组件
                num_labels, labeled = cv2.connectedComponents(color_mask)
                
                for label_id in range(1, min(num_labels, 50)):  # 限制每色的组件数
                    mask = (labeled == label_id).astype(np.uint8) * 255
                    area = np.sum(mask > 0)
                    
                    if area > 50:
                        masks.append({
                            'mask': mask,
                            'area': area,
                            'method': f'color_{n_colors}'
                        })
        
        return masks
    
    def generate_edge_masks(self, img: np.ndarray) -> list:
        """边缘检测分割"""
        
        masks = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 多种Canny阈值
        thresholds = [(30, 100), (50, 150), (80, 200), (100, 250)]
        
        for low, high in thresholds:
            edges = cv2.Canny(gray, low, high)
            
            # 闭操作连接边缘
            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 反转得到区域
            regions = 255 - closed
            
            # 找连通组件
            num_labels, labeled = cv2.connectedComponents(regions)
            
            for label_id in range(1, min(num_labels, 200)):
                mask = (labeled == label_id).astype(np.uint8) * 255
                area = np.sum(mask > 0)
                
                if 50 < area < img.shape[0] * img.shape[1] * 0.5:
                    masks.append({
                        'mask': mask,
                        'area': area,
                        'method': f'edge_{low}_{high}'
                    })
        
        return masks
    
    def generate_meanshift_masks(self, img: np.ndarray) -> list:
        """均值漂移分割"""
        
        masks = []
        
        # 缩小图像加速
        h, w = img.shape[:2]
        scale = min(1.0, 400 / max(h, w))
        small_img = cv2.resize(img, None, fx=scale, fy=scale)
        
        # 均值漂移
        for sp in [10, 20, 30]:  # spatial radius
            for sr in [30, 50, 70]:  # color radius
                shifted = cv2.pyrMeanShiftFiltering(small_img, sp, sr)
                
                # 量化颜色
                gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
                quantized = (gray // 25) * 25
                
                # 放大回原始尺寸
                quantized_full = cv2.resize(
                    quantized, (w, h), interpolation=cv2.INTER_NEAREST
                )
                
                # 找连通组件
                for value in np.unique(quantized_full)[::3]:  # 采样部分值
                    color_mask = (quantized_full == value).astype(np.uint8) * 255
                    
                    num_labels, labeled = cv2.connectedComponents(color_mask)
                    
                    for label_id in range(1, min(num_labels, 30)):
                        mask = (labeled == label_id).astype(np.uint8) * 255
                        area = np.sum(mask > 0)
                        
                        if area > 100:
                            masks.append({
                                'mask': mask,
                                'area': area,
                                'method': f'meanshift_{sp}_{sr}'
                            })
        
        return masks
    
    def generate_watershed_masks(self, img: np.ndarray) -> list:
        """分水岭分割"""
        
        masks = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 不同阈值
        for thresh_val in [50, 100, 150]:
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            
            # 距离变换
            dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
            
            # 找局部最大值
            for thresh_dist in [0.2, 0.4, 0.6]:
                _, sure_fg = cv2.threshold(dist, thresh_dist * dist.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)
                
                # 标记
                _, markers = cv2.connectedComponents(sure_fg)
                
                # 分水岭
                markers = cv2.watershed(img, markers)
                
                for marker_id in np.unique(markers):
                    if marker_id <= 0:
                        continue
                    
                    mask = (markers == marker_id).astype(np.uint8) * 255
                    area = np.sum(mask > 0)
                    
                    if 100 < area < img.shape[0] * img.shape[1] * 0.3:
                        masks.append({
                            'mask': mask,
                            'area': area,
                            'method': f'watershed_{thresh_val}'
                        })
        
        return masks
    
    def generate_felzenszwalb_masks(self, img: np.ndarray) -> list:
        """Felzenszwalb分割"""
        
        masks = []
        
        # 多种参数
        for scale in [50, 100, 200, 300]:
            segments = segmentation.felzenszwalb(
                img, scale=scale, sigma=0.5, min_size=50
            )
            
            for seg_id in np.unique(segments):
                mask = (segments == seg_id).astype(np.uint8) * 255
                area = np.sum(mask > 0)
                
                if area > 50:
                    masks.append({
                        'mask': mask,
                        'area': area,
                        'method': f'felzenszwalb_{scale}'
                    })
        
        return masks
    
    def deduplicate_fast(self, masks: list) -> list:
        """快速去重"""
        
        if not masks:
            return []
        
        # 按面积排序
        masks.sort(key=lambda x: x['area'], reverse=True)
        
        unique = []
        
        # 使用采样点快速比较
        sample_rate = 100  # 每100个像素采样一个
        
        for mask_data in masks:
            mask = mask_data['mask']
            
            # 计算mask的哈希特征
            mask_flat = mask.flatten()
            sample_indices = np.arange(0, len(mask_flat), sample_rate)
            mask_sample = mask_flat[sample_indices] > 127
            
            is_dup = False
            
            for u in unique[-100:]:  # 只与最近100个比较
                u_mask = u['mask']
                u_flat = u_mask.flatten()
                u_sample = u_flat[sample_indices] > 127
                
                # 快速相似度计算
                same = np.sum(mask_sample == u_sample)
                similarity = same / len(mask_sample)
                
                if similarity > 0.9:  # 90%相似
                    is_dup = True
                    break
            
            if not is_dup:
                unique.append(mask_data)
                
                # 限制最大数量
                if len(unique) >= self.target_masks:
                    break
        
        return unique
    
    def extract_colors(self, img: np.ndarray, masks: list) -> list:
        """提取每个mask的颜色"""
        
        for mask_data in masks:
            mask = mask_data['mask']
            pixels = img[mask > 127]
            
            if len(pixels) > 0:
                # 平均颜色
                mean_color = np.mean(pixels, axis=0)
                r, g, b = mean_color.astype(int)
                mask_data['color'] = f"#{r:02x}{g:02x}{b:02x}"
        
        return masks
    
    def save_visualizations(self, img: np.ndarray, masks: list, output_path: Path):
        """保存可视化"""
        
        h, w = img.shape[:2]
        
        # 创建彩色叠加
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        
        np.random.seed(42)
        colors = np.random.rand(len(masks), 3) * 255
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['mask']
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h))
            
            binary = mask > 127
            
            for c in range(3):
                overlay[:, :, c] += binary * colors[i, c] * 0.15
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        composite = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        
        Image.fromarray(composite).save(output_path / "thousand_composite.png")
        
        # 边缘图
        edges = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['mask']
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h))
            
            binary = (mask > 127).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            color = colors[i].astype(int).tolist()
            cv2.drawContours(edges, contours, -1, color, 1)
        
        Image.fromarray(edges).save(output_path / "thousand_edges.png")
        
        # HTML
        self.create_html(output_path, len(masks))
    
    def create_html(self, output_path: Path, count: int):
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Thousand Masks</title>
            <style>
                body {{ margin:0; font-family:sans-serif; background:#0a0a0a; color:white; }}
                .header {{ text-align:center; padding:50px; background:linear-gradient(135deg,#667eea,#764ba2); }}
                h1 {{ font-size:4em; margin:0; }}
                .count {{ font-size:3em; color:#FFD700; margin-top:20px; }}
                .container {{ max-width:1600px; margin:40px auto; padding:0 20px; }}
                .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:30px; }}
                .card {{ background:#1a1a1a; border-radius:20px; overflow:hidden; }}
                .card-header {{ padding:20px; background:#2a2a2a; font-size:1.3em; }}
                img {{ width:100%; display:block; }}
                .info {{ text-align:center; padding:30px; background:#1a1a1a; margin-top:30px; border-radius:20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>💎 THOUSAND MASKS</h1>
                <div class="count">{count} Masks Generated!</div>
            </div>
            <div class="container">
                <div class="grid">
                    <div class="card">
                        <div class="card-header">🎨 All Masks Overlay</div>
                        <img src="thousand_composite.png">
                    </div>
                    <div class="card">
                        <div class="card-header">📐 All Edges</div>
                        <img src="thousand_edges.png">
                    </div>
                </div>
                <div class="info">
                    <h2>Methods Used</h2>
                    <p>SLIC Superpixels (5 levels) + Color Quantization (4 levels) + 
                    Edge Detection (4 thresholds) + Mean Shift + Watershed + Felzenszwalb</p>
                </div>
            </div>
        </body>
        </html>
        """
        with open(output_path / "thousand_showcase.html", 'w') as f:
            f.write(html)


def main():
    generator = ThousandMasks(target_masks=1500)
    return generator.generate("01_输入图片/Ladygaga_2.jpg")


if __name__ == "__main__":
    main()
