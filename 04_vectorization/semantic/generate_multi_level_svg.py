#!/usr/bin/env python3
"""
生成多级别细节的SVG科研组图

输出：
- (a) 原始图片
- (b-f) 5个不同细节级别的SVG
"""

import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import svgwrite
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, "/Volumes/Seagate/SAM3/模型库/01_SAM3核心模型")

# 14个细节级别的参数配置 - 从基础到300MB高清
DETAIL_LEVELS = {
    1:  {'name': 'L1',  'n_colors_large': 1,  'n_colors_medium': 1,  'n_colors_small': 1,  'min_area': 8000, 'epsilon_factor': 0.025,   'max_prompts': 3},
    2:  {'name': 'L2',  'n_colors_large': 1,  'n_colors_medium': 1,  'n_colors_small': 1,  'min_area': 3000, 'epsilon_factor': 0.018,   'max_prompts': 5},
    3:  {'name': 'L3',  'n_colors_large': 2,  'n_colors_medium': 1,  'n_colors_small': 1,  'min_area': 1500, 'epsilon_factor': 0.012,   'max_prompts': 8},
    4:  {'name': 'L4',  'n_colors_large': 3,  'n_colors_medium': 2,  'n_colors_small': 1,  'min_area': 800,  'epsilon_factor': 0.008,   'max_prompts': 12},
    5:  {'name': 'L5',  'n_colors_large': 4,  'n_colors_medium': 3,  'n_colors_small': 2,  'min_area': 400,  'epsilon_factor': 0.005,   'max_prompts': 16},
    6:  {'name': 'L6',  'n_colors_large': 6,  'n_colors_medium': 4,  'n_colors_small': 3,  'min_area': 200,  'epsilon_factor': 0.003,   'max_prompts': 20},
    7:  {'name': 'L7',  'n_colors_large': 10, 'n_colors_medium': 6,  'n_colors_small': 4,  'min_area': 100,  'epsilon_factor': 0.002,   'max_prompts': 25},
    8:  {'name': 'L8',  'n_colors_large': 15, 'n_colors_medium': 10, 'n_colors_small': 6,  'min_area': 50,   'epsilon_factor': 0.0012,  'max_prompts': 28},
    9:  {'name': 'L9',  'n_colors_large': 20, 'n_colors_medium': 14, 'n_colors_small': 8,  'min_area': 25,   'epsilon_factor': 0.0008,  'max_prompts': 30},
    10: {'name': 'L10', 'n_colors_large': 28, 'n_colors_medium': 18, 'n_colors_small': 12, 'min_area': 12,   'epsilon_factor': 0.0005,  'max_prompts': 32},
    11: {'name': 'L11', 'n_colors_large': 38, 'n_colors_medium': 25, 'n_colors_small': 16, 'min_area': 6,    'epsilon_factor': 0.0003,  'max_prompts': 33},
    12: {'name': 'L12', 'n_colors_large': 50, 'n_colors_medium': 35, 'n_colors_small': 22, 'min_area': 4,    'epsilon_factor': 0.0002,  'max_prompts': 33},
    13: {'name': 'L13', 'n_colors_large': 65, 'n_colors_medium': 45, 'n_colors_small': 30, 'min_area': 2,    'epsilon_factor': 0.00015, 'max_prompts': 33},
    14: {'name': 'L14', 'n_colors_large': 80, 'n_colors_medium': 55, 'n_colors_small': 40, 'min_area': 1,    'epsilon_factor': 0.0001,  'max_prompts': 33},
}


def process_region_with_level(args):
    """处理单个区域 - 带级别参数"""
    mask, color, area, img, h, w, level_config = args
    
    paths = []
    
    min_area = level_config['min_area']
    epsilon_factor = level_config['epsilon_factor']
    
    if area > 50000:
        n_colors = level_config['n_colors_large']
    elif area > 10000:
        n_colors = level_config['n_colors_medium']
    else:
        n_colors = level_config['n_colors_small']
    
    if n_colors > 1 and area > 3000:
        inner_paths = quantize_region(img, mask, h, w, n_colors, min_area, epsilon_factor)
        paths.extend(inner_paths)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(simplified) >= 3:
                points = simplified.squeeze()
                if points.ndim == 1:
                    points = points.reshape(-1, 2)
                
                paths.append({
                    'points': points,
                    'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                    'area': cv2.contourArea(contour),
                    'layer': 'foreground'
                })
    
    return paths


def quantize_region(img, mask, h, w, n_colors, min_area, epsilon_factor):
    """区域颜色量化"""
    paths = []
    
    masked_img = img.copy()
    masked_img[mask < 127] = 0
    
    lab = cv2.cvtColor(masked_img, cv2.COLOR_RGB2LAB)
    mask_flat = mask.flatten() > 127
    pixels = lab.reshape(-1, 3)[mask_flat].astype(np.float32)
    
    if len(pixels) < 100:
        return paths
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
    n_colors = min(n_colors, len(pixels) // 100)
    if n_colors < 2:
        return paths
    
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    
    label_img = np.zeros(h * w, dtype=np.int32)
    label_img[mask_flat] = labels.flatten()
    label_img = label_img.reshape(h, w)
    
    centers_lab = centers.astype(np.uint8).reshape(1, -1, 3)
    centers_rgb = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2RGB).reshape(-1, 3)
    
    for cid, color_rgb in enumerate(centers_rgb):
        color_mask = ((label_img == cid) & (mask > 127)).astype(np.uint8) * 255
        
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area // 2:
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(simplified) >= 3:
                    points = simplified.squeeze()
                    if points.ndim == 1:
                        points = points.reshape(-1, 2)
                    
                    temp_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(temp_mask, [contour], -1, 255, -1)
                    pixels_rgb = img[temp_mask > 127]
                    if len(pixels_rgb) > 0:
                        actual_color = np.mean(pixels_rgb, axis=0).astype(int)
                    else:
                        actual_color = color_rgb
                    
                    paths.append({
                        'points': points,
                        'color': f"#{actual_color[0]:02x}{actual_color[1]:02x}{actual_color[2]:02x}",
                        'area': area,
                        'layer': 'detail'
                    })
    
    return paths


def points_to_path(points, use_curves=True):
    """点转SVG路径"""
    n = len(points)
    if n < 3:
        return ""
    
    pts = points.astype(int)
    
    if use_curves and n >= 4:
        path_d = f"M{pts[0][0]},{pts[0][1]}"
        for i in range(n):
            p0 = pts[(i - 1) % n]
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            p3 = pts[(i + 2) % n]
            
            c1x = int(p1[0] + (p2[0] - p0[0]) / 6)
            c1y = int(p1[1] + (p2[1] - p0[1]) / 6)
            c2x = int(p2[0] - (p3[0] - p1[0]) / 6)
            c2y = int(p2[1] - (p3[1] - p1[1]) / 6)
            
            path_d += f"C{c1x},{c1y} {c2x},{c2y} {p2[0]},{p2[1]}"
    else:
        path_d = f"M{pts[0][0]},{pts[0][1]}"
        for i in range(1, n):
            path_d += f"L{pts[i][0]},{pts[i][1]}"
    
    path_d += "Z"
    return path_d


class MultiLevelVectorizer:
    def __init__(self):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        print("🚀 Loading SAM3...")
        self.sam3_model = build_sam3_image_model(device='cpu')
        self.sam3_processor = Sam3Processor(self.sam3_model, device='cpu', confidence_threshold=0.1)
        print("✅ SAM3 loaded!")
        
        self.key_prompts = [
            "person", "woman", "man", "face", "hair", "eyes", "lips", "skin",
            "clothing", "dress", "shirt", "jacket", "accessories", "jewelry",
            "hand", "arm", "body", "background", "sky", "wall", "floor",
            "light", "shadow", "highlights", "texture", "pattern", "fabric",
            "decoration", "object", "detail", "foreground", "middle ground",
            "edge", "outline"
        ]
    
    def generate_all_levels(self, image_path, output_dir):
        """生成5个级别的SVG"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载图片
        img_pil = Image.open(image_path).convert('RGB')
        img = np.array(img_pil)
        h, w = img.shape[:2]
        
        print(f"\n📷 Image: {w}x{h}")
        
        # SAM3分割（只做一次，所有级别共用）
        print("\n🎯 Running SAM3 segmentation...")
        state = self.sam3_processor.set_image(img_pil)
        all_regions = self._segment_image(img, state, h, w)
        print(f"   Total regions: {len(all_regions)}")
        
        results = []
        
        for level in range(1, 15):
            print(f"\n{'='*50}")
            print(f"📊 Generating Level {level}: {DETAIL_LEVELS[level]['name']}")
            print(f"{'='*50}")
            
            level_start = time.time()
            level_config = DETAIL_LEVELS[level]
            
            # 过滤regions（根据max_prompts）
            max_prompts = level_config['max_prompts']
            filtered_regions = all_regions[:max_prompts * 20]  # 估算
            
            # 生成paths
            paths = []
            
            # 背景
            bg_paths = self._background(img, h, w, level_config)
            paths.extend(bg_paths)
            
            # 前景
            fg_paths = self._foreground(img, filtered_regions, h, w, level_config)
            paths.extend(fg_paths)
            
            # 生成SVG
            svg_path = output_dir / f"level_{level}.svg"
            stats = self._create_svg(paths, w, h, str(svg_path))
            
            level_time = time.time() - level_start
            
            results.append({
                'level': level,
                'name': level_config['name'],
                'paths': stats['paths'],
                'size_kb': stats['size_kb'],
                'time': level_time,
                'svg_path': svg_path
            })
            
            print(f"   Paths: {stats['paths']:,} | Size: {stats['size_kb']:.0f} KB | Time: {level_time:.1f}s")
        
        return results
    
    def _segment_image(self, img, state, h, w):
        """SAM3分割"""
        regions = []
        
        for prompt in self.key_prompts:
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
                        kernel = np.ones((3, 3), np.uint8)
                        binary = cv2.dilate(binary, kernel, iterations=1)
                        binary = cv2.GaussianBlur(binary, (3, 3), 0)
                        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
                        
                        area = np.sum(binary > 0)
                        
                        if area > 500 and area < h * w * 0.9:
                            orig_binary = (mask > 0.5).astype(np.uint8) * 255
                            pixels = img[orig_binary > 127]
                            if len(pixels) > 0:
                                color = np.mean(pixels, axis=0).astype(int)
                                regions.append({
                                    'mask': binary,
                                    'color': color,
                                    'area': area,
                                    'prompt': prompt
                                })
            except:
                pass
        
        regions.sort(key=lambda x: -x['area'])
        return regions
    
    def _background(self, img, h, w, level_config):
        """背景处理"""
        paths = []
        
        edge_pixels = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
        bg_color = np.mean(edge_pixels, axis=0).astype(int)
        
        paths.append({
            'points': np.array([[0, 0], [w, 0], [w, h], [0, h]]),
            'color': f"#{bg_color[0]:02x}{bg_color[1]:02x}{bg_color[2]:02x}",
            'area': h * w,
            'layer': 'base'
        })
        
        return paths
    
    def _foreground(self, img, regions, h, w, level_config):
        """前景处理 - 并行"""
        
        tasks = [(r['mask'], r['color'], r['area'], img, h, w, level_config) for r in regions]
        
        all_paths = []
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            results = list(executor.map(process_region_with_level, tasks))
            for paths in results:
                all_paths.extend(paths)
        
        return all_paths
    
    def _create_svg(self, paths, width, height, output_path):
        """创建SVG"""
        dwg = svgwrite.Drawing(output_path, size=(width, height))
        dwg.viewbox(0, 0, width, height)
        
        layer_order = {'base': -1, 'background': 0, 'foreground': 1, 'detail': 2}
        paths.sort(key=lambda x: (layer_order.get(x.get('layer', 'detail'), 2), -x['area']))
        
        path_count = 0
        for path_data in paths:
            points = path_data['points']
            color = path_data['color']
            
            if len(points) < 3:
                continue
            
            path_d = points_to_path(points)
            if path_d:
                dwg.add(dwg.path(d=path_d, fill=color, stroke=color, stroke_width=1))
                path_count += 1
        
        dwg.save()
        
        return {
            'paths': path_count,
            'size_kb': Path(output_path).stat().st_size / 1024
        }


def create_comparison_figure(image_path, results, output_path):
    """创建科研组图 - 5行3列布局（1原图 + 14 SVG）"""
    
    print("\n📊 Creating comparison figure...")
    
    # 读取原图
    original = Image.open(image_path).convert('RGB')
    
    # 创建5行3列的图
    fig, axes = plt.subplots(5, 3, figsize=(15, 25))
    fig.patch.set_facecolor('white')
    
    # 获取原图大小
    import os
    orig_size = os.path.getsize(image_path) / 1024
    
    # (a) 原始图片
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('(a) Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 0].text(0.5, -0.05, f'{orig_size:.0f} KB', 
                    transform=axes[0, 0].transAxes, ha='center', fontsize=10)
    
    # (b-o) 14个SVG级别
    labels = [chr(ord('b') + i) for i in range(14)]  # b, c, d, ..., o
    
    from cairosvg import svg2png
    from io import BytesIO
    
    for i, result in enumerate(results):
        # 计算位置：从(0,1)开始
        pos = i + 1  # 跳过(0,0)
        row = pos // 3
        col = pos % 3
        
        # 将SVG转为PNG用于显示
        svg_path = result['svg_path']
        png_data = svg2png(url=str(svg_path), output_width=600)
        svg_img = Image.open(BytesIO(png_data)).convert('RGB')
        
        axes[row, col].imshow(svg_img)
        axes[row, col].set_title(f"({labels[i]}) {result['name']}", fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
        
        # 格式化大小显示
        size_kb = result['size_kb']
        if size_kb >= 1024:
            size_str = f"{size_kb/1024:.1f} MB"
        else:
            size_str = f"{size_kb:.0f} KB"
        
        # 格式化时间显示
        time_sec = result.get('time', 0)
        if time_sec >= 60:
            time_str = f"{time_sec/60:.1f}min"
        else:
            time_str = f"{time_sec:.1f}s"
        
        axes[row, col].text(0.5, -0.05, 
                            f"{result['paths']:,} paths | {size_str} | {time_str}",
                            transform=axes[row, col].transAxes, ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    print(f"✅ Saved: {output_path.replace('.png', '.pdf')}")
    
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate multi-level SVG comparison')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('-o', '--output', default='multi_level_output', help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vectorizer = MultiLevelVectorizer()
    results = vectorizer.generate_all_levels(args.image, output_dir)
    
    # 创建组图
    figure_path = str(output_dir / 'comparison_figure.png')
    create_comparison_figure(args.image, results, figure_path)
    
    # 打印总结
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"{'Level':<8} {'Name':<15} {'Paths':>10} {'Size (KB)':>12}")
    print("-"*50)
    for r in results:
        print(f"{r['level']:<8} {r['name']:<15} {r['paths']:>10,} {r['size_kb']:>12,.0f}")


if __name__ == "__main__":
    main()
