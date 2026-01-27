#!/usr/bin/env python3
"""
科研绘图精确矢量化工具
针对技术图纸、电路图、示意图的高质量SVG转换
"""

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import svgwrite

def load_image(path):
    """加载图像"""
    img = cv2.imread(str(path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb

def extract_colors(img_rgb, n_colors=8):
    """提取主要颜色"""
    from sklearn.cluster import KMeans
    
    pixels = img_rgb.reshape(-1, 3)
    # 过滤白色和接近白色的像素
    mask = np.all(pixels < 250, axis=1)
    colored_pixels = pixels[mask]
    
    if len(colored_pixels) < n_colors:
        return [(255, 255, 255)]
    
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(colored_pixels)
    
    colors = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in colors]

def color_segment(img_rgb, target_color, tolerance=30):
    """根据颜色分割"""
    lower = np.array([max(0, c - tolerance) for c in target_color])
    upper = np.array([min(255, c + tolerance) for c in target_color])
    mask = cv2.inRange(img_rgb, lower, upper)
    return mask

def extract_edges(img_gray):
    """提取边缘"""
    # Canny边缘检测
    edges = cv2.Canny(img_gray, 50, 150)
    return edges

def contours_to_svg_path(contours, simplify_epsilon=1.5):
    """将轮廓转换为SVG路径"""
    paths = []
    for contour in contours:
        if len(contour) < 3:
            continue
        
        # 简化轮廓
        approx = cv2.approxPolyDP(contour, simplify_epsilon, True)
        if len(approx) < 3:
            continue
        
        points = approx.squeeze()
        if len(points.shape) == 1:
            continue
        
        # 构建路径
        path_data = f"M {points[0][0]},{points[0][1]}"
        for point in points[1:]:
            path_data += f" L {point[0]},{point[1]}"
        path_data += " Z"
        paths.append(path_data)
    
    return paths

def detect_shapes(mask):
    """检测形状并分类"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = {
        'rectangles': [],
        'circles': [],
        'triangles': [],
        'lines': [],
        'complex': []
    }
    
    for contour in contours:
        if cv2.contourArea(contour) < 50:  # 过滤噪点
            continue
        
        # 近似轮廓
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        vertices = len(approx)
        
        if vertices == 3:
            shapes['triangles'].append(contour)
        elif vertices == 4:
            # 检查是否为矩形
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.8 < aspect_ratio < 1.2:
                shapes['rectangles'].append(contour)
            else:
                shapes['rectangles'].append(contour)
        elif vertices > 6:
            # 检查是否为圆形
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.7:
                    shapes['circles'].append(contour)
                else:
                    shapes['complex'].append(contour)
            else:
                shapes['complex'].append(contour)
        else:
            shapes['complex'].append(contour)
    
    return shapes

def detect_lines(img_gray):
    """检测直线"""
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    return lines

def detect_arrows(img_gray, img_rgb):
    """检测箭头"""
    # 使用模板匹配或特征检测
    # 简化：基于三角形检测
    edges = cv2.Canny(img_gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    arrows = []
    for contour in contours:
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 3:  # 三角形可能是箭头头部
            arrows.append(contour)
    
    return arrows

def rgb_to_hex(rgb):
    """RGB转十六进制"""
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def create_precise_svg(img_rgb, output_path, simplify=1.5):
    """创建精确的SVG"""
    height, width = img_rgb.shape[:2]
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 创建SVG
    dwg = svgwrite.Drawing(str(output_path), size=(width, height), viewBox=f"0 0 {width} {height}")
    
    # 添加白色背景
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='white'))
    
    # 1. 提取主要颜色
    print("提取主要颜色...")
    colors = extract_colors(img_rgb, n_colors=10)
    
    # 定义目标颜色（科研绘图常用色）
    target_colors = {
        'red': (200, 50, 50),
        'blue': (50, 100, 180),
        'dark_blue': (30, 60, 120),
        'orange': (220, 150, 80),
        'gray': (100, 100, 100),
        'black': (30, 30, 30),
        'light_gray': (180, 180, 180),
    }
    
    # 2. 按颜色分割并矢量化
    print("按颜色分割...")
    layers = {}
    
    for color_name, target in target_colors.items():
        mask = color_segment(img_rgb, target, tolerance=50)
        
        # 形态学操作清理
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        if np.sum(mask) > 100:  # 有足够的像素
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                layers[color_name] = {
                    'contours': contours,
                    'color': target
                }
                print(f"  ✓ {color_name}: {len(contours)} 个轮廓")
    
    # 3. 添加到SVG
    print("生成SVG路径...")
    for layer_name, data in layers.items():
        group = dwg.g(id=layer_name)
        hex_color = rgb_to_hex(data['color'])
        
        paths = contours_to_svg_path(data['contours'], simplify)
        for path_d in paths:
            group.add(dwg.path(d=path_d, fill=hex_color, stroke='none', fill_opacity=0.9))
        
        dwg.add(group)
    
    # 4. 添加边缘线条
    print("提取边缘线条...")
    edges = extract_edges(img_gray)
    edge_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    lines_group = dwg.g(id='edges', stroke='#333333', stroke_width=1, fill='none')
    for contour in edge_contours:
        if cv2.arcLength(contour, False) > 20:  # 过滤短线
            approx = cv2.approxPolyDP(contour, 1.0, False)
            if len(approx) >= 2:
                points = approx.squeeze()
                if len(points.shape) == 1:
                    continue
                polyline = dwg.polyline(points=points.tolist(), fill='none')
                lines_group.add(polyline)
    dwg.add(lines_group)
    
    # 5. 检测并添加直线
    print("检测直线...")
    lines = detect_lines(img_gray)
    if lines is not None:
        straight_lines = dwg.g(id='straight_lines', stroke='#666666', stroke_width=1)
        for line in lines[:50]:  # 限制数量
            x1, y1, x2, y2 = [int(v) for v in line[0]]
            straight_lines.add(dwg.line(start=(x1, y1), end=(x2, y2)))
        dwg.add(straight_lines)
    
    # 保存
    dwg.save()
    print(f"✅ SVG已保存: {output_path}")
    
    return output_path

def create_layered_svg(img_path, output_dir):
    """创建分层SVG（每个颜色一层）"""
    img, img_rgb = load_image(img_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_path = Path(output_dir) / f"科研绘图_精确_{timestamp}.svg"
    create_precise_svg(img_rgb, output_path)
    
    # 同时创建简化版本
    simple_path = Path(output_dir) / f"科研绘图_简化_{timestamp}.svg"
    create_precise_svg(img_rgb, simple_path, simplify=3.0)
    
    return output_path, simple_path

def main():
    input_path = "/Volumes/Seagate/SAM3/01_input/科研绘图1.png"
    output_dir = Path("/Volumes/Seagate/SAM3/02_output")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 50)
    print("科研绘图精确矢量化")
    print("=" * 50)
    
    precise_svg, simple_svg = create_layered_svg(input_path, output_dir)
    
    print(f"\n📁 输出文件:")
    print(f"  精确版: {precise_svg}")
    print(f"  简化版: {simple_svg}")

if __name__ == "__main__":
    main()
