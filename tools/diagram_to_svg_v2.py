#!/usr/bin/env python3
"""
科研流程图转SVG v2
使用OpenCV轮廓 + 贝塞尔曲线拟合
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import svgwrite
from scipy.interpolate import splprep, splev


def simplify_contour(contour, tolerance=2.0):
    """简化轮廓点"""
    epsilon = tolerance
    return cv2.approxPolyDP(contour, epsilon, True)


def contour_to_path(contour):
    """将轮廓转换为SVG路径"""
    if len(contour) < 3:
        return None
    
    points = contour.squeeze()
    if len(points.shape) == 1:
        return None
    
    # 构建路径
    path_data = f"M {points[0][0]},{points[0][1]} "
    
    for i in range(1, len(points)):
        path_data += f"L {points[i][0]},{points[i][1]} "
    
    path_data += "Z"
    return path_data


def smooth_contour_to_path(contour, smoothing=0.5):
    """将轮廓转换为平滑的贝塞尔曲线路径"""
    if len(contour) < 4:
        return contour_to_path(contour)
    
    points = contour.squeeze()
    if len(points.shape) == 1 or len(points) < 4:
        return contour_to_path(contour)
    
    try:
        # 闭合曲线
        x = np.append(points[:, 0], points[0, 0])
        y = np.append(points[:, 1], points[0, 1])
        
        # B样条拟合
        tck, u = splprep([x, y], s=smoothing * len(points), per=True)
        
        # 生成平滑点
        u_new = np.linspace(0, 1, max(20, len(points) // 2))
        x_new, y_new = splev(u_new, tck)
        
        # 构建路径
        path_data = f"M {x_new[0]:.1f},{y_new[0]:.1f} "
        for i in range(1, len(x_new)):
            path_data += f"L {x_new[i]:.1f},{y_new[i]:.1f} "
        path_data += "Z"
        
        return path_data
    except:
        return contour_to_path(contour)


def diagram_to_svg(input_path: str, output_path: str = None):
    """将科研流程图转换为SVG"""
    
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix('.svg')
    else:
        output_path = Path(output_path)
    
    print(f"\n🎨 Converting diagram to SVG (v2)")
    print(f"   Input: {input_path}")
    
    # 读取图像
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"   ❌ Failed to read image")
        return None
    
    h, w = img.shape[:2]
    print(f"   Size: {w}x{h}")
    
    # 创建SVG
    dwg = svgwrite.Drawing(str(output_path), size=(w, h), viewBox=f"0 0 {w} {h}")
    dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill='white'))
    
    # 转换颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 颜色定义
    color_ranges = {
        'red': {
            'ranges': [
                ([0, 80, 80], [10, 255, 255]),
                ([160, 80, 80], [180, 255, 255])
            ],
            'fill': '#CC3333',
            'name': 'Red Elements'
        },
        'blue': {
            'ranges': [
                ([90, 50, 50], [130, 255, 255])
            ],
            'fill': '#3366CC',
            'name': 'Blue Elements'
        },
        'light_blue': {
            'ranges': [
                ([90, 20, 200], [130, 80, 255])
            ],
            'fill': '#99CCFF',
            'name': 'Light Blue'
        }
    }
    
    total_paths = 0
    
    # 处理每种颜色
    for color_name, color_info in color_ranges.items():
        print(f"   Processing {color_info['name']}...")
        
        # 合并多个颜色范围
        mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in color_info['ranges']:
            range_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, range_mask)
        
        if np.sum(mask) == 0:
            continue
        
        # 形态学处理
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建组
        group = dwg.g(id=color_name)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:  # 跳过太小的区域
                continue
            
            # 简化轮廓
            simplified = simplify_contour(contour, 1.5)
            path_data = smooth_contour_to_path(simplified, 0.3)
            
            if path_data:
                group.add(dwg.path(d=path_data, fill=color_info['fill'], stroke='none'))
                total_paths += 1
        
        dwg.add(group)
    
    # 处理黑色/深色线条（最后绘制，在顶层）
    print("   Processing black lines...")
    
    # 提取深色区域
    _, black_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 排除彩色区域
    for color_name, color_info in color_ranges.items():
        for lower, upper in color_info['ranges']:
            color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            # 膨胀一点以确保完全排除
            color_mask = cv2.dilate(color_mask, np.ones((3, 3), np.uint8))
            black_mask = cv2.bitwise_and(black_mask, cv2.bitwise_not(color_mask))
    
    # 形态学处理
    kernel = np.ones((2, 2), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    
    # 找轮廓
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    black_group = dwg.g(id='black_lines')
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5:
            continue
        
        simplified = simplify_contour(contour, 1.0)
        path_data = smooth_contour_to_path(simplified, 0.2)
        
        if path_data:
            black_group.add(dwg.path(d=path_data, fill='#333333', stroke='none'))
            total_paths += 1
    
    dwg.add(black_group)
    
    # 保存
    dwg.save()
    
    size_kb = output_path.stat().st_size / 1024
    print(f"\n   ✅ Saved: {output_path}")
    print(f"   Paths: {total_paths}")
    print(f"   Size: {size_kb:.1f} KB")
    
    return output_path


def main():
    import argparse
    import subprocess
    
    parser = argparse.ArgumentParser(description='Convert diagram to SVG v2')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output SVG path')
    args = parser.parse_args()
    
    result = diagram_to_svg(args.input, args.output)
    
    if result:
        subprocess.run(['open', str(result)])


if __name__ == "__main__":
    main()
