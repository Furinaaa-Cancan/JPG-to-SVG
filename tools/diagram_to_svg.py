#!/usr/bin/env python3
"""
科研流程图转SVG
支持彩色图像的高质量矢量化
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import subprocess
import tempfile
import os


def diagram_to_svg(input_path: str, output_path: str = None):
    """将科研流程图转换为SVG"""
    
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix('.svg')
    else:
        output_path = Path(output_path)
    
    print(f"\n🎨 Converting diagram to SVG")
    print(f"   Input: {input_path}")
    
    # 读取图像
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"   ❌ Failed to read image")
        return None
    
    h, w = img.shape[:2]
    print(f"   Size: {w}x{h}")
    
    # 提取主要颜色通道并分别矢量化
    # 对于科研图，通常有：黑色线条、红色、蓝色等
    
    # 转换到不同颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 定义颜色范围
    colors = {
        'red': ([0, 100, 100], [10, 255, 255]),      # 红色
        'red2': ([160, 100, 100], [180, 255, 255]),  # 红色（另一范围）
        'blue': ([100, 100, 100], [130, 255, 255]),  # 蓝色
        'black': None,  # 特殊处理
    }
    
    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">')
    svg_parts.append(f'  <rect width="{w}" height="{h}" fill="white"/>')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        
        # 1. 处理黑色/灰色线条（主体）
        print("   Processing black lines...")
        _, black_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 排除彩色区域
        for name, range_vals in colors.items():
            if range_vals:
                lower = np.array(range_vals[0])
                upper = np.array(range_vals[1])
                color_mask = cv2.inRange(hsv, lower, upper)
                black_mask = cv2.bitwise_and(black_mask, cv2.bitwise_not(color_mask))
        
        black_svg = process_mask_to_svg(black_mask, tmpdir, 'black', '#333333')
        if black_svg:
            svg_parts.append(f'  <g id="black_lines">{black_svg}</g>')
        
        # 2. 处理红色
        print("   Processing red areas...")
        red_mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        red_svg = process_mask_to_svg(red_mask, tmpdir, 'red', '#CC3333')
        if red_svg:
            svg_parts.append(f'  <g id="red_elements">{red_svg}</g>')
        
        # 3. 处理蓝色
        print("   Processing blue areas...")
        blue_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
        
        blue_svg = process_mask_to_svg(blue_mask, tmpdir, 'blue', '#3366CC')
        if blue_svg:
            svg_parts.append(f'  <g id="blue_elements">{blue_svg}</g>')
    
    svg_parts.append('</svg>')
    
    # 写入文件
    svg_content = '\n'.join(svg_parts)
    with open(output_path, 'w') as f:
        f.write(svg_content)
    
    size_kb = output_path.stat().st_size / 1024
    print(f"\n   ✅ Saved: {output_path}")
    print(f"   Size: {size_kb:.1f} KB")
    
    return output_path


def process_mask_to_svg(mask, tmpdir, name, color):
    """将mask转换为SVG路径"""
    
    if mask is None or np.sum(mask) == 0:
        return None
    
    # 形态学处理，清理噪点
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 保存为PBM（potrace输入格式）
    pbm_path = os.path.join(tmpdir, f'{name}.pbm')
    svg_path = os.path.join(tmpdir, f'{name}.svg')
    
    # 转为PBM格式
    h, w = mask.shape
    with open(pbm_path, 'wb') as f:
        f.write(f'P4\n{w} {h}\n'.encode())
        # Pack bits
        for row in mask:
            row_bits = (row > 127).astype(np.uint8)
            # Pack 8 pixels per byte
            packed = np.packbits(row_bits)
            f.write(packed.tobytes())
    
    # 用potrace转换
    result = subprocess.run([
        'potrace', pbm_path,
        '-s',  # SVG output
        '-o', svg_path,
        '-t', '2',  # 简化阈值
        '-O', '0.2',  # 优化曲线
    ], capture_output=True)
    
    if result.returncode != 0 or not os.path.exists(svg_path):
        return None
    
    # 读取并提取路径
    with open(svg_path, 'r') as f:
        content = f.read()
    
    # 提取<path>元素
    import re
    paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', content)
    
    if not paths:
        # 尝试另一种格式
        paths = re.findall(r'd="([^"]*)"', content)
    
    if paths:
        svg_paths = '\n    '.join([f'<path d="{p}" fill="{color}" stroke="none"/>' for p in paths])
        return svg_paths
    
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert diagram to SVG')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output SVG path')
    args = parser.parse_args()
    
    result = diagram_to_svg(args.input, args.output)
    
    if result:
        # 打开查看
        subprocess.run(['open', str(result)])


if __name__ == "__main__":
    main()
