#!/usr/bin/env python3
"""
科研绘图专业矢量化工具
- OCR文字识别 → SVG text
- 精确形状检测 → 几何图元
- 颜色分层 → 可编辑图层
"""

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple, Dict
import svgwrite

@dataclass
class TextRegion:
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float

@dataclass 
class Shape:
    type: str  # rect, circle, line, polygon, diamond
    points: List[Tuple[int, int]]
    color: Tuple[int, int, int]
    area: float

def ocr_extract_text(img_rgb) -> List[TextRegion]:
    """使用Tesseract命令行提取文字及位置"""
    text_regions = []
    
    # 图像预处理：放大并增强对比度
    img_pil = Image.fromarray(img_rgb)
    # 放大2倍提高OCR准确率
    img_pil = img_pil.resize((img_pil.width * 2, img_pil.height * 2), Image.LANCZOS)
    
    # 保存临时图像
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        img_pil.save(tmp_path)
    
    try:
        # 调用tesseract生成TSV输出 (psm 6效果更好)
        result = subprocess.run(
            ['tesseract', tmp_path, 'stdout', '--psm', '6', '-l', 'eng', 'tsv'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                # TSV固定格式: level,page_num,block_num,par_num,line_num,word_num,left,top,width,height,conf,text
                for line in lines[1:]:
                    cols = line.split('\t')
                    if len(cols) >= 12:
                        text = cols[11].strip()
                        try:
                            conf = float(cols[10])
                        except:
                            conf = -1
                        
                        if text and conf > 30:
                            try:
                                # 除以2因为图像放大了2倍
                                text_regions.append(TextRegion(
                                    text=text,
                                    x=int(cols[6]) // 2,
                                    y=int(cols[7]) // 2,
                                    width=int(cols[8]) // 2,
                                    height=int(cols[9]) // 2,
                                    confidence=conf
                                ))
                            except:
                                pass
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    
    return text_regions

def detect_geometric_shapes(img_gray, img_rgb) -> List[Shape]:
    """检测几何形状"""
    shapes = []
    
    # 边缘检测
    edges = cv2.Canny(img_gray, 50, 150)
    
    # 膨胀边缘以连接断开的线
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # 过滤噪点
            continue
        
        # 近似轮廓
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        # 获取颜色（轮廓内部平均颜色）
        mask = np.zeros(img_gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_color = cv2.mean(img_rgb, mask=mask)[:3]
        color = tuple(int(c) for c in mean_color)
        
        points = [tuple(p[0]) for p in approx]
        
        # 形状分类
        if vertices == 3:
            shape_type = 'triangle'
        elif vertices == 4:
            # 检查是否为菱形
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / h if h > 0 else 1
            
            # 计算角度
            if 0.8 < aspect < 1.2:
                # 检查是否为菱形（对角线垂直）
                cx, cy = x + w/2, y + h/2
                corners = np.array(points)
                dists = np.sqrt(np.sum((corners - [cx, cy])**2, axis=1))
                if np.std(dists) < 5:  # 到中心距离相近
                    shape_type = 'diamond'
                else:
                    shape_type = 'rect'
            else:
                shape_type = 'rect'
        elif vertices > 6:
            # 检查是否为圆
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            if circularity > 0.7:
                shape_type = 'circle'
            else:
                shape_type = 'polygon'
        else:
            shape_type = 'polygon'
        
        shapes.append(Shape(
            type=shape_type,
            points=points,
            color=color,
            area=area
        ))
    
    return shapes

def detect_lines(img_gray) -> List[Tuple[int, int, int, int]]:
    """检测直线"""
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                            minLineLength=20, maxLineGap=5)
    
    if lines is None:
        return []
    
    return [(int(l[0][0]), int(l[0][1]), int(l[0][2]), int(l[0][3])) for l in lines]

def detect_arrows(img_gray) -> List[Dict]:
    """检测箭头"""
    arrows = []
    edges = cv2.Canny(img_gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 箭头通常是三角形或7边形（三角形+线）
        if 3 <= len(approx) <= 7:
            area = cv2.contourArea(contour)
            if 50 < area < 500:  # 箭头大小范围
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    arrows.append({
                        'center': (cx, cy),
                        'contour': contour,
                        'points': [tuple(p[0]) for p in approx]
                    })
    
    return arrows

def color_segmentation(img_rgb) -> Dict[str, np.ndarray]:
    """颜色分割"""
    # 定义科研绘图常用颜色
    color_ranges = {
        'red': ([150, 30, 30], [255, 100, 100]),
        'blue': ([30, 60, 120], [100, 150, 220]),
        'light_blue': ([100, 150, 200], [180, 220, 255]),
        'orange': ([180, 100, 50], [255, 180, 120]),
        'green': ([30, 100, 30], [100, 200, 100]),
        'black': ([0, 0, 0], [60, 60, 60]),
        'gray': ([80, 80, 80], [180, 180, 180]),
    }
    
    masks = {}
    for name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(img_rgb, np.array(lower), np.array(upper))
        if np.sum(mask) > 500:  # 有足够像素
            masks[name] = mask
    
    return masks

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def create_professional_svg(img_path, output_path):
    """创建专业级SVG"""
    # 加载图像
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape
    
    print(f"图像尺寸: {width} x {height}")
    
    # 创建SVG
    dwg = svgwrite.Drawing(str(output_path), size=(width, height), 
                          viewBox=f"0 0 {width} {height}")
    
    # 添加样式
    dwg.defs.add(dwg.style('''
        .text-label { font-family: Arial, sans-serif; }
        .shape { stroke-width: 1; }
    '''))
    
    # 白色背景
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='white'))
    
    # 1. OCR提取文字
    print("正在识别文字...")
    text_regions = ocr_extract_text(img_rgb)
    print(f"  识别到 {len(text_regions)} 个文字区域")
    
    text_group = dwg.g(id='text_layer')
    for tr in text_regions:
        # 估算字体大小
        font_size = max(8, min(tr.height, 24))
        text_elem = dwg.text(
            tr.text,
            insert=(tr.x, tr.y + tr.height * 0.8),  # 基线调整
            font_size=font_size,
            font_family='Arial, sans-serif',
            fill='#333333'
        )
        text_group.add(text_elem)
    dwg.add(text_group)
    
    # 2. 颜色分割
    print("正在分割颜色区域...")
    color_masks = color_segmentation(img_rgb)
    
    for color_name, mask in color_masks.items():
        # 形态学清理
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # 获取该颜色的代表色
        color_mean = cv2.mean(img_rgb, mask=mask)[:3]
        hex_color = rgb_to_hex(color_mean)
        
        group = dwg.g(id=f'{color_name}_layer', fill=hex_color, fill_opacity=0.9)
        
        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue
            
            # 简化轮廓
            epsilon = 1.5
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) < 3:
                continue
            
            points = [(int(p[0][0]), int(p[0][1])) for p in approx]
            
            # 创建路径
            path_data = f"M {points[0][0]},{points[0][1]}"
            for px, py in points[1:]:
                path_data += f" L {px},{py}"
            path_data += " Z"
            
            group.add(dwg.path(d=path_data, stroke='none'))
        
        dwg.add(group)
        print(f"  {color_name}: {len(contours)} 个区域")
    
    # 3. 检测直线
    print("正在检测直线...")
    lines = detect_lines(img_gray)
    
    lines_group = dwg.g(id='lines_layer', stroke='#444444', stroke_width=1, fill='none')
    for x1, y1, x2, y2 in lines[:100]:  # 限制数量
        lines_group.add(dwg.line(start=(x1, y1), end=(x2, y2)))
    dwg.add(lines_group)
    print(f"  检测到 {len(lines)} 条直线")
    
    # 4. 检测几何形状
    print("正在检测几何形状...")
    shapes = detect_geometric_shapes(img_gray, img_rgb)
    
    shapes_group = dwg.g(id='shapes_layer')
    shape_counts = {}
    
    for shape in shapes:
        shape_counts[shape.type] = shape_counts.get(shape.type, 0) + 1
        hex_color = rgb_to_hex(shape.color)
        
        if shape.type == 'circle' and len(shape.points) > 0:
            # 计算圆心和半径
            pts = np.array(shape.points)
            cx, cy = pts.mean(axis=0)
            radius = np.sqrt(shape.area / np.pi)
            shapes_group.add(dwg.circle(
                center=(int(cx), int(cy)), r=int(radius),
                fill=hex_color, stroke='#333', stroke_width=1
            ))
        elif shape.type in ['rect', 'diamond']:
            if len(shape.points) >= 4:
                path_data = f"M {shape.points[0][0]},{shape.points[0][1]}"
                for px, py in shape.points[1:]:
                    path_data += f" L {px},{py}"
                path_data += " Z"
                shapes_group.add(dwg.path(
                    d=path_data, fill=hex_color, 
                    stroke='#333', stroke_width=1
                ))
    
    dwg.add(shapes_group)
    print(f"  形状统计: {shape_counts}")
    
    # 保存
    dwg.save()
    print(f"\n✅ SVG已保存: {output_path}")
    
    return text_regions, shapes

def main():
    input_path = Path("/Volumes/Seagate/SAM3/01_input/科研绘图1.png")
    output_dir = Path("/Volumes/Seagate/SAM3/02_output")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"科研绘图_OCR_{timestamp}.svg"
    
    print("=" * 60)
    print("科研绘图专业矢量化 (含OCR)")
    print("=" * 60)
    
    text_regions, shapes = create_professional_svg(input_path, output_path)
    
    # 输出文字识别结果
    print("\n📝 识别的文字内容:")
    for tr in sorted(text_regions, key=lambda x: (x.y, x.x)):
        print(f"  [{tr.confidence}%] '{tr.text}' at ({tr.x}, {tr.y})")
    
    print(f"\n📁 输出: {output_path}")

if __name__ == "__main__":
    main()
