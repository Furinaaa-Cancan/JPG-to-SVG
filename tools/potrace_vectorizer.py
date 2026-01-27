#!/usr/bin/env python3
"""
使用Potrace进行高质量矢量化
Potrace是工业级矢量化工具，精度更高
"""

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile
import xml.etree.ElementTree as ET

def image_to_potrace_svg(img_path, output_svg, color_mode='color'):
    """使用Potrace将图像转换为SVG"""
    
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    if color_mode == 'color':
        return color_potrace(img_rgb, output_svg)
    else:
        return bw_potrace(img_path, output_svg)

def bw_potrace(img_path, output_svg):
    """黑白模式Potrace"""
    with tempfile.NamedTemporaryFile(suffix='.pbm', delete=False) as tmp_pbm:
        pbm_path = tmp_pbm.name
    
    # 转换为PBM格式
    img = Image.open(img_path).convert('L')
    # 二值化
    threshold = 200
    img = img.point(lambda p: 255 if p > threshold else 0, '1')
    img.save(pbm_path)
    
    # 调用Potrace
    result = subprocess.run(
        ['potrace', pbm_path, '-s', '-o', str(output_svg), 
         '--turdsize', '2', '--alphamax', '1.0', '--opttolerance', '0.2'],
        capture_output=True, text=True
    )
    
    Path(pbm_path).unlink(missing_ok=True)
    return result.returncode == 0

def color_potrace(img_rgb, output_svg):
    """彩色分层Potrace矢量化"""
    height, width = img_rgb.shape[:2]
    
    # 定义颜色层
    color_layers = {
        'red': ([150, 0, 0], [255, 100, 100], '#c83232'),
        'blue': ([0, 50, 100], [100, 150, 255], '#3264c8'),
        'orange': ([180, 80, 0], [255, 180, 100], '#dc9650'),
        'gray': ([60, 60, 60], [200, 200, 200], '#a0a0a0'),
        'black': ([0, 0, 0], [60, 60, 60], '#323232'),
    }
    
    svg_paths = []
    
    for name, (lower, upper, hex_color) in color_layers.items():
        # 颜色分割
        mask = cv2.inRange(img_rgb, np.array(lower), np.array(upper))
        
        # 形态学清理
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        if np.sum(mask) < 500:
            continue
        
        # 保存为PBM
        with tempfile.NamedTemporaryFile(suffix='.pbm', delete=False) as tmp:
            pbm_path = tmp.name
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
            tmp_svg = tmp.name
        
        # 转换mask为PBM
        mask_img = Image.fromarray(mask)
        mask_img.save(pbm_path)
        
        # 调用Potrace
        result = subprocess.run(
            ['potrace', pbm_path, '-s', '-o', tmp_svg,
             '--turdsize', '2', '--alphamax', '1.0', '--opttolerance', '0.2'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            # 读取生成的SVG路径
            try:
                tree = ET.parse(tmp_svg)
                root = tree.getroot()
                ns = {'svg': 'http://www.w3.org/2000/svg'}
                
                for path in root.findall('.//svg:path', ns) or root.findall('.//path'):
                    d = path.get('d')
                    if d:
                        svg_paths.append({
                            'color': hex_color,
                            'name': name,
                            'd': d
                        })
            except:
                pass
        
        Path(pbm_path).unlink(missing_ok=True)
        Path(tmp_svg).unlink(missing_ok=True)
    
    # 组合生成最终SVG
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <rect fill="white" width="{width}" height="{height}"/>
'''
    
    for layer_name in color_layers.keys():
        layer_paths = [p for p in svg_paths if p['name'] == layer_name]
        if layer_paths:
            svg_content += f'  <g id="{layer_name}" fill="{layer_paths[0]["color"]}">\n'
            for p in layer_paths:
                svg_content += f'    <path d="{p["d"]}"/>\n'
            svg_content += '  </g>\n'
    
    svg_content += '</svg>'
    
    with open(output_svg, 'w') as f:
        f.write(svg_content)
    
    return True

def add_ocr_text_to_svg(svg_path, img_path):
    """将OCR文字添加到SVG"""
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 放大图像进行OCR
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((img_pil.width * 2, img_pil.height * 2), Image.LANCZOS)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        img_pil.save(tmp_path)
    
    # OCR
    result = subprocess.run(
        ['tesseract', tmp_path, 'stdout', '--psm', '6', '-l', 'eng', 'tsv'],
        capture_output=True, text=True
    )
    
    Path(tmp_path).unlink(missing_ok=True)
    
    text_elements = []
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for line in lines[1:]:
            cols = line.split('\t')
            if len(cols) >= 12:
                text = cols[11].strip()
                try:
                    conf = float(cols[10])
                except:
                    conf = -1
                
                if text and conf > 50:  # 更高置信度阈值
                    x = int(cols[6]) // 2
                    y = int(cols[7]) // 2
                    h = int(cols[9]) // 2
                    font_size = max(8, min(h, 14))
                    text_elements.append(
                        f'    <text x="{x}" y="{y + int(h*0.8)}" '
                        f'font-family="Arial" font-size="{font_size}" fill="#333">{text}</text>'
                    )
    
    # 读取原SVG并添加文字层
    with open(svg_path, 'r') as f:
        content = f.read()
    
    # 在</svg>前插入文字层
    text_layer = '\n  <g id="text_layer">\n' + '\n'.join(text_elements) + '\n  </g>\n'
    content = content.replace('</svg>', text_layer + '</svg>')
    
    with open(svg_path, 'w') as f:
        f.write(content)
    
    return len(text_elements)

def main():
    input_path = Path("/Volumes/Seagate/SAM3/01_input/科研绘图1.png")
    output_dir = Path("/Volumes/Seagate/SAM3/02_output")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("Potrace高质量矢量化")
    print("=" * 60)
    
    # 彩色分层矢量化
    color_svg = output_dir / f"科研绘图_potrace_color_{timestamp}.svg"
    print("\n1. 彩色分层矢量化...")
    color_potrace(cv2.cvtColor(cv2.imread(str(input_path)), cv2.COLOR_BGR2RGB), color_svg)
    print(f"   ✅ {color_svg}")
    
    # 添加OCR文字
    print("\n2. 添加OCR文字...")
    text_count = add_ocr_text_to_svg(color_svg, input_path)
    print(f"   ✅ 添加了 {text_count} 个文字元素")
    
    # 黑白轮廓版本
    bw_svg = output_dir / f"科研绘图_potrace_bw_{timestamp}.svg"
    print("\n3. 黑白轮廓版本...")
    bw_potrace(input_path, bw_svg)
    print(f"   ✅ {bw_svg}")
    
    print(f"\n📁 输出文件:")
    print(f"   彩色版: {color_svg}")
    print(f"   黑白版: {bw_svg}")

if __name__ == "__main__":
    main()
