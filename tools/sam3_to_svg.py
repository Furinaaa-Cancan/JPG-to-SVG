#!/usr/bin/env python3
"""
SAM3分割 + SVG矢量化工具
用于科研绘图的复刻
"""

import sys
sys.path.insert(0, '/Volumes/Seagate/SAM3/models/sam3')

import numpy as np
from PIL import Image
import torch
from pathlib import Path
from datetime import datetime

# SAM3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def load_sam3_model(device='mps'):
    """加载SAM3模型和处理器"""
    print(f"Loading SAM3 on {device}...")
    model = build_sam3_image_model(device=device, load_from_HF=True)
    model.eval()
    processor = Sam3Processor(model, device=device, confidence_threshold=0.3)
    print("✅ SAM3 loaded!")
    return processor

def segment_with_sam3(processor, image_path, prompts=None):
    """
    使用SAM3进行图像分割
    prompts: 文本提示列表，如 ["circuit", "sensor", "arrow", "text"]
    """
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")
    
    results = {}
    
    if prompts:
        # 使用文本提示分割
        for prompt in prompts:
            print(f"Segmenting with prompt: '{prompt}'...")
            try:
                # 设置图像
                state = processor.set_image(image)
                # 设置文本提示并运行推理
                state = processor.set_text_prompt(prompt, state)
                
                if 'masks' in state and len(state['masks']) > 0:
                    masks = state['masks'].cpu().numpy()
                    scores = state['scores'].cpu().numpy()
                    
                    # 合并所有检测到的mask
                    combined_mask = np.zeros(masks[0].shape[-2:], dtype=bool)
                    for i in range(len(masks)):
                        mask = masks[i].squeeze()
                        combined_mask = combined_mask | mask
                    
                    best_score = scores.max() if len(scores) > 0 else 0.0
                    results[prompt] = {
                        'mask': combined_mask,
                        'score': float(best_score),
                        'count': len(masks)
                    }
                    print(f"  ✓ Found {len(masks)} masks for '{prompt}', best score: {best_score:.3f}")
                else:
                    print(f"  ✗ No masks found for '{prompt}'")
            except Exception as e:
                print(f"  ✗ Failed for '{prompt}': {e}")
                import traceback
                traceback.print_exc()
    
    return results, image

def mask_to_svg_path(mask, simplify_tolerance=2.0):
    """将二值mask转换为SVG路径"""
    import cv2
    
    # 确保是uint8类型
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    paths = []
    for contour in contours:
        if len(contour) < 3:
            continue
        
        # 简化轮廓
        epsilon = simplify_tolerance
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 3:
            continue
        
        # 转换为SVG路径
        points = approx.squeeze()
        if len(points.shape) == 1:
            continue
            
        path_data = f"M {points[0][0]},{points[0][1]}"
        for point in points[1:]:
            path_data += f" L {point[0]},{point[1]}"
        path_data += " Z"
        
        paths.append(path_data)
    
    return paths

def create_svg(image_size, segments, output_path):
    """创建SVG文件"""
    width, height = image_size
    
    # SVG头
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <style>
      .segment {{ fill-opacity: 0.7; stroke: #333; stroke-width: 1; }}
    </style>
  </defs>
'''
    
    # 颜色映射
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    
    for i, (name, data) in enumerate(segments.items()):
        color = colors[i % len(colors)]
        paths = mask_to_svg_path(data['mask'])
        
        svg_content += f'  <!-- {name} (score: {data.get("score", 0):.3f}) -->\n'
        svg_content += f'  <g id="{name}" class="segment">\n'
        
        for path in paths:
            svg_content += f'    <path d="{path}" fill="{color}"/>\n'
        
        svg_content += '  </g>\n'
    
    svg_content += '</svg>'
    
    with open(output_path, 'w') as f:
        f.write(svg_content)
    
    print(f"✅ SVG saved to: {output_path}")
    return output_path

def main():
    # 输入图像
    input_path = "/Volumes/Seagate/SAM3/01_input/科研绘图1.png"
    output_dir = Path("/Volumes/Seagate/SAM3/02_output")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 加载模型 (MPS有bug，使用CPU)
    device = 'cpu'
    processor = load_sam3_model(device)
    
    # 定义要分割的元素 - 针对惠斯通电桥科研图
    prompts = [
        "sensor",
        "circuit diagram", 
        "resistor",
        "red rectangle",
        "blue rectangle",
        "arrow",
        "wire",
        "diamond shape"
    ]
    
    # 分割
    segments, image = segment_with_sam3(processor, input_path, prompts)
    
    # 保存分割结果可视化
    if segments:
        # 创建SVG
        svg_path = output_dir / f"科研绘图1_sam3_{timestamp}.svg"
        create_svg(image.size, segments, svg_path)
        
        # 保存mask可视化
        vis_path = output_dir / f"科研绘图1_masks_{timestamp}.png"
        save_mask_visualization(image, segments, vis_path)
        
        print(f"\n📊 分割结果汇总:")
        for name, data in segments.items():
            print(f"  - {name}: score={data['score']:.3f}, count={data.get('count', 1)}")
    else:
        print("❌ No segments found!")

def save_mask_visualization(image, segments, output_path):
    """保存mask可视化"""
    import cv2
    
    image_np = np.array(image)
    overlay = image_np.copy()
    
    colors = [
        (231, 76, 60),   # red
        (52, 152, 219),  # blue
        (46, 204, 113),  # green
        (243, 156, 18),  # orange
        (155, 89, 182),  # purple
        (26, 188, 156),  # teal
    ]
    
    for i, (name, data) in enumerate(segments.items()):
        mask = data['mask']
        color = colors[i % len(colors)]
        
        # 应用颜色到mask区域
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = (
            overlay[mask_bool] * 0.5 + 
            np.array(color) * 0.5
        ).astype(np.uint8)
    
    # 保存
    result = Image.fromarray(overlay)
    result.save(output_path)
    print(f"✅ Visualization saved to: {output_path}")

if __name__ == "__main__":
    main()
