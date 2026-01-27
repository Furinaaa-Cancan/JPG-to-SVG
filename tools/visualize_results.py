#!/usr/bin/env python3
"""
可视化语义分层结果
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from pathlib import Path


def visualize_semantic_layers():
    """可视化语义分层结果"""
    
    # 读取原始图像
    original_path = "01_输入图片/Ladygaga_2.jpg"
    original = Image.open(original_path)
    original_np = np.array(original)
    
    # 获取最新的结果
    masks_dir = Path("02_输出结果/masks")
    json_files = list(masks_dir.glob("*_semantic_layers.json"))
    if not json_files:
        print("No semantic layers found! Please run test_modules.py first.")
        return
    
    # 获取最新的文件
    latest_json = sorted(json_files, key=lambda x: x.stem)[-1]
    latest_timestamp = latest_json.stem.replace("_semantic_layers", "")
    
    # 读取metadata
    json_path = masks_dir / f"{latest_timestamp}_semantic_layers.json"
    with open(json_path) as f:
        metadata = json.load(f)
    
    # 创建可视化
    num_layers = metadata['metadata']['num_layers']
    cols = 3
    rows = (num_layers + cols - 1) // cols + 1  # 额外一行显示原图和合成
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()
    
    # 显示原图
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 创建彩色合成图
    h, w = original_np.shape[:2]
    composite = np.zeros((h, w, 3), dtype=np.uint8)
    colors = plt.cm.tab10(np.linspace(0, 1, num_layers))
    
    # 显示每个层的mask
    for i, layer_info in enumerate(metadata['layers']):
        # 读取visible mask
        mask_path = masks_dir / f"{latest_timestamp}_layer_{i}_visible.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # 显示mask
        ax = axes[i + cols]  # 跳过第一行
        ax.imshow(mask, cmap='gray')
        ax.set_title(f"Layer {i}: {layer_info['name'].replace('layer_' + str(i) + '_', '')}\n"
                     f"Semantic: {layer_info['semantic']}, Z-order: {layer_info['z_order']}", 
                     fontsize=10)
        ax.axis('off')
        
        # 添加到彩色合成图
        mask_3d = np.stack([mask > 127] * 3, axis=-1)
        color_rgb = (colors[i][:3] * 255).astype(np.uint8)
        composite = np.where(mask_3d, color_rgb, composite)
    
    # 显示彩色合成图
    axes[1].imshow(composite)
    axes[1].set_title("Semantic Layers Composite", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 显示叠加图
    overlay = cv2.addWeighted(original_np, 0.6, composite, 0.4, 0)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay Visualization", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # 隐藏多余的axes
    for idx in range(num_layers + cols, len(axes)):
        axes[idx].axis('off')
    
    # 添加质量信息
    quality = metadata['quality']
    fig.suptitle(f"Semantic Layer Extraction Results\n"
                 f"Completeness: {quality['completeness']:.1%} | "
                 f"Separation: {quality['separation']:.1%} | "
                 f"Coverage: {quality['coverage']:.1%} | "
                 f"Confidence: {quality['confidence']:.1%}",
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = f"02_输出结果/visualization_{latest_timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    
    plt.show()


def print_analysis_summary():
    """打印分析摘要"""
    
    # 读取分析结果
    analysis_path = "02_输出结果/analysis.json"
    with open(analysis_path) as f:
        analysis = json.load(f)
    
    print("\n" + "="*60)
    print("📊 ANALYSIS SUMMARY")
    print("="*60)
    
    print("\n🎨 Vectorization Strategy:")
    print("-"*40)
    
    # 统计每种方法的使用次数和总大小
    method_stats = {}
    for region in analysis['strategy']['region_strategies']:
        method = region['method']
        if method not in method_stats:
            method_stats[method] = {'count': 0, 'size': 0}
        method_stats[method]['count'] += 1
        method_stats[method]['size'] += region['estimated_size']
    
    for method, stats in method_stats.items():
        print(f"  {method.upper()}:")
        print(f"    - Used for {stats['count']} regions")
        print(f"    - Total size: {stats['size']} KB")
    
    print("\n📈 Performance Metrics:")
    print("-"*40)
    perf = analysis['performance_estimate']
    print(f"  Total file size: {perf['estimated_file_size_kb']} KB")
    print(f"  Processing time: {perf['estimated_processing_time_s']:.1f} seconds")
    print(f"  Quality score: {perf['estimated_quality_score']:.1%}")
    
    print("\n🔍 Global Image Features:")
    print("-"*40)
    features = analysis['global_features']
    print(f"  Edge density: {features['edge_density']:.1%}")
    print(f"  Texture complexity: {features['texture_complexity']:.1%}")
    print(f"  Color diversity: {features['color_diversity']:.1%}")
    print(f"  Gradient strength: {features['gradient_strength']:.1%}")
    print(f"  Overall complexity: {features['overall_complexity']:.2f}")
    
    print("="*60)


if __name__ == "__main__":
    visualize_semantic_layers()
    print_analysis_summary()
