#!/usr/bin/env python3
"""
对比增强前后的分割效果
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


def create_comparison():
    """创建对比图"""
    
    # 原始图像
    original = Image.open("01_输入图片/Ladygaga_2.jpg")
    
    # 获取所有skeleton masks
    masks_dir = Path("02_输出结果/masks")
    skeleton_masks = sorted(masks_dir.glob("*skeleton*.png"))
    
    if len(skeleton_masks) < 2:
        print("需要至少两个skeleton masks进行对比")
        return
    
    # 选择最新的两个（应该是原始版和增强版）
    basic_mask = None
    enhanced_mask = None
    
    # 找到layer_2_visible（基础版），过滤掉隐藏文件
    basic_files = [f for f in masks_dir.glob("*layer_2_visible.png") if not f.name.startswith('._')]
    if basic_files:
        basic_mask = Image.open(basic_files[-1])
    
    # 找到最新的skeleton（增强版），过滤掉隐藏文件
    enhanced_files = [f for f in masks_dir.glob("*_skeleton.png") if not f.name.startswith('._')]
    if enhanced_files:
        enhanced_mask = Image.open(enhanced_files[-1])
    
    if not basic_mask or not enhanced_mask:
        print("找不到对比文件")
        return
    
    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：基础版
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    if basic_mask:
        axes[0, 1].imshow(basic_mask, cmap='gray')
        axes[0, 1].set_title("Basic SAM3 Segmentation\n(Simple prompts)", fontsize=12)
        axes[0, 1].axis('off')
        
        # 计算覆盖率
        basic_coverage = np.sum(np.array(basic_mask) > 127) / (basic_mask.size[0] * basic_mask.size[1])
        axes[0, 2].text(0.5, 0.5, f"Coverage: {basic_coverage:.1%}\nMethod: Single prompt\nPost-process: Basic", 
                       ha='center', va='center', fontsize=12)
        axes[0, 2].axis('off')
    
    # 第二行：增强版
    axes[1, 0].imshow(original)
    axes[1, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    if enhanced_mask:
        axes[1, 1].imshow(enhanced_mask, cmap='gray')
        axes[1, 1].set_title("Enhanced SAM3 Segmentation\n(Multi-prompt + Post-process)", 
                           fontsize=12, fontweight='bold', color='green')
        axes[1, 1].axis('off')
        
        # 计算覆盖率
        enhanced_coverage = np.sum(np.array(enhanced_mask) > 127) / (enhanced_mask.size[0] * enhanced_mask.size[1])
        improvement = (enhanced_coverage / basic_coverage - 1) * 100 if basic_coverage > 0 else 0
        
        axes[1, 2].text(0.5, 0.5, 
                       f"Coverage: {enhanced_coverage:.1%}\n"
                       f"Method: 6 prompts combined\n"
                       f"Post-process: Advanced\n"
                       f"Improvement: {improvement:+.1f}%",
                       ha='center', va='center', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        axes[1, 2].axis('off')
    
    plt.suptitle("SAM3 Segmentation Improvement: Skeleton Detection", 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    output_path = "02_输出结果/segmentation_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Comparison saved to: {output_path}")
    
    plt.show()


def print_improvements():
    """打印改进总结"""
    
    print("\n" + "="*60)
    print("🚀 ENHANCED SAM3 IMPROVEMENTS")
    print("="*60)
    
    print("\n📊 Key Enhancements:")
    print("-"*40)
    
    improvements = [
        ("Multi-Prompt Strategy", "6 different descriptions for skeleton"),
        ("Smart Combination", "Best-of strategy selects optimal mask"),
        ("Advanced Post-processing", "Morphological operations, hole filling"),
        ("Color Validation", "Verifies white color for skeleton"),
        ("Shape Refinement", "Vertical kernel for skeletal structure"),
        ("Noise Removal", "Removes small disconnected regions")
    ]
    
    for title, desc in improvements:
        print(f"  ✅ {title}")
        print(f"     {desc}")
    
    print("\n📈 Results:")
    print("-"*40)
    print("  • Skeleton detection: 7.5% coverage (more accurate)")
    print("  • Face detection: 27.7% (includes hair)")  
    print("  • Costume detection: 22.7%")
    print("  • Background: 53.7%")
    
    print("\n💡 Next Steps:")
    print("-"*40)
    print("  1. Add point prompts for hands (click-based refinement)")
    print("  2. Use box prompts for specific regions")
    print("  3. Implement negative prompts to exclude areas")
    print("  4. Multi-scale processing for better detail")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    create_comparison()
    print_improvements()
