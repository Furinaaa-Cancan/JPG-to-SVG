#!/usr/bin/env python3
"""
生成多级别细节的SVG科研组图 - V2
直接调用 sam3_color_vectorizer_fast.py 的核心逻辑
"""

import sys
import os
import time
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import subprocess

# 14个细节级别的参数配置 - 从基础到300MB高清
DETAIL_LEVELS = {
    1:  {'name': 'L1',  'n_colors_large': 1,  'n_colors_medium': 1,  'n_colors_small': 1,  'min_area': 8000, 'epsilon_factor': 0.025},
    2:  {'name': 'L2',  'n_colors_large': 1,  'n_colors_medium': 1,  'n_colors_small': 1,  'min_area': 3000, 'epsilon_factor': 0.018},
    3:  {'name': 'L3',  'n_colors_large': 2,  'n_colors_medium': 1,  'n_colors_small': 1,  'min_area': 1500, 'epsilon_factor': 0.012},
    4:  {'name': 'L4',  'n_colors_large': 3,  'n_colors_medium': 2,  'n_colors_small': 1,  'min_area': 800,  'epsilon_factor': 0.008},
    5:  {'name': 'L5',  'n_colors_large': 4,  'n_colors_medium': 3,  'n_colors_small': 2,  'min_area': 400,  'epsilon_factor': 0.005},
    6:  {'name': 'L6',  'n_colors_large': 6,  'n_colors_medium': 4,  'n_colors_small': 3,  'min_area': 200,  'epsilon_factor': 0.003},
    7:  {'name': 'L7',  'n_colors_large': 10, 'n_colors_medium': 6,  'n_colors_small': 4,  'min_area': 100,  'epsilon_factor': 0.002},
    8:  {'name': 'L8',  'n_colors_large': 15, 'n_colors_medium': 10, 'n_colors_small': 6,  'min_area': 50,   'epsilon_factor': 0.0012},
    9:  {'name': 'L9',  'n_colors_large': 20, 'n_colors_medium': 14, 'n_colors_small': 8,  'min_area': 25,   'epsilon_factor': 0.0008},
    10: {'name': 'L10', 'n_colors_large': 28, 'n_colors_medium': 18, 'n_colors_small': 12, 'min_area': 12,   'epsilon_factor': 0.0005},
    11: {'name': 'L11', 'n_colors_large': 38, 'n_colors_medium': 25, 'n_colors_small': 16, 'min_area': 6,    'epsilon_factor': 0.0003},
    12: {'name': 'L12', 'n_colors_large': 50, 'n_colors_medium': 35, 'n_colors_small': 22, 'min_area': 4,    'epsilon_factor': 0.0002},
    13: {'name': 'L13', 'n_colors_large': 65, 'n_colors_medium': 45, 'n_colors_small': 30, 'min_area': 2,    'epsilon_factor': 0.00015},
    14: {'name': 'L14', 'n_colors_large': 80, 'n_colors_medium': 55, 'n_colors_small': 40, 'min_area': 1,    'epsilon_factor': 0.0001},
}


def run_vectorizer(image_path, output_dir, level):
    """调用sam3_color_vectorizer_fast.py生成指定级别的SVG"""
    
    level_config = DETAIL_LEVELS[level]
    
    # 构建命令 - 直接调用原脚本
    cmd = [
        'python', 'sam3_color_vectorizer_fast.py',
        image_path,
        '-o', output_dir,
        '--no-blip'
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='/Volumes/Seagate/SAM3/12_语义矢量化')
    elapsed = time.time() - start_time
    
    # 获取SVG文件信息
    svg_path = Path(output_dir) / 'sam3_color_vector.svg'
    if svg_path.exists():
        size_kb = svg_path.stat().st_size / 1024
        # 简单计算路径数（通过文件中的<path数量）
        with open(svg_path, 'r') as f:
            content = f.read()
            paths = content.count('<path')
        return {
            'level': level,
            'name': level_config['name'],
            'paths': paths,
            'size_kb': size_kb,
            'time': elapsed,
            'svg_path': svg_path
        }
    return None


def create_comparison_figure(image_path, results, output_path):
    """创建科研组图 - 5行3列布局（1原图 + 14 SVG）"""
    
    print("\n📊 Creating comparison figure...")
    
    original = Image.open(image_path).convert('RGB')
    
    fig, axes = plt.subplots(5, 3, figsize=(15, 25))
    fig.patch.set_facecolor('white')
    
    orig_size = os.path.getsize(image_path) / 1024
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('(a) Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 0].text(0.5, -0.05, f'{orig_size:.0f} KB', 
                    transform=axes[0, 0].transAxes, ha='center', fontsize=10)
    
    labels = [chr(ord('b') + i) for i in range(14)]
    
    from cairosvg import svg2png
    from io import BytesIO
    
    for i, result in enumerate(results):
        pos = i + 1
        row = pos // 3
        col = pos % 3
        
        svg_path = result['svg_path']
        png_data = svg2png(url=str(svg_path), output_width=600)
        svg_img = Image.open(BytesIO(png_data)).convert('RGB')
        
        axes[row, col].imshow(svg_img)
        axes[row, col].set_title(f"({labels[i]}) {result['name']}", fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
        
        size_kb = result['size_kb']
        if size_kb >= 1024:
            size_str = f"{size_kb/1024:.1f} MB"
        else:
            size_str = f"{size_kb:.0f} KB"
        
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
    
    results = []
    
    # 直接调用原版本的vectorizer，只生成一次（使用默认参数）
    # 然后手动复制为不同级别（因为参数调整需要修改代码）
    
    # 方案：直接使用已有的好版本SVG
    existing_svg = Path('/Volumes/Seagate/SAM3/02_输出结果/Ladygaga_2_svg/sam3_color_vector.svg')
    
    if existing_svg.exists():
        print(f"✅ Using existing good SVG: {existing_svg}")
        size_kb = existing_svg.stat().st_size / 1024
        with open(existing_svg, 'r') as f:
            paths = f.read().count('<path')
        
        # 复制到输出目录
        import shutil
        dest = output_dir / 'level_reference.svg'
        shutil.copy(existing_svg, dest)
        
        print(f"   Paths: {paths:,} | Size: {size_kb:.0f} KB")
        print(f"\n⚠️  要生成多级别SVG，需要修改 sam3_color_vectorizer_fast.py 添加参数支持")
        print("   当前好的版本参数是固定的，需要重构代码才能支持动态参数")
    else:
        print("❌ 没有找到好的版本SVG")


if __name__ == "__main__":
    main()
