#!/usr/bin/env python3
"""
生成多级别细节的SVG科研组图
直接调用 sam3_color_vectorizer_fast.py
"""

import sys
import os
import time
import subprocess
import shutil
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def run_level(image_path, output_dir, level):
    """运行指定级别的矢量化"""
    
    level_output = Path(output_dir) / f"level_{level}"
    level_output.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'python', 'sam3_color_vectorizer_fast.py',
        str(image_path),
        '-o', str(level_output),
        '-l', str(level),
        '--no-blip'
    ]
    
    print(f"\n{'='*50}")
    print(f"📊 Generating Level {level}")
    print(f"{'='*50}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, 
                          cwd='/Volumes/Seagate/SAM3/12_语义矢量化')
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"   ❌ Error: {result.stderr[-500:]}")
        return None
    
    # 获取SVG文件信息
    svg_path = level_output / 'sam3_color_vector.svg'
    if svg_path.exists():
        size_kb = svg_path.stat().st_size / 1024
        # 计算路径数
        with open(svg_path, 'r') as f:
            content = f.read()
            paths = content.count('<path')
        
        # 重命名为level_X.svg
        final_svg = Path(output_dir) / f"level_{level}.svg"
        shutil.copy(svg_path, final_svg)
        
        print(f"   Paths: {paths:,} | Size: {size_kb:.0f} KB | Time: {elapsed:.1f}s")
        
        return {
            'level': level,
            'name': f'L{level}',
            'paths': paths,
            'size_kb': size_kb,
            'time': elapsed,
            'svg_path': final_svg
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
        try:
            png_data = svg2png(url=str(svg_path), output_width=600)
            svg_img = Image.open(BytesIO(png_data)).convert('RGB')
        except Exception as e:
            print(f"   ⚠️ Error converting {svg_path}: {e}")
            svg_img = original.copy()
        
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
    parser.add_argument('--levels', type=str, default='1-14', help='Level range (e.g., 1-14 or 1,3,7,14)')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析级别
    if '-' in args.levels:
        start, end = map(int, args.levels.split('-'))
        levels = list(range(start, end + 1))
    else:
        levels = [int(x) for x in args.levels.split(',')]
    
    print(f"\n🎯 Generating {len(levels)} SVG levels: {levels}")
    
    results = []
    total_start = time.time()
    
    for level in levels:
        result = run_level(args.image, output_dir, level)
        if result:
            results.append(result)
    
    total_time = time.time() - total_start
    
    # 打印汇总
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"{'Level':<8} {'Name':<8} {'Paths':>12} {'Size':>12} {'Time':>10}")
    print("-"*60)
    for r in results:
        size_str = f"{r['size_kb']:.0f} KB" if r['size_kb'] < 1024 else f"{r['size_kb']/1024:.1f} MB"
        time_str = f"{r['time']:.1f}s" if r['time'] < 60 else f"{r['time']/60:.1f}min"
        print(f"{r['level']:<8} {r['name']:<8} {r['paths']:>12,} {size_str:>12} {time_str:>10}")
    print(f"\nTotal time: {total_time/60:.1f} min")
    
    # 创建组图（如果有14个级别）
    if len(results) == 14:
        fig_path = str(output_dir / 'comparison_figure.png')
        create_comparison_figure(args.image, results, fig_path)


if __name__ == "__main__":
    main()
