#!/usr/bin/env python3
"""
生成科研组图：1张原图 + 5张SVG（逐步递进到300MB）
2行3列布局
"""

import sys
import os
import time
import subprocess
import shutil
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# 5个级别：从小到大，最高级到300MB
# 级别参数对应 sam3_color_vectorizer_fast.py 的 -l 参数
FIVE_LEVELS = [
    {'level': 1,  'name': 'Minimal',     'target': '~500KB'},
    {'level': 4,  'name': 'Simple',      'target': '~5MB'},
    {'level': 8,  'name': 'Medium',      'target': '~80MB'},
    {'level': 12, 'name': 'Detailed',    'target': '~200MB'},
    {'level': 14, 'name': 'Full Detail', 'target': '~300MB'},
]


def run_level(image_path, output_dir, level_info):
    """运行指定级别的矢量化"""
    
    level = level_info['level']
    name = level_info['name']
    
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
    print(f"📊 Generating {name} (Level {level})")
    print(f"{'='*50}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, 
                          cwd='/Volumes/Seagate/SAM3/12_语义矢量化')
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"   ❌ Error: {result.stderr[-500:] if result.stderr else 'Unknown'}")
        return None
    
    # 获取SVG文件信息
    svg_path = level_output / 'sam3_color_vector.svg'
    if svg_path.exists():
        size_kb = svg_path.stat().st_size / 1024
        # 计算路径数
        with open(svg_path, 'r') as f:
            content = f.read()
            paths = content.count('<path')
        
        # 复制到主目录
        final_svg = Path(output_dir) / f"{name.lower().replace(' ', '_')}.svg"
        shutil.copy(svg_path, final_svg)
        
        print(f"   ✅ Paths: {paths:,} | Size: {size_kb/1024:.1f} MB | Time: {elapsed:.1f}s")
        
        return {
            'level': level,
            'name': name,
            'paths': paths,
            'size_kb': size_kb,
            'time': elapsed,
            'svg_path': final_svg
        }
    
    return None


def create_figure(image_path, results, output_path):
    """创建科研组图 - 2行3列布局（1原图 + 5 SVG）"""
    
    print("\n📊 Creating comparison figure...")
    
    original = Image.open(image_path).convert('RGB')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('white')
    
    # 原图大小
    orig_size = os.path.getsize(image_path) / 1024
    
    # (a) 原始图片
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('(a) Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 0].text(0.5, -0.08, f'Size: {orig_size:.0f} KB', 
                    transform=axes[0, 0].transAxes, ha='center', fontsize=12)
    
    # (b-f) 5个SVG级别
    labels = ['(b)', '(c)', '(d)', '(e)', '(f)']
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    
    from cairosvg import svg2png
    from io import BytesIO
    
    for i, result in enumerate(results):
        row, col = positions[i]
        
        # SVG转PNG显示
        svg_path = result['svg_path']
        try:
            png_data = svg2png(url=str(svg_path), output_width=800)
            svg_img = Image.open(BytesIO(png_data)).convert('RGB')
        except Exception as e:
            print(f"   ⚠️ Error converting {svg_path}: {e}")
            svg_img = original.copy()
        
        axes[row, col].imshow(svg_img)
        axes[row, col].set_title(f"{labels[i]} {result['name']}", fontsize=14, fontweight='bold')
        axes[row, col].axis('off')
        
        # 格式化大小
        size_kb = result['size_kb']
        if size_kb >= 1024:
            size_str = f"{size_kb/1024:.1f} MB"
        else:
            size_str = f"{size_kb:.0f} KB"
        
        # 格式化时间
        time_sec = result.get('time', 0)
        if time_sec >= 60:
            time_str = f"{time_sec/60:.1f} min"
        else:
            time_str = f"{time_sec:.1f}s"
        
        axes[row, col].text(0.5, -0.08, 
                            f"Paths: {result['paths']:,} | Size: {size_str} | Time: {time_str}",
                            transform=axes[row, col].transAxes, ha='center', fontsize=11)
    
    plt.tight_layout()
    
    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    print(f"✅ Saved: {pdf_path}")
    
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate 5-level SVG comparison figure')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('-o', '--output', default='five_level_output', help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎯 Generating 5 SVG levels (Original + 5 SVG)")
    
    results = []
    total_start = time.time()
    
    for level_info in FIVE_LEVELS:
        result = run_level(args.image, output_dir, level_info)
        if result:
            results.append(result)
    
    total_time = time.time() - total_start
    
    # 打印汇总
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"{'Name':<15} {'Paths':>12} {'Size':>12} {'Time':>12}")
    print("-"*70)
    for r in results:
        size_str = f"{r['size_kb']:.0f} KB" if r['size_kb'] < 1024 else f"{r['size_kb']/1024:.1f} MB"
        time_str = f"{r['time']:.1f}s" if r['time'] < 60 else f"{r['time']/60:.1f} min"
        print(f"{r['name']:<15} {r['paths']:>12,} {size_str:>12} {time_str:>12}")
    print(f"\nTotal time: {total_time/60:.1f} min")
    
    # 创建组图
    if len(results) == 5:
        fig_path = str(output_dir / 'comparison_figure.png')
        create_figure(args.image, results, fig_path)
    
    # 打开输出目录
    subprocess.run(['open', str(output_dir)])


if __name__ == "__main__":
    main()
