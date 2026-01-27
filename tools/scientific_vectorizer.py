#!/usr/bin/env python3
"""
科研图自动矢量化工具 - 完整Pipeline
Scientific Figure Auto-Vectorizer

功能：
1. 自动分层分割（文字、颜色、SAM3）
2. 高质量SVG生成（文字可编辑）
3. OCR后处理（拼写纠正）
4. 批量处理支持

用法：
    python scientific_vectorizer.py input.png              # 处理单张图
    python scientific_vectorizer.py input_dir/ -o out/     # 批量处理
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from scientific_figure_v3 import ScientificFigureV3
from masks_to_svg import MasksToSVG
from ocr_postprocess import correct_svg_file


class ScientificVectorizer:
    """科研图自动矢量化器"""
    
    def __init__(self, output_dir: str = None):
        self.processor = ScientificFigureV3()
        self.svg_generator = MasksToSVG()
        self.output_dir = output_dir or "/Volumes/Seagate/SAM3/02_output/vectorized"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
    def process(self, image_path: str, correct_ocr: bool = True) -> dict:
        """
        完整处理流程
        
        Args:
            image_path: 输入图片路径
            correct_ocr: 是否进行OCR后处理
            
        Returns:
            处理结果字典
        """
        print("\n" + "="*70)
        print("🎯 科研图自动矢量化")
        print("="*70)
        print(f"   输入: {image_path}")
        
        start_time = datetime.now()
        image_name = Path(image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建输出子目录
        output_subdir = Path(self.output_dir) / f"{image_name}_{timestamp}"
        output_subdir.mkdir(parents=True, exist_ok=True)
        masks_dir = output_subdir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        result = {
            "input": image_path,
            "output_dir": str(output_subdir),
            "success": False
        }
        
        try:
            # Step 1: 分层分割
            print("\n" + "-"*50)
            print("📋 Step 1/3: 分层分割")
            print("-"*50)
            
            # 获取文字信息
            text_regions, text_mask = self.processor.detect_text_precise(image_path)
            
            # 颜色分离
            color_masks = self.processor.separate_colors_clean(image_path, text_mask)
            
            # SAM3分割3D结构
            beam_result = self.processor.segment_3d_beam_with_sam3(image_path, text_mask)
            
            # 层级细化
            layers = self.processor.refine_masks(
                text_mask, color_masks, beam_result
            )
            
            # 保存masks
            import cv2
            for layer_name, mask in layers.items():
                mask_path = masks_dir / f"{layer_name}.png"
                cv2.imwrite(str(mask_path), mask)
            
            result["layers"] = list(layers.keys())
            result["text_count"] = len(text_regions)
            
            # Step 2: SVG生成
            print("\n" + "-"*50)
            print("📋 Step 2/3: SVG生成")
            print("-"*50)
            
            svg_path = output_subdir / f"{image_name}.svg"
            self.svg_generator.generate_layered_svg(
                image_path=image_path,
                masks_dir=str(masks_dir),
                text_regions=text_regions,
                output_path=str(svg_path)
            )
            
            result["svg_raw"] = str(svg_path)
            
            # Step 3: OCR后处理
            if correct_ocr:
                print("\n" + "-"*50)
                print("📋 Step 3/3: OCR后处理")
                print("-"*50)
                
                corrected_path = output_subdir / f"{image_name}_final.svg"
                ocr_result = correct_svg_file(str(svg_path), str(corrected_path))
                
                result["svg_final"] = str(corrected_path)
                result["ocr_corrections"] = ocr_result.get("corrections", 0)
            else:
                result["svg_final"] = str(svg_path)
                result["ocr_corrections"] = 0
            
            # 计算统计
            elapsed = (datetime.now() - start_time).total_seconds()
            svg_size = os.path.getsize(result["svg_final"])
            
            result["success"] = True
            result["elapsed_seconds"] = elapsed
            result["svg_size_kb"] = svg_size / 1024
            
            # 生成简单的HTML预览
            self._generate_preview(result, output_subdir)
            
            print("\n" + "="*70)
            print("✅ 矢量化完成!")
            print("="*70)
            print(f"   耗时: {elapsed:.1f}秒")
            print(f"   SVG大小: {svg_size/1024:.1f} KB")
            print(f"   文字区域: {len(text_regions)}个")
            print(f"   OCR纠正: {result['ocr_corrections']}处")
            print(f"   输出目录: {output_subdir}")
            
        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            result["error"] = str(e)
            import traceback
            traceback.print_exc()
        
        return result
    
    def _generate_preview(self, result: dict, output_dir: Path):
        """生成HTML预览页面"""
        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>矢量化结果 - {Path(result["input"]).stem}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ text-align: center; color: #333; }}
        .container {{ display: flex; gap: 20px; max-width: 1400px; margin: 0 auto; }}
        .panel {{ flex: 1; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .panel h3 {{ margin-top: 0; color: #555; }}
        .panel img, .panel object {{ width: 100%; height: auto; border: 1px solid #ddd; }}
        .stats {{ background: #e8f4f8; padding: 15px; border-radius: 8px; margin: 20px auto; max-width: 600px; }}
        .stats table {{ width: 100%; }}
        .stats td {{ padding: 5px 10px; }}
        .success {{ color: #28a745; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>🎯 科研图矢量化结果</h1>
    
    <div class="stats">
        <table>
            <tr><td>处理时间</td><td>{result.get("elapsed_seconds", 0):.1f}秒</td></tr>
            <tr><td>SVG大小</td><td class="success">{result.get("svg_size_kb", 0):.1f} KB</td></tr>
            <tr><td>文字区域</td><td>{result.get("text_count", 0)}个</td></tr>
            <tr><td>OCR纠正</td><td>{result.get("ocr_corrections", 0)}处</td></tr>
        </table>
    </div>
    
    <div class="container">
        <div class="panel">
            <h3>📷 原始图片</h3>
            <img src="{result["input"]}" alt="Original">
        </div>
        <div class="panel">
            <h3>📄 矢量化SVG</h3>
            <object type="image/svg+xml" data="{Path(result["svg_final"]).name}"></object>
        </div>
    </div>
</body>
</html>'''
        
        preview_path = output_dir / "preview.html"
        with open(preview_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        result["preview"] = str(preview_path)
    
    def batch_process(self, input_dir: str, extensions: list = None) -> list:
        """
        批量处理目录中的图片
        
        Args:
            input_dir: 输入目录
            extensions: 文件扩展名列表，默认 ['.png', '.jpg', '.jpeg']
            
        Returns:
            处理结果列表
        """
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg']
        
        input_path = Path(input_dir)
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        print(f"\n找到 {len(image_files)} 个图片文件")
        
        results = []
        for i, img_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 处理: {img_path.name}")
            result = self.process(str(img_path))
            results.append(result)
        
        # 统计
        success_count = sum(1 for r in results if r.get("success"))
        print(f"\n" + "="*70)
        print(f"📊 批量处理完成: {success_count}/{len(results)} 成功")
        print("="*70)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="科研图自动矢量化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scientific_vectorizer.py image.png
    python scientific_vectorizer.py input_dir/ -o output/
    python scientific_vectorizer.py image.png --no-ocr
        """
    )
    parser.add_argument("input", help="输入图片或目录")
    parser.add_argument("-o", "--output", default=None, help="输出目录")
    parser.add_argument("--no-ocr", action="store_true", help="跳过OCR后处理")
    
    args = parser.parse_args()
    
    vectorizer = ScientificVectorizer(output_dir=args.output)
    
    input_path = Path(args.input)
    if input_path.is_dir():
        vectorizer.batch_process(str(input_path))
    elif input_path.is_file():
        vectorizer.process(str(input_path), correct_ocr=not args.no_ocr)
    else:
        print(f"❌ 输入路径不存在: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    # 如果没有命令行参数，使用默认测试
    if len(sys.argv) == 1:
        vectorizer = ScientificVectorizer()
        vectorizer.process("/Volumes/Seagate/SAM3/01_input/科研绘图1.png")
    else:
        main()
