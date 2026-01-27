#!/usr/bin/env python3
"""
Mask转SVG生成器
将分层Mask转换为可编辑的SVG文件

特点：
1. 文字层 → <text> 标签（可编辑）
2. 图形层 → <path> 矢量轮廓
3. 保持层次结构
4. 支持颜色保留
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import subprocess
import tempfile
import os


class MasksToSVG:
    """Mask转SVG生成器"""
    
    def __init__(self):
        self.has_potrace = self._check_potrace()
        
    def _check_potrace(self) -> bool:
        """检查potrace是否可用"""
        try:
            result = subprocess.run(['potrace', '--version'], 
                                   capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def mask_to_svg_path(self, mask: np.ndarray, color: str = "black",
                         simplify: bool = True) -> str:
        """
        将mask转换为SVG path
        使用potrace进行高质量矢量化
        """
        if np.sum(mask) == 0:
            return ""
        
        # 确保是二值图
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 直接使用OpenCV生成正确坐标的path
        return self._opencv_vectorize(binary, color, simplify)
    
    def _potrace_vectorize(self, binary: np.ndarray, color: str) -> str:
        """使用potrace进行高质量矢量化，修正坐标缩放"""
        with tempfile.NamedTemporaryFile(suffix='.pbm', delete=False) as f_pbm:
            pbm_path = f_pbm.name
        
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f_svg:
            svg_path = f_svg.name
        
        try:
            h, w = binary.shape
            
            # 保存为PBM格式
            with open(pbm_path, 'wb') as f:
                f.write(f"P4\n{w} {h}\n".encode())
                packed = np.packbits(binary > 0, axis=1)
                f.write(packed.tobytes())
            
            # 调用potrace，使用正确的尺寸参数
            subprocess.run([
                'potrace', pbm_path,
                '-s',  # SVG输出
                '-o', svg_path,
                '-t', '5',
                '-a', '1.0',
                '-O', '0.2',
                '-W', str(w),  # 指定输出宽度
                '-H', str(h),  # 指定输出高度
            ], capture_output=True)
            
            with open(svg_path, 'r') as f:
                svg_content = f.read()
            
            import re
            # 提取transform如果有
            transform_match = re.search(r'transform="([^"]+)"', svg_content)
            paths = re.findall(r'd="([^"]+)"', svg_content)
            
            if paths:
                path_str = ' '.join(paths)
                # 如果有transform，应用它
                if transform_match:
                    transform = transform_match.group(1)
                    return f'<g transform="{transform}"><path d="{path_str}" fill="{color}" fill-rule="evenodd"/></g>'
                return f'<path d="{path_str}" fill="{color}" fill-rule="evenodd"/>'
            
            return ""
            
        finally:
            if os.path.exists(pbm_path):
                os.remove(pbm_path)
            if os.path.exists(svg_path):
                os.remove(svg_path)
    
    def _opencv_vectorize(self, binary: np.ndarray, color: str, 
                          simplify: bool = True) -> str:
        """使用OpenCV轮廓作为备用方法"""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return ""
        
        path_parts = []
        for cnt in contours:
            if len(cnt) < 3:
                continue
            
            if simplify:
                epsilon = 0.5 * cv2.arcLength(cnt, True) / len(cnt)
                cnt = cv2.approxPolyDP(cnt, epsilon, True)
            
            # 转换为SVG path
            points = cnt.reshape(-1, 2)
            if len(points) < 3:
                continue
            
            path_d = f"M {points[0][0]},{points[0][1]}"
            for p in points[1:]:
                path_d += f" L {p[0]},{p[1]}"
            path_d += " Z"
            path_parts.append(path_d)
        
        if path_parts:
            combined = ' '.join(path_parts)
            return f'<path d="{combined}" fill="{color}" fill-rule="evenodd"/>'
        
        return ""
    
    def text_regions_to_svg(self, text_regions: list) -> str:
        """将文字区域转换为SVG text元素"""
        text_elements = []
        
        for region in text_regions:
            x, y, w, h = region["bbox"]
            text = region.get("text", "")
            conf = region.get("confidence", 0)
            
            if not text or conf < 0.3:
                continue
            
            # 文字位置在bbox底部
            text_x = x
            text_y = y + h - 2
            
            # 估算字体大小
            font_size = min(h - 2, 14)
            
            # 根据文字颜色判断（如果是红色或蓝色文字）
            fill_color = "black"  # 默认黑色
            
            text_elem = f'<text x="{text_x}" y="{text_y}" font-family="Arial, sans-serif" font-size="{font_size}" fill="{fill_color}">{text}</text>'
            text_elements.append(text_elem)
        
        return '\n    '.join(text_elements)
    
    def generate_layered_svg(self, image_path: str, masks_dir: str,
                             text_regions: list, output_path: str) -> str:
        """
        生成分层SVG
        """
        print("\n" + "="*60)
        print("📄 生成分层SVG")
        print("="*60)
        
        # 获取图像尺寸
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # 查找最新的mask文件
        masks_path = Path(masks_dir)
        mask_files = {}
        
        layer_patterns = {
            "L1_text": "*L1_text*.png",
            "L2_red": "*L2_red*.png",
            "L3_blue": "*L3_blue*.png",
            "L4_beam_3d": "*L4_beam_3d*.png",
            "L5_black_lines": "*L5_black_lines*.png",
        }
        
        for layer_name, pattern in layer_patterns.items():
            files = list(masks_path.glob(pattern))
            if files:
                # 取最新的
                mask_files[layer_name] = sorted(files)[-1]
        
        print(f"   找到 {len(mask_files)} 个mask文件")
        
        # 层颜色映射
        layer_colors = {
            "L2_red": "#CC0000",
            "L3_blue": "#0066CC",
            "L4_beam_3d": "#CCCCCC",
            "L5_black_lines": "#333333",
        }
        
        # 开始生成SVG
        svg_parts = []
        svg_parts.append('<?xml version="1.0" encoding="UTF-8"?>')
        svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
        svg_parts.append('  <title>Scientific Figure - Layered SVG</title>')
        svg_parts.append('  <desc>Generated by SAM3 Scientific Figure Processor</desc>')
        
        # 背景层
        svg_parts.append('  <g id="background">')
        svg_parts.append(f'    <rect x="0" y="0" width="{w}" height="{h}" fill="white"/>')
        svg_parts.append('  </g>')
        
        # 图形层（从底到顶）
        layer_order = ["L4_beam_3d", "L5_black_lines", "L3_blue", "L2_red"]
        
        for layer_name in layer_order:
            if layer_name not in mask_files:
                continue
            
            print(f"   处理 {layer_name}...")
            mask = cv2.imread(str(mask_files[layer_name]), cv2.IMREAD_GRAYSCALE)
            
            if mask is None or np.sum(mask) == 0:
                continue
            
            color = layer_colors.get(layer_name, "black")
            path_svg = self.mask_to_svg_path(mask, color)
            
            if path_svg:
                svg_parts.append(f'  <g id="{layer_name}">')
                svg_parts.append(f'    {path_svg}')
                svg_parts.append('  </g>')
                print(f"      ✓ 已添加")
        
        # 文字层（最顶层）
        print("   处理文字层...")
        if text_regions:
            svg_parts.append('  <g id="text-layer">')
            text_svg = self.text_regions_to_svg(text_regions)
            if text_svg:
                svg_parts.append(f'    {text_svg}')
            svg_parts.append('  </g>')
            print(f"      ✓ 添加 {len(text_regions)} 个文字")
        
        svg_parts.append('</svg>')
        
        # 写入文件
        svg_content = '\n'.join(svg_parts)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        # 计算文件大小
        file_size = os.path.getsize(output_path)
        print(f"\n   ✅ SVG已保存: {output_path}")
        print(f"   📦 文件大小: {file_size / 1024:.1f} KB")
        
        return output_path


def main():
    """主函数：从v3结果生成SVG"""
    import sys
    sys.path.insert(0, "/Volumes/Seagate/SAM3/tools")
    
    # 先运行v3获取text_regions
    from scientific_figure_v3 import ScientificFigureV3
    
    print("="*70)
    print("🎯 科研图完整矢量化流程")
    print("="*70)
    
    image_path = "/Volumes/Seagate/SAM3/01_input/科研绘图1.png"
    masks_dir = "/Volumes/Seagate/SAM3/02_output/scientific_v3"
    output_dir = "/Volumes/Seagate/SAM3/02_output/scientific_svg"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: 运行v3获取mask和文字信息
    print("\n📋 Step 1: 运行分割处理器...")
    processor = ScientificFigureV3()
    
    # 只获取文字信息，不重新生成mask
    text_regions, _ = processor.detect_text_precise(image_path)
    
    # Step 2: 生成SVG
    print("\n📋 Step 2: 生成SVG...")
    svg_generator = MasksToSVG()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    svg_path = f"{output_dir}/scientific_figure_{timestamp}.svg"
    
    svg_generator.generate_layered_svg(
        image_path=image_path,
        masks_dir=masks_dir,
        text_regions=text_regions,
        output_path=svg_path
    )
    
    print("\n" + "="*70)
    print("✅ 完成！")
    print("="*70)
    print(f"输出SVG: {svg_path}")


if __name__ == "__main__":
    main()
