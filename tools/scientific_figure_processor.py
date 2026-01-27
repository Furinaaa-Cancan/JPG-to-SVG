#!/usr/bin/env python3
"""
科研绘图矢量化处理器
核心思路：文字与图形分离处理

Pipeline:
1. 文字检测与提取 (OCR)
2. 文字区域mask生成
3. 图形区域分割 (SAM3)
4. 几何图形识别 (线/矩形/菱形)
5. 分层SVG生成
"""

import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
from datetime import datetime

# 添加SAM3路径
sys.path.insert(0, "/Volumes/Seagate/SAM3")


class ScientificFigureProcessor:
    """科研图处理器"""
    
    def __init__(self):
        self.ocr_engine = None
        self.sam3_model = None
        
    def analyze_image(self, image_path: str) -> dict:
        """
        第一步：分析图像结构
        识别文字区域、图形区域、连接线
        """
        print("\n" + "="*60)
        print("📊 STEP 1: 图像结构分析")
        print("="*60)
        
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        analysis = {
            "size": (w, h),
            "text_regions": [],
            "graphic_regions": [],
            "line_regions": [],
            "color_analysis": {}
        }
        
        # 1. 颜色分析
        print("\n🎨 颜色分析...")
        analysis["color_analysis"] = self._analyze_colors(img_rgb)
        
        # 2. 边缘检测 - 识别线条和几何形状
        print("📐 边缘检测...")
        edges = cv2.Canny(img, 50, 150)
        analysis["edge_density"] = np.sum(edges > 0) / (w * h)
        
        # 3. 连通域分析
        print("🔗 连通域分析...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 分类连通域
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:  # 太小，忽略
                continue
            
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / ch if ch > 0 else 0
            
            region_info = {
                "bbox": [x, y, cw, ch],
                "area": area,
                "aspect_ratio": aspect_ratio
            }
            
            # 启发式分类
            if aspect_ratio > 3 and ch < 30:
                # 宽扁形状，可能是文字
                region_info["type"] = "text_candidate"
                analysis["text_regions"].append(region_info)
            elif 0.8 < aspect_ratio < 1.2 and area > 500:
                # 接近正方形，可能是图形元素
                region_info["type"] = "graphic_candidate"
                analysis["graphic_regions"].append(region_info)
            else:
                analysis["graphic_regions"].append(region_info)
        
        print(f"\n📋 分析结果:")
        print(f"   - 图像尺寸: {w}x{h}")
        print(f"   - 边缘密度: {analysis['edge_density']:.2%}")
        print(f"   - 文字候选区域: {len(analysis['text_regions'])}")
        print(f"   - 图形候选区域: {len(analysis['graphic_regions'])}")
        
        return analysis
    
    def _analyze_colors(self, img_rgb: np.ndarray) -> dict:
        """分析图像主要颜色"""
        # 转HSV更容易分析
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        # 检测主要颜色区域
        colors = {
            "red": {"lower": (0, 100, 100), "upper": (10, 255, 255)},
            "blue": {"lower": (100, 100, 100), "upper": (130, 255, 255)},
            "white": {"lower": (0, 0, 200), "upper": (180, 30, 255)},
            "black": {"lower": (0, 0, 0), "upper": (180, 255, 50)},
        }
        
        result = {}
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        
        for color_name, ranges in colors.items():
            mask = cv2.inRange(img_hsv, ranges["lower"], ranges["upper"])
            pixel_count = np.sum(mask > 0)
            result[color_name] = {
                "pixel_count": int(pixel_count),
                "percentage": pixel_count / total_pixels
            }
        
        return result
    
    def detect_text_regions(self, image_path: str) -> list:
        """
        第二步：检测文字区域
        使用多种方法：
        1. MSER (Maximally Stable Extremal Regions)
        2. 形态学操作
        3. OCR引擎（如果可用）
        """
        print("\n" + "="*60)
        print("📝 STEP 2: 文字区域检测")
        print("="*60)
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        
        text_regions = []
        
        # 方法1: MSER检测
        print("\n🔍 方法1: MSER文字检测...")
        mser = cv2.MSER_create()
        mser.setMinArea(60)
        mser.setMaxArea(5000)
        
        regions, _ = mser.detectRegions(gray)
        
        # 合并重叠区域
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        
        # 获取边界框
        mser_boxes = []
        for hull in hulls:
            x, y, bw, bh = cv2.boundingRect(hull)
            # 过滤太大或太小的区域
            if 10 < bw < w*0.5 and 8 < bh < 50:
                mser_boxes.append([x, y, bw, bh])
        
        print(f"   MSER检测到 {len(mser_boxes)} 个候选区域")
        
        # 方法2: 形态学文字检测
        print("\n🔍 方法2: 形态学文字检测...")
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 水平膨胀连接文字
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        dilated = cv2.dilate(binary, kernel_h, iterations=1)
        
        # 垂直膨胀
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        dilated = cv2.dilate(dilated, kernel_v, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        morph_boxes = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / bh if bh > 0 else 0
            # 文字通常是宽扁的
            if aspect > 1.5 and 8 < bh < 40 and bw > 20:
                morph_boxes.append([x, y, bw, bh])
        
        print(f"   形态学检测到 {len(morph_boxes)} 个候选区域")
        
        # 合并两种方法的结果
        all_boxes = mser_boxes + morph_boxes
        merged_boxes = self._merge_overlapping_boxes(all_boxes)
        
        print(f"\n✅ 合并后共 {len(merged_boxes)} 个文字区域")
        
        # 转换为标准格式
        for box in merged_boxes:
            text_regions.append({
                "bbox": box,
                "confidence": 0.8,
                "text": None  # 需要OCR填充
            })
        
        return text_regions
    
    def _merge_overlapping_boxes(self, boxes: list, overlap_thresh: float = 0.3) -> list:
        """合并重叠的边界框"""
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        
        # 转换为 (x1, y1, x2, y2) 格式
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        indices = np.argsort(y1)
        
        merged = []
        while len(indices) > 0:
            i = indices[0]
            merged.append([int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])])
            
            # 计算IoU
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])
            
            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            intersection = inter_w * inter_h
            
            iou = intersection / (areas[i] + areas[indices[1:]] - intersection + 1e-6)
            
            # 保留IoU小于阈值的
            remaining = np.where(iou < overlap_thresh)[0]
            indices = indices[remaining + 1]
        
        return merged
    
    def create_text_mask(self, image_path: str, text_regions: list, padding: int = 3) -> np.ndarray:
        """
        第三步：创建文字区域Mask
        用于后续将文字从图像中分离
        """
        print("\n" + "="*60)
        print("🎭 STEP 3: 生成文字Mask")
        print("="*60)
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # 创建mask
        text_mask = np.zeros((h, w), dtype=np.uint8)
        
        for region in text_regions:
            x, y, bw, bh = region["bbox"]
            # 添加padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + bw + padding)
            y2 = min(h, y + bh + padding)
            
            text_mask[y1:y2, x1:x2] = 255
        
        # 稍微膨胀确保覆盖完整
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)
        
        coverage = np.sum(text_mask > 0) / (w * h)
        print(f"   文字mask覆盖率: {coverage:.1%}")
        
        return text_mask
    
    def detect_geometric_elements(self, image_path: str, text_mask: np.ndarray) -> dict:
        """
        第四步：检测几何图形元素
        - 直线
        - 矩形
        - 菱形
        - 锯齿线（电阻符号）
        - 箭头
        """
        print("\n" + "="*60)
        print("📐 STEP 4: 几何图形检测")
        print("="*60)
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        
        # 排除文字区域
        gray_no_text = gray.copy()
        gray_no_text[text_mask > 0] = 255  # 将文字区域设为白色
        
        elements = {
            "lines": [],
            "rectangles": [],
            "diamonds": [],
            "arrows": [],
            "zigzags": []  # 电阻符号
        }
        
        # 1. 直线检测 (Hough)
        print("\n🔍 检测直线...")
        edges = cv2.Canny(gray_no_text, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.degrees(np.arctan2(y2-y1, x2-x1))
                elements["lines"].append({
                    "start": (int(x1), int(y1)),
                    "end": (int(x2), int(y2)),
                    "length": float(length),
                    "angle": float(angle)
                })
        
        print(f"   检测到 {len(elements['lines'])} 条直线")
        
        # 2. 矩形检测
        print("\n🔍 检测矩形...")
        _, binary = cv2.threshold(gray_no_text, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            
            # 多边形近似
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            if len(approx) == 4:
                # 检查是否接近矩形
                x, y, bw, bh = cv2.boundingRect(approx)
                rect_area = bw * bh
                if area / rect_area > 0.8:  # 填充率高，是矩形
                    elements["rectangles"].append({
                        "bbox": [int(x), int(y), int(bw), int(bh)],
                        "vertices": approx.reshape(-1, 2).tolist()
                    })
                else:
                    # 可能是菱形
                    elements["diamonds"].append({
                        "vertices": approx.reshape(-1, 2).tolist(),
                        "center": (int(x + bw/2), int(y + bh/2))
                    })
        
        print(f"   检测到 {len(elements['rectangles'])} 个矩形")
        print(f"   检测到 {len(elements['diamonds'])} 个菱形")
        
        # 3. 箭头检测（通过三角形端点）
        print("\n🔍 检测箭头...")
        for cnt in contours:
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            if len(approx) == 3:  # 三角形
                area = cv2.contourArea(cnt)
                if 50 < area < 500:  # 箭头大小范围
                    elements["arrows"].append({
                        "vertices": approx.reshape(-1, 2).tolist()
                    })
        
        print(f"   检测到 {len(elements['arrows'])} 个箭头")
        
        return elements
    
    def segment_with_sam3(self, image_path: str, text_mask: np.ndarray) -> list:
        """
        第五步：使用SAM3分割复杂区域
        对于无法用几何方法处理的区域
        """
        print("\n" + "="*60)
        print("🧠 STEP 5: SAM3智能分割")
        print("="*60)
        
        try:
            sys.path.insert(0, "/Volumes/Seagate/SAM3/模型库/01_SAM3核心模型")
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            print("加载SAM3模型...")
            model = build_sam3_image_model(device="cpu")
            processor = Sam3Processor(model, device="cpu")
            
            img = Image.open(image_path)
            state = processor.set_image(img)
            
            # 针对科研图的关键prompt
            prompts = [
                "3D beam structure",
                "strain gauge", 
                "electronic circuit",
                "resistor symbol",
                "diamond bridge circuit",
                "connecting wire"
            ]
            
            masks = []
            for prompt in prompts:
                print(f"   分割: {prompt}")
                try:
                    state = processor.set_text_prompt(prompt, state)
                    if state and "masks" in state:
                        for mask in state["masks"]:
                            mask_array = np.array(mask)
                            if np.sum(mask_array) > 100:
                                masks.append({
                                    "prompt": prompt,
                                    "mask": mask_array,
                                    "area": int(np.sum(mask_array > 0))
                                })
                except Exception as e:
                    print(f"      警告: {e}")
            
            print(f"\n✅ SAM3分割得到 {len(masks)} 个mask")
            return masks
            
        except Exception as e:
            print(f"⚠️ SAM3加载失败: {e}")
            print("   将跳过SAM3分割，使用纯几何方法")
            return []
    
    def generate_svg(self, image_path: str, text_regions: list, 
                     geometric_elements: dict, sam3_masks: list,
                     output_path: str) -> str:
        """
        第六步：生成分层SVG
        """
        print("\n" + "="*60)
        print("📄 STEP 6: 生成分层SVG")
        print("="*60)
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        svg_parts = []
        svg_parts.append(f'<?xml version="1.0" encoding="UTF-8"?>')
        svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
        
        # 背景层
        svg_parts.append('  <g id="background">')
        svg_parts.append(f'    <rect x="0" y="0" width="{w}" height="{h}" fill="white"/>')
        svg_parts.append('  </g>')
        
        # 几何图形层
        svg_parts.append('  <g id="geometric-elements">')
        
        # 直线
        for line in geometric_elements.get("lines", [])[:50]:  # 限制数量
            x1, y1 = line["start"]
            x2, y2 = line["end"]
            svg_parts.append(f'    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="1"/>')
        
        # 矩形
        for rect in geometric_elements.get("rectangles", []):
            x, y, bw, bh = rect["bbox"]
            svg_parts.append(f'    <rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="none" stroke="black" stroke-width="1"/>')
        
        # 菱形
        for diamond in geometric_elements.get("diamonds", []):
            points = " ".join([f"{p[0]},{p[1]}" for p in diamond["vertices"]])
            svg_parts.append(f'    <polygon points="{points}" fill="none" stroke="black" stroke-width="1"/>')
        
        svg_parts.append('  </g>')
        
        # 文字层（占位符，需要OCR填充）
        svg_parts.append('  <g id="text-layer">')
        for i, region in enumerate(text_regions):
            x, y, bw, bh = region["bbox"]
            text_content = region.get("text", f"[Text_{i}]")
            # 文字位置在box底部居中
            svg_parts.append(f'    <text x="{x + bw//2}" y="{y + bh - 2}" font-size="{min(bh-2, 14)}" text-anchor="middle" fill="black">{text_content}</text>')
        svg_parts.append('  </g>')
        
        svg_parts.append('</svg>')
        
        svg_content = '\n'.join(svg_parts)
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        print(f"✅ SVG已保存: {output_path}")
        print(f"   - 直线: {len(geometric_elements.get('lines', []))}")
        print(f"   - 矩形: {len(geometric_elements.get('rectangles', []))}")
        print(f"   - 菱形: {len(geometric_elements.get('diamonds', []))}")
        print(f"   - 文字区域: {len(text_regions)}")
        
        return output_path
    
    def process(self, image_path: str, output_dir: str = None) -> dict:
        """
        完整处理流程
        """
        print("\n" + "="*70)
        print("🎯 科研绘图矢量化处理器")
        print("="*70)
        print(f"输入: {image_path}")
        
        if output_dir is None:
            output_dir = "/Volumes/Seagate/SAM3/02_output/scientific"
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 1: 分析图像
        analysis = self.analyze_image(image_path)
        
        # Step 2: 检测文字区域
        text_regions = self.detect_text_regions(image_path)
        
        # Step 3: 生成文字mask
        text_mask = self.create_text_mask(image_path, text_regions)
        
        # 保存文字mask
        mask_path = f"{output_dir}/text_mask_{timestamp}.png"
        cv2.imwrite(mask_path, text_mask)
        print(f"   文字mask已保存: {mask_path}")
        
        # Step 4: 检测几何元素
        geometric_elements = self.detect_geometric_elements(image_path, text_mask)
        
        # Step 5: SAM3分割（可选）
        sam3_masks = []  # self.segment_with_sam3(image_path, text_mask)
        
        # Step 6: 生成SVG
        svg_path = f"{output_dir}/scientific_{timestamp}.svg"
        self.generate_svg(image_path, text_regions, geometric_elements, sam3_masks, svg_path)
        
        # 生成可视化对比图
        self._create_visualization(image_path, text_regions, geometric_elements, 
                                   f"{output_dir}/analysis_{timestamp}.png")
        
        return {
            "analysis": analysis,
            "text_regions": len(text_regions),
            "geometric_elements": {k: len(v) for k, v in geometric_elements.items()},
            "output_svg": svg_path
        }
    
    def _create_visualization(self, image_path: str, text_regions: list, 
                              geometric_elements: dict, output_path: str):
        """创建分析可视化图"""
        img = cv2.imread(image_path)
        vis = img.copy()
        
        # 绘制文字区域（绿色）
        for region in text_regions:
            x, y, w, h = region["bbox"]
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 绘制检测到的直线（蓝色）
        for line in geometric_elements.get("lines", [])[:30]:
            pt1 = line["start"]
            pt2 = line["end"]
            cv2.line(vis, pt1, pt2, (255, 0, 0), 2)
        
        # 绘制矩形（红色）
        for rect in geometric_elements.get("rectangles", []):
            x, y, w, h = rect["bbox"]
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        cv2.imwrite(output_path, vis)
        print(f"   可视化图已保存: {output_path}")


def main():
    processor = ScientificFigureProcessor()
    
    # 处理科研绘图
    image_path = "/Volumes/Seagate/SAM3/01_input/科研绘图1.png"
    result = processor.process(image_path)
    
    print("\n" + "="*70)
    print("📊 处理完成")
    print("="*70)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
