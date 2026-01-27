#!/usr/bin/env python3
"""
科研绘图矢量化处理器 v2
核心改进：
1. 使用EasyOCR精确检测文字
2. 颜色分离处理（红/蓝/黑分开）
3. 更精确的几何检测
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import json


class ScientificFigureV2:
    """科研图处理器 v2"""
    
    def __init__(self):
        self.ocr = None
        
    def _init_ocr(self):
        """延迟加载OCR"""
        if self.ocr is None:
            try:
                import easyocr
                print("🔤 加载EasyOCR...")
                self.ocr = easyocr.Reader(['en'], gpu=False)
                print("✅ EasyOCR加载成功")
            except ImportError:
                print("⚠️ EasyOCR未安装，使用备用方法")
                self.ocr = "fallback"
    
    def detect_text_precise(self, image_path: str) -> tuple:
        """
        精确文字检测
        返回: (text_regions, text_mask)
        """
        print("\n" + "="*60)
        print("📝 精确文字检测")
        print("="*60)
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        text_mask = np.zeros((h, w), dtype=np.uint8)
        text_regions = []
        
        self._init_ocr()
        
        if self.ocr != "fallback":
            # 使用EasyOCR
            results = self.ocr.readtext(image_path)
            
            for (bbox, text, conf) in results:
                if conf < 0.3:  # 过滤低置信度
                    continue
                
                # bbox是4个点的列表
                pts = np.array(bbox, dtype=np.int32)
                x, y, bw, bh = cv2.boundingRect(pts)
                
                # 添加少量padding
                pad = 2
                x1, y1 = max(0, x-pad), max(0, y-pad)
                x2, y2 = min(w, x+bw+pad), min(h, y+bh+pad)
                
                text_mask[y1:y2, x1:x2] = 255
                
                text_regions.append({
                    "bbox": [x1, y1, x2-x1, y2-y1],
                    "text": text,
                    "confidence": conf,
                    "points": pts.tolist()
                })
                
            print(f"   EasyOCR检测到 {len(text_regions)} 个文字区域")
        else:
            # 备用：基于颜色的文字检测
            text_regions, text_mask = self._detect_text_by_color(img)
        
        coverage = np.sum(text_mask > 0) / (w * h)
        print(f"   文字mask覆盖率: {coverage:.1%}")
        
        return text_regions, text_mask
    
    def _detect_text_by_color(self, img: np.ndarray) -> tuple:
        """基于颜色检测文字（备用方法）"""
        h, w = img.shape[:2]
        text_mask = np.zeros((h, w), dtype=np.uint8)
        text_regions = []
        
        # 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值 - 更好地处理不同亮度区域
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # 形态学处理 - 连接文字笔画
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
        
        # 查找连通域
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            
            # 文字特征过滤
            aspect = bw / bh if bh > 0 else 0
            density = area / (bw * bh) if bw * bh > 0 else 0
            
            # 严格的文字判断条件
            is_text = (
                5 < bh < 25 and        # 合理的文字高度
                bw > 10 and            # 最小宽度
                aspect > 0.5 and       # 不能太窄
                density > 0.2          # 不能太稀疏
            )
            
            if is_text:
                text_mask[y:y+bh, x:x+bw] = 255
                text_regions.append({
                    "bbox": [x, y, bw, bh],
                    "text": None,
                    "confidence": 0.5
                })
        
        return text_regions, text_mask
    
    def separate_by_color(self, image_path: str) -> dict:
        """
        按颜色分离图层
        科研图通常使用标准颜色：红、蓝、黑
        """
        print("\n" + "="*60)
        print("🎨 颜色分离")
        print("="*60)
        
        img = cv2.imread(image_path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        
        layers = {}
        
        # 红色 (HSV: H在0-10或170-180)
        red_lower1 = np.array([0, 70, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 70, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(img_hsv, red_lower1, red_upper1) | cv2.inRange(img_hsv, red_lower2, red_upper2)
        layers["red"] = red_mask
        print(f"   红色区域: {np.sum(red_mask > 0) / (w*h) * 100:.2f}%")
        
        # 蓝色 (HSV: H在100-130)
        blue_lower = np.array([100, 70, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(img_hsv, blue_lower, blue_upper)
        layers["blue"] = blue_mask
        print(f"   蓝色区域: {np.sum(blue_mask > 0) / (w*h) * 100:.2f}%")
        
        # 黑色/深灰 (低亮度)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        # 排除已识别的红蓝区域
        black_mask = black_mask & ~red_mask & ~blue_mask
        layers["black"] = black_mask
        print(f"   黑色区域: {np.sum(black_mask > 0) / (w*h) * 100:.2f}%")
        
        return layers
    
    def detect_geometric_precise(self, image_path: str, text_mask: np.ndarray, 
                                  color_layers: dict) -> dict:
        """
        精确几何检测
        基于颜色分层处理
        """
        print("\n" + "="*60)
        print("📐 精确几何检测")
        print("="*60)
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        elements = {
            "lines": [],
            "rectangles": [],
            "diamonds": [],
            "zigzags": [],  # 电阻符号
            "arrows": [],
            "circles": []
        }
        
        # 对每个颜色层单独处理
        for color_name, mask in color_layers.items():
            print(f"\n   处理 {color_name} 层...")
            
            # 排除文字区域
            clean_mask = mask.copy()
            clean_mask[text_mask > 0] = 0
            
            # 1. 直线检测
            lines = cv2.HoughLinesP(clean_mask, 1, np.pi/180, 30,
                                    minLineLength=20, maxLineGap=5)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
                    
                    elements["lines"].append({
                        "start": (int(x1), int(y1)),
                        "end": (int(x2), int(y2)),
                        "length": float(length),
                        "angle": float(angle),
                        "color": color_name
                    })
            
            # 2. 轮廓检测
            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 50:
                    continue
                
                # 多边形近似
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                vertices = len(approx)
                
                x, y, bw, bh = cv2.boundingRect(cnt)
                
                if vertices == 4:
                    # 判断是矩形还是菱形
                    rect_area = bw * bh
                    fill_ratio = area / rect_area if rect_area > 0 else 0
                    
                    # 检查是否是45度旋转的菱形
                    pts = approx.reshape(4, 2)
                    center = pts.mean(axis=0)
                    
                    # 计算对角线
                    diag1 = np.linalg.norm(pts[0] - pts[2])
                    diag2 = np.linalg.norm(pts[1] - pts[3])
                    
                    if fill_ratio > 0.85:
                        elements["rectangles"].append({
                            "bbox": [int(x), int(y), int(bw), int(bh)],
                            "vertices": pts.tolist(),
                            "color": color_name
                        })
                    elif 0.4 < diag1/diag2 < 2.5:  # 对角线比例接近
                        elements["diamonds"].append({
                            "center": (int(center[0]), int(center[1])),
                            "vertices": pts.tolist(),
                            "color": color_name
                        })
                
                elif vertices == 3:
                    # 三角形 -> 可能是箭头
                    elements["arrows"].append({
                        "vertices": approx.reshape(3, 2).tolist(),
                        "color": color_name
                    })
                
                elif vertices > 6:
                    # 多边形 -> 检查是否是圆
                    circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
                    if circularity > 0.7:
                        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                        elements["circles"].append({
                            "center": (int(cx), int(cy)),
                            "radius": int(radius),
                            "color": color_name
                        })
        
        # 3. 检测锯齿线（电阻符号）
        elements["zigzags"] = self._detect_zigzag_pattern(color_layers.get("black", np.zeros((h, w), dtype=np.uint8)))
        
        print(f"\n📊 检测结果:")
        for elem_type, elem_list in elements.items():
            if elem_list:
                print(f"   - {elem_type}: {len(elem_list)}")
        
        return elements
    
    def _detect_zigzag_pattern(self, mask: np.ndarray) -> list:
        """检测锯齿线（电阻符号）"""
        zigzags = []
        
        # 使用形态学检测
        # 锯齿线特征：窄长、高频变化
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 15 or h > 30:  # 锯齿线通常是窄长的
                continue
            
            # 计算轮廓复杂度
            peri = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)
            
            if peri > 0 and area > 0:
                complexity = peri * peri / area  # 周长平方/面积 - 越大越复杂
                if complexity > 50:  # 高复杂度 = 锯齿
                    zigzags.append({
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "complexity": float(complexity)
                    })
        
        return zigzags
    
    def create_high_quality_mask(self, image_path: str, text_mask: np.ndarray,
                                  color_layers: dict, elements: dict) -> dict:
        """
        生成高质量分层Mask
        """
        print("\n" + "="*60)
        print("🎭 生成高质量Mask")
        print("="*60)
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        masks = {
            "text": text_mask,
            "red_elements": color_layers.get("red", np.zeros((h, w), dtype=np.uint8)),
            "blue_elements": color_layers.get("blue", np.zeros((h, w), dtype=np.uint8)),
            "black_lines": color_layers.get("black", np.zeros((h, w), dtype=np.uint8)),
            "background": np.zeros((h, w), dtype=np.uint8)
        }
        
        # 背景 = 白色区域 - 所有其他mask
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        combined_elements = masks["text"] | masks["red_elements"] | masks["blue_elements"] | masks["black_lines"]
        masks["background"] = white_mask & ~combined_elements
        
        # 打印统计
        total_pixels = w * h
        print("\n   Mask覆盖统计:")
        for name, mask in masks.items():
            coverage = np.sum(mask > 0) / total_pixels * 100
            print(f"   - {name}: {coverage:.2f}%")
        
        return masks
    
    def visualize_results(self, image_path: str, text_regions: list,
                          elements: dict, masks: dict, output_dir: str) -> str:
        """生成可视化结果"""
        print("\n" + "="*60)
        print("📊 生成可视化")
        print("="*60)
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 文字检测可视化
        vis_text = img.copy()
        for region in text_regions:
            x, y, bw, bh = region["bbox"]
            cv2.rectangle(vis_text, (x, y), (x+bw, y+bh), (0, 255, 0), 1)
            if region.get("text"):
                cv2.putText(vis_text, region["text"][:10], (x, y-2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # 2. 几何检测可视化
        vis_geom = img.copy()
        
        # 画直线
        for line in elements.get("lines", [])[:50]:
            color = {"red": (0, 0, 255), "blue": (255, 0, 0), "black": (128, 128, 128)}.get(line.get("color"), (0, 0, 0))
            cv2.line(vis_geom, line["start"], line["end"], color, 2)
        
        # 画矩形
        for rect in elements.get("rectangles", []):
            color = {"red": (0, 0, 255), "blue": (255, 0, 0), "black": (128, 128, 128)}.get(rect.get("color"), (0, 0, 0))
            x, y, bw, bh = rect["bbox"]
            cv2.rectangle(vis_geom, (x, y), (x+bw, y+bh), color, 2)
        
        # 画菱形
        for diamond in elements.get("diamonds", []):
            pts = np.array(diamond["vertices"], dtype=np.int32)
            cv2.polylines(vis_geom, [pts], True, (255, 0, 255), 2)
        
        # 3. Mask组合可视化
        vis_mask = np.zeros((h, w, 3), dtype=np.uint8)
        vis_mask[masks["text"] > 0] = [0, 255, 0]       # 文字=绿
        vis_mask[masks["red_elements"] > 0] = [0, 0, 255]  # 红色元素=红
        vis_mask[masks["blue_elements"] > 0] = [255, 0, 0]  # 蓝色元素=蓝
        vis_mask[masks["black_lines"] > 0] = [128, 128, 128]  # 黑色=灰
        
        # 4. 拼接对比图
        row1 = np.hstack([img, vis_text])
        row2 = np.hstack([vis_geom, vis_mask])
        comparison = np.vstack([row1, row2])
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 20), font, 0.5, (0, 0, 0), 1)
        cv2.putText(comparison, "Text Detection", (w+10, 20), font, 0.5, (0, 255, 0), 1)
        cv2.putText(comparison, "Geometry Detection", (10, h+20), font, 0.5, (255, 0, 255), 1)
        cv2.putText(comparison, "Color Masks", (w+10, h+20), font, 0.5, (255, 255, 0), 1)
        
        # 保存
        output_path = f"{output_dir}/analysis_v2_{timestamp}.png"
        cv2.imwrite(output_path, comparison)
        print(f"   对比图已保存: {output_path}")
        
        # 保存单独的高质量mask
        for name, mask in masks.items():
            mask_path = f"{output_dir}/mask_{name}_{timestamp}.png"
            cv2.imwrite(mask_path, mask)
        print(f"   Mask文件已保存到: {output_dir}")
        
        return output_path
    
    def process(self, image_path: str, output_dir: str = None) -> dict:
        """完整处理流程"""
        print("\n" + "="*70)
        print("🎯 科研绘图矢量化处理器 v2")
        print("="*70)
        print(f"输入: {image_path}")
        
        if output_dir is None:
            output_dir = "/Volumes/Seagate/SAM3/02_output/scientific_v2"
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Step 1: 精确文字检测
        text_regions, text_mask = self.detect_text_precise(image_path)
        
        # Step 2: 颜色分离
        color_layers = self.separate_by_color(image_path)
        
        # Step 3: 几何检测
        elements = self.detect_geometric_precise(image_path, text_mask, color_layers)
        
        # Step 4: 生成高质量Mask
        masks = self.create_high_quality_mask(image_path, text_mask, color_layers, elements)
        
        # Step 5: 可视化
        vis_path = self.visualize_results(image_path, text_regions, elements, masks, output_dir)
        
        result = {
            "text_count": len(text_regions),
            "text_regions": text_regions,
            "elements": {k: len(v) for k, v in elements.items()},
            "masks": list(masks.keys()),
            "visualization": vis_path
        }
        
        print("\n" + "="*70)
        print("✅ 处理完成")
        print("="*70)
        
        return result


def main():
    processor = ScientificFigureV2()
    
    image_path = "/Volumes/Seagate/SAM3/01_input/科研绘图1.png"
    result = processor.process(image_path)
    
    print("\n📊 结果摘要:")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
