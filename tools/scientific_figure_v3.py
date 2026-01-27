#!/usr/bin/env python3
"""
科研绘图矢量化处理器 v3
核心改进：
1. 解决文字与颜色重叠问题
2. SAM3处理3D悬臂梁复杂区域
3. 生成高质量分层Mask
"""

import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import json

# 添加SAM3路径
sys.path.insert(0, "/Volumes/Seagate/SAM3")
sys.path.insert(0, "/Volumes/Seagate/SAM3/models/sam3")


class ScientificFigureV3:
    """科研图处理器 v3 - 高质量Mask"""
    
    def __init__(self):
        self.ocr = None
        self.sam3_processor = None
        self.sam3_model = None
        
    def _init_ocr(self):
        """延迟加载OCR"""
        if self.ocr is None:
            try:
                import easyocr
                print("🔤 加载EasyOCR...")
                self.ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
                print("✅ EasyOCR加载成功")
            except ImportError:
                print("⚠️ EasyOCR未安装")
                self.ocr = "fallback"
    
    def _init_sam3(self):
        """延迟加载SAM3"""
        if self.sam3_processor is None:
            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
                
                print("🧠 加载SAM3...")
                self.sam3_model = build_sam3_image_model(device="cpu")
                self.sam3_processor = Sam3Processor(self.sam3_model, device="cpu")
                print("✅ SAM3加载成功")
            except Exception as e:
                print(f"⚠️ SAM3加载失败: {e}")
                self.sam3_processor = "fallback"
    
    def detect_text_precise(self, image_path: str) -> tuple:
        """精确文字检测，返回(text_regions, text_mask)"""
        print("\n" + "="*60)
        print("📝 STEP 1: 精确文字检测")
        print("="*60)
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        text_mask = np.zeros((h, w), dtype=np.uint8)
        text_regions = []
        
        self._init_ocr()
        
        if self.ocr != "fallback":
            results = self.ocr.readtext(image_path)
            
            for (bbox, text, conf) in results:
                if conf < 0.3:
                    continue
                
                pts = np.array(bbox, dtype=np.int32)
                x, y, bw, bh = cv2.boundingRect(pts)
                
                # 精确多边形mask而非矩形
                cv2.fillPoly(text_mask, [pts], 255)
                
                text_regions.append({
                    "bbox": [x, y, bw, bh],
                    "text": text,
                    "confidence": conf,
                    "polygon": pts.tolist()
                })
            
            print(f"   检测到 {len(text_regions)} 个文字区域")
        
        # 轻微膨胀确保完全覆盖
        kernel = np.ones((3, 3), np.uint8)
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)
        
        coverage = np.sum(text_mask > 0) / (w * h)
        print(f"   文字mask覆盖率: {coverage:.1%}")
        
        return text_regions, text_mask
    
    def separate_colors_clean(self, image_path: str, text_mask: np.ndarray) -> dict:
        """
        颜色分离 - 关键改进：从颜色mask中排除文字区域
        """
        print("\n" + "="*60)
        print("🎨 STEP 2: 颜色分离（排除文字）")
        print("="*60)
        
        img = cv2.imread(image_path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        
        # 创建非文字区域mask
        non_text_mask = ~text_mask.astype(bool)
        
        layers = {}
        
        # 红色检测
        red_lower1 = np.array([0, 70, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 70, 50])
        red_upper2 = np.array([180, 255, 255])
        red_raw = cv2.inRange(img_hsv, red_lower1, red_upper1) | cv2.inRange(img_hsv, red_lower2, red_upper2)
        # 排除文字区域
        layers["red"] = (red_raw & (non_text_mask * 255).astype(np.uint8))
        
        # 蓝色检测
        blue_lower = np.array([100, 70, 50])
        blue_upper = np.array([130, 255, 255])
        blue_raw = cv2.inRange(img_hsv, blue_lower, blue_upper)
        layers["blue"] = (blue_raw & (non_text_mask * 255).astype(np.uint8))
        
        # 黑色检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, black_raw = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        # 排除文字和已识别的红蓝
        black_clean = black_raw & (non_text_mask * 255).astype(np.uint8)
        black_clean = black_clean & ~layers["red"] & ~layers["blue"]
        layers["black"] = black_clean
        
        # 打印统计
        total = w * h
        for name, mask in layers.items():
            pct = np.sum(mask > 0) / total * 100
            print(f"   {name}: {pct:.2f}%")
        
        return layers
    
    def segment_3d_beam_with_sam3(self, image_path: str, text_mask: np.ndarray) -> dict:
        """
        使用SAM3分割3D悬臂梁区域
        """
        print("\n" + "="*60)
        print("🧠 STEP 3: SAM3分割3D悬臂梁")
        print("="*60)
        
        self._init_sam3()
        
        if self.sam3_processor == "fallback":
            print("   跳过SAM3，使用备用方法")
            return self._segment_beam_fallback(image_path)
        
        img = Image.open(image_path)
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # 设置图像
        state = self.sam3_processor.set_image(img)
        
        # 针对3D悬臂梁的prompt
        beam_prompts = [
            "3D cantilever beam",
            "mechanical beam structure", 
            "strain gauge mounting plate",
            "metal beam with strain gauges"
        ]
        
        beam_masks = []
        
        for prompt in beam_prompts:
            print(f"   尝试: '{prompt}'")
            try:
                state = self.sam3_processor.set_text_prompt(prompt, state)
                
                if state and "masks" in state and len(state["masks"]) > 0:
                    for mask in state["masks"]:
                        mask_array = np.array(mask)
                        
                        # 处理多维mask (squeeze掉batch维度)
                        while mask_array.ndim > 2:
                            mask_array = mask_array.squeeze(0)
                        
                        if mask_array.dtype == bool:
                            mask_array = mask_array.astype(np.uint8) * 255
                        elif mask_array.max() <= 1:
                            mask_array = (mask_array * 255).astype(np.uint8)
                        
                        area = np.sum(mask_array > 0)
                        if area > 1000:  # 过滤太小的
                            beam_masks.append({
                                "prompt": prompt,
                                "mask": mask_array,
                                "area": int(area)
                            })
                            print(f"      ✓ 找到mask, 面积: {area}")
            except Exception as e:
                print(f"      ✗ 失败: {e}")
        
        if not beam_masks:
            print("   SAM3未找到悬臂梁，使用备用方法")
            return self._segment_beam_fallback(image_path)
        
        # 选择最佳mask（面积最大的）
        best_mask = max(beam_masks, key=lambda x: x["area"])
        print(f"\n   ✅ 选择最佳mask: '{best_mask['prompt']}', 面积: {best_mask['area']}")
        
        return {
            "beam_3d": best_mask["mask"],
            "method": "sam3",
            "prompt": best_mask["prompt"]
        }
    
    def _segment_beam_fallback(self, image_path: str) -> dict:
        """备用方法：基于位置和颜色分割悬臂梁"""
        print("   使用位置+颜色备用方法")
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # 悬臂梁大约在图像左侧1/3区域
        beam_region = np.zeros((h, w), dtype=np.uint8)
        
        # 基于灰度和位置
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 左侧区域
        left_region = gray[:, :int(w*0.4)]
        
        # 悬臂梁是浅灰色带阴影
        _, beam_mask = cv2.threshold(left_region, 180, 255, cv2.THRESH_BINARY)
        beam_mask_inv = cv2.bitwise_not(beam_mask)
        
        # 形态学处理
        kernel = np.ones((5, 5), np.uint8)
        beam_mask_clean = cv2.morphologyEx(beam_mask_inv, cv2.MORPH_CLOSE, kernel)
        beam_mask_clean = cv2.morphologyEx(beam_mask_clean, cv2.MORPH_OPEN, kernel)
        
        # 放回完整尺寸
        beam_region[:, :int(w*0.4)] = beam_mask_clean
        
        # 查找最大连通域
        contours, _ = cv2.findContours(beam_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            final_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(final_mask, [largest], -1, 255, -1)
            
            area = cv2.contourArea(largest)
            print(f"   ✅ 备用方法找到悬臂梁, 面积: {area}")
            
            return {
                "beam_3d": final_mask,
                "method": "fallback",
                "prompt": "position+color"
            }
        
        return {
            "beam_3d": np.zeros((h, w), dtype=np.uint8),
            "method": "failed",
            "prompt": None
        }
    
    def refine_masks(self, text_mask: np.ndarray, color_layers: dict, 
                     beam_result: dict) -> dict:
        """
        STEP 4: 优化和整合所有mask
        确保层次清晰，无重叠
        """
        print("\n" + "="*60)
        print("🔧 STEP 4: Mask优化与整合")
        print("="*60)
        
        h, w = text_mask.shape
        
        # 创建优先级层次（高优先级覆盖低优先级）
        # 优先级: 文字 > 红色 > 蓝色 > 3D悬臂梁 > 黑色线条 > 背景
        
        final_masks = {
            "L1_text": text_mask.copy(),
            "L2_red": np.zeros((h, w), dtype=np.uint8),
            "L3_blue": np.zeros((h, w), dtype=np.uint8),
            "L4_beam_3d": np.zeros((h, w), dtype=np.uint8),
            "L5_black_lines": np.zeros((h, w), dtype=np.uint8),
            "L6_background": np.zeros((h, w), dtype=np.uint8)
        }
        
        # 已占用区域
        occupied = text_mask > 0
        
        # L2: 红色（排除文字）
        red_clean = color_layers["red"].copy()
        red_clean[occupied] = 0
        final_masks["L2_red"] = red_clean
        occupied = occupied | (red_clean > 0)
        
        # L3: 蓝色（排除文字和红色）
        blue_clean = color_layers["blue"].copy()
        blue_clean[occupied] = 0
        final_masks["L3_blue"] = blue_clean
        occupied = occupied | (blue_clean > 0)
        
        # L4: 3D悬臂梁（排除已占用）
        beam_mask = beam_result.get("beam_3d", np.zeros((h, w), dtype=np.uint8))
        beam_clean = beam_mask.copy()
        beam_clean[occupied] = 0
        final_masks["L4_beam_3d"] = beam_clean
        occupied = occupied | (beam_clean > 0)
        
        # L5: 黑色线条（排除已占用）
        black_clean = color_layers["black"].copy()
        black_clean[occupied] = 0
        final_masks["L5_black_lines"] = black_clean
        occupied = occupied | (black_clean > 0)
        
        # L6: 背景（剩余区域）
        final_masks["L6_background"] = (~occupied).astype(np.uint8) * 255
        
        # 打印统计
        print("\n   分层Mask统计:")
        total = h * w
        for name, mask in final_masks.items():
            pct = np.sum(mask > 0) / total * 100
            print(f"   - {name}: {pct:.2f}%")
        
        # 验证无重叠
        overlap_check = np.zeros((h, w), dtype=np.int32)
        for mask in final_masks.values():
            overlap_check += (mask > 0).astype(np.int32)
        
        max_overlap = np.max(overlap_check)
        if max_overlap > 1:
            print(f"   ⚠️ 警告: 检测到重叠区域 (最大重叠层数: {max_overlap})")
        else:
            print("   ✅ 验证通过: 无重叠")
        
        return final_masks
    
    def create_visualization(self, image_path: str, text_regions: list,
                             final_masks: dict, output_dir: str) -> str:
        """生成高质量可视化"""
        print("\n" + "="*60)
        print("📊 STEP 5: 生成可视化")
        print("="*60)
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建彩色mask叠加图
        vis_overlay = img.copy()
        
        # 定义每层颜色 (BGR)
        layer_colors = {
            "L1_text": (0, 255, 0),        # 绿色
            "L2_red": (0, 0, 255),          # 红色
            "L3_blue": (255, 0, 0),         # 蓝色
            "L4_beam_3d": (128, 128, 0),    # 青色
            "L5_black_lines": (128, 128, 128),  # 灰色
        }
        
        for layer_name, color in layer_colors.items():
            mask = final_masks.get(layer_name)
            if mask is not None and np.sum(mask) > 0:
                # 半透明叠加
                overlay_region = vis_overlay.copy()
                overlay_region[mask > 0] = color
                vis_overlay = cv2.addWeighted(vis_overlay, 0.7, overlay_region, 0.3, 0)
        
        # 创建分层展示图 (2x3网格)
        cell_h, cell_w = h, w
        grid = np.ones((cell_h * 2, cell_w * 3, 3), dtype=np.uint8) * 255
        
        # Row 1: 原图, 文字mask, 红色mask
        grid[0:cell_h, 0:cell_w] = img
        grid[0:cell_h, cell_w:cell_w*2] = cv2.cvtColor(final_masks["L1_text"], cv2.COLOR_GRAY2BGR)
        grid[0:cell_h, cell_w*2:cell_w*3] = cv2.cvtColor(final_masks["L2_red"], cv2.COLOR_GRAY2BGR)
        
        # Row 2: 蓝色mask, 3D悬臂梁mask, 叠加结果
        grid[cell_h:cell_h*2, 0:cell_w] = cv2.cvtColor(final_masks["L3_blue"], cv2.COLOR_GRAY2BGR)
        grid[cell_h:cell_h*2, cell_w:cell_w*2] = cv2.cvtColor(final_masks["L4_beam_3d"], cv2.COLOR_GRAY2BGR)
        grid[cell_h:cell_h*2, cell_w*2:cell_w*3] = vis_overlay
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = [
            ("Original", (10, 20)),
            ("L1: Text", (cell_w + 10, 20)),
            ("L2: Red Elements", (cell_w*2 + 10, 20)),
            ("L3: Blue Elements", (10, cell_h + 20)),
            ("L4: 3D Beam (SAM3)", (cell_w + 10, cell_h + 20)),
            ("Final Overlay", (cell_w*2 + 10, cell_h + 20)),
        ]
        
        for label, pos in labels:
            cv2.putText(grid, label, pos, font, 0.5, (0, 0, 0), 1)
        
        # 保存
        grid_path = f"{output_dir}/v3_layers_{timestamp}.png"
        cv2.imwrite(grid_path, grid)
        print(f"   分层图已保存: {grid_path}")
        
        # 保存单独的高质量mask
        for name, mask in final_masks.items():
            mask_path = f"{output_dir}/v3_{name}_{timestamp}.png"
            cv2.imwrite(mask_path, mask)
        print(f"   各层Mask已保存到: {output_dir}")
        
        # 保存叠加图
        overlay_path = f"{output_dir}/v3_overlay_{timestamp}.png"
        cv2.imwrite(overlay_path, vis_overlay)
        
        return grid_path
    
    def process(self, image_path: str, output_dir: str = None) -> dict:
        """完整处理流程"""
        print("\n" + "="*70)
        print("🎯 科研绘图矢量化处理器 v3 - 高质量Mask")
        print("="*70)
        print(f"输入: {image_path}")
        
        if output_dir is None:
            output_dir = "/Volumes/Seagate/SAM3/02_output/scientific_v3"
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Step 1: 精确文字检测
        text_regions, text_mask = self.detect_text_precise(image_path)
        
        # Step 2: 颜色分离（排除文字）
        color_layers = self.separate_colors_clean(image_path, text_mask)
        
        # Step 3: SAM3分割3D悬臂梁
        beam_result = self.segment_3d_beam_with_sam3(image_path, text_mask)
        
        # Step 4: 优化整合
        final_masks = self.refine_masks(text_mask, color_layers, beam_result)
        
        # Step 5: 可视化
        vis_path = self.create_visualization(image_path, text_regions, final_masks, output_dir)
        
        result = {
            "text_count": len(text_regions),
            "beam_method": beam_result.get("method"),
            "layers": list(final_masks.keys()),
            "visualization": vis_path
        }
        
        print("\n" + "="*70)
        print("✅ 处理完成")
        print("="*70)
        
        return result


def main():
    processor = ScientificFigureV3()
    
    image_path = "/Volumes/Seagate/SAM3/01_input/科研绘图1.png"
    result = processor.process(image_path)
    
    print("\n📊 结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
