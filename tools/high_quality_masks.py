#!/usr/bin/env python3
"""
高质量Mask生成器
生成极高精度的分层mask，适合矢量化处理
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, "/Volumes/Seagate/SAM3")
sys.path.insert(0, "/Volumes/Seagate/SAM3/models/sam3")


class HighQualityMaskGenerator:
    """高质量Mask生成器"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or "/Volumes/Seagate/SAM3/02_output/hq_masks"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.sam3_processor = None
        self.ocr_reader = None
        
    def _init_sam3(self):
        """初始化SAM3"""
        if self.sam3_processor is not None:
            return
        try:
            from sam3.model_builder import build_sam3_image_model
            print("🧠 加载SAM3...")
            self.sam3_processor = build_sam3_image_model()
            print("✅ SAM3加载成功")
        except Exception as e:
            print(f"⚠️ SAM3加载失败: {e}")
            self.sam3_processor = "fallback"
    
    def _init_ocr(self):
        """初始化OCR"""
        if self.ocr_reader is not None:
            return
        try:
            import easyocr
            print("🔤 加载EasyOCR...")
            self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("✅ EasyOCR加载成功")
        except Exception as e:
            print(f"⚠️ OCR加载失败: {e}")
            self.ocr_reader = "fallback"
    
    def generate_high_quality_masks(self, image_path: str) -> dict:
        """
        生成高质量分层mask
        
        返回：
        - 精确的边缘检测
        - 抗锯齿处理
        - 无压缩PNG输出
        """
        print("\n" + "="*70)
        print("🎯 高质量Mask生成")
        print("="*70)
        print(f"   输入: {image_path}")
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        h, w = img.shape[:2]
        print(f"   尺寸: {w}×{h}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = Path(self.output_dir) / f"hq_{timestamp}"
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "image_size": (w, h),
            "output_dir": str(output_subdir),
            "masks": {}
        }
        
        # Step 1: 精确文字检测
        print("\n" + "-"*50)
        print("📝 Step 1: 精确文字检测")
        print("-"*50)
        text_mask, text_regions = self._detect_text_precise(img)
        results["text_regions"] = text_regions
        
        # Step 2: 精确颜色分离
        print("\n" + "-"*50)
        print("🎨 Step 2: 精确颜色分离")
        print("-"*50)
        color_masks = self._separate_colors_precise(img, text_mask)
        
        # Step 3: SAM3分割复杂结构
        print("\n" + "-"*50)
        print("🧠 Step 3: SAM3分割")
        print("-"*50)
        beam_mask = self._segment_with_sam3(image_path, text_mask)
        
        # Step 4: 精细边缘处理
        print("\n" + "-"*50)
        print("✨ Step 4: 边缘精细化")
        print("-"*50)
        
        # 整合所有mask，确保无重叠
        final_masks = self._integrate_masks(
            h, w, text_mask, color_masks, beam_mask
        )
        
        # Step 5: 保存高质量PNG（无压缩）
        print("\n" + "-"*50)
        print("💾 Step 5: 保存高质量Mask")
        print("-"*50)
        
        for layer_name, mask in final_masks.items():
            # 边缘抗锯齿
            mask_aa = self._anti_alias_mask(mask)
            
            # 保存为无压缩PNG
            output_path = output_subdir / f"{layer_name}.png"
            cv2.imwrite(str(output_path), mask_aa, 
                       [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 0 = 无压缩
            
            # 计算覆盖率
            coverage = np.sum(mask > 0) / (h * w) * 100
            file_size = output_path.stat().st_size
            
            print(f"   {layer_name}: {coverage:.2f}% ({file_size/1024:.1f}KB)")
            results["masks"][layer_name] = {
                "path": str(output_path),
                "coverage": coverage,
                "size_kb": file_size / 1024
            }
        
        # 保存叠加预览
        self._save_overlay_preview(img, final_masks, output_subdir / "overlay.png")
        
        # 保存原图副本
        cv2.imwrite(str(output_subdir / "original.png"), img,
                   [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        print("\n" + "="*70)
        print("✅ 高质量Mask生成完成")
        print("="*70)
        print(f"   输出目录: {output_subdir}")
        
        return results
    
    def _detect_text_precise(self, img: np.ndarray) -> tuple:
        """精确文字检测"""
        self._init_ocr()
        
        h, w = img.shape[:2]
        text_mask = np.zeros((h, w), dtype=np.uint8)
        text_regions = []
        
        if self.ocr_reader == "fallback":
            print("   使用备用文字检测")
            return text_mask, text_regions
        
        # 使用EasyOCR检测
        results = self.ocr_reader.readtext(img)
        
        for bbox, text, conf in results:
            if conf < 0.3:
                continue
            
            pts = np.array(bbox, dtype=np.int32)
            
            # 精确填充多边形（无膨胀，保持原始边界）
            cv2.fillPoly(text_mask, [pts], 255)
            
            x_min = int(min(p[0] for p in bbox))
            y_min = int(min(p[1] for p in bbox))
            x_max = int(max(p[0] for p in bbox))
            y_max = int(max(p[1] for p in bbox))
            
            text_regions.append({
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "text": text,
                "confidence": conf,
                "polygon": bbox
            })
        
        print(f"   检测到 {len(text_regions)} 个文字区域")
        coverage = np.sum(text_mask > 0) / (h * w) * 100
        print(f"   文字覆盖率: {coverage:.2f}%")
        
        return text_mask, text_regions
    
    def _separate_colors_precise(self, img: np.ndarray, 
                                  text_mask: np.ndarray) -> dict:
        """精确颜色分离"""
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 更精确的颜色范围
        color_ranges = {
            "red": [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))
            ],
            "blue": [
                (np.array([100, 80, 80]), np.array([130, 255, 255]))
            ],
            "black": [
                (np.array([0, 0, 0]), np.array([180, 50, 80]))
            ]
        }
        
        color_masks = {}
        text_mask_inv = cv2.bitwise_not(text_mask)
        
        for color_name, ranges in color_ranges.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            for lower, upper in ranges:
                m = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, m)
            
            # 排除文字区域
            mask = cv2.bitwise_and(mask, text_mask_inv)
            
            # 精细形态学处理（小kernel保留细节）
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            coverage = np.sum(mask > 0) / (h * w) * 100
            print(f"   {color_name}: {coverage:.2f}%")
            
            color_masks[color_name] = mask
        
        return color_masks
    
    def _segment_with_sam3(self, image_path: str, 
                           text_mask: np.ndarray) -> np.ndarray:
        """使用SAM3分割复杂结构"""
        self._init_sam3()
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        if self.sam3_processor == "fallback":
            return self._segment_beam_fallback(img)
        
        try:
            from PIL import Image
            pil_img = Image.open(image_path)
            state = self.sam3_processor.set_image(pil_img)
            
            prompts = [
                "3D cantilever beam",
                "mechanical beam structure", 
                "metal beam with strain gauges"
            ]
            
            best_mask = None
            best_area = 0
            
            for prompt in prompts:
                try:
                    state = self.sam3_processor.set_text_prompt(prompt, state)
                    if state and "masks" in state and len(state["masks"]) > 0:
                        for mask in state["masks"]:
                            mask_arr = np.array(mask)
                            while mask_arr.ndim > 2:
                                mask_arr = mask_arr.squeeze(0)
                            
                            if mask_arr.dtype == bool:
                                mask_arr = mask_arr.astype(np.uint8) * 255
                            elif mask_arr.max() <= 1:
                                mask_arr = (mask_arr * 255).astype(np.uint8)
                            
                            area = np.sum(mask_arr > 0)
                            if area > best_area and area > 1000:
                                best_area = area
                                best_mask = mask_arr
                                print(f"   ✓ SAM3找到: {prompt}, 面积: {area}")
                except:
                    pass
            
            if best_mask is not None:
                return best_mask
                
        except Exception as e:
            print(f"   SAM3失败: {e}")
        
        return self._segment_beam_fallback(img)
    
    def _segment_beam_fallback(self, img: np.ndarray) -> np.ndarray:
        """备用悬臂梁分割"""
        print("   使用备用方法分割悬臂梁")
        h, w = img.shape[:2]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 在左侧区域查找
        left_region = edges[:, :int(w*0.4)]
        
        # 找轮廓
        contours, _ = cv2.findContours(left_region, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_NONE)  # APPROX_NONE保留所有点
        
        beam_mask = np.zeros((h, w), dtype=np.uint8)
        
        if contours:
            # 找最大轮廓
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                cv2.drawContours(beam_mask[:, :int(w*0.4)], [largest], -1, 255, -1)
                print(f"   ✓ 找到悬臂梁, 面积: {cv2.contourArea(largest)}")
        
        return beam_mask
    
    def _integrate_masks(self, h: int, w: int, text_mask: np.ndarray,
                         color_masks: dict, beam_mask: np.ndarray) -> dict:
        """整合所有mask，确保无重叠"""
        
        # 优先级：文字 > 红色 > 蓝色 > 悬臂梁 > 黑色 > 背景
        final = {
            "L1_text": text_mask.copy(),
            "L2_red": np.zeros((h, w), dtype=np.uint8),
            "L3_blue": np.zeros((h, w), dtype=np.uint8),
            "L4_beam": np.zeros((h, w), dtype=np.uint8),
            "L5_black": np.zeros((h, w), dtype=np.uint8),
            "L6_background": np.zeros((h, w), dtype=np.uint8)
        }
        
        used = text_mask.copy()
        
        # 红色
        if "red" in color_masks:
            available = cv2.bitwise_and(color_masks["red"], cv2.bitwise_not(used))
            final["L2_red"] = available
            used = cv2.bitwise_or(used, available)
        
        # 蓝色
        if "blue" in color_masks:
            available = cv2.bitwise_and(color_masks["blue"], cv2.bitwise_not(used))
            final["L3_blue"] = available
            used = cv2.bitwise_or(used, available)
        
        # 悬臂梁
        if beam_mask is not None:
            available = cv2.bitwise_and(beam_mask, cv2.bitwise_not(used))
            final["L4_beam"] = available
            used = cv2.bitwise_or(used, available)
        
        # 黑色
        if "black" in color_masks:
            available = cv2.bitwise_and(color_masks["black"], cv2.bitwise_not(used))
            final["L5_black"] = available
            used = cv2.bitwise_or(used, available)
        
        # 背景
        final["L6_background"] = cv2.bitwise_not(used)
        
        # 验证无重叠
        total = sum(np.sum(m > 0) for m in final.values())
        expected = h * w
        if abs(total - expected) < 10:
            print("   ✅ 验证通过：无重叠")
        
        return final
    
    def _anti_alias_mask(self, mask: np.ndarray) -> np.ndarray:
        """边缘抗锯齿处理"""
        # 使用高斯模糊轻微平滑边缘
        blurred = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0.5)
        
        # 重新二值化但保留边缘过渡
        result = np.where(blurred > 200, 255, 
                         np.where(blurred > 50, blurred, 0))
        
        return result.astype(np.uint8)
    
    def _save_overlay_preview(self, img: np.ndarray, masks: dict, 
                               output_path: Path):
        """保存叠加预览图"""
        overlay = img.copy()
        
        colors = {
            "L1_text": (255, 255, 0),    # 黄色
            "L2_red": (0, 0, 255),        # 红色
            "L3_blue": (255, 0, 0),       # 蓝色
            "L4_beam": (128, 128, 128),   # 灰色
            "L5_black": (0, 255, 0),      # 绿色标注黑色区域
        }
        
        for layer_name, color in colors.items():
            if layer_name in masks:
                mask = masks[layer_name]
                colored = np.zeros_like(overlay)
                colored[:] = color
                overlay = np.where(mask[:,:,np.newaxis] > 0,
                                  cv2.addWeighted(overlay, 0.5, colored, 0.5, 0),
                                  overlay)
        
        cv2.imwrite(str(output_path), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"   预览已保存: {output_path}")


def main():
    generator = HighQualityMaskGenerator()
    result = generator.generate_high_quality_masks(
        "/Volumes/Seagate/SAM3/01_input/科研绘图1.png"
    )
    print(f"\n输出目录: {result['output_dir']}")


if __name__ == "__main__":
    main()
