#!/usr/bin/env python3
"""
SAM3语义分割器
使用SAM3的文本提示功能对科研图进行精确分割
输出：高质量mask PNG文件
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
import sys

sys.path.insert(0, "/Volumes/Seagate/SAM3")
sys.path.insert(0, "/Volumes/Seagate/SAM3/models/sam3")


class SAM3SemanticSegmenter:
    """SAM3语义分割器"""
    
    def __init__(self):
        self.output_dir = Path("/Volumes/Seagate/SAM3/02_output/sam3_masks")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sam3 = None
        
    def _init_sam3(self):
        """初始化SAM3"""
        if self.sam3 is not None:
            return True
            
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            print("🧠 加载SAM3模型...")
            model = build_sam3_image_model(device="cpu")
            self.sam3 = Sam3Processor(model, device="cpu")
            print("✅ SAM3加载成功")
            return True
        except Exception as e:
            print(f"❌ SAM3加载失败: {e}")
            return False
    
    def segment_with_prompt(self, image_path: str, prompt: str, 
                            output_name: str) -> np.ndarray:
        """使用文本提示分割"""
        if not self._init_sam3():
            return None
            
        try:
            pil_img = Image.open(image_path)
            state = self.sam3.set_image(pil_img)
            state = self.sam3.set_text_prompt(prompt, state)
            
            if state and "masks" in state and len(state["masks"]) > 0:
                # 获取最佳mask
                best_mask = None
                best_area = 0
                
                for mask in state["masks"]:
                    mask_arr = np.array(mask)
                    while mask_arr.ndim > 2:
                        mask_arr = mask_arr.squeeze(0)
                    
                    if mask_arr.dtype == bool:
                        mask_arr = mask_arr.astype(np.uint8) * 255
                    elif mask_arr.max() <= 1:
                        mask_arr = (mask_arr * 255).astype(np.uint8)
                    
                    area = np.sum(mask_arr > 0)
                    if area > best_area:
                        best_area = area
                        best_mask = mask_arr
                
                if best_mask is not None:
                    print(f"   ✓ '{prompt}': 面积 {best_area} 像素")
                    return best_mask
                    
        except Exception as e:
            print(f"   ✗ '{prompt}': {e}")
        
        return None
    
    def segment_scientific_figure(self, image_path: str):
        """分割科研图的所有元素"""
        print("\n" + "="*70)
        print("🎯 SAM3语义分割 - 科研图")
        print("="*70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.output_dir / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        print(f"   图像: {w}×{h}")
        
        # 保存原图
        cv2.imwrite(str(out_dir / "original.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # 定义要分割的元素及其提示词
        elements = [
            ("beam", "3D cantilever beam structure"),
            ("beam2", "metal beam with mounted sensors"),
            ("red_sensors", "red strain gauge sensors"),
            ("blue_sensors", "blue strain gauge sensors"),
            ("red_arrows", "red arrows"),
            ("blue_arrows", "blue arrows"),
            ("circuit", "Wheatstone bridge circuit diagram"),
            ("resistors", "electrical resistor symbols"),
            ("text_labels", "text labels and annotations"),
        ]
        
        masks = {}
        
        print("\n📦 SAM3分割各元素:")
        for name, prompt in elements:
            print(f"\n   处理: {name}")
            mask = self.segment_with_prompt(image_path, prompt, name)
            
            if mask is not None:
                masks[name] = mask
                # 保存mask
                mask_path = out_dir / f"{name}_mask.png"
                cv2.imwrite(str(mask_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                
                # 保存提取的区域
                extracted = cv2.bitwise_and(img, img, mask=mask)
                cv2.imwrite(str(out_dir / f"{name}_extracted.png"), extracted,
                           [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # 合并相似元素
        print("\n🔧 合并分层:")
        final_masks = {}
        
        # 悬臂梁
        if "beam" in masks or "beam2" in masks:
            beam_combined = np.zeros((h, w), dtype=np.uint8)
            if "beam" in masks:
                beam_combined = cv2.bitwise_or(beam_combined, masks["beam"])
            if "beam2" in masks:
                beam_combined = cv2.bitwise_or(beam_combined, masks["beam2"])
            final_masks["L1_beam"] = beam_combined
            cv2.imwrite(str(out_dir / "L1_beam.png"), beam_combined, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # 红色元素
        if "red_sensors" in masks or "red_arrows" in masks:
            red_combined = np.zeros((h, w), dtype=np.uint8)
            if "red_sensors" in masks:
                red_combined = cv2.bitwise_or(red_combined, masks["red_sensors"])
            if "red_arrows" in masks:
                red_combined = cv2.bitwise_or(red_combined, masks["red_arrows"])
            final_masks["L2_red"] = red_combined
            cv2.imwrite(str(out_dir / "L2_red.png"), red_combined, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # 蓝色元素
        if "blue_sensors" in masks or "blue_arrows" in masks:
            blue_combined = np.zeros((h, w), dtype=np.uint8)
            if "blue_sensors" in masks:
                blue_combined = cv2.bitwise_or(blue_combined, masks["blue_sensors"])
            if "blue_arrows" in masks:
                blue_combined = cv2.bitwise_or(blue_combined, masks["blue_arrows"])
            final_masks["L3_blue"] = blue_combined
            cv2.imwrite(str(out_dir / "L3_blue.png"), blue_combined, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # 电路
        if "circuit" in masks or "resistors" in masks:
            circuit_combined = np.zeros((h, w), dtype=np.uint8)
            if "circuit" in masks:
                circuit_combined = cv2.bitwise_or(circuit_combined, masks["circuit"])
            if "resistors" in masks:
                circuit_combined = cv2.bitwise_or(circuit_combined, masks["resistors"])
            final_masks["L4_circuit"] = circuit_combined
            cv2.imwrite(str(out_dir / "L4_circuit.png"), circuit_combined, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # 文字
        if "text_labels" in masks:
            final_masks["L5_text"] = masks["text_labels"]
            cv2.imwrite(str(out_dir / "L5_text.png"), masks["text_labels"], [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # 统计
        print("\n📊 最终分层统计:")
        for name, mask in final_masks.items():
            coverage = np.sum(mask > 0) / (h * w) * 100
            print(f"   {name}: {coverage:.2f}%")
        
        print(f"\n✅ 输出目录: {out_dir}")
        print("   文件列表:")
        for f in sorted(out_dir.glob("*.png")):
            size_kb = f.stat().st_size / 1024
            print(f"   - {f.name} ({size_kb:.1f}KB)")
        
        return str(out_dir)


def main():
    segmenter = SAM3SemanticSegmenter()
    segmenter.segment_scientific_figure("/Volumes/Seagate/SAM3/01_input/科研绘图1.png")


if __name__ == "__main__":
    main()
