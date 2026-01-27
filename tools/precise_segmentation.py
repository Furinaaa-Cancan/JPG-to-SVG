#!/usr/bin/env python3
"""
精确科研图分割器 v2
针对应变片电桥电路图的专门优化
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, "/Volumes/Seagate/SAM3")
sys.path.insert(0, "/Volumes/Seagate/SAM3/models/sam3")


class PreciseSegmenter:
    """精确分割器"""
    
    def __init__(self):
        self.output_dir = "/Volumes/Seagate/SAM3/02_output/precise_masks"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
    def segment(self, image_path: str) -> dict:
        """执行精确分割"""
        print("\n" + "="*70)
        print("🎯 精确科研图分割 v2")
        print("="*70)
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        print(f"   图像尺寸: {w}×{h}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.output_dir) / f"seg_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 转换颜色空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        masks = {}
        
        # ============ Step 1: 检测红色区域（传感器T1, T2 + 箭头）============
        print("\n🔴 Step 1: 红色区域检测")
        red_mask = self._detect_red_precise(img, hsv)
        masks["L1_red"] = red_mask
        print(f"   覆盖率: {np.sum(red_mask>0)/(h*w)*100:.2f}%")
        
        # ============ Step 2: 检测蓝色区域（传感器C1, C2 + 箭头）============
        print("\n🔵 Step 2: 蓝色区域检测")
        blue_mask = self._detect_blue_precise(img, hsv)
        masks["L2_blue"] = blue_mask
        print(f"   覆盖率: {np.sum(blue_mask>0)/(h*w)*100:.2f}%")
        
        # ============ Step 3: 检测3D悬臂梁结构 ============
        print("\n📦 Step 3: 3D悬臂梁检测")
        beam_mask = self._detect_beam_structure(img, gray, hsv)
        masks["L3_beam"] = beam_mask
        print(f"   覆盖率: {np.sum(beam_mask>0)/(h*w)*100:.2f}%")
        
        # ============ Step 4: 检测电路图区域 ============
        print("\n⚡ Step 4: 电路图检测")
        circuit_mask = self._detect_circuit(img, gray, w)
        masks["L4_circuit"] = circuit_mask
        print(f"   覆盖率: {np.sum(circuit_mask>0)/(h*w)*100:.2f}%")
        
        # ============ Step 5: 文字检测 ============
        print("\n📝 Step 5: 文字区域检测")
        text_mask, text_regions = self._detect_text(img)
        masks["L5_text"] = text_mask
        print(f"   检测到 {len(text_regions)} 个文字区域")
        print(f"   覆盖率: {np.sum(text_mask>0)/(h*w)*100:.2f}%")
        
        # ============ Step 6: 黑色线条和边框 ============
        print("\n⬛ Step 6: 黑色线条检测")
        black_mask = self._detect_black_lines(img, hsv, gray)
        masks["L6_black"] = black_mask
        print(f"   覆盖率: {np.sum(black_mask>0)/(h*w)*100:.2f}%")
        
        # ============ Step 7: 层级整合，确保无重叠 ============
        print("\n🔧 Step 7: 层级整合")
        final_masks = self._integrate_layers(h, w, masks)
        
        # ============ Step 8: 保存结果 ============
        print("\n💾 Step 8: 保存高质量Mask")
        self._save_results(img, final_masks, text_regions, out_dir)
        
        print("\n" + "="*70)
        print("✅ 分割完成")
        print("="*70)
        print(f"   输出目录: {out_dir}")
        
        return {"output_dir": str(out_dir), "masks": final_masks}
    
    def _detect_red_precise(self, img: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        """精确检测红色区域"""
        h, w = img.shape[:2]
        
        # 红色在HSV中分布在两端
        mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # 也检测BGR中的红色（更直接）
        b, g, r = cv2.split(img)
        red_dominant = (r.astype(np.int16) - np.maximum(g, b).astype(np.int16)) > 30
        red_bright = r > 150
        red_bgr = (red_dominant & red_bright).astype(np.uint8) * 255
        
        # 合并两种检测
        red_mask = cv2.bitwise_or(red_mask, red_bgr)
        
        # 形态学处理：填充小孔洞
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return red_mask
    
    def _detect_blue_precise(self, img: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        """精确检测蓝色区域"""
        # HSV检测
        blue_mask = cv2.inRange(hsv, np.array([100, 70, 70]), np.array([130, 255, 255]))
        
        # BGR检测
        b, g, r = cv2.split(img)
        blue_dominant = (b.astype(np.int16) - np.maximum(r, g).astype(np.int16)) > 20
        blue_bright = b > 120
        blue_bgr = (blue_dominant & blue_bright).astype(np.uint8) * 255
        
        blue_mask = cv2.bitwise_or(blue_mask, blue_bgr)
        
        # 形态学处理
        kernel = np.ones((3, 3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return blue_mask
    
    def _detect_beam_structure(self, img: np.ndarray, gray: np.ndarray, 
                                hsv: np.ndarray) -> np.ndarray:
        """检测3D悬臂梁结构"""
        h, w = img.shape[:2]
        beam_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 悬臂梁在左侧40%区域
        left_w = int(w * 0.45)
        
        # 方法1：检测灰色区域（悬臂梁主体）
        s = hsv[:, :left_w, 1]  # 饱和度
        v = hsv[:, :left_w, 2]  # 亮度
        
        # 低饱和度（灰色）+ 中等亮度
        gray_region = (s < 50) & (v > 100) & (v < 240)
        beam_mask[:, :left_w] = gray_region.astype(np.uint8) * 255
        
        # 方法2：边缘检测增强悬臂梁轮廓
        edges = cv2.Canny(gray[:, :left_w], 30, 100)
        
        # 膨胀边缘
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # 填充轮廓
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        temp_mask = np.zeros((h, left_w), dtype=np.uint8)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # 过滤小区域
                cv2.drawContours(temp_mask, [cnt], -1, 255, -1)
        
        # 合并方法1和方法2
        beam_mask[:, :left_w] = cv2.bitwise_or(beam_mask[:, :left_w], temp_mask)
        
        # 最终形态学处理
        kernel = np.ones((5, 5), np.uint8)
        beam_mask = cv2.morphologyEx(beam_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        beam_mask = cv2.morphologyEx(beam_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return beam_mask
    
    def _detect_circuit(self, img: np.ndarray, gray: np.ndarray, 
                        width: int) -> np.ndarray:
        """检测电路图区域"""
        h, w = img.shape[:2]
        circuit_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 电路在右侧55%区域
        right_start = int(w * 0.45)
        right_region = gray[:, right_start:]
        
        # 检测电路中的细线条（电阻符号等）
        edges = cv2.Canny(right_region, 50, 150)
        
        # 膨胀连接
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        temp_mask = np.zeros_like(right_region)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                cv2.drawContours(temp_mask, [cnt], -1, 255, -1)
        
        circuit_mask[:, right_start:] = temp_mask
        
        # 形态学处理
        kernel = np.ones((3, 3), np.uint8)
        circuit_mask = cv2.morphologyEx(circuit_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return circuit_mask
    
    def _detect_text(self, img: np.ndarray) -> tuple:
        """检测文字区域"""
        h, w = img.shape[:2]
        text_mask = np.zeros((h, w), dtype=np.uint8)
        text_regions = []
        
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            results = reader.readtext(img)
            
            for bbox, text, conf in results:
                if conf < 0.3:
                    continue
                
                pts = np.array(bbox, dtype=np.int32)
                
                # 稍微扩展边界以完整覆盖文字
                x_min = max(0, int(min(p[0] for p in bbox)) - 2)
                y_min = max(0, int(min(p[1] for p in bbox)) - 2)
                x_max = min(w, int(max(p[0] for p in bbox)) + 2)
                y_max = min(h, int(max(p[1] for p in bbox)) + 2)
                
                cv2.rectangle(text_mask, (x_min, y_min), (x_max, y_max), 255, -1)
                
                text_regions.append({
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "text": text,
                    "confidence": conf
                })
                
        except Exception as e:
            print(f"   OCR失败: {e}")
        
        return text_mask, text_regions
    
    def _detect_black_lines(self, img: np.ndarray, hsv: np.ndarray, 
                            gray: np.ndarray) -> np.ndarray:
        """检测黑色线条和边框"""
        h, w = img.shape[:2]
        
        # 方法1：HSV检测黑色
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        black_hsv = ((v < 80) & (s < 50)).astype(np.uint8) * 255
        
        # 方法2：灰度阈值
        _, black_gray = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        # 合并
        black_mask = cv2.bitwise_or(black_hsv, black_gray)
        
        # 形态学处理：保留细线
        kernel = np.ones((2, 2), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        
        return black_mask
    
    def _integrate_layers(self, h: int, w: int, masks: dict) -> dict:
        """层级整合，确保无重叠"""
        # 优先级：红色 > 蓝色 > 文字 > 黑色线条 > 电路 > 悬臂梁 > 背景
        priority = ["L1_red", "L2_blue", "L5_text", "L6_black", "L4_circuit", "L3_beam"]
        
        final = {}
        used = np.zeros((h, w), dtype=np.uint8)
        
        for layer_name in priority:
            if layer_name in masks:
                available = cv2.bitwise_and(masks[layer_name], cv2.bitwise_not(used))
                final[layer_name] = available
                used = cv2.bitwise_or(used, available)
                
                coverage = np.sum(available > 0) / (h * w) * 100
                print(f"   {layer_name}: {coverage:.2f}%")
        
        # 背景
        final["L7_background"] = cv2.bitwise_not(used)
        bg_coverage = np.sum(final["L7_background"] > 0) / (h * w) * 100
        print(f"   L7_background: {bg_coverage:.2f}%")
        
        # 验证
        total = sum(np.sum(m > 0) for m in final.values())
        if abs(total - h * w) < 10:
            print("   ✅ 验证通过：无重叠")
        
        return final
    
    def _save_results(self, img: np.ndarray, masks: dict, 
                      text_regions: list, out_dir: Path):
        """保存结果"""
        h, w = img.shape[:2]
        
        # 保存原图
        cv2.imwrite(str(out_dir / "original.png"), img, 
                   [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # 颜色映射
        colors = {
            "L1_red": (0, 0, 255),
            "L2_blue": (255, 0, 0),
            "L3_beam": (128, 128, 128),
            "L4_circuit": (0, 255, 255),
            "L5_text": (0, 255, 0),
            "L6_black": (255, 255, 0),
            "L7_background": (255, 255, 255)
        }
        
        # 保存各层mask
        for layer_name, mask in masks.items():
            # 无压缩PNG
            path = out_dir / f"{layer_name}.png"
            cv2.imwrite(str(path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            size_kb = path.stat().st_size / 1024
            coverage = np.sum(mask > 0) / (h * w) * 100
            print(f"   {layer_name}: {coverage:.2f}% ({size_kb:.1f}KB)")
        
        # 创建分层提取预览 - 每层单独显示原图对应区域
        for layer_name, mask in masks.items():
            if layer_name != "L7_background":
                # 提取该层对应的原图区域
                extracted = cv2.bitwise_and(img, img, mask=mask)
                cv2.imwrite(str(out_dir / f"{layer_name}_extracted.png"), extracted,
                           [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # 创建合成预览 - 原图
        cv2.imwrite(str(out_dir / "overlay.png"), img, 
                   [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # 保存索引页面
        self._create_index_html(out_dir, masks, text_regions)
    
    def _create_index_html(self, out_dir: Path, masks: dict, text_regions: list):
        """创建索引HTML"""
        html = '''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>精确分割结果</title>
<style>
body{font-family:Arial;margin:20px;background:#1a1a1a;color:#fff}
h1{text-align:center;color:#4CAF50}
h2{color:#888;margin-top:30px}
.row{display:flex;gap:20px;margin:20px 0}
.col{flex:1;text-align:center}
.col img{width:100%;border:1px solid #333;border-radius:4px}
.col p{margin:5px 0;font-size:12px;color:#888}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:20px 0}
.item{background:#222;padding:10px;border-radius:8px;text-align:center}
.item img{width:100%;border-radius:4px}
.item p{margin:5px 0;font-size:11px;color:#666}
</style></head><body>
<h1>🎯 精确分割结果</h1>

<div class="row">
<div class="col"><img src="original.png"><p>原图</p></div>
</div>

<h2>纯黑白Mask（用于矢量化）</h2>
<div class="grid">
'''
        for name in masks.keys():
            html += f'<div class="item"><img src="{name}.png"><p>{name}</p></div>\n'
        
        html += '''</div>

<h2>提取的原图区域</h2>
<div class="grid">
'''
        for name in masks.keys():
            if name != "L7_background":
                html += f'<div class="item"><img src="{name}_extracted.png"><p>{name} 提取</p></div>\n'
        
        html += '</div></body></html>'
        
        with open(out_dir / "index.html", 'w') as f:
            f.write(html)


def main():
    segmenter = PreciseSegmenter()
    segmenter.segment("/Volumes/Seagate/SAM3/01_input/科研绘图1.png")


if __name__ == "__main__":
    main()
