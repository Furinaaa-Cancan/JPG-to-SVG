#!/usr/bin/env python3
"""
像素级精确分割 v3
直接基于颜色值精确匹配，逐模块验证
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


class PixelAccurateSegmenter:
    """像素级精确分割器"""
    
    def __init__(self):
        self.output_dir = Path("/Volumes/Seagate/SAM3/02_output/pixel_accurate")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def segment(self, image_path: str) -> dict:
        """执行像素级精确分割"""
        print("\n" + "="*70)
        print("🎯 像素级精确分割 v3")
        print("="*70)
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        print(f"   图像尺寸: {w}×{h}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.output_dir / f"v3_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原图
        cv2.imwrite(str(out_dir / "00_original.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # 逐模块分割并验证
        masks = {}
        
        # ========== 模块1: 纯红色像素 ==========
        print("\n" + "-"*50)
        print("🔴 模块1: 红色像素检测")
        red_mask = self._detect_red_pixels(img)
        masks["red"] = red_mask
        self._verify_and_save(img, red_mask, "01_red", out_dir)
        
        # ========== 模块2: 纯蓝色像素 ==========
        print("\n" + "-"*50)
        print("🔵 模块2: 蓝色像素检测")
        blue_mask = self._detect_blue_pixels(img)
        masks["blue"] = blue_mask
        self._verify_and_save(img, blue_mask, "02_blue", out_dir)
        
        # ========== 模块3: 黑色像素（线条、文字）==========
        print("\n" + "-"*50)
        print("⬛ 模块3: 黑色像素检测")
        black_mask = self._detect_black_pixels(img)
        masks["black"] = black_mask
        self._verify_and_save(img, black_mask, "03_black", out_dir)
        
        # ========== 模块4: 灰色像素（悬臂梁）==========
        print("\n" + "-"*50)
        print("🔘 模块4: 灰色像素检测（悬臂梁）")
        gray_mask = self._detect_gray_pixels(img)
        masks["gray"] = gray_mask
        self._verify_and_save(img, gray_mask, "04_gray", out_dir)
        
        # ========== 模块5: 白色/背景 ==========
        print("\n" + "-"*50)
        print("⬜ 模块5: 白色/背景检测")
        white_mask = self._detect_white_pixels(img)
        masks["white"] = white_mask
        self._verify_and_save(img, white_mask, "05_white", out_dir)
        
        # ========== 验证覆盖率 ==========
        print("\n" + "-"*50)
        print("📊 覆盖率验证")
        total_covered = np.zeros((h, w), dtype=np.uint8)
        for name, mask in masks.items():
            coverage = np.sum(mask > 0) / (h * w) * 100
            print(f"   {name}: {coverage:.2f}%")
            total_covered = cv2.bitwise_or(total_covered, mask)
        
        uncovered = cv2.bitwise_not(total_covered)
        uncovered_pct = np.sum(uncovered > 0) / (h * w) * 100
        print(f"   未覆盖: {uncovered_pct:.2f}%")
        
        # 保存未覆盖区域
        cv2.imwrite(str(out_dir / "06_uncovered.png"), uncovered)
        uncovered_vis = cv2.bitwise_and(img, img, mask=uncovered)
        cv2.imwrite(str(out_dir / "06_uncovered_vis.png"), uncovered_vis)
        
        # ========== 生成最终分层结果 ==========
        print("\n" + "-"*50)
        print("🔧 生成无重叠分层")
        final_masks = self._create_non_overlapping_layers(h, w, masks)
        
        for name, mask in final_masks.items():
            cv2.imwrite(str(out_dir / f"final_{name}.png"), mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            coverage = np.sum(mask > 0) / (h * w) * 100
            print(f"   final_{name}: {coverage:.2f}%")
        
        # 创建索引页面
        self._create_index(out_dir, masks, final_masks)
        
        print("\n" + "="*70)
        print("✅ 分割完成")
        print(f"   输出目录: {out_dir}")
        print("="*70)
        
        return {"output_dir": str(out_dir)}
    
    def _detect_red_pixels(self, img: np.ndarray) -> np.ndarray:
        """检测红色像素 - 基于BGR值"""
        b, g, r = cv2.split(img)
        
        # 红色：R通道高，且R明显大于G和B
        red_high = r > 150
        r_dominant = (r.astype(np.int16) - g.astype(np.int16) > 50) & \
                     (r.astype(np.int16) - b.astype(np.int16) > 50)
        
        red_mask = (red_high & r_dominant).astype(np.uint8) * 255
        
        # 形态学清理
        kernel = np.ones((2, 2), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        return red_mask
    
    def _detect_blue_pixels(self, img: np.ndarray) -> np.ndarray:
        """检测蓝色像素 - 基于BGR值"""
        b, g, r = cv2.split(img)
        
        # 蓝色：B通道高，且B明显大于R
        blue_high = b > 120
        b_dominant = (b.astype(np.int16) - r.astype(np.int16) > 30)
        
        blue_mask = (blue_high & b_dominant).astype(np.uint8) * 255
        
        kernel = np.ones((2, 2), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        
        return blue_mask
    
    def _detect_black_pixels(self, img: np.ndarray) -> np.ndarray:
        """检测黑色像素"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 黑色：亮度很低
        black_mask = (gray < 60).astype(np.uint8) * 255
        
        return black_mask
    
    def _detect_gray_pixels(self, img: np.ndarray) -> np.ndarray:
        """检测灰色像素（悬臂梁）"""
        b, g, r = cv2.split(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 灰色：R≈G≈B，且亮度在中间范围
        diff_rg = np.abs(r.astype(np.int16) - g.astype(np.int16))
        diff_rb = np.abs(r.astype(np.int16) - b.astype(np.int16))
        diff_gb = np.abs(g.astype(np.int16) - b.astype(np.int16))
        
        is_neutral = (diff_rg < 30) & (diff_rb < 30) & (diff_gb < 30)
        mid_brightness = (gray > 80) & (gray < 220)
        
        gray_mask = (is_neutral & mid_brightness).astype(np.uint8) * 255
        
        # 只保留左侧区域（悬臂梁位置）
        h, w = img.shape[:2]
        gray_mask[:, int(w*0.5):] = 0
        
        kernel = np.ones((3, 3), np.uint8)
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
        
        return gray_mask
    
    def _detect_white_pixels(self, img: np.ndarray) -> np.ndarray:
        """检测白色/浅色背景"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 白色：亮度很高
        white_mask = (gray > 240).astype(np.uint8) * 255
        
        return white_mask
    
    def _verify_and_save(self, img: np.ndarray, mask: np.ndarray, 
                         name: str, out_dir: Path):
        """验证并保存mask"""
        h, w = img.shape[:2]
        coverage = np.sum(mask > 0) / (h * w) * 100
        print(f"   覆盖率: {coverage:.2f}%")
        
        # 保存mask
        cv2.imwrite(str(out_dir / f"{name}_mask.png"), mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # 保存提取的区域
        extracted = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(str(out_dir / f"{name}_extracted.png"), extracted, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        print(f"   ✓ 已保存: {name}_mask.png, {name}_extracted.png")
    
    def _create_non_overlapping_layers(self, h: int, w: int, masks: dict) -> dict:
        """创建无重叠分层"""
        # 优先级：红 > 蓝 > 黑 > 灰 > 白
        priority = ["red", "blue", "black", "gray", "white"]
        
        final = {}
        used = np.zeros((h, w), dtype=np.uint8)
        
        for name in priority:
            if name in masks:
                available = cv2.bitwise_and(masks[name], cv2.bitwise_not(used))
                final[name] = available
                used = cv2.bitwise_or(used, available)
        
        # 剩余区域
        final["other"] = cv2.bitwise_not(used)
        
        return final
    
    def _create_index(self, out_dir: Path, raw_masks: dict, final_masks: dict):
        """创建索引HTML"""
        html = '''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>像素级分割结果</title>
<style>
body{font-family:Arial;margin:20px;background:#1a1a1a;color:#fff}
h1,h2{text-align:center}
h2{color:#888;margin-top:40px}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:20px 0}
.item{background:#222;padding:10px;border-radius:8px;text-align:center}
.item img{width:100%;border-radius:4px}
.item p{margin:5px 0;font-size:11px;color:#888}
.full{grid-column:span 5}
.half{grid-column:span 2}
</style></head><body>
<h1>🎯 像素级精确分割结果</h1>

<h2>原图</h2>
<div class="grid">
<div class="item full"><img src="00_original.png"></div>
</div>

<h2>各颜色通道检测</h2>
<div class="grid">
<div class="item"><img src="01_red_mask.png"><p>红色 mask</p></div>
<div class="item"><img src="01_red_extracted.png"><p>红色 提取</p></div>
<div class="item"><img src="02_blue_mask.png"><p>蓝色 mask</p></div>
<div class="item"><img src="02_blue_extracted.png"><p>蓝色 提取</p></div>
<div class="item"><img src="06_uncovered_vis.png"><p>未覆盖区域</p></div>
</div>

<div class="grid">
<div class="item"><img src="03_black_mask.png"><p>黑色 mask</p></div>
<div class="item"><img src="03_black_extracted.png"><p>黑色 提取</p></div>
<div class="item"><img src="04_gray_mask.png"><p>灰色 mask</p></div>
<div class="item"><img src="04_gray_extracted.png"><p>灰色 提取</p></div>
<div class="item"><img src="05_white_mask.png"><p>白色 mask</p></div>
</div>

<h2>最终无重叠分层</h2>
<div class="grid">
'''
        for name in final_masks.keys():
            html += f'<div class="item"><img src="final_{name}.png"><p>{name}</p></div>\n'
        
        html += '</div></body></html>'
        
        with open(out_dir / "index.html", 'w') as f:
            f.write(html)


def main():
    segmenter = PixelAccurateSegmenter()
    segmenter.segment("/Volumes/Seagate/SAM3/01_input/科研绘图1.png")


if __name__ == "__main__":
    main()
