#!/usr/bin/env python3
"""
SAM3 + 颜色量化 混合矢量化 - 多核加速版 (自动化)

使用多进程并行处理：
1. BLIP2自动分析图片生成提示词
2. SAM3分割并行（多prompt同时处理）
3. 前景细节提取并行（多区域同时处理）
"""

import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import svgwrite
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, "/Volumes/Seagate/SAM3/models/sam3")

# BLIP2用于自动图片分析
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False


# 全局变量用于多进程共享
_img_global = None
_h_global = None
_w_global = None


def process_region_batch(args):
    """处理单个区域的颜色量化（在子进程中运行）- 优化版"""
    mask, color, area, img, h, w, level_config = args
    
    paths = []
    
    # 从level_config获取参数
    cfg_min_area = level_config.get('min_area', 80)
    cfg_epsilon = level_config.get('epsilon', 0.002)
    n_large = level_config.get('n_colors_large', 8)
    n_medium = level_config.get('n_colors_medium', 5)
    n_small = level_config.get('n_colors_small', 3)
    
    # 自适应参数：大区域更多颜色，小区域少颜色
    if area > 50000:
        n_colors = n_large
        min_area = cfg_min_area
        epsilon_factor = cfg_epsilon * 0.75
    elif area > 10000:
        n_colors = n_medium
        min_area = cfg_min_area
        epsilon_factor = cfg_epsilon
    elif area > 3000:
        n_colors = n_small
        min_area = cfg_min_area
        epsilon_factor = cfg_epsilon * 1.5
    else:
        # 小区域直接用平均色
        n_colors = 1
        min_area = cfg_min_area
        epsilon_factor = cfg_epsilon * 2
    
    if n_colors > 1 and area > 3000:
        inner_paths = quantize_region_standalone(img, mask, h, w, n_colors=n_colors, min_area=cfg_min_area, epsilon=cfg_epsilon)
        paths.extend(inner_paths)
    
    # 用平均色填充整个mask轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(simplified) >= 3:
                    points = simplified.squeeze()
                    if points.ndim == 1:
                        points = points.reshape(-1, 2)
                    
                    paths.append({
                        'points': points,
                        'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                        'area': cv2.contourArea(contour),
                        'layer': 'foreground'
                    })
    
    return paths


def quantize_region_standalone(img, mask, h, w, n_colors=5, min_area=50, epsilon=0.002):
    """独立的区域颜色量化函数"""
    
    paths = []
    
    masked_img = img.copy()
    masked_img[mask < 127] = 0
    
    lab = cv2.cvtColor(masked_img, cv2.COLOR_RGB2LAB)
    
    mask_flat = mask.flatten() > 127
    pixels = lab.reshape(-1, 3)[mask_flat].astype(np.float32)
    
    if len(pixels) < 100:
        return paths
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
    n_colors = min(n_colors, len(pixels) // 100)
    if n_colors < 2:
        return paths
    
    # 使用KMEANS_PP_CENTERS更好的初始化，只尝试1次
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    
    label_img = np.zeros(h * w, dtype=np.int32)
    label_img[mask_flat] = labels.flatten()
    label_img = label_img.reshape(h, w)
    
    centers_lab = centers.astype(np.uint8).reshape(1, -1, 3)
    centers_rgb = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2RGB).reshape(-1, 3)
    
    for cid, color_rgb in enumerate(centers_rgb):
        color_mask = ((label_img == cid) & (mask > 127)).astype(np.uint8) * 255
        
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                eps = epsilon * cv2.arcLength(contour, True)
                simplified = cv2.approxPolyDP(contour, eps, True)
                
                if len(simplified) >= 3:
                    points = simplified.squeeze()
                    if points.ndim == 1:
                        points = points.reshape(-1, 2)
                    
                    temp_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(temp_mask, [contour], -1, 255, -1)
                    pixels_rgb = img[temp_mask > 127]
                    if len(pixels_rgb) > 0:
                        actual_color = np.mean(pixels_rgb, axis=0).astype(int)
                    else:
                        actual_color = color_rgb
                    
                    paths.append({
                        'points': points,
                        'color': f"#{actual_color[0]:02x}{actual_color[1]:02x}{actual_color[2]:02x}",
                        'area': area,
                        'layer': 'detail'
                    })
    
    return paths


class SAM3ColorVectorizerFast:
    """SAM3 + 颜色量化混合矢量化 - 多核加速版"""
    
    def __init__(self, n_workers=None, use_blip=True, use_mps=True, level_config=None):
        self.n_workers = n_workers or min(cpu_count(), 12)
        # 默认级别7的配置
        self.level_config = level_config or {
            'n_colors_large': 8, 'n_colors_medium': 5, 'n_colors_small': 3,
            'min_area': 80, 'epsilon': 0.002
        }
        print(f"\n🚀 Initializing SAM3 + Color Vectorizer (Fast)")
        print(f"   CPU cores: {self.n_workers}")
        
        import torch
        
        # 设备选择：优先MPS，其次CUDA，最后CPU
        if use_mps and torch.backends.mps.is_available():
            self.device = "mps"
            self.sam3_device = "cpu"  # SAM3文本编码器MPS有bug，必须CPU
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.sam3_device = "cuda"
        else:
            self.device = "cpu"
            self.sam3_device = "cpu"
        
        torch.set_num_threads(self.n_workers)
        print(f"   Device: {self.device} (SAM3: {self.sam3_device})")
        print(f"   PyTorch threads: {torch.get_num_threads()}")
        
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        self.sam3_model = build_sam3_image_model(device=self.sam3_device)
        self.sam3_processor = Sam3Processor(self.sam3_model, device=self.sam3_device, confidence_threshold=0.1)
        print(f"✅ SAM3 loaded!")
        
        # BLIP2可选（启用可提高质量但会增加约1分钟处理时间）
        self.blip_processor = None
        self.blip_model = None
        self.use_blip = use_blip
        
        if use_blip and BLIP_AVAILABLE:
            try:
                print("   Loading BLIP2 for auto-captioning...")
                self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                # BLIP2用MPS/CUDA加速
                blip_dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32
                self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b", 
                    torch_dtype=blip_dtype
                ).to(self.device)
                print(f"   ✅ BLIP2 loaded on {self.device}!")
            except Exception as e:
                print(f"   ⚠️ BLIP2 not available: {e}")
                self.blip_processor = None
        elif not use_blip:
            print("   ⏭️ BLIP2 skipped (fast mode)")
    
    def vectorize(self, image_path: str, output_dir: str = "02_输出结果/sam3_color_svg", max_size: int = 2048):
        """混合矢量化 - 多尺度处理
        
        Args:
            max_size: SAM3分割用的最大尺寸（默认2048），颜色细节始终用原图
        """
        
        print("\n" + "="*70)
        print(f"💎 SAM3 + COLOR HYBRID VECTORIZATION (Multi-Scale)")
        print("="*70)
        
        start_time = time.time()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载原图（用于颜色细节提取）
        orig_pil = Image.open(image_path).convert("RGB")
        orig_w, orig_h = orig_pil.size
        orig_img = np.array(orig_pil)
        
        # 多尺度处理：SAM3用小图分割，颜色用原图
        if max_size > 0 and max(orig_w, orig_h) > max_size:
            scale = max_size / max(orig_w, orig_h)
            sam_w = int(orig_w * scale)
            sam_h = int(orig_h * scale)
            sam_pil = orig_pil.resize((sam_w, sam_h), Image.LANCZOS)
            sam_img = np.array(sam_pil)
            print(f"\n📷 Original: {orig_w}x{orig_h} (for color detail)")
            print(f"   SAM3 uses: {sam_w}x{sam_h} (for segmentation)")
        else:
            scale = 1.0
            sam_pil = orig_pil
            sam_img = orig_img
            sam_w, sam_h = orig_w, orig_h
            print(f"\n📷 Image: {orig_w}x{orig_h}")
        
        all_paths = []
        
        # Step 1: SAM3语义分割（用小图）
        print("\n🎯 Step 1: SAM3 Semantic Segmentation...")
        t1 = time.time()
        sam3_regions = self.sam3_segment_parallel(sam_pil, sam_img, sam_h, sam_w)
        print(f"   SAM3 regions: {len(sam3_regions)} ({time.time()-t1:.1f}s)")
        
        # 将SAM3 mask放大到原图尺寸
        if scale < 1.0:
            print("   📐 Upscaling masks to original size...")
            for region in sam3_regions:
                mask_small = region['mask']
                mask_big = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                region['mask'] = mask_big
                # 重新计算原图区域的平均颜色
                pixels = orig_img[mask_big > 127]
                if len(pixels) > 0:
                    region['color'] = np.mean(pixels, axis=0).astype(int)
                region['area'] = np.sum(mask_big > 0)
        
        # 使用原图尺寸进行后续处理
        h, w = orig_h, orig_w
        img = orig_img
        
        # Step 2: 找出未被SAM3覆盖的区域
        print("\n🔍 Step 2: Fill uncovered regions...")
        t2 = time.time()
        uncovered_paths = self.fill_uncovered_regions(img, sam3_regions, h, w)
        all_paths.extend(uncovered_paths)
        print(f"   Uncovered paths: {len(uncovered_paths)} ({time.time()-t2:.1f}s)")
        
        # Step 3: 背景基底层
        print("\n🌅 Step 3: Background base...")
        t3 = time.time()
        bg_paths = self.background_quantize(img, h, w)
        all_paths.extend(bg_paths)
        print(f"   Background paths: {len(bg_paths)} ({time.time()-t3:.1f}s)")
        
        # Step 4: 前景区域并行处理（用原图提取细节）
        print("\n🎨 Step 4: Foreground detail extraction (Original Resolution)...")
        t4 = time.time()
        fg_paths = self.foreground_detail_parallel(img, sam3_regions, h, w)
        all_paths.extend(fg_paths)
        print(f"   Foreground paths: {len(fg_paths)} ({time.time()-t4:.1f}s)")
        
        # Step 4: 生成SVG
        print("\n✨ Step 4: Generate SVG...")
        svg_path = output_path / "sam3_color_vector.svg"
        stats = self.create_svg(all_paths, w, h, str(svg_path))
        
        process_time = time.time() - start_time
        
        # 保存对比图片
        self.save_comparison_image(image_path, str(svg_path), output_path, stats, process_time)
        
        print(f"\n" + "="*70)
        print(f"✅ COMPLETE!")
        print(f"   Paths: {stats['paths']}")
        print(f"   Size: {stats['size_kb']:.1f} KB")
        print(f"   Time: {process_time:.1f}s ⚡")
        print("="*70)
        
        return stats
    
    def auto_generate_prompts(self, img_pil) -> list:
        """使用BLIP2自动分析图片生成提示词"""
        
        # 通用基础提示词（适用于任何图片）
        base_prompts = [
            # 通用物体
            "person", "people", "man", "woman", "face", "head", "hair",
            "hand", "hands", "arm", "leg", "body",
            # 通用服装
            "clothing", "dress", "shirt", "pants", "jacket", "coat",
            "hat", "shoes", "boots",
            # 通用场景
            "background", "floor", "ground", "wall", "sky",
            # 通用属性
            "light", "shadow", "dark area", "bright area",
            "white object", "black object", "colorful object",
        ]
        
        # 如果BLIP2可用，自动分析图片内容
        if self.blip_processor is not None and self.blip_model is not None:
            try:
                import torch
                print("   🤖 Auto-analyzing image with BLIP2...")
                
                # 生成图片描述
                inputs = self.blip_processor(img_pil, return_tensors="pt")
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = self.blip_model.generate(**inputs, max_new_tokens=50)
                caption = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(f"   📝 Caption: {caption}")
                
                # 从描述中提取关键词
                caption_words = caption.lower().replace(',', ' ').replace('.', ' ').split()
                
                # 颜色词
                colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", 
                         "white", "black", "gray", "grey", "brown", "gold", "silver", "blonde"]
                
                # 提取颜色+物体组合
                for i, word in enumerate(caption_words):
                    if word in colors and i + 1 < len(caption_words):
                        base_prompts.append(f"{word} {caption_words[i+1]}")
                    if len(word) > 3 and word not in ['the', 'and', 'with', 'that', 'this']:
                        base_prompts.append(word)
                
                # 注：问答式提取已移除（太慢，每个问题约1分钟）
                # 如需更精确的提示词，可以手动指定
                        
            except Exception as e:
                print(f"   ⚠️ BLIP2 analysis failed: {e}")
        
        # 去重
        unique_prompts = list(dict.fromkeys(base_prompts))
        print(f"   📋 Generated {len(unique_prompts)} prompts")
        
        return unique_prompts
    
    def auto_detect_regions(self, img, h, w) -> list:
        """自动检测图片中的显著区域（基于颜色和边缘）"""
        
        regions = []
        
        # 1. 基于颜色聚类找显著区域
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        pixels = lab.reshape(-1, 3).astype(np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 8, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        
        label_img = labels.reshape(h, w)
        
        for cid in range(8):
            color_mask = (label_img == cid).astype(np.uint8) * 255
            area = np.sum(color_mask > 0)
            
            if area > 1000 and area < h * w * 0.8:
                # 找到该颜色区域的边界框
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 500:
                        x, y, bw, bh = cv2.boundingRect(contour)
                        # 转换为归一化坐标 [cx, cy, w, h]
                        cx = (x + bw/2) / w
                        cy = (y + bh/2) / h
                        nw = bw / w
                        nh = bh / h
                        regions.append({
                            'box': [cx, cy, nw, nh],
                            'area': cv2.contourArea(contour)
                        })
        
        # 按面积排序，取top区域
        regions.sort(key=lambda x: -x['area'])
        return regions[:10]
    
    def sam3_segment_parallel(self, img_pil, img, h, w) -> list:
        """SAM3语义分割 - 自动化版本"""
        
        regions = []
        state = self.sam3_processor.set_image(img_pil)
        
        # 1. 自动生成提示词
        key_prompts = self.auto_generate_prompts(img_pil)
        
        # 2. 自动检测显著区域并用边界框分割
        print("   🔍 Auto-detecting salient regions...")
        auto_regions = self.auto_detect_regions(img, h, w)
        
        for i, region in enumerate(auto_regions):
            try:
                self.sam3_processor.reset_all_prompts(state)
                result = self.sam3_processor.add_geometric_prompt(region['box'], True, state)
                if result and 'masks' in result and result['masks'] is not None:
                    masks = result['masks'].cpu().numpy()
                    for mask in masks:
                        if len(mask.shape) > 2:
                            mask = mask.squeeze()
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask.astype(np.float32), (w, h))
                        binary = (mask > 0.5).astype(np.uint8) * 255
                        # 膨胀填补轮廓空隙
                        kernel = np.ones((5, 5), np.uint8)
                        binary = cv2.dilate(binary, kernel, iterations=3)
                        binary = cv2.GaussianBlur(binary, (3, 3), 0)
                        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
                        area = np.sum(binary > 0)
                        if area > 100:
                            orig_mask = (mask > 0.5).astype(np.uint8) * 255
                            pixels = img[orig_mask > 127]
                            if len(pixels) > 0:
                                color = np.mean(pixels, axis=0).astype(int)
                                regions.append({
                                    'mask': binary,
                                    'color': color,
                                    'area': area,
                                    'prompt': f'auto_region_{i}'
                                })
            except:
                pass
        
        print(f"   📦 Auto regions: {len(regions)}")
        
        # 3. 文本提示分割
        for prompt in key_prompts:
            try:
                self.sam3_processor.reset_all_prompts(state)
                result = self.sam3_processor.set_text_prompt(prompt, state)
                
                if result and 'masks' in result and result['masks'] is not None:
                    masks = result['masks'].cpu().numpy()
                    
                    for mask in masks:
                        if len(mask.shape) > 2:
                            mask = mask.squeeze()
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask.astype(np.float32), (w, h))
                        
                        binary = (mask > 0.5).astype(np.uint8) * 255
                        
                        # 膨胀填补轮廓空隙
                        kernel = np.ones((5, 5), np.uint8)
                        binary = cv2.dilate(binary, kernel, iterations=3)
                        binary = cv2.GaussianBlur(binary, (3, 3), 0)
                        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
                        
                        area = np.sum(binary > 0)
                        
                        if area > 500 and area < h * w * 0.9:
                            orig_binary = (mask > 0.5).astype(np.uint8) * 255
                            pixels = img[orig_binary > 127]
                            if len(pixels) > 0:
                                color = np.mean(pixels, axis=0).astype(int)
                                
                                regions.append({
                                    'mask': binary,
                                    'color': color,
                                    'area': area,
                                    'prompt': prompt
                                })
            except:
                pass
        
        regions.sort(key=lambda x: -x['area'])
        return regions
    
    def fill_uncovered_regions(self, img, sam3_regions, h, w) -> list:
        """找出SAM3未覆盖的区域，用原图颜色量化填充"""
        
        paths = []
        
        # 合并所有SAM3 mask
        covered_mask = np.zeros((h, w), dtype=np.uint8)
        for region in sam3_regions:
            covered_mask = np.maximum(covered_mask, region['mask'])
        
        # 未覆盖区域
        uncovered_mask = (covered_mask < 127).astype(np.uint8) * 255
        uncovered_area = np.sum(uncovered_mask > 0)
        
        print(f"   Uncovered area: {uncovered_area / (h*w) * 100:.1f}%")
        
        if uncovered_area < 1000:
            return paths
        
        # 对未覆盖区域做颜色量化
        masked_img = img.copy()
        masked_img[uncovered_mask < 127] = 0
        
        lab = cv2.cvtColor(masked_img, cv2.COLOR_RGB2LAB)
        mask_flat = uncovered_mask.flatten() > 127
        pixels = lab.reshape(-1, 3)[mask_flat].astype(np.float32)
        
        if len(pixels) < 100:
            return paths
        
        # 更多颜色来捕捉细节
        n_colors = min(15, len(pixels) // 100)
        if n_colors < 2:
            return paths
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        
        label_img = np.zeros(h * w, dtype=np.int32)
        label_img[mask_flat] = labels.flatten()
        label_img = label_img.reshape(h, w)
        
        centers_lab = centers.astype(np.uint8).reshape(1, -1, 3)
        centers_rgb = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2RGB).reshape(-1, 3)
        
        for cid, color_rgb in enumerate(centers_rgb):
            color_mask = ((label_img == cid) & (uncovered_mask > 127)).astype(np.uint8) * 255
            
            kernel = np.ones((3, 3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    epsilon = 0.001 * cv2.arcLength(contour, True)
                    simplified = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(simplified) >= 3:
                        points = simplified.squeeze()
                        if points.ndim == 1:
                            points = points.reshape(-1, 2)
                        
                        # 获取实际颜色
                        temp_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.drawContours(temp_mask, [contour], -1, 255, -1)
                        pixels_rgb = img[temp_mask > 127]
                        if len(pixels_rgb) > 0:
                            actual_color = np.mean(pixels_rgb, axis=0).astype(int)
                        else:
                            actual_color = color_rgb
                        
                        paths.append({
                            'points': points,
                            'color': f"#{actual_color[0]:02x}{actual_color[1]:02x}{actual_color[2]:02x}",
                            'area': area,
                            'layer': 'uncovered'
                        })
        
        return paths
    
    def background_quantize(self, img, h, w) -> list:
        """背景处理 - 简化版，只用精细条带填充"""
        
        paths = []
        
        # 1. 全画布基底层（用四边像素平均色）
        edge_pixels = np.concatenate([
            img[0, :], img[-1, :], img[:, 0], img[:, -1]
        ])
        bg_color = np.mean(edge_pixels, axis=0).astype(int)
        
        paths.append({
            'points': np.array([[0, 0], [w, 0], [w, h], [0, h]]),
            'color': f"#{bg_color[0]:02x}{bg_color[1]:02x}{bg_color[2]:02x}",
            'area': h * w,
            'layer': 'base'
        })
        
        # 2. 底部精细条带填充（从90%位置开始，更细的条带）
        bottom_start = int(h * 0.90)  # 从90%位置开始
        strip_height = 10  # 更细的条带
        
        for y in range(bottom_start, h, strip_height):
            y_end = min(y + strip_height, h)
            strip_pixels = img[y:y_end, :].reshape(-1, 3)
            strip_color = np.mean(strip_pixels, axis=0).astype(int)
            
            paths.append({
                'points': np.array([[0, y], [w, y], [w, y_end], [0, y_end]]),
                'color': f"#{strip_color[0]:02x}{strip_color[1]:02x}{strip_color[2]:02x}",
                'area': w * (y_end - y),
                'layer': 'background'
            })
        
        return paths
    
    def foreground_detail_parallel(self, img, sam3_regions, h, w) -> list:
        """前景区域并行处理 - 使用进程池"""
        
        # 准备任务参数（包含level_config）
        tasks = []
        for region in sam3_regions:
            tasks.append((
                region['mask'],
                region['color'],
                region['area'],
                img,
                h,
                w,
                self.level_config
            ))
        
        all_paths = []
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(process_region_batch, tasks))
        
        for paths in results:
            all_paths.extend(paths)
        
        return all_paths
    
    def create_svg(self, paths: list, width: int, height: int, output_path: str) -> dict:
        """创建SVG"""
        
        dwg = svgwrite.Drawing(output_path, size=(width, height))
        dwg.viewbox(0, 0, width, height)
        
        layer_order = {'base': -1, 'background': 0, 'foreground': 1, 'detail': 2}
        paths.sort(key=lambda x: (layer_order.get(x.get('layer', 'detail'), 2), -x['area']))
        
        path_count = 0
        
        for path_data in paths:
            points = path_data['points']
            color = path_data['color']
            
            if len(points) < 3:
                continue
            
            path_d = self.points_to_path(points)
            
            if path_d:
                dwg.add(dwg.path(d=path_d, fill=color, stroke=color, stroke_width=1))
                path_count += 1
        
        dwg.save()
        
        # 生成SVGZ压缩版本
        import gzip
        svgz_path = output_path.replace('.svg', '.svgz')
        with open(output_path, 'rb') as f_in:
            with gzip.open(svgz_path, 'wb', compresslevel=9) as f_out:
                f_out.write(f_in.read())
        
        svg_size = Path(output_path).stat().st_size / 1024
        svgz_size = Path(svgz_path).stat().st_size / 1024
        
        return {
            'paths': path_count,
            'size_kb': svg_size,
            'svgz_kb': svgz_size
        }
    
    def points_to_path(self, points: np.ndarray, use_curves=True) -> str:
        """点转路径 - 优化版（整数坐标）"""
        
        n = len(points)
        if n < 3:
            return ""
        
        # 转为整数坐标
        pts = points.astype(int)
        
        if use_curves and n >= 4:
            # 贝塞尔曲线（平滑但数据量大）
            path_d = f"M{pts[0][0]},{pts[0][1]}"
            
            for i in range(n):
                p0 = pts[(i - 1) % n]
                p1 = pts[i]
                p2 = pts[(i + 1) % n]
                p3 = pts[(i + 2) % n]
                
                c1x = int(p1[0] + (p2[0] - p0[0]) / 6)
                c1y = int(p1[1] + (p2[1] - p0[1]) / 6)
                c2x = int(p2[0] - (p3[0] - p1[0]) / 6)
                c2y = int(p2[1] - (p3[1] - p1[1]) / 6)
                
                path_d += f"C{c1x},{c1y} {c2x},{c2y} {p2[0]},{p2[1]}"
        else:
            # 直线路径（更紧凑）
            path_d = f"M{pts[0][0]},{pts[0][1]}"
            for i in range(1, n):
                path_d += f"L{pts[i][0]},{pts[i][1]}"
        
        path_d += "Z"
        return path_d
    
    def save_comparison_image(self, original: str, svg: str, output_path: Path, stats: dict, process_time: float):
        """保存对比图片（PNG格式）"""
        import matplotlib.pyplot as plt
        from cairosvg import svg2png
        from io import BytesIO
        
        # 读取原图
        orig_img = Image.open(original).convert('RGB')
        
        # SVG转PNG
        try:
            png_data = svg2png(url=svg, output_width=1200)
            svg_img = Image.open(BytesIO(png_data)).convert('RGB')
        except Exception as e:
            print(f"   ⚠️ SVG转换失败: {e}")
            svg_img = orig_img.copy()
        
        # 创建对比图
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        fig.patch.set_facecolor('white')
        
        # 原图
        axes[0].imshow(orig_img)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        orig_size = Path(original).stat().st_size / 1024
        axes[0].text(0.5, -0.05, f'Size: {orig_size:.0f} KB', 
                    transform=axes[0].transAxes, ha='center', fontsize=11)
        
        # SVG
        axes[1].imshow(svg_img)
        axes[1].set_title('SVG Vector', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # 格式化大小
        size_kb = stats['size_kb']
        if size_kb >= 1024:
            size_str = f"{size_kb/1024:.1f} MB"
        else:
            size_str = f"{size_kb:.0f} KB"
        
        # 格式化时间
        if process_time >= 60:
            time_str = f"{process_time/60:.1f} min"
        else:
            time_str = f"{process_time:.1f}s"
        
        axes[1].text(0.5, -0.05, f'Paths: {stats["paths"]:,} | Size: {size_str} | Time: {time_str}', 
                    transform=axes[1].transAxes, ha='center', fontsize=11)
        
        plt.tight_layout()
        
        # 保存PNG和PDF
        png_path = output_path / "comparison.png"
        pdf_path = output_path / "comparison.pdf"
        plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Saved: {png_path}")
        print(f"   ✅ Saved: {pdf_path}")


# 细节级别配置
DETAIL_LEVELS = {
    1:  {'n_colors_large': 1,  'n_colors_medium': 1,  'n_colors_small': 1,  'min_area': 8000, 'epsilon': 0.025},
    2:  {'n_colors_large': 1,  'n_colors_medium': 1,  'n_colors_small': 1,  'min_area': 3000, 'epsilon': 0.018},
    3:  {'n_colors_large': 2,  'n_colors_medium': 1,  'n_colors_small': 1,  'min_area': 1500, 'epsilon': 0.012},
    4:  {'n_colors_large': 3,  'n_colors_medium': 2,  'n_colors_small': 1,  'min_area': 800,  'epsilon': 0.008},
    5:  {'n_colors_large': 4,  'n_colors_medium': 3,  'n_colors_small': 2,  'min_area': 400,  'epsilon': 0.005},
    6:  {'n_colors_large': 6,  'n_colors_medium': 4,  'n_colors_small': 3,  'min_area': 200,  'epsilon': 0.003},
    7:  {'n_colors_large': 8,  'n_colors_medium': 5,  'n_colors_small': 3,  'min_area': 80,   'epsilon': 0.002},   # 默认级别
    8:  {'n_colors_large': 10, 'n_colors_medium': 6,  'n_colors_small': 4,  'min_area': 50,   'epsilon': 0.0015},
    9:  {'n_colors_large': 12, 'n_colors_medium': 8,  'n_colors_small': 5,  'min_area': 30,   'epsilon': 0.001},
    10: {'n_colors_large': 15, 'n_colors_medium': 10, 'n_colors_small': 6,  'min_area': 20,   'epsilon': 0.0008},
    11: {'n_colors_large': 18, 'n_colors_medium': 12, 'n_colors_small': 8,  'min_area': 12,   'epsilon': 0.0006},
    12: {'n_colors_large': 22, 'n_colors_medium': 15, 'n_colors_small': 10, 'min_area': 8,    'epsilon': 0.0004},
    13: {'n_colors_large': 28, 'n_colors_medium': 18, 'n_colors_small': 12, 'min_area': 5,    'epsilon': 0.0003},
    14: {'n_colors_large': 35, 'n_colors_medium': 22, 'n_colors_small': 15, 'min_area': 3,    'epsilon': 0.0002},
}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SAM3 + Color Vectorization (Auto)')
    parser.add_argument('image', nargs='?', default=None, help='Input image path')
    parser.add_argument('-o', '--output', default=None, help='Output directory')
    parser.add_argument('-w', '--workers', type=int, default=12, help='Number of workers')
    parser.add_argument('-s', '--max-size', type=int, default=2048, help='Max image size (default 2048, 0=no resize)')
    parser.add_argument('-l', '--level', type=int, default=7, choices=range(1,15), help='Detail level 1-14 (default 7)')
    parser.add_argument('--no-blip', action='store_true', help='Skip BLIP2 (faster, ~1min saved)')
    parser.add_argument('--no-mps', action='store_true', help='Disable MPS, use CPU only')
    
    args = parser.parse_args()
    
    # 如果没有指定图片，提示用户
    if args.image is None:
        print("\n" + "="*60)
        print("🎨 SAM3 + Color Vectorization (Auto)")
        print("="*60)
        print("\nUsage: python sam3_color_vectorizer_fast.py <image_path>")
        print("\nExample:")
        print("  python sam3_color_vectorizer_fast.py photo.jpg")
        print("  python sam3_color_vectorizer_fast.py photo.jpg --no-blip  # 快速模式")
        print("  python sam3_color_vectorizer_fast.py photo.jpg -s 1024    # 缩小到1024")
        print("\nOptions:")
        print("  -s, --max-size  图片最大边长（默认2048，设0禁用缩放）")
        print("  --no-blip       跳过BLIP2分析（更快）")
        print("  --no-mps        禁用MPS加速，只用CPU")
        print("  -w N            设置并行worker数量")
        print("\nFeatures:")
        print("  • BLIP2 auto-captioning (MPS accelerated)")
        print("  • Auto-detection of salient regions")
        print("  • SAM3 semantic segmentation")
        print("  • Color quantization for details")
        print("  • Multi-core parallel processing")
        print("="*60)
        return
    
    # 检查图片是否存在
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image not found: {args.image}")
        return
    
    # 设置输出目录
    if args.output:
        output_dir = args.output
    else:
        output_dir = f"02_输出结果/{image_path.stem}_svg"
    
    # 获取级别配置
    level_config = DETAIL_LEVELS.get(args.level, DETAIL_LEVELS[7])
    
    vectorizer = SAM3ColorVectorizerFast(
        n_workers=args.workers,
        use_blip=not args.no_blip,
        use_mps=not args.no_mps,
        level_config=level_config
    )
    print(f"   Detail Level: {args.level} (min_area={level_config['min_area']}, epsilon={level_config['epsilon']})")
    return vectorizer.vectorize(str(image_path), output_dir, max_size=args.max_size)


if __name__ == "__main__":
    main()
