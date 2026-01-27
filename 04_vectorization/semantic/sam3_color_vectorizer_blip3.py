#!/usr/bin/env python3
"""
SAM3 + 颜色量化 混合矢量化 - BLIP-3 (xGen-MM) 版本

使用BLIP-3替代BLIP-2进行图像理解和提示词生成
BLIP-3特点：更强的推理能力、支持任意分辨率、多图理解
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

sys.path.insert(0, "/Volumes/Seagate/SAM3/模型库/01_SAM3核心模型")

# BLIP-3 (xGen-MM) 
try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    BLIP3_AVAILABLE = True
except ImportError:
    BLIP3_AVAILABLE = False


# 全局变量用于多进程共享
_img_global = None
_h_global = None
_w_global = None


def process_region_batch(args):
    """处理单个区域的颜色量化（在子进程中运行）"""
    mask, color, area, img, h, w, level_config = args
    
    paths = []
    
    cfg_min_area = level_config.get('min_area', 80)
    cfg_epsilon = level_config.get('epsilon', 0.002)
    n_large = level_config.get('n_colors_large', 8)
    n_medium = level_config.get('n_colors_medium', 5)
    n_small = level_config.get('n_colors_small', 3)
    
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
        n_colors = 1
        min_area = cfg_min_area
        epsilon_factor = cfg_epsilon * 2
    
    if n_colors > 1 and area > 3000:
        inner_paths = quantize_region_standalone(img, mask, h, w, n_colors=n_colors, min_area=cfg_min_area, epsilon=cfg_epsilon)
        paths.extend(inner_paths)
    
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


class SAM3ColorVectorizerBLIP3:
    """SAM3 + BLIP-3 混合矢量化器"""
    
    def __init__(self, n_workers=None, use_blip3=True, use_mps=True, level_config=None):
        self.n_workers = n_workers or min(cpu_count(), 12)
        self.level_config = level_config or {
            'n_colors_large': 8, 'n_colors_medium': 5, 'n_colors_small': 3,
            'min_area': 80, 'epsilon': 0.002
        }
        print(f"\n🚀 Initializing SAM3 + Color Vectorizer (BLIP-3 Version)")
        print(f"   CPU cores: {self.n_workers}")
        
        import torch
        
        if use_mps and torch.backends.mps.is_available():
            self.device = "mps"
            self.sam3_device = "cpu"
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.sam3_device = "cuda"
        else:
            self.device = "cpu"
            self.sam3_device = "cpu"
        
        torch.set_num_threads(self.n_workers)
        print(f"   Device: {self.device} (SAM3: {self.sam3_device})")
        
        # 加载SAM3
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        self.sam3_model = build_sam3_image_model(device=self.sam3_device)
        self.sam3_processor = Sam3Processor(self.sam3_model, device=self.sam3_device, confidence_threshold=0.1)
        print(f"✅ SAM3 loaded!")
        
        # 加载BLIP-3 (xGen-MM)
        self.blip3_processor = None
        self.blip3_model = None
        self.use_blip3 = use_blip3
        
        if use_blip3 and BLIP3_AVAILABLE:
            try:
                print("   Loading BLIP-3 (xGen-MM) for advanced image understanding...")
                
                # xGen-MM模型 - Salesforce的BLIP-3
                model_name = "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
                
                self.blip3_processor = AutoProcessor.from_pretrained(
                    model_name, 
                    trust_remote_code=True
                )
                
                import torch
                blip3_dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32
                
                self.blip3_model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=blip3_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                # MPS可能不完全支持，尝试加载
                try:
                    self.blip3_model = self.blip3_model.to(self.device)
                    print(f"   ✅ BLIP-3 (xGen-MM) loaded on {self.device}!")
                except Exception as e:
                    print(f"   ⚠️ BLIP-3 fallback to CPU: {e}")
                    self.blip3_model = self.blip3_model.to("cpu")
                    self.device = "cpu"
                    
            except Exception as e:
                print(f"   ⚠️ BLIP-3 not available: {e}")
                print(f"   💡 Falling back to base prompts")
                self.blip3_processor = None
        elif not use_blip3:
            print("   ⏭️ BLIP-3 skipped (fast mode)")
        
        # 打印level配置
        level_num = 7  # 默认
        print(f"   Detail Level: {level_num} (min_area={self.level_config['min_area']}, epsilon={self.level_config['epsilon']})")
    
    def auto_generate_prompts_blip3(self, img_pil) -> list:
        """使用BLIP-3生成更智能的提示词"""
        
        # 基础通用提示词
        base_prompts = [
            "person", "people", "man", "woman", "face", "head", "hair",
            "hand", "hands", "arm", "body", "clothing", "dress", "shirt",
            "background", "sky", "ground", "floor", "wall"
        ]
        
        if self.blip3_model is None or self.blip3_processor is None:
            print(f"   📋 Using base prompts: {len(base_prompts)}")
            return base_prompts
        
        try:
            import torch
            
            # BLIP-3风格的对话式提问
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "List all the distinct objects and elements visible in this image. Be specific and detailed. Format: comma-separated list."}
                    ]
                }
            ]
            
            prompt = self.blip3_processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True
            )
            
            inputs = self.blip3_processor(
                images=img_pil, 
                text=prompt, 
                return_tensors="pt"
            ).to(self.blip3_model.device)
            
            with torch.no_grad():
                outputs = self.blip3_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False
                )
            
            response = self.blip3_processor.decode(outputs[0], skip_special_tokens=True)
            
            # 解析响应，提取物体列表
            if "assistant" in response.lower():
                response = response.split("assistant")[-1]
            
            # 清理并分割
            objects = []
            for item in response.replace('\n', ',').split(','):
                item = item.strip().lower()
                if item and len(item) > 1 and len(item) < 30:
                    # 过滤掉非物体词
                    if not any(skip in item for skip in ['the', 'a ', 'an ', 'is', 'are', 'and']):
                        objects.append(item)
            
            # 合并基础提示词和BLIP-3生成的
            all_prompts = list(set(base_prompts + objects[:20]))
            print(f"   📋 BLIP-3 generated {len(objects)} objects, total prompts: {len(all_prompts)}")
            
            return all_prompts
            
        except Exception as e:
            print(f"   ⚠️ BLIP-3 inference failed: {e}")
            return base_prompts
    
    def auto_detect_regions(self, img, h, w) -> list:
        """自动检测图像中的显著区域"""
        
        regions = []
        
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        pixels = lab.reshape(-1, 3).astype(np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        _, labels, _ = cv2.kmeans(pixels, 8, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        
        label_img = labels.reshape(h, w)
        
        for cid in range(8):
            color_mask = (label_img == cid).astype(np.uint8) * 255
            area = np.sum(color_mask > 0)
            
            if area > 1000 and area < h * w * 0.8:
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 500:
                        x, y, bw, bh = cv2.boundingRect(contour)
                        cx = (x + bw/2) / w
                        cy = (y + bh/2) / h
                        nw = bw / w
                        nh = bh / h
                        regions.append({
                            'box': [cx, cy, nw, nh],
                            'area': cv2.contourArea(contour)
                        })
        
        regions.sort(key=lambda x: -x['area'])
        return regions[:10]
    
    def sam3_segment_parallel(self, img_pil, img, h, w) -> list:
        """SAM3语义分割 - BLIP-3版本"""
        
        regions = []
        state = self.sam3_processor.set_image(img_pil)
        
        # 使用BLIP-3生成提示词
        key_prompts = self.auto_generate_prompts_blip3(img_pil)
        
        # 自动检测显著区域
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
        
        # 文本提示分割
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
    
    def vectorize(self, image_path: str, output_dir: str = "02_输出结果/sam3_blip3_svg", max_size: int = 2048):
        """混合矢量化主函数"""
        
        print("\n" + "="*70)
        print(f"💎 SAM3 + BLIP-3 HYBRID VECTORIZATION")
        print("="*70)
        
        start_time = time.time()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        orig_pil = Image.open(image_path).convert("RGB")
        orig_w, orig_h = orig_pil.size
        orig_img = np.array(orig_pil)
        
        if max_size > 0 and max(orig_w, orig_h) > max_size:
            scale = max_size / max(orig_w, orig_h)
            sam_w = int(orig_w * scale)
            sam_h = int(orig_h * scale)
            sam_pil = orig_pil.resize((sam_w, sam_h), Image.LANCZOS)
            sam_img = np.array(sam_pil)
            print(f"\n📷 Original: {orig_w}x{orig_h}")
            print(f"   SAM3 uses: {sam_w}x{sam_h}")
        else:
            scale = 1.0
            sam_pil = orig_pil
            sam_img = orig_img
            sam_w, sam_h = orig_w, orig_h
            print(f"\n📷 Image: {orig_w}x{orig_h}")
        
        # 用原图尺寸作为最终输出
        img = orig_img
        h, w = orig_h, orig_w
        
        all_paths = []
        
        # Step 1: SAM3分割
        print("\n🎯 Step 1: SAM3 + BLIP-3 Segmentation...")
        t1 = time.time()
        sam3_regions = self.sam3_segment_parallel(sam_pil, sam_img, sam_h, sam_w)
        
        # 缩放mask到原图尺寸
        if scale != 1.0:
            for region in sam3_regions:
                region['mask'] = cv2.resize(region['mask'], (w, h), interpolation=cv2.INTER_NEAREST)
                region['area'] = np.sum(region['mask'] > 0)
        
        print(f"   SAM3 regions: {len(sam3_regions)} ({time.time()-t1:.1f}s)")
        
        # Step 2: 背景处理
        print("\n🌅 Step 2: Background processing...")
        t2 = time.time()
        bg_paths = self.background_quantize(img, h, w)
        all_paths.extend(bg_paths)
        print(f"   Background paths: {len(bg_paths)} ({time.time()-t2:.1f}s)")
        
        # Step 3: 前景处理
        print("\n🎨 Step 3: Foreground detail extraction...")
        t3 = time.time()
        fg_paths = self.foreground_detail_parallel(img, sam3_regions, h, w)
        all_paths.extend(fg_paths)
        print(f"   Foreground paths: {len(fg_paths)} ({time.time()-t3:.1f}s)")
        
        # Step 4: 生成SVG
        print("\n✨ Step 4: Generate SVG...")
        svg_path = output_path / "sam3_blip3_vector.svg"
        stats = self.create_svg(all_paths, w, h, str(svg_path))
        
        process_time = time.time() - start_time
        
        # 保存对比图
        self.save_comparison_image(image_path, str(svg_path), output_path, stats, process_time)
        
        print(f"\n" + "="*70)
        print(f"✅ COMPLETE!")
        print(f"   Paths: {stats['paths']}")
        print(f"   Size: {stats['size_kb']:.1f} KB")
        print(f"   Time: {process_time:.1f}s")
        print("="*70)
        
        return stats
    
    def background_quantize(self, img, h, w) -> list:
        """背景处理"""
        
        paths = []
        
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
        
        return paths
    
    def foreground_detail_parallel(self, img, sam3_regions, h, w) -> list:
        """前景区域并行处理"""
        
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
    
    def points_to_path(self, points) -> str:
        """将点转换为SVG路径"""
        
        pts = np.array(points)
        n = len(pts)
        
        if n < 3:
            return ""
        
        if n > 20:
            path_d = f"M{pts[0][0]},{pts[0][1]}"
            for i in range(1, n):
                p0 = pts[i-1]
                p1 = pts[i]
                p2 = pts[(i+1) % n]
                
                c1x = p0[0] + (p1[0] - p0[0]) * 0.5
                c1y = p0[1] + (p1[1] - p0[1]) * 0.5
                c2x = p1[0] + (p2[0] - p1[0]) * 0.5
                c2y = p1[1] + (p2[1] - p1[1]) * 0.5
                
                path_d += f"C{c1x},{c1y} {c2x},{c2y} {p2[0]},{p2[1]}"
        else:
            path_d = f"M{pts[0][0]},{pts[0][1]}"
            for i in range(1, n):
                path_d += f"L{pts[i][0]},{pts[i][1]}"
        
        path_d += "Z"
        return path_d
    
    def save_comparison_image(self, original: str, svg: str, output_path: Path, stats: dict, process_time: float):
        """保存对比图片"""
        import matplotlib.pyplot as plt
        from cairosvg import svg2png
        from io import BytesIO
        
        orig_img = Image.open(original).convert('RGB')
        
        try:
            png_data = svg2png(url=svg, output_width=1200)
            svg_img = Image.open(BytesIO(png_data)).convert('RGB')
        except Exception as e:
            print(f"   ⚠️ SVG转换失败: {e}")
            svg_img = orig_img.copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        fig.patch.set_facecolor('white')
        
        axes[0].imshow(orig_img)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(svg_img)
        axes[1].set_title('SVG (BLIP-3 Version)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        size_kb = stats['size_kb']
        size_str = f"{size_kb/1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.0f} KB"
        time_str = f"{process_time/60:.1f} min" if process_time >= 60 else f"{process_time:.1f}s"
        
        axes[1].text(0.5, -0.05, f'Paths: {stats["paths"]:,} | Size: {size_str} | Time: {time_str}', 
                    transform=axes[1].transAxes, ha='center', fontsize=11)
        
        plt.tight_layout()
        
        png_path = output_path / "comparison.png"
        pdf_path = output_path / "comparison.pdf"
        plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Saved: {png_path}")


# 细节级别配置（与BLIP2版本相同）
DETAIL_LEVELS = {
    1:  {'n_colors_large': 1,  'n_colors_medium': 1,  'n_colors_small': 1,  'min_area': 8000, 'epsilon': 0.025},
    7:  {'n_colors_large': 8,  'n_colors_medium': 5,  'n_colors_small': 3,  'min_area': 80,   'epsilon': 0.002},
    14: {'n_colors_large': 25, 'n_colors_medium': 18, 'n_colors_small': 12, 'min_area': 3,    'epsilon': 0.0003},
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SAM3 + BLIP-3 Color Vectorization')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('-o', '--output', default='02_输出结果/sam3_blip3_svg', help='Output directory')
    parser.add_argument('-l', '--level', type=int, default=7, help='Detail level (1-14)')
    parser.add_argument('--no-blip3', action='store_true', help='Skip BLIP-3')
    parser.add_argument('--no-mps', action='store_true', help='Disable MPS')
    
    args = parser.parse_args()
    
    level_config = DETAIL_LEVELS.get(args.level, DETAIL_LEVELS[7])
    
    vectorizer = SAM3ColorVectorizerBLIP3(
        use_blip3=not args.no_blip3,
        use_mps=not args.no_mps,
        level_config=level_config
    )
    
    vectorizer.vectorize(args.image, args.output)


if __name__ == "__main__":
    main()
