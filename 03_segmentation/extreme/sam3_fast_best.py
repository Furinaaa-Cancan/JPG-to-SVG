#!/usr/bin/env python3
"""
SAM3快速高质量分割
使用正确的API + 多核CPU并行
"""

import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor
import threading

sys.path.insert(0, "/Volumes/Seagate/SAM3/模型库/01_SAM3核心模型")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3FastBest:
    """SAM3快速高质量分割"""
    
    def __init__(self):
        print("\n🚀 Initializing SAM3 Fast Best...")
        
        # CPU模式最稳定
        self.model = build_sam3_image_model(device="cpu")
        self.processor = Sam3Processor(self.model, device="cpu")
        
        # 线程锁
        self.lock = threading.Lock()
        
        print("✅ SAM3 loaded!")
    
    def segment_fast(self, image_path: str, output_dir: str = "02_输出结果/sam3_fast"):
        """快速分割"""
        
        print("\n" + "="*70)
        print("⚡ SAM3 FAST BEST SEGMENTATION")
        print("="*70)
        
        start_time = time.time()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载图像
        img = Image.open(image_path)
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        print(f"\n📷 Input: {image_path}")
        print(f"   Size: {w}x{h}")
        
        # 设置图像
        print("\n🔧 Setting image...")
        state = self.processor.set_image(img)
        
        # 超详细提示词 - 分批处理
        print("\n📝 Text Prompting (Batch Processing)...")
        all_prompts = self.get_all_prompts()
        
        all_masks = []
        batch_size = 20
        total_batches = (len(all_prompts) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_prompts))
            batch_prompts = all_prompts[start_idx:end_idx]
            
            batch_masks = self.process_batch(batch_prompts, state, h, w)
            all_masks.extend(batch_masks)
            
            print(f"   Batch {batch_idx+1}/{total_batches}: {len(batch_masks)} masks")
        
        print(f"\n   Total raw masks: {len(all_masks)}")
        
        # 去重
        print("\n🔄 Deduplicating...")
        unique_masks = self.deduplicate(all_masks)
        print(f"   Unique masks: {len(unique_masks)}")
        
        # 生成可视化
        print("\n🎨 Generating visualizations...")
        self.save_visualizations(img_array, unique_masks, output_path)
        
        process_time = time.time() - start_time
        
        print(f"\n" + "="*70)
        print(f"✅ COMPLETE!")
        print(f"   Total masks: {len(unique_masks)}")
        print(f"   Time: {process_time:.1f}s")
        print(f"   Speed: {len(unique_masks)/process_time:.2f} masks/sec")
        print("="*70)
        
        # 打开展示
        import subprocess
        subprocess.run(["open", str(output_path / "fast_showcase.html")])
        
        return {'masks': unique_masks, 'time': process_time}
    
    def get_all_prompts(self):
        """获取所有提示词"""
        return [
            # 装饰 - 超详细
            "gold decoration", "golden embroidery", "metallic gold trim",
            "gold thread pattern", "embroidered design", "decorative stitching",
            "sequins", "sparkle", "shiny decoration",
            "button", "decorative button", "gold button",
            "buckle", "belt buckle", "metallic buckle",
            "belt", "waist belt", "decorative belt",
            "pearl", "pearl decoration", "bead",
            
            # 服装部分
            "blue dress", "royal blue dress", "blue costume",
            "blue velvet", "velvet fabric", "blue fabric",
            "dress bodice", "upper dress", "chest area",
            "dress skirt", "lower dress", "hem",
            "sleeve", "left sleeve", "right sleeve",
            "collar", "dress collar", "neckline",
            "shoulder", "shoulder area", "shoulder decoration",
            "cuff", "sleeve cuff", "wrist area",
            
            # 服装细节
            "fold", "fabric fold", "dress fold",
            "wrinkle", "fabric wrinkle", "crease",
            "pleat", "dress pleat", "skirt pleat",
            "shadow on dress", "fabric shadow",
            "highlight on fabric", "shiny fabric",
            "texture", "fabric texture", "velvet texture",
            
            # 人物
            "face", "woman face", "singer face",
            "head", "human head", "person head",
            "blonde hair", "wavy hair", "curly blonde hair",
            "hair strand", "hair lock", "hair wave",
            "eye", "closed eye", "eyelash",
            "nose", "mouth", "red lips",
            "lips", "lipstick", "red lipstick",
            "chin", "jaw", "cheek",
            "forehead", "eyebrow",
            "neck", "throat", "neck area",
            "ear", "earring",
            
            # 手臂和手
            "arm", "left arm", "right arm",
            "hand", "left hand", "right hand",
            "finger", "fingers", "thumb",
            "wrist", "elbow",
            "skin", "skin tone", "exposed skin",
            
            # 骷髅道具
            "skeleton", "full skeleton", "skeleton prop",
            "skeleton decoration", "decorative skeleton",
            "skull", "skull head", "skull face",
            "skull bone", "cranium",
            "rib cage", "ribs", "rib bone",
            "spine", "backbone", "vertebra",
            "pelvis", "hip bone",
            "arm bone", "leg bone", "thigh bone",
            "hand bones", "finger bones", "skeleton hand",
            "foot bones", "skeleton foot",
            "white bone", "bone structure",
            
            # 背景
            "background", "stage background",
            "blue background", "gradient background",
            "smoke", "fog", "mist",
            "stage light", "lighting", "spotlight",
            "shadow", "dark area",
            
            # 颜色区域
            "white area", "white object",
            "black area", "dark region",
            "blue area", "blue region",
            "gold area", "golden region",
        ]
    
    def process_batch(self, prompts, state, h, w):
        """处理一批提示词"""
        masks = []
        
        for prompt in prompts:
            try:
                prompt_state = self.processor.set_text_prompt(prompt, state)
                
                if prompt_state and 'masks' in prompt_state:
                    mask_data = prompt_state['masks']
                    if mask_data is not None and hasattr(mask_data, 'shape'):
                        if mask_data.shape[0] > 0:
                            mask = mask_data[0] if len(mask_data.shape) > 2 else mask_data
                            
                            if hasattr(mask, 'cpu'):
                                mask = mask.cpu().numpy()
                            
                            # 确保尺寸正确
                            if len(mask.shape) > 2:
                                mask = mask.squeeze()
                            
                            # 检查mask有效性
                            area = np.sum(mask > 0.5)
                            coverage = area / (h * w) * 100
                            
                            if coverage > 0.01:  # 至少0.01%覆盖
                                masks.append({
                                    'mask': mask,
                                    'prompt': prompt,
                                    'coverage': coverage
                                })
            except Exception as e:
                pass
        
        return masks
    
    def deduplicate(self, masks):
        """去重"""
        if not masks:
            return []
        
        # 按覆盖率排序
        masks.sort(key=lambda x: x['coverage'], reverse=True)
        
        unique = []
        
        for mask_data in masks:
            mask = mask_data['mask']
            
            is_dup = False
            for u in unique:
                u_mask = u['mask']
                
                # 确保尺寸匹配
                if mask.shape != u_mask.shape:
                    continue
                
                # 计算IOU
                intersection = np.logical_and(mask > 0.5, u_mask > 0.5).sum()
                union = np.logical_or(mask > 0.5, u_mask > 0.5).sum()
                
                if union > 0:
                    iou = intersection / union
                    if iou > 0.85:
                        is_dup = True
                        break
            
            if not is_dup:
                unique.append(mask_data)
        
        return unique
    
    def save_visualizations(self, img, masks, output_path):
        """保存可视化"""
        
        h, w = img.shape[:2]
        
        # 保存单独的masks
        masks_dir = output_path / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        for i, m in enumerate(masks):
            mask = m['mask']
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            
            binary = (mask > 0.5).astype(np.uint8) * 255
            prompt = m.get('prompt', 'unknown')[:25].replace(' ', '_')
            cov = m.get('coverage', 0)
            
            Image.fromarray(binary).save(masks_dir / f"{i:03d}_{prompt}_cov{cov:.1f}.png")
        
        # 彩色叠加
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        np.random.seed(42)
        colors = np.random.rand(len(masks), 3) * 255
        
        for i, m in enumerate(masks):
            mask = m['mask']
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h))
            
            binary = mask > 0.5
            for c in range(3):
                overlay[:, :, c] += binary * colors[i, c] * 0.3
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        composite = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        Image.fromarray(composite).save(output_path / "fast_composite.png")
        
        # 边缘图
        edges = np.zeros((h, w, 3), dtype=np.uint8)
        for i, m in enumerate(masks):
            mask = m['mask']
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h))
            
            binary = (mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = colors[i].astype(int).tolist()
            cv2.drawContours(edges, contours, -1, color, 2)
        
        Image.fromarray(edges).save(output_path / "fast_edges.png")
        
        # HTML
        self.create_html(output_path, len(masks))
    
    def create_html(self, output_path, mask_count):
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SAM3 Fast Best</title>
            <style>
                body {{ margin:0; font-family:sans-serif; background:#1a1a2e; color:white; }}
                .header {{ text-align:center; padding:40px; background:rgba(0,0,0,0.3); }}
                h1 {{ font-size:3em; color:#00ff88; margin:0; }}
                .container {{ max-width:1400px; margin:40px auto; padding:0 20px; }}
                .stat {{ font-size:2em; color:#ffd700; }}
                .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
                .card {{ background:rgba(255,255,255,0.1); border-radius:15px; overflow:hidden; }}
                .card-header {{ padding:15px; background:rgba(0,0,0,0.3); font-weight:bold; }}
                img {{ width:100%; display:block; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>⚡ SAM3 Fast Best</h1>
                <p class="stat">{mask_count} Masks Extracted</p>
            </div>
            <div class="container">
                <div class="grid">
                    <div class="card">
                        <div class="card-header">🎨 Composite</div>
                        <img src="fast_composite.png">
                    </div>
                    <div class="card">
                        <div class="card-header">📐 Edges</div>
                        <img src="fast_edges.png">
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        with open(output_path / "fast_showcase.html", 'w') as f:
            f.write(html)


def main():
    segmenter = SAM3FastBest()
    return segmenter.segment_fast("01_输入图片/Ladygaga_2.jpg")


if __name__ == "__main__":
    main()
