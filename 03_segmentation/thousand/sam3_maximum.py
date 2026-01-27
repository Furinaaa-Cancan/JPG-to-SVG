#!/usr/bin/env python3
"""
SAM3极限分割 - 使用SAM3的全部能力
- 低置信度阈值获取更多masks
- 密集边界框提示
- 超细文本提示
"""

import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
import time

sys.path.insert(0, "/Volumes/Seagate/SAM3/模型库/01_SAM3核心模型")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3Maximum:
    """SAM3极限分割"""
    
    def __init__(self, confidence_threshold: float = 0.1):
        """
        confidence_threshold: 降低到0.1获取更多masks（默认0.5太高）
        """
        print(f"\n🚀 SAM3 Maximum - Confidence: {confidence_threshold}")
        
        self.model = build_sam3_image_model(device="cpu")
        self.processor = Sam3Processor(
            self.model, 
            device="cpu",
            confidence_threshold=confidence_threshold  # 关键！降低阈值
        )
        
        print("✅ SAM3 loaded!")
    
    def segment_maximum(self, image_path: str, output_dir: str = "02_输出结果/sam3_max"):
        """使用SAM3的全部能力"""
        
        print("\n" + "="*70)
        print("💎 SAM3 MAXIMUM SEGMENTATION")
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
        
        all_masks = []
        
        # 方法1: 超详细文本提示
        print("\n📝 Method 1: Exhaustive Text Prompts")
        prompts = self.get_exhaustive_prompts()
        
        for i, prompt in enumerate(prompts):
            try:
                # 重置提示
                self.processor.reset_all_prompts(state)
                
                # 设置新提示
                prompt_state = self.processor.set_text_prompt(prompt, state)
                
                if prompt_state and 'masks' in prompt_state:
                    masks = prompt_state['masks']
                    if masks is not None:
                        # 可能有多个masks
                        masks_np = masks.cpu().numpy() if hasattr(masks, 'cpu') else np.array(masks)
                        
                        # 添加所有找到的masks
                        for j in range(masks_np.shape[0]):
                            mask = masks_np[j]
                            if len(mask.shape) > 2:
                                mask = mask.squeeze()
                            
                            area = np.sum(mask > 0.5)
                            if area > 100:
                                all_masks.append({
                                    'mask': mask,
                                    'area': area,
                                    'prompt': prompt,
                                    'score': float(prompt_state['scores'][j].cpu()) if 'scores' in prompt_state else 1.0
                                })
                
                if (i + 1) % 20 == 0:
                    print(f"   Progress: {i+1}/{len(prompts)}, found {len(all_masks)} masks")
                    
            except Exception as e:
                pass
        
        print(f"   Text prompts found: {len(all_masks)} masks")
        
        # 跳过边界框方法（太慢）
        # 方法2: 不同置信度阈值
        print("\n🎯 Method 3: Multi-threshold Sweep")
        for thresh in [0.05, 0.15, 0.25]:
            self.processor.set_confidence_threshold(thresh, state)
            
            # 重新运行一些关键提示
            key_prompts = ["object", "thing", "part", "detail", "element"]
            for prompt in key_prompts:
                try:
                    self.processor.reset_all_prompts(state)
                    prompt_state = self.processor.set_text_prompt(prompt, state)
                    
                    if prompt_state and 'masks' in prompt_state:
                        masks = prompt_state['masks']
                        if masks is not None:
                            masks_np = masks.cpu().numpy() if hasattr(masks, 'cpu') else np.array(masks)
                            
                            for j in range(masks_np.shape[0]):
                                mask = masks_np[j]
                                if len(mask.shape) > 2:
                                    mask = mask.squeeze()
                                
                                area = np.sum(mask > 0.5)
                                if area > 50:
                                    all_masks.append({
                                        'mask': mask,
                                        'area': area,
                                        'prompt': f"{prompt}@{thresh}",
                                        'score': float(prompt_state['scores'][j].cpu()) if 'scores' in prompt_state else 1.0
                                    })
                except:
                    pass
        
        print(f"\n📊 Total raw masks: {len(all_masks)}")
        
        # 按面积排序，只保留前2000个
        print("\n🔄 Sorting by area (skip dedup for speed)...")
        all_masks.sort(key=lambda x: x['area'], reverse=True)
        unique_masks = all_masks[:2000]
        print(f"   Selected: {len(unique_masks)} masks")
        
        # 提取颜色
        print("\n🎨 Extracting colors...")
        for mask_data in unique_masks:
            mask = mask_data['mask']
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            
            # 确保尺寸匹配
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h))
            
            pixels = img_array[mask > 0.5]
            if len(pixels) > 0:
                mean_color = np.mean(pixels, axis=0).astype(int)
                mask_data['color'] = f"#{mean_color[0]:02x}{mean_color[1]:02x}{mean_color[2]:02x}"
        
        # 保存
        print("\n💾 Saving results...")
        self.save_results(img_array, unique_masks, output_path)
        
        process_time = time.time() - start_time
        
        print(f"\n" + "="*70)
        print(f"✅ SAM3 MAXIMUM COMPLETE!")
        print(f"   Total masks: {len(unique_masks)}")
        print(f"   Time: {process_time:.1f}s")
        print("="*70)
        
        import subprocess
        subprocess.run(["open", str(output_path / "max_showcase.html")])
        
        return {'masks': unique_masks, 'count': len(unique_masks)}
    
    def get_exhaustive_prompts(self):
        """超详细的提示词列表"""
        
        return [
            # 基础对象
            "object", "thing", "item", "element", "part", "piece", "section",
            "region", "area", "zone", "segment", "portion",
            
            # 服装 - 极详细
            "blue dress", "royal blue dress", "velvet dress", "costume",
            "dress bodice", "dress top", "upper dress",
            "dress skirt", "lower dress", "skirt hem",
            "dress fold", "fabric fold", "cloth fold",
            "dress wrinkle", "fabric wrinkle", "crease",
            "dress pleat", "fabric pleat",
            "sleeve", "left sleeve", "right sleeve", "sleeve cuff",
            "collar", "dress collar", "neckline", "neck trim",
            "shoulder", "shoulder pad", "shoulder decoration",
            
            # 装饰 - 极详细
            "gold decoration", "golden trim", "gold embroidery",
            "gold thread", "golden thread", "metallic thread",
            "embroidered pattern", "embroidery design", "stitching",
            "decorative pattern", "ornament", "ornamental design",
            "button", "gold button", "decorative button", "fastener",
            "buckle", "belt buckle", "metal buckle",
            "belt", "waist belt", "decorative belt",
            "sequin", "sparkle", "glitter", "shiny decoration",
            "bead", "pearl", "jewelry",
            "lace", "lace trim", "lace pattern",
            "ribbon", "bow", "tassel",
            
            # 人物 - 极详细
            "face", "woman face", "facial features",
            "forehead", "temple",
            "eye", "left eye", "right eye", "eyelid", "eyelash", "eyebrow",
            "nose", "nose bridge", "nostril",
            "mouth", "lips", "upper lip", "lower lip", "red lips",
            "teeth", "tongue",
            "cheek", "cheekbone", "chin", "jaw", "jawline",
            "ear", "left ear", "right ear", "earring",
            "neck", "throat", "neck skin",
            "hair", "blonde hair", "wavy hair", "curly hair",
            "hair strand", "hair lock", "hair wave", "hair curl",
            "head", "head shape",
            
            # 手臂和手
            "arm", "left arm", "right arm", "forearm", "upper arm",
            "elbow", "wrist",
            "hand", "left hand", "right hand", "palm",
            "finger", "thumb", "index finger", "fingernail",
            "skin", "skin tone", "exposed skin",
            
            # 骷髅道具
            "skeleton", "skeleton prop", "decorative skeleton",
            "full skeleton", "complete skeleton",
            "skull", "skull head", "skull face", "skull bone",
            "skull teeth", "skull jaw",
            "rib cage", "ribs", "rib bone", "chest bones",
            "spine", "backbone", "vertebra",
            "pelvis", "hip bone",
            "arm bone", "humerus", "radius", "ulna",
            "leg bone", "femur", "tibia",
            "hand bones", "finger bones", "skeleton hand",
            "foot bones", "skeleton foot",
            "white bone", "bone", "bone structure",
            
            # 背景
            "background", "blue background", "stage background",
            "gradient", "gradient background",
            "smoke", "fog", "mist", "haze",
            "light", "spotlight", "stage light", "lighting",
            "shadow", "dark area", "dark region",
            
            # 颜色区域
            "white", "white area", "white region", "white object",
            "black", "black area", "black region", "dark object",
            "blue", "blue area", "blue region", "blue object",
            "gold", "golden area", "golden region",
            "red", "pink", "red area",
            
            # 纹理
            "texture", "fabric texture", "velvet texture",
            "pattern", "design", "motif",
            "shiny surface", "glossy area", "reflection",
            "matte surface", "rough area",
            
            # 边缘和细节
            "edge", "boundary", "outline", "contour",
            "detail", "fine detail", "small detail",
            "highlight", "bright area", "shiny spot",
        ]
    
    def dense_box_prompts(self, state, w, h) -> list:
        """密集边界框提示"""
        
        masks = []
        
        # 生成网格边界框
        grid_sizes = [4, 8, 16]  # 不同密度
        
        for grid in grid_sizes:
            cell_w = w / grid
            cell_h = h / grid
            
            for i in range(grid):
                for j in range(grid):
                    # 计算边界框（归一化坐标，中心+宽高格式）
                    cx = (i + 0.5) / grid
                    cy = (j + 0.5) / grid
                    bw = 1.0 / grid * 0.9  # 稍微小一点避免重叠
                    bh = 1.0 / grid * 0.9
                    
                    try:
                        self.processor.reset_all_prompts(state)
                        
                        # 添加边界框提示
                        prompt_state = self.processor.add_geometric_prompt(
                            [cx, cy, bw, bh],  # 中心坐标 + 宽高
                            True,  # 正样本
                            state
                        )
                        
                        if prompt_state and 'masks' in prompt_state:
                            masks_data = prompt_state['masks']
                            if masks_data is not None:
                                masks_np = masks_data.cpu().numpy() if hasattr(masks_data, 'cpu') else np.array(masks_data)
                                
                                for k in range(masks_np.shape[0]):
                                    mask = masks_np[k]
                                    if len(mask.shape) > 2:
                                        mask = mask.squeeze()
                                    
                                    area = np.sum(mask > 0.5)
                                    if area > 50:
                                        masks.append({
                                            'mask': mask,
                                            'area': area,
                                            'prompt': f'box_{grid}_{i}_{j}',
                                            'score': 1.0
                                        })
                    except:
                        pass
        
        return masks
    
    def smart_dedupe(self, masks: list) -> list:
        """智能去重"""
        
        if not masks:
            return []
        
        # 按分数排序
        masks.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        unique = []
        
        for mask_data in masks:
            mask = mask_data['mask']
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            
            is_dup = False
            
            # 只与最近的比较（速度优化）
            for u in unique[-50:]:
                u_mask = u['mask']
                if len(u_mask.shape) > 2:
                    u_mask = u_mask.squeeze()
                
                # 确保尺寸匹配
                if mask.shape != u_mask.shape:
                    continue
                
                # 计算IOU
                intersection = np.logical_and(mask > 0.5, u_mask > 0.5).sum()
                union = np.logical_or(mask > 0.5, u_mask > 0.5).sum()
                
                if union > 0:
                    iou = intersection / union
                    if iou > 0.7:  # 70%重叠
                        is_dup = True
                        break
            
            if not is_dup:
                unique.append(mask_data)
        
        return unique
    
    def save_results(self, img: np.ndarray, masks: list, output_path: Path):
        """保存结果"""
        
        h, w = img.shape[:2]
        
        # 彩色叠加
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        np.random.seed(42)
        colors = np.random.rand(len(masks), 3) * 255
        
        for i, m in enumerate(masks):
            mask = m['mask']
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h))
            
            binary = mask > 0.5
            for c in range(3):
                overlay[:, :, c] += binary * colors[i, c] * 0.2
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        composite = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        Image.fromarray(composite).save(output_path / "max_composite.png")
        
        # 边缘
        edges = np.zeros((h, w, 3), dtype=np.uint8)
        for i, m in enumerate(masks):
            mask = m['mask']
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h))
            
            binary = (mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = colors[i].astype(int).tolist()
            cv2.drawContours(edges, contours, -1, color, 1)
        
        Image.fromarray(edges).save(output_path / "max_edges.png")
        
        # HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>SAM3 Maximum</title>
        <style>
            body {{ margin:0; background:#000; color:#fff; font-family:sans-serif; }}
            .header {{ text-align:center; padding:50px; background:linear-gradient(135deg,#f093fb,#f5576c); }}
            h1 {{ font-size:4em; margin:0; }}
            .count {{ font-size:3em; color:#FFD700; }}
            .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; padding:40px; max-width:1600px; margin:0 auto; }}
            .card {{ background:#1a1a1a; border-radius:15px; overflow:hidden; }}
            .card-header {{ padding:15px; background:#2a2a2a; font-weight:bold; }}
            img {{ width:100%; }}
        </style>
        </head>
        <body>
            <div class="header">
                <h1>💎 SAM3 MAXIMUM</h1>
                <div class="count">{len(masks)} Masks</div>
            </div>
            <div class="grid">
                <div class="card">
                    <div class="card-header">🎨 Composite</div>
                    <img src="max_composite.png">
                </div>
                <div class="card">
                    <div class="card-header">📐 Edges</div>
                    <img src="max_edges.png">
                </div>
            </div>
        </body>
        </html>
        """
        with open(output_path / "max_showcase.html", 'w') as f:
            f.write(html)


def main():
    # 使用低置信度阈值获取更多masks
    segmenter = SAM3Maximum(confidence_threshold=0.1)
    return segmenter.segment_maximum("01_输入图片/Ladygaga_2.jpg")


if __name__ == "__main__":
    main()
