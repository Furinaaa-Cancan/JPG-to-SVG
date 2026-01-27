#!/usr/bin/env python3
"""
SAM3最强分割
使用SAM3的正确API进行极致细节分割
"""

import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
from typing import Dict, List
import time

# 添加SAM3路径
sys.path.insert(0, "/Volumes/Seagate/SAM3/模型库/01_SAM3核心模型")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3BestSegmentation:
    """SAM3最强分割器"""
    
    def __init__(self):
        print("\n🚀 Initializing SAM3 Best Segmentation...")
        
        # 加载SAM3模型
        self.model = build_sam3_image_model(device="cpu")
        self.processor = Sam3Processor(self.model, device="cpu")
        
        print("✅ SAM3 loaded successfully!")
        
    def segment_with_best_prompts(self, image_path: str, output_dir: str = "02_输出结果/sam3_best"):
        """使用最佳提示策略进行分割"""
        
        print("\n" + "="*70)
        print("🎯 SAM3 BEST SEGMENTATION - MAXIMUM QUALITY")
        print("="*70)
        
        start_time = time.time()
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载图像
        img = Image.open(image_path)
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        print(f"\n📷 Input: {image_path}")
        print(f"   Size: {w}x{h}")
        
        # Step 1: 设置图像
        print("\n🔧 Step 1: Setting image in SAM3")
        state = self.processor.set_image(img)
        
        # Step 2: 详细的文本提示分割
        print("\n📝 Step 2: Comprehensive Text Prompting")
        
        # 超详细的提示词列表
        detailed_prompts = [
            # === 装饰细节 ===
            "gold decoration", "golden trim", "gold ornament",
            "metallic decoration", "shiny decoration",
            "embroidery", "embroidered pattern",
            "sequins", "sparkles", "glitter",
            "beads", "pearls", "jewels",
            "lace", "lace pattern", "lace trim",
            "ribbon", "bow", "tassel",
            
            # === 服装细节 ===
            "blue dress", "blue fabric", "blue costume",
            "dress", "gown", "costume",
            "collar", "neckline", "neck area",
            "sleeve", "arm covering", "shoulder",
            "bodice", "chest area", "torso",
            "skirt", "dress bottom", "hem",
            "fold", "wrinkle", "crease", "pleat",
            "fabric texture", "cloth pattern",
            "shadow on dress", "highlight on dress",
            
            # === 服装部件 ===
            "button", "fastener", "clasp",
            "zipper", "hook", "buckle",
            "belt", "sash", "waistband",
            "pocket", "seam", "stitching",
            
            # === 人物部分 ===
            "face", "head", "facial features",
            "blonde hair", "hair", "hairstyle",
            "eye", "eyes", "eyebrow",
            "nose", "mouth", "lips", "red lips",
            "chin", "cheek", "forehead",
            "neck", "throat",
            "hand", "hands", "finger", "fingers",
            "arm", "elbow", "wrist",
            "skin", "skin tone",
            
            # === 配饰 ===
            "jewelry", "accessory",
            "necklace", "chain", "pendant",
            "bracelet", "ring", "earring",
            "brooch", "pin",
            
            # === 骷髅道具 ===
            "skeleton", "skeletal figure", "bones",
            "skull", "skull head", "skull face",
            "rib cage", "ribs", "spine",
            "bone", "white bones", "skeletal structure",
            "skeleton decoration", "prop skeleton",
            
            # === 背景元素 ===
            "background", "backdrop",
            "blue background", "gradient background",
            "stage", "performance area",
            "lighting", "stage light", "spotlight",
            "smoke", "fog", "mist", "haze",
            "shadow", "dark area",
            
            # === 颜色特定区域 ===
            "white area", "white object",
            "black area", "black object",
            "blue area", "blue region",
            "gold area", "golden region",
            "red area", "pink area",
            
            # === 纹理和图案 ===
            "pattern", "design", "motif",
            "texture", "surface detail",
            "shiny surface", "matte surface",
            "smooth area", "rough area",
            "reflection", "glossy area",
            
            # === 边缘和轮廓 ===
            "edge", "boundary", "outline",
            "contour", "silhouette",
            "sharp edge", "soft edge",
            "transition area", "gradient area"
        ]
        
        all_masks = []
        successful_prompts = []
        
        for i, prompt in enumerate(detailed_prompts):
            try:
                # 设置文本提示
                prompt_state = self.processor.set_text_prompt(prompt, state)
                
                # 检查是否有mask
                if prompt_state is not None:
                    # 尝试不同方式获取mask
                    mask = None
                    
                    # 方式1：直接从prompt_state获取masks
                    if isinstance(prompt_state, dict) and 'masks' in prompt_state:
                        masks = prompt_state['masks']
                        if masks is not None and hasattr(masks, 'shape'):
                            if masks.shape[0] > 0:
                                mask = masks[0] if len(masks.shape) > 2 else masks
                    
                    # 方式2：从state获取
                    elif isinstance(state, dict) and 'masks' in state:
                        masks = state['masks']
                        if masks is not None and hasattr(masks, 'shape'):
                            if masks.shape[0] > 0:
                                mask = masks[0] if len(masks.shape) > 2 else masks
                    
                    if mask is not None:
                        # 转换为numpy数组
                        if hasattr(mask, 'cpu'):
                            mask = mask.cpu().numpy()
                        else:
                            mask = np.array(mask)
                        
                        # 检查mask有效性
                        if mask.size > 0 and np.any(mask > 0.1):
                            all_masks.append({
                                'mask': mask,
                                'prompt': prompt,
                                'coverage': np.sum(mask > 0.5) / (h * w) * 100
                            })
                            successful_prompts.append(prompt)
                            print(f"   ✓ [{i+1}/{len(detailed_prompts)}] Found: {prompt}")
                
            except Exception as e:
                # 静默处理错误，继续下一个
                pass
            
            # 显示进度
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{len(detailed_prompts)} prompts processed")
        
        print(f"\n   Successfully segmented {len(all_masks)} regions")
        
        # Step 3: 去重
        print("\n🔄 Step 3: Deduplicating Masks")
        unique_masks = self.deduplicate_masks(all_masks)
        print(f"   Unique masks: {len(unique_masks)}")
        
        # Step 4: 生成可视化
        print("\n🎨 Step 4: Generating Visualizations")
        
        # 保存每个mask
        self.save_all_masks(unique_masks, output_path)
        
        # 创建组合可视化
        composite = self.create_composite_visualization(img_array, unique_masks)
        composite_path = output_path / "sam3_composite.png"
        Image.fromarray(composite).save(composite_path)
        print(f"   Saved composite: {composite_path}")
        
        # 创建边缘可视化
        edges = self.create_edge_visualization(unique_masks, h, w)
        edges_path = output_path / "sam3_edges.png"
        Image.fromarray(edges).save(edges_path)
        print(f"   Saved edges: {edges_path}")
        
        # Step 5: 生成报告
        print("\n📊 Step 5: Generating Report")
        stats = self.generate_report(unique_masks, successful_prompts, output_path)
        
        process_time = time.time() - start_time
        
        print(f"\n" + "="*70)
        print(f"✅ SEGMENTATION COMPLETE!")
        print(f"   Total masks: {len(unique_masks)}")
        print(f"   Processing time: {process_time:.1f}s")
        print(f"   Average coverage: {stats['avg_coverage']:.2f}%")
        print(f"   Output: {output_path}")
        print("="*70)
        
        # 创建HTML展示
        self.create_html_showcase(output_path, stats)
        
        return {
            'masks': unique_masks,
            'stats': stats,
            'output_dir': str(output_path)
        }
    
    def deduplicate_masks(self, masks: List[Dict]) -> List[Dict]:
        """去重masks"""
        
        if not masks:
            return []
        
        # 按覆盖率排序
        masks.sort(key=lambda x: x.get('coverage', 0), reverse=True)
        
        unique_masks = []
        
        for mask_data in masks:
            mask = mask_data['mask']
            
            # 检查是否与已有mask重复
            is_duplicate = False
            
            for unique in unique_masks:
                unique_mask = unique['mask']
                
                # 计算IOU
                intersection = np.logical_and(mask > 0.5, unique_mask > 0.5).sum()
                union = np.logical_or(mask > 0.5, unique_mask > 0.5).sum()
                
                if union > 0:
                    iou = intersection / union
                    if iou > 0.85:  # 85%重叠视为重复
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_masks.append(mask_data)
        
        return unique_masks
    
    def save_all_masks(self, masks: List[Dict], output_path: Path):
        """保存所有masks"""
        
        masks_dir = output_path / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['mask']
            prompt = mask_data.get('prompt', 'unknown')
            coverage = mask_data.get('coverage', 0)
            
            # 二值化
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            
            # 确保是2D数组
            if len(binary_mask.shape) == 3:
                binary_mask = binary_mask[:, :, 0]
            elif len(binary_mask.shape) > 2:
                binary_mask = binary_mask.squeeze()
            
            # 生成文件名
            safe_prompt = prompt.replace(' ', '_').replace('/', '_')[:30]
            filename = f"{i:03d}_{safe_prompt}_cov{coverage:.1f}.png"
            
            mask_path = masks_dir / filename
            Image.fromarray(binary_mask).save(mask_path)
    
    def create_composite_visualization(self, img: np.ndarray, masks: List[Dict]) -> np.ndarray:
        """创建彩色组合可视化"""
        
        h, w = img.shape[:2]
        
        # 创建彩色叠加
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        
        # 生成不同的颜色
        np.random.seed(42)
        colors = np.random.rand(len(masks), 3) * 255
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['mask']
            
            # 确保mask是2D的
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            
            # 确保尺寸匹配
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h))
            
            binary_mask = (mask > 0.5)
            
            for c in range(3):
                overlay[:, :, c] += binary_mask * colors[i, c] * 0.3
        
        # 限制值范围
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # 与原图混合
        composite = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        
        return composite
    
    def create_edge_visualization(self, masks: List[Dict], h: int, w: int) -> np.ndarray:
        """创建边缘可视化"""
        
        edges = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 生成颜色
        np.random.seed(42)
        colors = np.random.rand(len(masks), 3) * 255
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['mask']
            
            # 确保mask是2D的
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            
            # 确保尺寸匹配
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h))
            
            binary_mask = (mask > 0.5).astype(np.uint8)
            
            # 找轮廓
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 绘制彩色轮廓
            color = colors[i].astype(int).tolist()
            cv2.drawContours(edges, contours, -1, color, 2)
        
        return edges
    
    def generate_report(self, masks: List[Dict], prompts: List[str], output_path: Path) -> Dict:
        """生成详细报告"""
        
        # 计算统计
        coverages = [m['coverage'] for m in masks]
        
        stats = {
            'total_masks': len(masks),
            'successful_prompts': len(prompts),
            'avg_coverage': np.mean(coverages) if coverages else 0,
            'min_coverage': np.min(coverages) if coverages else 0,
            'max_coverage': np.max(coverages) if coverages else 0,
            'prompts': prompts[:20]  # 保存前20个成功的提示
        }
        
        # 保存JSON报告
        report_path = output_path / "segmentation_report.json"
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def create_html_showcase(self, output_path: Path, stats: Dict):
        """创建HTML展示页面"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SAM3 Best Segmentation Results</title>
            <meta charset="utf-8">
            <style>
                body {{
                    margin: 0;
                    font-family: -apple-system, sans-serif;
                    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
                    color: white;
                    min-height: 100vh;
                }}
                .header {{
                    text-align: center;
                    padding: 60px 20px;
                    background: rgba(0,0,0,0.3);
                }}
                h1 {{
                    font-size: 3.5em;
                    margin: 0;
                    background: linear-gradient(45deg, #f093fb, #f5576c);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }}
                .subtitle {{
                    font-size: 1.3em;
                    margin-top: 10px;
                    opacity: 0.9;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 40px 20px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 40px;
                }}
                .stat-card {{
                    background: rgba(255,255,255,0.1);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    padding: 30px;
                    text-align: center;
                    border: 1px solid rgba(255,255,255,0.2);
                }}
                .stat-number {{
                    font-size: 3em;
                    font-weight: bold;
                    color: #f5576c;
                }}
                .stat-label {{
                    margin-top: 10px;
                    opacity: 0.8;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .images-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 30px;
                }}
                .image-card {{
                    background: rgba(255,255,255,0.05);
                    border-radius: 20px;
                    overflow: hidden;
                    border: 1px solid rgba(255,255,255,0.1);
                }}
                .image-header {{
                    background: rgba(0,0,0,0.3);
                    padding: 20px;
                    font-size: 1.2em;
                    font-weight: bold;
                }}
                img {{
                    width: 100%;
                    display: block;
                }}
                .success-prompts {{
                    background: rgba(255,255,255,0.05);
                    border-radius: 20px;
                    padding: 30px;
                    margin-top: 40px;
                }}
                .prompt-list {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin-top: 20px;
                }}
                .prompt-tag {{
                    background: rgba(255,255,255,0.1);
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 SAM3 最强分割</h1>
                <div class="subtitle">使用 Segment Anything Model 3 达到最高分割质量</div>
            </div>
            
            <div class="container">
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{stats['total_masks']}</div>
                        <div class="stat-label">Total Masks</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['successful_prompts']}</div>
                        <div class="stat-label">Successful Prompts</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['avg_coverage']:.1f}%</div>
                        <div class="stat-label">Avg Coverage</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['max_coverage']:.1f}%</div>
                        <div class="stat-label">Max Coverage</div>
                    </div>
                </div>
                
                <div class="images-grid">
                    <div class="image-card">
                        <div class="image-header">🎨 Segmentation Overlay</div>
                        <img src="sam3_composite.png" alt="Composite">
                    </div>
                    <div class="image-card">
                        <div class="image-header">📐 Edge Detection</div>
                        <img src="sam3_edges.png" alt="Edges">
                    </div>
                </div>
                
                <div class="success-prompts">
                    <h2>✅ Successfully Detected Elements</h2>
                    <div class="prompt-list">
                        {"".join(f'<span class="prompt-tag">{p}</span>' for p in stats.get('prompts', [])[:20])}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        html_path = output_path / "sam3_showcase.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        # 自动打开
        import subprocess
        subprocess.run(["open", str(html_path)])


def main():
    """运行SAM3最强分割"""
    
    segmenter = SAM3BestSegmentation()
    result = segmenter.segment_with_best_prompts("01_输入图片/Ladygaga_2.jpg")
    
    return result


if __name__ == "__main__":
    main()
