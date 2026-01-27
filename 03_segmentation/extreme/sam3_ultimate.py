#!/usr/bin/env python3
"""
SAM3极限分割系统
- MPS GPU加速
- 自动mask生成
- 多尺度处理
- 稳定性评分
- 最大细节提取
"""

import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
import time
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# 添加SAM3路径
sys.path.insert(0, "/Volumes/Seagate/SAM3/模型库/01_SAM3核心模型")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3Ultimate:
    """SAM3极限分割器"""
    
    def __init__(self, use_mps: bool = True, num_workers: int = 12):
        print("\n🚀 Initializing SAM3 Ultimate...")
        
        # 选择设备
        if use_mps and torch.backends.mps.is_available():
            self.device = "mps"
            print("   Using MPS (Metal Performance Shaders) acceleration ⚡")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("   Using CUDA acceleration ⚡")
        else:
            self.device = "cpu"
            print("   Using CPU (will be slower)")
        
        self.num_workers = num_workers
        print(f"   CPU workers: {num_workers}")
        
        # 加载SAM3模型
        try:
            self.model = build_sam3_image_model(device=self.device)
            self.processor = Sam3Processor(self.model, device=self.device)
            print("✅ SAM3 loaded successfully!")
        except Exception as e:
            print(f"⚠️  MPS failed, falling back to CPU: {e}")
            self.device = "cpu"
            self.model = build_sam3_image_model(device="cpu")
            self.processor = Sam3Processor(self.model, device="cpu")
            print("✅ SAM3 loaded on CPU")
    
    def segment_everything_ultimate(self, image_path: str, output_dir: str = "02_输出结果/sam3_ultimate"):
        """使用SAM3的所有高级功能进行极限分割"""
        
        print("\n" + "="*70)
        print("💎 SAM3 ULTIMATE SEGMENTATION - MAXIMUM POWER")
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
        print(f"   Device: {self.device}")
        
        # Step 1: 设置图像
        print("\n🔧 Step 1: Setting image in SAM3")
        state = self.processor.set_image(img)
        
        # Step 2: 使用SAM3的自动mask生成功能
        print("\n🎯 Step 2: Automatic Mask Generation")
        auto_masks = self.automatic_mask_generation(state, img_array)
        print(f"   Generated {len(auto_masks)} automatic masks")
        
        # Step 3: 超详细文本提示 - 并行处理
        print("\n📝 Step 3: Parallel Text Prompting")
        text_masks = self.parallel_text_prompting(state, img_array)
        print(f"   Generated {len(text_masks)} text-prompted masks")
        
        # Step 4: 多尺度处理
        print("\n🔬 Step 4: Multi-scale Processing")
        scale_masks = self.multiscale_processing(img, img_array)
        print(f"   Generated {len(scale_masks)} multi-scale masks")
        
        # Step 5: 稳定性评分过滤
        print("\n⭐ Step 5: Stability Score Filtering")
        all_masks = auto_masks + text_masks + scale_masks
        filtered_masks = self.filter_by_stability(all_masks)
        print(f"   Filtered to {len(filtered_masks)} high-quality masks")
        
        # Step 6: 智能去重
        print("\n🔄 Step 6: Intelligent Deduplication")
        unique_masks = self.smart_deduplication(filtered_masks)
        print(f"   Final unique masks: {len(unique_masks)}")
        
        # Step 7: 生成可视化
        print("\n🎨 Step 7: Generating Visualizations")
        self.save_all_visualizations(img_array, unique_masks, output_path)
        
        # Step 8: 生成报告
        stats = self.generate_ultimate_report(unique_masks, output_path)
        
        process_time = time.time() - start_time
        
        print(f"\n" + "="*70)
        print(f"✅ ULTIMATE SEGMENTATION COMPLETE!")
        print(f"   Total masks: {len(unique_masks)}")
        print(f"   Processing time: {process_time:.1f}s")
        print(f"   Speed: {len(unique_masks)/process_time:.1f} masks/sec")
        print(f"   Device: {self.device}")
        print("="*70)
        
        # 自动打开展示
        import subprocess
        subprocess.run(["open", str(output_path / "ultimate_showcase.html")])
        
        return {
            'masks': unique_masks,
            'stats': stats,
            'time': process_time
        }
    
    def automatic_mask_generation(self, state, img_array: np.ndarray) -> list:
        """使用SAM3的自动mask生成功能"""
        
        masks = []
        h, w = img_array.shape[:2]
        
        # 使用不同的网格密度
        grid_sizes = [16, 32, 48]  # 多种网格密度
        
        for grid_size in grid_sizes:
            print(f"   Grid {grid_size}x{grid_size}...")
            
            x_step = w // grid_size
            y_step = h // grid_size
            
            # 批量采样点
            sample_points = []
            for i in range(grid_size):
                for j in range(grid_size):
                    x = i * x_step + x_step // 2
                    y = j * y_step + y_step // 2
                    sample_points.append((x, y))
            
            # 每次处理一批点
            batch_size = 50
            for batch_idx in range(0, len(sample_points), batch_size):
                batch = sample_points[batch_idx:batch_idx + batch_size]
                
                for x, y in batch:
                    try:
                        prompt_state = self.processor.set_text_prompt(f"object at {x},{y}", state)
                        
                        if prompt_state and 'masks' in prompt_state:
                            mask_data = prompt_state['masks']
                            if mask_data is not None and hasattr(mask_data, 'shape'):
                                if mask_data.shape[0] > 0:
                                    mask = mask_data[0] if len(mask_data.shape) > 2 else mask_data
                                    
                                    if hasattr(mask, 'cpu'):
                                        mask = mask.cpu().numpy()
                                    
                                    masks.append({
                                        'mask': mask,
                                        'point': (x, y),
                                        'grid': grid_size,
                                        'score': 1.0
                                    })
                    except:
                        continue
        
        return masks
    
    def parallel_text_prompting(self, state, img_array: np.ndarray) -> list:
        """并行文本提示处理"""
        
        # 超详细的提示词
        prompts = self.get_comprehensive_prompts()
        
        # 使用多线程并行处理
        masks = []
        
        def process_prompt(prompt):
            try:
                prompt_state = self.processor.set_text_prompt(prompt, state)
                
                if prompt_state and 'masks' in prompt_state:
                    mask_data = prompt_state['masks']
                    if mask_data is not None and hasattr(mask_data, 'shape'):
                        if mask_data.shape[0] > 0:
                            mask = mask_data[0] if len(mask_data.shape) > 2 else mask_data
                            
                            if hasattr(mask, 'cpu'):
                                mask = mask.cpu().numpy()
                            
                            return {
                                'mask': mask,
                                'prompt': prompt,
                                'score': 1.0
                            }
            except:
                pass
            return None
        
        # 并行处理（文本提示可以并行）
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_prompt, prompts))
        
        masks = [r for r in results if r is not None]
        
        print(f"   Parallel processing: {len(masks)}/{len(prompts)} successful")
        
        return masks
    
    def get_comprehensive_prompts(self) -> list:
        """获取超全面的提示词列表"""
        
        return [
            # 服装装饰 - 超详细
            "gold embroidery", "golden thread", "gold metallic trim",
            "embroidered pattern", "decorative embroidery",
            "sequins", "sparkles", "glitter", "shiny decorations",
            "beads", "pearl beads", "jewelry beads",
            "lace trim", "lace pattern", "delicate lace",
            "ribbon detail", "bow decoration", "tassel",
            "button", "decorative button", "metallic button",
            "zipper", "buckle", "clasp", "fastener",
            "belt detail", "waist decoration",
            
            # 服装结构 - 极详细
            "blue velvet dress", "royal blue costume", "cobalt blue fabric",
            "dress bodice", "corset", "chest piece",
            "sleeve detail", "puffed sleeve", "shoulder detail",
            "collar decoration", "neckline", "neck trim",
            "skirt fold", "dress hem", "bottom trim",
            "fabric wrinkle", "cloth crease", "fold line",
            "shadow on dress", "highlight on fabric",
            "dress texture", "velvet texture", "fabric weave",
            
            # 人物细节 - 最详细
            "face", "facial skin", "facial features",
            "blonde wavy hair", "curly hair", "hair strand",
            "eye", "eyelash", "eyebrow", "eyelid",
            "nose", "nostril", "nose bridge",
            "mouth", "lips", "red lipstick", "teeth",
            "chin", "jaw", "cheek", "cheekbone",
            "forehead", "temple",
            "ear", "earring",
            "neck", "throat", "collarbone",
            "hand", "palm", "finger", "fingernail",
            "wrist", "arm", "elbow", "shoulder",
            "skin", "skin tone", "skin texture",
            
            # 骷髅道具 - 完整分割
            "skeleton", "full skeleton", "complete skeleton",
            "skull", "skull head", "skull face", "skull teeth",
            "rib cage", "ribs", "rib bones",
            "spine", "vertebrae", "backbone",
            "arm bones", "leg bones", "hand bones",
            "bone", "white bone", "skeletal bone",
            "bone joint", "bone connection",
            "skeleton prop", "prop skeleton", "stage prop",
            
            # 背景元素
            "background", "blue background", "gradient background",
            "stage background", "backdrop",
            "light", "lighting", "spotlight", "stage light",
            "smoke", "fog", "mist", "haze effect",
            "shadow area", "dark region",
            
            # 颜色区域 - 精确
            "white region", "bright white area",
            "black region", "dark black area",
            "blue region", "deep blue area",
            "gold region", "golden area",
            "red region", "pink area",
            
            # 特殊效果
            "reflection", "shiny surface", "glossy area",
            "matte surface", "texture detail",
            "edge", "boundary", "outline", "contour",
            "transition area", "gradient transition"
        ]
    
    def multiscale_processing(self, img: Image.Image, img_array: np.ndarray) -> list:
        """多尺度处理 - 不同分辨率捕捉不同细节"""
        
        masks = []
        scales = [0.5, 0.75, 1.0, 1.25]  # 多个缩放级别
        
        for scale in scales:
            if scale == 1.0:
                continue  # 原始尺寸已经处理过了
            
            print(f"   Processing at {scale}x scale...")
            
            # 缩放图像
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            scaled_img = img.resize((new_w, new_h), Image.LANCZOS)
            
            # 处理缩放后的图像
            try:
                state = self.processor.set_image(scaled_img)
                
                # 在缩放图像上进行采样
                grid_size = 24
                x_step = new_w // grid_size
                y_step = new_h // grid_size
                
                for i in range(0, grid_size, 3):  # 稀疏采样
                    for j in range(0, grid_size, 3):
                        x = i * x_step + x_step // 2
                        y = j * y_step + y_step // 2
                        
                        try:
                            prompt_state = self.processor.set_text_prompt(f"region at {x},{y}", state)
                            
                            if prompt_state and 'masks' in prompt_state:
                                mask_data = prompt_state['masks']
                                if mask_data is not None and hasattr(mask_data, 'shape'):
                                    if mask_data.shape[0] > 0:
                                        mask = mask_data[0] if len(mask_data.shape) > 2 else mask_data
                                        
                                        if hasattr(mask, 'cpu'):
                                            mask = mask.cpu().numpy()
                                        
                                        # 缩放回原始大小
                                        mask_resized = cv2.resize(
                                            mask.astype(np.float32),
                                            (img.width, img.height),
                                            interpolation=cv2.INTER_LINEAR
                                        )
                                        
                                        masks.append({
                                            'mask': mask_resized,
                                            'scale': scale,
                                            'score': 1.0
                                        })
                        except:
                            continue
            except:
                continue
        
        return masks
    
    def filter_by_stability(self, masks: list) -> list:
        """根据稳定性评分过滤mask"""
        
        filtered = []
        
        for mask_data in masks:
            mask = mask_data['mask']
            
            # 计算稳定性指标
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            
            # 1. 面积不能太小
            area = np.sum(mask > 0.5)
            if area < 100:
                continue
            
            # 2. 计算紧凑度
            if area > 0:
                perimeter = self.calculate_perimeter(mask > 0.5)
                compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # 过滤太分散的mask
                if compactness < 0.01:
                    continue
            
            # 3. 计算填充度
            bbox = self.get_bbox(mask > 0.5)
            if bbox:
                x1, y1, x2, y2 = bbox
                bbox_area = (x2 - x1) * (y2 - y1)
                fill_ratio = area / bbox_area if bbox_area > 0 else 0
                
                # 过滤填充度太低的
                if fill_ratio < 0.2:
                    continue
            
            # 计算综合评分
            score = compactness * 0.5 + fill_ratio * 0.5
            mask_data['stability_score'] = score
            
            filtered.append(mask_data)
        
        # 按评分排序
        filtered.sort(key=lambda x: x.get('stability_score', 0), reverse=True)
        
        return filtered
    
    def calculate_perimeter(self, binary_mask: np.ndarray) -> float:
        """计算mask周长"""
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            return cv2.arcLength(contours[0], True)
        return 0
    
    def get_bbox(self, binary_mask: np.ndarray) -> tuple:
        """获取bounding box"""
        coords = np.argwhere(binary_mask)
        if len(coords) > 0:
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            return (x1, y1, x2, y2)
        return None
    
    def smart_deduplication(self, masks: list) -> list:
        """智能去重 - 考虑多种因素"""
        
        if not masks:
            return []
        
        # 按稳定性评分排序
        masks.sort(key=lambda x: x.get('stability_score', 0), reverse=True)
        
        unique_masks = []
        
        for mask_data in masks:
            mask = mask_data['mask']
            
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            
            # 检查与已有mask的重叠
            is_duplicate = False
            
            for unique in unique_masks:
                unique_mask = unique['mask']
                if len(unique_mask.shape) > 2:
                    unique_mask = unique_mask.squeeze()
                
                # 确保尺寸匹配
                if mask.shape != unique_mask.shape:
                    unique_mask = cv2.resize(unique_mask, (mask.shape[1], mask.shape[0]))
                
                # 计算IOU
                intersection = np.logical_and(mask > 0.5, unique_mask > 0.5).sum()
                union = np.logical_or(mask > 0.5, unique_mask > 0.5).sum()
                
                if union > 0:
                    iou = intersection / union
                    if iou > 0.8:  # 80%重叠
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_masks.append(mask_data)
                
                # 限制最大数量
                if len(unique_masks) >= 200:
                    break
        
        return unique_masks
    
    def save_all_visualizations(self, img: np.ndarray, masks: list, output_path: Path):
        """保存所有可视化"""
        
        # 保存单独的masks
        masks_dir = output_path / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        for i, mask_data in enumerate(masks[:100]):
            mask = mask_data['mask']
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            
            # 生成描述性文件名
            score = mask_data.get('stability_score', 0)
            prompt = mask_data.get('prompt', f'auto_{i}')[:30].replace(' ', '_')
            
            filename = f"{i:03d}_{prompt}_score{score:.2f}.png"
            Image.fromarray(binary_mask).save(masks_dir / filename)
        
        # 创建彩色组合
        composite = self.create_colorful_composite(img, masks)
        Image.fromarray(composite).save(output_path / "ultimate_composite.png")
        
        # 创建边缘图
        edges = self.create_detailed_edges(masks, img.shape[:2])
        Image.fromarray(edges).save(output_path / "ultimate_edges.png")
        
        print(f"   Saved visualizations to {output_path}")
    
    def create_colorful_composite(self, img: np.ndarray, masks: list) -> np.ndarray:
        """创建彩色组合"""
        
        h, w = img.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        
        np.random.seed(42)
        colors = np.random.rand(len(masks), 3) * 255
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['mask']
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h))
            
            binary_mask = (mask > 0.5)
            
            for c in range(3):
                overlay[:, :, c] += binary_mask * colors[i, c] * 0.3
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        composite = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        
        return composite
    
    def create_detailed_edges(self, masks: list, shape: tuple) -> np.ndarray:
        """创建详细边缘图"""
        
        h, w = shape
        edges = np.zeros((h, w, 3), dtype=np.uint8)
        
        np.random.seed(42)
        colors = np.random.rand(len(masks), 3) * 255
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['mask']
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h))
            
            binary_mask = (mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            color = colors[i].astype(int).tolist()
            cv2.drawContours(edges, contours, -1, color, 2)
        
        return edges
    
    def generate_ultimate_report(self, masks: list, output_path: Path) -> dict:
        """生成终极报告"""
        
        stats = {
            'total_masks': len(masks),
            'device': self.device,
            'workers': self.num_workers,
            'avg_stability': np.mean([m.get('stability_score', 0) for m in masks]),
            'generation_methods': {}
        }
        
        # 统计生成方法
        for mask in masks:
            if 'prompt' in mask:
                method = 'text_prompt'
            elif 'grid' in mask:
                method = f'auto_grid_{mask["grid"]}'
            elif 'scale' in mask:
                method = f'multiscale_{mask["scale"]}'
            else:
                method = 'other'
            
            stats['generation_methods'][method] = stats['generation_methods'].get(method, 0) + 1
        
        # 保存JSON
        with open(output_path / "ultimate_report.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 创建HTML展示
        self.create_ultimate_html(output_path, stats)
        
        return stats
    
    def create_ultimate_html(self, output_path: Path, stats: dict):
        """创建终极HTML展示"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SAM3 Ultimate Segmentation</title>
            <meta charset="utf-8">
            <style>
                body {{
                    margin: 0;
                    font-family: -apple-system, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }}
                .header {{
                    text-align: center;
                    padding: 60px 20px;
                    background: rgba(0,0,0,0.3);
                }}
                h1 {{
                    font-size: 4em;
                    margin: 0;
                    text-shadow: 0 0 20px rgba(255,255,255,0.5);
                }}
                .subtitle {{
                    font-size: 1.5em;
                    margin-top: 10px;
                }}
                .container {{
                    max-width: 1600px;
                    margin: 40px auto;
                    padding: 0 20px;
                }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 40px;
                }}
                .stat-card {{
                    background: rgba(255,255,255,0.1);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    padding: 30px;
                    text-align: center;
                }}
                .stat-number {{
                    font-size: 3em;
                    font-weight: bold;
                    color: #FFD700;
                }}
                .stat-label {{
                    margin-top: 10px;
                    text-transform: uppercase;
                    letter-spacing: 2px;
                }}
                .images {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 30px;
                }}
                .image-card {{
                    background: rgba(0,0,0,0.3);
                    border-radius: 20px;
                    overflow: hidden;
                }}
                .image-header {{
                    padding: 20px;
                    font-size: 1.3em;
                    font-weight: bold;
                }}
                img {{
                    width: 100%;
                    display: block;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>⚡ SAM3 ULTIMATE</h1>
                <div class="subtitle">Maximum Power Segmentation</div>
            </div>
            
            <div class="container">
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number">{stats['total_masks']}</div>
                        <div class="stat-label">Total Masks</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['device'].upper()}</div>
                        <div class="stat-label">Device</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['workers']}</div>
                        <div class="stat-label">CPU Workers</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['avg_stability']:.2f}</div>
                        <div class="stat-label">Avg Stability</div>
                    </div>
                </div>
                
                <div class="images">
                    <div class="image-card">
                        <div class="image-header">🎨 Ultimate Composite</div>
                        <img src="ultimate_composite.png" alt="Composite">
                    </div>
                    <div class="image-card">
                        <div class="image-header">📐 Ultimate Edges</div>
                        <img src="ultimate_edges.png" alt="Edges">
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path / "ultimate_showcase.html", 'w') as f:
            f.write(html_content)


def main():
    """运行SAM3极限分割"""
    
    # 使用MPS加速和12核CPU
    segmenter = SAM3Ultimate(use_mps=True, num_workers=12)
    
    result = segmenter.segment_everything_ultimate("01_输入图片/Ladygaga_2.jpg")
    
    return result


if __name__ == "__main__":
    main()
