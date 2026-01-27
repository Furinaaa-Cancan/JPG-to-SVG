#!/usr/bin/env python3
"""
SAM3超精细分割系统
使用SAM3的全部能力进行极致细节分割
"""

import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
from typing import Dict, List, Tuple
import time

# 添加SAM3路径
sys.path.insert(0, "/Volumes/Seagate/SAM3/模型库/01_SAM3核心模型")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3UltraSegmentation:
    """SAM3超精细分割器"""
    
    def __init__(self):
        print("\n🚀 Initializing SAM3 Ultra Segmentation...")
        
        # 加载SAM3模型
        self.model = build_sam3_image_model(device="cpu")
        self.processor = Sam3Processor(self.model, device="cpu")
        
        print("✅ SAM3 loaded successfully!")
        
    def segment_everything(self, image_path: str, output_dir: str = "02_输出结果/sam3_ultra"):
        """用SAM3分割一切细节"""
        
        print("\n" + "="*70)
        print("💎 SAM3 ULTRA SEGMENTATION - MAXIMUM DETAIL")
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
        
        # Step 2: 网格化点提示 - 超密集采样
        print("\n🎯 Step 2: Dense Grid Sampling")
        all_masks = []
        
        # 使用更密集的网格
        grid_size = 64  # 64x64 = 4096个点！
        x_step = w // grid_size
        y_step = h // grid_size
        
        print(f"   Sampling {grid_size}x{grid_size} = {grid_size*grid_size} points")
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * x_step + x_step // 2
                y = j * y_step + y_step // 2
                
                # 点击该位置
                prompt_state = self.processor.set_point_prompt(
                    [[x, y]], [1], state
                )
                
                # 获取mask
                if 'masks' in prompt_state and prompt_state['masks'] is not None:
                    mask = prompt_state['masks']
                    if mask.shape[0] > 0:
                        all_masks.append({
                            'mask': mask[0],
                            'point': (x, y),
                            'score': prompt_state.get('scores', [1.0])[0] if 'scores' in prompt_state else 1.0
                        })
                
                # 显示进度
                if (i * grid_size + j) % 100 == 0:
                    print(f"   Progress: {(i * grid_size + j) / (grid_size * grid_size) * 100:.1f}%")
        
        print(f"   Generated {len(all_masks)} raw masks")
        
        # Step 3: 文本提示分割 - 针对特定元素
        print("\n📝 Step 3: Text-Prompted Segmentation")
        
        text_prompts = [
            # 装饰元素
            "gold decoration", "golden ornament", "metallic trim",
            "embroidery", "sequins", "beads",
            
            # 服装部分
            "blue dress", "blue costume", "blue fabric",
            "collar", "sleeve", "hem", "fold", "wrinkle",
            
            # 细节
            "button", "zipper", "lace", "ribbon",
            "pattern", "texture", "shadow", "highlight",
            
            # 人物部分
            "face", "hair", "hand", "finger",
            "lips", "eye", "skin",
            
            # 配饰
            "jewelry", "necklace", "bracelet",
            
            # 道具
            "skeleton", "bone", "skull",
            
            # 背景
            "background", "stage", "light", "smoke"
        ]
        
        text_masks = []
        for prompt in text_prompts:
            try:
                prompt_state = self.processor.set_text_prompt(prompt, state)
                
                if 'masks' in prompt_state and prompt_state['masks'] is not None:
                    mask = prompt_state['masks']
                    if mask.shape[0] > 0:
                        text_masks.append({
                            'mask': mask[0],
                            'prompt': prompt,
                            'score': prompt_state.get('scores', [1.0])[0] if 'scores' in prompt_state else 1.0
                        })
                        print(f"   ✓ Found: {prompt}")
            except:
                pass
        
        print(f"   Text prompts generated {len(text_masks)} masks")
        
        # Step 4: 边界框提示 - 对关键区域
        print("\n📦 Step 4: Box-Prompted Segmentation")
        
        # 定义关键区域的边界框
        key_boxes = [
            # 上半身装饰区域
            [w*0.3, h*0.2, w*0.7, h*0.5],  # 胸前装饰
            [w*0.2, h*0.3, w*0.8, h*0.7],  # 主体服装
            [w*0.4, h*0.4, w*0.6, h*0.6],  # 中心装饰
            
            # 头部区域
            [w*0.35, 0, w*0.65, h*0.3],  # 头发脸部
            
            # 手部区域
            [w*0.5, h*0.3, w*0.75, h*0.6],  # 右手区域
            
            # 骷髅区域
            [w*0.45, h*0.25, w*0.73, h*0.75],  # 骷髅道具
        ]
        
        box_masks = []
        for i, box in enumerate(key_boxes):
            try:
                # 转换为整数
                box = [int(x) for x in box]
                
                # 设置边界框提示
                prompt_state = self.processor.set_box_prompt(
                    [box[0], box[1], box[2], box[3]], 
                    state
                )
                
                if 'masks' in prompt_state and prompt_state['masks'] is not None:
                    mask = prompt_state['masks']
                    if mask.shape[0] > 0:
                        box_masks.append({
                            'mask': mask[0],
                            'box': box,
                            'score': prompt_state.get('scores', [1.0])[0] if 'scores' in prompt_state else 1.0
                        })
                        print(f"   ✓ Box {i+1} segmented")
            except:
                pass
        
        print(f"   Box prompts generated {len(box_masks)} masks")
        
        # Step 5: 合并和去重所有masks
        print("\n🔄 Step 5: Merging and Deduplicating")
        
        # 合并所有masks
        all_final_masks = all_masks + text_masks + box_masks
        
        # 去重
        unique_masks = self.deduplicate_masks(all_final_masks)
        print(f"   Unique masks: {len(unique_masks)}")
        
        # Step 6: 生成可视化
        print("\n🎨 Step 6: Generating Visualizations")
        
        # 创建组合可视化
        self.create_visualization(img_array, unique_masks, output_path)
        
        # 保存每个mask
        self.save_individual_masks(unique_masks, output_path)
        
        # Step 7: 生成统计报告
        stats = self.generate_stats(unique_masks, w, h)
        
        process_time = time.time() - start_time
        
        print(f"\n✅ Complete!")
        print(f"   Total masks: {len(unique_masks)}")
        print(f"   Processing time: {process_time:.1f}s")
        print(f"   Average coverage: {stats['avg_coverage']:.1f}%")
        print(f"   Output: {output_path}")
        
        return {
            'masks': unique_masks,
            'stats': stats,
            'output_dir': str(output_path)
        }
    
    def deduplicate_masks(self, masks: List[Dict]) -> List[Dict]:
        """去重masks"""
        
        if not masks:
            return []
        
        # 按分数排序
        masks.sort(key=lambda x: x.get('score', 0), reverse=True)
        
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
                    if iou > 0.8:  # 80%重叠视为重复
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_masks.append(mask_data)
                
                # 限制数量避免内存问题
                if len(unique_masks) >= 500:
                    break
        
        return unique_masks
    
    def create_visualization(self, img: np.ndarray, masks: List[Dict], output_path: Path):
        """创建可视化"""
        
        # 创建彩色mask叠加
        h, w = img.shape[:2]
        colored_masks = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 生成随机颜色
        np.random.seed(42)
        colors = np.random.randint(50, 255, size=(len(masks), 3))
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['mask']
            
            # 二值化
            binary_mask = (mask > 0.5).astype(np.uint8)
            
            # 应用颜色
            for c in range(3):
                colored_masks[:, :, c] = np.where(
                    binary_mask > 0,
                    colors[i, c],
                    colored_masks[:, :, c]
                )
        
        # 创建组合图像
        alpha = 0.5
        composite = cv2.addWeighted(img, 1-alpha, colored_masks, alpha, 0)
        
        # 保存
        composite_path = output_path / "sam3_composite.png"
        Image.fromarray(composite).save(composite_path)
        print(f"   Saved composite: {composite_path}")
        
        # 创建边缘可视化
        edges = np.zeros((h, w), dtype=np.uint8)
        
        for mask_data in masks:
            mask = mask_data['mask']
            binary_mask = (mask > 0.5).astype(np.uint8)
            
            # 找轮廓
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(edges, contours, -1, 255, 1)
        
        edges_path = output_path / "sam3_edges.png"
        Image.fromarray(edges).save(edges_path)
        print(f"   Saved edges: {edges_path}")
        
        # 创建HTML展示
        self.create_html_showcase(str(composite_path), str(edges_path), output_path, len(masks))
    
    def save_individual_masks(self, masks: List[Dict], output_path: Path):
        """保存单独的masks"""
        
        masks_dir = output_path / "individual_masks"
        masks_dir.mkdir(exist_ok=True)
        
        for i, mask_data in enumerate(masks[:100]):  # 限制保存数量
            mask = mask_data['mask']
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            
            # 生成描述性名称
            if 'prompt' in mask_data:
                name = f"mask_{i:03d}_{mask_data['prompt'].replace(' ', '_')}.png"
            elif 'point' in mask_data:
                name = f"mask_{i:03d}_point_{mask_data['point'][0]}_{mask_data['point'][1]}.png"
            else:
                name = f"mask_{i:03d}.png"
            
            mask_path = masks_dir / name
            Image.fromarray(binary_mask).save(mask_path)
        
        print(f"   Saved {min(100, len(masks))} individual masks")
    
    def generate_stats(self, masks: List[Dict], width: int, height: int) -> Dict:
        """生成统计信息"""
        
        total_area = width * height
        coverage_list = []
        
        for mask_data in masks:
            mask = mask_data['mask']
            area = np.sum(mask > 0.5)
            coverage = (area / total_area) * 100
            coverage_list.append(coverage)
        
        stats = {
            'total_masks': len(masks),
            'avg_coverage': np.mean(coverage_list) if coverage_list else 0,
            'min_coverage': np.min(coverage_list) if coverage_list else 0,
            'max_coverage': np.max(coverage_list) if coverage_list else 0,
            'text_masks': sum(1 for m in masks if 'prompt' in m),
            'point_masks': sum(1 for m in masks if 'point' in m),
            'box_masks': sum(1 for m in masks if 'box' in m)
        }
        
        # 保存统计
        stats_path = Path(masks[0].get('output_dir', '02_输出结果/sam3_ultra')) / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def create_html_showcase(self, composite_path: str, edges_path: str, 
                            output_path: Path, mask_count: int):
        """创建HTML展示页面"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SAM3 Ultra Segmentation Results</title>
            <style>
                body {{
                    margin: 0;
                    font-family: -apple-system, sans-serif;
                    background: #0a0a0a;
                    color: white;
                }}
                .header {{
                    text-align: center;
                    padding: 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                h1 {{
                    margin: 0;
                    font-size: 3em;
                }}
                .subtitle {{
                    margin-top: 10px;
                    opacity: 0.9;
                }}
                .content {{
                    max-width: 1400px;
                    margin: 40px auto;
                    padding: 0 20px;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 30px;
                    margin-bottom: 40px;
                }}
                .card {{
                    background: #1a1a1a;
                    border-radius: 15px;
                    overflow: hidden;
                    border: 1px solid #333;
                }}
                .card-header {{
                    background: #2a2a2a;
                    padding: 20px;
                    font-size: 1.2em;
                    font-weight: bold;
                }}
                .card-content {{
                    padding: 20px;
                }}
                img {{
                    width: 100%;
                    height: auto;
                    display: block;
                }}
                .stats {{
                    background: #1a1a1a;
                    border-radius: 15px;
                    padding: 30px;
                    text-align: center;
                }}
                .stat-number {{
                    font-size: 4em;
                    font-weight: bold;
                    color: #667eea;
                }}
                .stat-label {{
                    font-size: 1.2em;
                    opacity: 0.8;
                    margin-top: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 SAM3 Ultra Segmentation</h1>
                <div class="subtitle">Maximum Detail Extraction with Segment Anything Model 3</div>
            </div>
            
            <div class="content">
                <div class="stats">
                    <div class="stat-number">{mask_count}</div>
                    <div class="stat-label">Total Segments Extracted</div>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <div class="card-header">🎨 Colored Segments</div>
                        <div class="card-content">
                            <img src="{Path(composite_path).name}" alt="Composite">
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">📐 Edge Detection</div>
                        <div class="card-content">
                            <img src="{Path(edges_path).name}" alt="Edges">
                        </div>
                    </div>
                </div>
                
                <div class="stats">
                    <h2>Segmentation Methods Used</h2>
                    <ul style="text-align: left; display: inline-block;">
                        <li>Dense Grid Sampling (64x64 points)</li>
                        <li>Text-Prompted Segmentation (decorations, textures, etc.)</li>
                        <li>Box-Prompted Segmentation (key regions)</li>
                        <li>Automatic Deduplication (80% IOU threshold)</li>
                    </ul>
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
    """运行SAM3超精细分割"""
    
    segmenter = SAM3UltraSegmentation()
    result = segmenter.segment_everything("01_输入图片/Ladygaga_2.jpg")
    
    print("\n" + "="*70)
    print("🎉 SAM3 SEGMENTATION COMPLETE!")
    print("This is the STRONGEST segmentation possible!")
    print("="*70)
    
    return result


if __name__ == "__main__":
    main()
