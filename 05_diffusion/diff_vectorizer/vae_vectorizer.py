#!/usr/bin/env python3
"""
VAE潜在空间矢量化
利用SD的VAE在潜在空间进行智能分割
- 计算量小（潜在空间是原图1/64）
- 更语义化的分割（VAE学习了图像的语义表示）
- 文件更小
"""

import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import svgwrite
import time
from sklearn.cluster import KMeans
from diffusers import AutoencoderKL


class VAEVectorizer:
    """VAE潜在空间矢量化"""
    
    def __init__(self, model_path: str = None):
        print("\n🚀 Loading SD VAE...")
        
        # 使用本地SDXL的VAE
        if model_path is None:
            model_path = "/Volumes/Seagate/SAM3/模型库/02_StableDiffusion模型/基础模型/sdxl-base"
        
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True
        ).to(self.device)
        self.vae.eval()
        print(f"✅ VAE loaded on {self.device}")
    
    def vectorize(self, image_path: str, output_dir: str = "02_输出结果/vae_svg"):
        """VAE潜在空间矢量化"""
        
        print("\n" + "="*70)
        print("💎 VAE LATENT SPACE VECTORIZATION")
        print("="*70)
        
        start_time = time.time()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载图像
        img_pil = Image.open(image_path).convert("RGB")
        img_array = np.array(img_pil)
        orig_h, orig_w = img_array.shape[:2]
        
        print(f"\n📷 Original: {orig_w}x{orig_h}")
        
        # 调整到VAE要求的尺寸（8的倍数）
        new_w = (orig_w // 8) * 8
        new_h = (orig_h // 8) * 8
        img_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
        
        print(f"   Resized: {new_w}x{new_h}")
        
        # Step 1: 编码到潜在空间
        print("\n🔧 Step 1: Encoding to latent space...")
        latents = self.encode_image(img_resized)
        latent_h, latent_w = latents.shape[2], latents.shape[3]
        print(f"   Latent shape: {latents.shape} ({latent_w}x{latent_h})")
        
        # Step 2: 在潜在空间进行语义聚类
        print("\n🎯 Step 2: Semantic clustering in latent space...")
        n_clusters = 64  # 少量聚类，但语义更强
        cluster_map = self.cluster_latents(latents, n_clusters)
        
        # Step 3: 上采样到原始尺寸
        print("\n📐 Step 3: Upsampling cluster map...")
        cluster_map_full = cv2.resize(
            cluster_map.astype(np.float32),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(int)
        
        # Step 4: 提取每个聚类的区域和颜色
        print("\n🎨 Step 4: Extracting regions...")
        regions = self.extract_regions(cluster_map_full, img_array, n_clusters)
        print(f"   Extracted {len(regions)} regions")
        
        # Step 5: 生成SVG
        print("\n✨ Step 5: Generating SVG...")
        svg_path = output_path / "vae_vector.svg"
        stats = self.create_svg(regions, orig_w, orig_h, str(svg_path))
        
        # 对比HTML
        self.create_html(image_path, str(svg_path), output_path, stats)
        
        process_time = time.time() - start_time
        
        print(f"\n" + "="*70)
        print(f"✅ VAE VECTORIZATION COMPLETE!")
        print(f"   Paths: {stats['paths']}")
        print(f"   Size: {stats['size_kb']:.1f} KB")
        print(f"   Time: {process_time:.1f}s")
        print("="*70)
        
        import subprocess
        subprocess.run(["open", str(output_path / "result.html")])
        
        return stats
    
    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """将图像编码到潜在空间"""
        
        # 转换为tensor
        img_tensor = torch.from_numpy(np.array(image)).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        img_tensor = (img_tensor / 127.5) - 1.0  # 归一化到[-1, 1]
        img_tensor = img_tensor.to(self.device)
        
        # 编码
        latents = self.vae.encode(img_tensor).latent_dist.sample()
        
        return latents
    
    def cluster_latents(self, latents: torch.Tensor, n_clusters: int) -> np.ndarray:
        """在潜在空间进行聚类"""
        
        # latents shape: [1, 4, H, W]
        latent_np = latents.cpu().numpy()[0]  # [4, H, W]
        
        h, w = latent_np.shape[1], latent_np.shape[2]
        
        # 重塑为 [H*W, 4]
        features = latent_np.transpose(1, 2, 0).reshape(-1, 4)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=50)
        labels = kmeans.fit_predict(features)
        
        # 重塑回 [H, W]
        cluster_map = labels.reshape(h, w)
        
        return cluster_map
    
    def extract_regions(self, cluster_map: np.ndarray, img: np.ndarray, n_clusters: int) -> list:
        """提取每个聚类的区域"""
        
        regions = []
        h, w = cluster_map.shape
        
        for cid in range(n_clusters):
            # 创建mask
            mask = (cluster_map == cid).astype(np.uint8) * 255
            area = np.sum(mask > 0)
            
            if area < 100:
                continue
            
            # 提取颜色
            pixels = img[mask > 127]
            if len(pixels) > 0:
                color = np.mean(pixels, axis=0).astype(int)
                
                # 找连通组件
                n_labels, labeled = cv2.connectedComponents(mask)
                
                for lid in range(1, n_labels):
                    component_mask = (labeled == lid).astype(np.uint8) * 255
                    component_area = np.sum(component_mask > 0)
                    
                    if component_area > 50:
                        # 提取该组件的精确颜色
                        component_pixels = img[component_mask > 127]
                        if len(component_pixels) > 0:
                            component_color = np.mean(component_pixels, axis=0).astype(int)
                            
                            regions.append({
                                'mask': component_mask,
                                'color': f"#{component_color[0]:02x}{component_color[1]:02x}{component_color[2]:02x}",
                                'area': component_area
                            })
        
        return regions
    
    def create_svg(self, regions: list, width: int, height: int, output_path: str) -> dict:
        """创建SVG"""
        
        dwg = svgwrite.Drawing(output_path, size=(width, height))
        dwg.viewbox(0, 0, width, height)
        
        # 按面积排序
        regions.sort(key=lambda x: x['area'], reverse=True)
        
        paths = 0
        
        for region in regions:
            mask = region['mask']
            color = region['color']
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 30:
                    continue
                
                # 简化轮廓
                epsilon = 2.0
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 3:
                    points = approx.squeeze()
                    if points.ndim == 1:
                        points = points.reshape(-1, 2)
                    if len(points) < 3:
                        continue
                    
                    # 使用二次贝塞尔曲线平滑
                    path_d = self.create_smooth_path(points)
                    
                    dwg.add(dwg.path(d=path_d, fill=color, stroke="none"))
                    paths += 1
        
        dwg.save()
        
        return {
            'paths': paths,
            'size_kb': Path(output_path).stat().st_size / 1024
        }
    
    def create_smooth_path(self, points: np.ndarray) -> str:
        """创建平滑的贝塞尔曲线路径"""
        
        if len(points) < 3:
            return ""
        
        path_d = f"M{points[0][0]},{points[0][1]}"
        
        for i in range(1, len(points)):
            curr = points[i]
            prev = points[i - 1]
            
            # 计算控制点
            if i < len(points) - 1:
                next_pt = points[i + 1]
                cx = (prev[0] + curr[0] + next_pt[0]) / 3
                cy = (prev[1] + curr[1] + next_pt[1]) / 3
                path_d += f" Q{curr[0]},{curr[1]} {cx},{cy}"
            else:
                path_d += f" L{curr[0]},{curr[1]}"
        
        path_d += " Z"
        return path_d
    
    def create_html(self, original: str, svg: str, output_path: Path, stats: dict):
        """创建对比HTML"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>VAE Vectorization</title>
            <style>
                body {{ margin:0; background:#0a0a0a; color:#fff; font-family:sans-serif; }}
                .header {{ text-align:center; padding:50px; background:linear-gradient(135deg,#00d2ff,#3a7bd5); }}
                h1 {{ font-size:3em; margin:0; }}
                .subtitle {{ margin-top:10px; opacity:0.9; }}
                .stats {{ display:flex; justify-content:center; gap:40px; margin-top:20px; }}
                .stat {{ background:rgba(0,0,0,0.3); padding:15px 30px; border-radius:25px; }}
                .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; padding:40px; max-width:1600px; margin:0 auto; }}
                .card {{ background:#1a1a1a; border-radius:15px; overflow:hidden; }}
                .card-header {{ padding:15px; background:#2a2a2a; font-weight:bold; text-align:center; }}
                img, object {{ width:100%; display:block; }}
                .tech {{ text-align:center; padding:30px; background:#1a1a1a; margin:20px 40px; border-radius:15px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 VAE 潜在空间矢量化</h1>
                <div class="subtitle">利用Stable Diffusion VAE的语义理解能力</div>
                <div class="stats">
                    <span class="stat">📊 {stats['paths']} 路径</span>
                    <span class="stat">📦 {stats['size_kb']:.0f} KB</span>
                </div>
            </div>
            <div class="grid">
                <div class="card">
                    <div class="card-header">📷 原图</div>
                    <img src="../../{original}">
                </div>
                <div class="card">
                    <div class="card-header">✨ SVG (VAE)</div>
                    <object data="{Path(svg).name}" type="image/svg+xml"></object>
                </div>
            </div>
            <div class="tech">
                <h2>💡 技术原理</h2>
                <p>SD的VAE将图像压缩到4通道潜在空间（1/64大小）</p>
                <p>在潜在空间聚类 → 语义更强 + 计算量更小</p>
            </div>
        </body>
        </html>
        """
        with open(output_path / "result.html", 'w') as f:
            f.write(html)


def main():
    vectorizer = VAEVectorizer()
    return vectorizer.vectorize("01_输入图片/Ladygaga_2.jpg")


if __name__ == "__main__":
    main()
