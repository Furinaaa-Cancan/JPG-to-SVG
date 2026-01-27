"""
姿势编辑器
流程：原图 + 新姿势 → ControlNet → 新图 → 矢量化
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from datetime import datetime
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import cv2


class PoseEditor:
    """姿势编辑器"""
    
    def __init__(self, device="mps"):
        self.device = device
        self.pipe = None
        
    def load_models(self):
        """加载ControlNet和SDXL"""
        print("Loading ControlNet OpenPose + SDXL...")
        
        # 加载ControlNet
        controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-openpose-sdxl-1.0",
            torch_dtype=torch.float32
        )
        
        # 加载SDXL + ControlNet
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float32
        ).to(self.device)
        
        print("✅ Models loaded!")
    
    def create_pose_image(self, keypoints: dict, size=(1024, 1024)) -> Image.Image:
        """
        从关键点创建姿势图像
        keypoints: 字典，包含身体各部位坐标
        """
        img = Image.new('RGB', size, 'black')
        draw = ImageDraw.Draw(img)
        
        # 骨架连接
        connections = [
            ('nose', 'neck'),
            ('neck', 'right_shoulder'), ('neck', 'left_shoulder'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('neck', 'right_hip'), ('neck', 'left_hip'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ]
        
        # 绘制骨架
        for start, end in connections:
            if start in keypoints and end in keypoints:
                p1 = keypoints[start]
                p2 = keypoints[end]
                if p1 and p2:
                    draw.line([p1, p2], fill='white', width=8)
        
        # 绘制关键点
        for name, pos in keypoints.items():
            if pos:
                x, y = pos
                r = 10
                draw.ellipse([x-r, y-r, x+r, y+r], fill='red')
        
        return img
    
    def extract_pose_from_image(self, img_path: str) -> Image.Image:
        """从图像提取姿势（简化版：使用边缘检测模拟）"""
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        # 使用Canny边缘检测作为简化的pose
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # 转为RGB
        pose_img = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
        
        return pose_img
    
    def edit_pose(
        self,
        original_image: str,
        pose_image: str = None,
        target_pose: dict = None,
        prompt: str = "a woman in elegant pose, high quality, detailed",
        negative_prompt: str = "blurry, distorted, ugly, deformed",
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 0.8,
        num_inference_steps: int = 30,
        output_dir: str = "output"
    ):
        """
        编辑姿势
        
        Args:
            original_image: 原始图像路径
            pose_image: 目标姿势图像路径（可选）
            target_pose: 目标姿势关键点（可选）
            prompt: 生成提示词
            strength: 变化强度
            controlnet_scale: ControlNet强度
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🎭 Pose Editor")
        print(f"{'='*60}")
        print(f"Input: {original_image}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}\n")
        
        # 加载原图
        orig_img = load_image(original_image)
        orig_img = orig_img.resize((1024, 1024))
        
        # 获取姿势图像
        if pose_image:
            pose_img = load_image(pose_image).resize((1024, 1024))
        elif target_pose:
            pose_img = self.create_pose_image(target_pose, (1024, 1024))
        else:
            # 从原图提取姿势
            pose_img = self.extract_pose_from_image(original_image)
            pose_img = pose_img.resize((1024, 1024))
        
        # 保存姿势图
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pose_path = output_path / f"pose_{timestamp}.png"
        pose_img.save(pose_path)
        print(f"📐 Pose image saved: {pose_path}")
        
        # 生成新图像
        print("🔄 Generating new pose...")
        
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pose_img,
            controlnet_conditioning_scale=controlnet_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images[0]
        
        # 保存结果
        result_path = output_path / f"new_pose_{timestamp}.png"
        result.save(result_path)
        print(f"✅ Result saved: {result_path}")
        
        # 打开结果
        import subprocess
        subprocess.run(["open", str(result_path)])
        
        return result_path


def main():
    """示例：手动指定新姿势"""
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    editor = PoseEditor(device=device)
    editor.load_models()
    
    # 原图
    original = "/Volumes/Seagate/SAM3/01_输入图片/Ladygaga_2.jpg"
    
    # 定义新姿势（坐标为1024x1024图像上的像素位置）
    # 示例：双手举起的姿势
    new_pose = {
        'nose': (512, 200),
        'neck': (512, 280),
        'right_shoulder': (400, 320),
        'left_shoulder': (624, 320),
        'right_elbow': (320, 200),  # 举起
        'left_elbow': (704, 200),   # 举起
        'right_wrist': (280, 100),  # 高举
        'left_wrist': (744, 100),   # 高举
        'right_hip': (450, 550),
        'left_hip': (574, 550),
        'right_knee': (430, 750),
        'left_knee': (594, 750),
        'right_ankle': (420, 950),
        'left_ankle': (604, 950),
    }
    
    editor.edit_pose(
        original_image=original,
        target_pose=new_pose,
        prompt="Lady Gaga in blue military costume, arms raised up, dramatic pose, high quality, detailed",
        strength=0.8,
        controlnet_scale=0.9,
        output_dir="/Volumes/Seagate/SAM3/13_SVG_Diffusion/output_pose"
    )


if __name__ == "__main__":
    main()
