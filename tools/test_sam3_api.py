#!/usr/bin/env python3
"""
测试SAM3的正确API
"""

import sys
sys.path.insert(0, "/Volumes/Seagate/SAM3/模型库/01_SAM3核心模型")

from PIL import Image
import numpy as np
import torch

# 测试SAM3的正确用法
print("Testing SAM3 API...")

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    # 加载模型
    print("Loading model...")
    model = build_sam3_image_model(device="cpu")
    processor = Sam3Processor(model, device="cpu")
    
    # 加载测试图像
    image_path = "01_输入图片/Ladygaga_2.jpg"
    image = Image.open(image_path).convert("RGB")
    print(f"Image loaded: {image.size}")
    
    # 设置图像
    print("\nTesting API methods:")
    state = processor.set_image(image)
    print(f"✓ set_image() returns: {type(state)}")
    
    # 测试文本提示
    prompt = "skeleton"
    prompt_state = processor.set_text_prompt(prompt, state)
    print(f"✓ set_text_prompt() returns: {type(prompt_state)}")
    
    # 查看processor有哪些方法
    print("\nAvailable methods in processor:")
    methods = [m for m in dir(processor) if not m.startswith('_')]
    for method in sorted(methods):
        print(f"  - {method}")
    
    # 尝试不同的方法获取mask
    print("\nTrying to get masks...")
    
    # 方法1：直接访问
    if hasattr(prompt_state, 'masks'):
        print("  Found masks attribute in prompt_state")
        masks = prompt_state.masks
    
    # 方法2：predict方法
    if hasattr(processor, 'predict'):
        print("  Found predict method")
        try:
            result = processor.predict(prompt_state)
            print(f"    predict() returns: {type(result)}")
            if isinstance(result, dict):
                print(f"    Keys in result: {list(result.keys())}")
        except Exception as e:
            print(f"    predict() failed: {e}")
    
    # 方法3：generate方法
    if hasattr(processor, 'generate'):
        print("  Found generate method")
        try:
            result = processor.generate(prompt_state)
            print(f"    generate() returns: {type(result)}")
        except Exception as e:
            print(f"    generate() failed: {e}")
    
    # 方法4：decode方法
    if hasattr(processor, 'decode'):
        print("  Found decode method")
        try:
            result = processor.decode(prompt_state)
            print(f"    decode() returns: {type(result)}")
        except Exception as e:
            print(f"    decode() failed: {e}")
    
    # 直接检查prompt_state的内容（它是dict）
    print("\nExamining prompt_state (dict)...")
    if isinstance(prompt_state, dict):
        print("  Keys in prompt_state:")
        for key, value in prompt_state.items():
            if isinstance(value, (list, tuple)):
                print(f"    - {key}: {type(value)} with length {len(value)}")
                if len(value) > 0 and hasattr(value[0], 'shape'):
                    print(f"      First item shape: {value[0].shape}")
            elif isinstance(value, np.ndarray):
                print(f"    - {key}: numpy array with shape {value.shape}")
            elif isinstance(value, torch.Tensor):
                print(f"    - {key}: torch tensor with shape {value.shape}")
            else:
                print(f"    - {key}: {type(value)}")
                
    # 检查是否有masks
    if 'masks' in prompt_state:
        masks = prompt_state['masks']
        print(f"\n  ✅ Found masks in prompt_state!")
        print(f"  Type: {type(masks)}")
        if isinstance(masks, (list, tuple)) and len(masks) > 0:
            print(f"  Number of masks: {len(masks)}")
            print(f"  First mask: {type(masks[0])}")
            if hasattr(masks[0], 'shape'):
                print(f"  First mask shape: {masks[0].shape}")
                
    # 也检查state的内容
    print("\nExamining state (dict)...")
    if isinstance(state, dict):
        print("  Keys in state:")
        for key in state.keys():
            print(f"    - {key}")
    
    # 尝试直接生成mask（基于之前的记忆）
    print("\nTrying direct mask generation...")
    if hasattr(prompt_state, 'segmentation') or hasattr(prompt_state, 'masks'):
        masks = getattr(prompt_state, 'masks', getattr(prompt_state, 'segmentation', None))
        if masks is not None:
            print(f"  ✅ Found masks: {type(masks)}")
            if isinstance(masks, (list, tuple)):
                print(f"  Number of masks: {len(masks)}")
                if len(masks) > 0:
                    print(f"  First mask shape: {masks[0].shape if hasattr(masks[0], 'shape') else 'N/A'}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
