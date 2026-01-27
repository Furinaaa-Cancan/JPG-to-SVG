#!/usr/bin/env python3
"""
测试脚本：用Lady Gaga图片测试模块0和模块1
"""

import sys
import os
sys.path.append("03_模块_Mask生成")

from module0_intelligent_analyzer import IntelligentAnalyzer
from module1_semantic_layer_extractor import SemanticLayerExtractor


def test_lady_gaga():
    """测试Lady Gaga图片"""
    
    image_path = "01_输入图片/Ladygaga_2.jpg"
    
    print("="*60)
    print("🎯 Testing JPG to SVG Modular System")
    print("="*60)
    
    # 模块1：语义分层提取
    print("\n📦 MODULE 1: Semantic Layer Extraction")
    print("-"*40)
    
    extractor = SemanticLayerExtractor(device="cpu")  # 使用CPU避免MPS bug
    layers_result = extractor.extract_semantic_layers(image_path)
    
    # 保存结果
    extractor.save_results(layers_result, "02_输出结果/masks")
    
    print(f"\n✅ Extracted {len(layers_result['layers'])} semantic layers")
    for layer in layers_result['layers']:
        print(f"  - Layer {layer.layer_id}: {layer.name}")
        print(f"    Semantic: {layer.semantic_label}, Z-order: {layer.z_order}")
    
    # 模块0：智能分析
    print("\n📦 MODULE 0: Intelligent Analysis")
    print("-"*40)
    
    analyzer = IntelligentAnalyzer()
    analysis = analyzer.analyze_image(image_path, layers_result['layers'])
    
    # 保存分析结果
    analyzer.save_analysis(analysis, "02_输出结果/analysis.json")
    
    print("\n📊 Analysis Results:")
    print(f"  Overall complexity: {analysis['global_features']['overall_complexity']:.2f}")
    
    # 打印每个区域的处理策略
    print("\n🎨 Vectorization Strategy for Each Layer:")
    for region in analysis['strategy']['region_strategies']:
        print(f"  Region {region['region_id']}:")
        print(f"    Method: {region['method']}")
        print(f"    Priority: {region['priority']:.2f}")
        print(f"    Est. size: {region['estimated_size']} KB")
    
    # 性能估算
    perf = analysis['performance_estimate']
    print("\n📈 Performance Estimates:")
    print(f"  Total file size: {perf['estimated_file_size_kb']} KB")
    print(f"  Processing time: {perf['estimated_processing_time_s']:.1f} seconds")
    print(f"  Quality score: {perf['estimated_quality_score']:.2%}")
    
    print("\n" + "="*60)
    print("✅ Test Complete!")
    print("="*60)
    
    return layers_result, analysis


if __name__ == "__main__":
    test_lady_gaga()
