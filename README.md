# 🎨 Intelligent Hybrid Vectorization System

## Project Structure

```
/Volumes/Seagate/SAM3/
├── 00_docs/                  # Documentation
├── 01_input/                 # Input images (Lady Gaga, etc.)
├── 02_output/                # Output results
├── 03_segmentation/          # SAM3 segmentation research
│   ├── fine/                 # Fine segmentation
│   ├── extreme/              # Extreme segmentation
│   └── thousand/             # 1000-level masks
├── 04_vectorization/         # Vectorization algorithms
│   ├── basic/                # Basic vectorizer
│   └── semantic/             # Semantic vectorizer
├── 05_diffusion/             # Diffusion-based SVG generation
│   ├── diff_vectorizer/      # Differentiable vectorizer
│   └── svg_dreamer/          # SVGDreamer + diffvg
├── 06_style_art/             # [ACTIVE] Style art generation
│   ├── lowpoly_art.py        # Low-poly / Cubism / Pop Art
│   └── output/               # Generated results
├── tools/                    # Utility scripts
├── archive/                  # Archived experiments
└── models/                   # AI models (~400GB)
    ├── sam3/                 # SAM3 segmentation model
    └── stable_diffusion/     # SD models (SDXL, SD3.5, etc.)
```

## 🚀 快速开始

### 测试整个系统
```bash
python test_modules.py
```

### 单独测试模块
```bash
# 模块1：语义分层提取
python 03_模块_Mask生成/module1_semantic_layer_extractor.py 01_输入图片/Ladygaga_2.jpg

# 模块0：智能分析
python 03_模块_Mask生成/module0_intelligent_analyzer.py 01_输入图片/Ladygaga_2.jpg
```

## 📋 核心理念

**不是用一种方法处理整张图，而是智能地为每个区域选择最适合的矢量化方法**

### 四种矢量化方法的智能选择：

| 方法 | 适用场景 | 文件大小 | 质量 | 示例 |
|------|---------|----------|------|------|
| **几何原语** | 简单形状、纯色 | 极小(~1KB) | ★★★ | 骷髅骨架 |
| **渐变网格** | 光影、背景 | 小(~5KB) | ★★★★ | 舞台烟雾 |
| **纹理Pattern** | 重复图案 | 中(~10KB) | ★★★★ | 服装纹理 |
| **关键细节** | 脸、手、文字 | 大(~20KB) | ★★★★★ | Lady Gaga的脸 |

## 🔬 模块详解

### 模块0：智能分析与决策 ✅
- **功能**：分析图像特征，为每个区域决定最佳矢量化方法
- **输入**：原始图像
- **输出**：处理策略映射
- **关键技术**：
  - 纹理复杂度分析
  - 几何规则性检测
  - 语义重要性评估
  - 颜色梯度分析

### 模块1：语义分层提取 ✅
- **功能**：不只是mask，而是理解层次结构
- **输入**：原始图像
- **输出**：带层次的语义masks
- **关键创新**：
  - **Amodal Completion**：补全被遮挡部分
  - **深度排序**：自动判断前后关系
  - **语义标注**：理解每个区域是什么

### 模块2：自适应表示生成（开发中）
- 根据分析结果生成不同类型的矢量表示
- DiffVG集成进行优化

### 模块3：可微分优化（待开发）
- 使用梯度下降优化矢量参数
- Score Distillation从SD模型提取知识

### 模块4：智能融合（待开发）
- 将多种表示无缝组合
- 文件大小优化

## 📊 Lady Gaga测试案例分析

### 预期处理策略：
```
背景（烟雾）     → 渐变网格法 (~5KB)
服装主体         → 色块+渐变 (~10KB)
骷髅道具         → 几何原语法 (~3KB)
Lady Gaga的脸    → 关键细节法 (~15KB)
手部             → 关键细节法 (~10KB)
服装装饰         → 纹理Pattern (~5KB)
-----------------------------------
预期总大小：      ~50KB
预期质量：        85%相似度
```

## 💡 核心创新

### 1. 分层而非拼图
- 传统：相邻色块拼接（移动一块留空洞）
- 我们：完整对象堆叠（可自由移动编辑）

### 2. 语义感知压缩
- 脸部：保持高精度（识别度关键）
- 背景：激进简化（不影响主体）
- 纹理：用Pattern（高效复用）

### 3. 混合表示
- 不是纯SVG或纯位图
- 而是"SVG框架 + 关键细节 + 纹理pattern"

## 🎯 性能目标

| 指标 | 目标值 | 当前状态 |
|------|--------|---------|
| 文件大小 | <100KB | 开发中 |
| 处理速度 | <5秒 | ~3秒(分割) |
| 视觉相似度 | >85% | 测试中 |
| 可编辑性 | 完全可编辑 | ✅ |

## 🔧 技术栈

- **SAM3**: 语义分割
- **SDXL**: 纹理优化（可选）
- **DiffVG**: 可微分渲染（待集成）
- **OpenCV**: 图像处理
- **NumPy**: 数值计算

## 📝 下一步计划

1. ✅ 模块0：智能分析
2. ✅ 模块1：语义分层  
3. ⏳ 模块2：矢量化实现
4. ⏳ 模块3：DiffVG优化
5. ⏳ 模块4：SVG生成

## 🤔 为什么这是第三条路？

| 方法 | 问题 | 我们的解决方案 |
|------|------|--------------|
| 超像素法 | 文件巨大，不可编辑 | 只在关键部位使用高精度 |
| 色块法 | 不逼真，像卡通 | 混合渐变和纹理 |
| 传统混合 | 缺乏智能 | 语义理解+自适应选择 |

**核心哲学：让每种方法在最适合的地方发光**
