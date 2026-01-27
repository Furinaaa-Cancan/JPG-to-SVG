<div align="center">

# 🎨 JPG-to-SVG

**Intelligent Hybrid Vectorization System**

*Transform raster images into editable, semantic-aware SVG with AI-powered segmentation*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![SAM3](https://img.shields.io/badge/Powered%20by-SAM3-orange.svg)](https://github.com/facebookresearch/sam3)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Roadmap](#-roadmap)

</div>

---

## ✨ Features

> **Core Philosophy**: Don't process the entire image with one method — intelligently select the best vectorization approach for each semantic region.

| Method | Best For | File Size | Quality |
|--------|----------|-----------|---------|
| **Geometric Primitives** | Simple shapes, solid colors | ~1KB | ★★★☆☆ |
| **Gradient Mesh** | Lighting, backgrounds, smoke | ~5KB | ★★★★☆ |
| **Texture Patterns** | Repeating patterns, fabrics | ~10KB | ★★★★☆ |
| **Detail Preservation** | Faces, hands, text | ~20KB | ★★★★★ |

### 🔑 Key Innovations

- **Layered Architecture** — Objects stack independently (no gaps when editing)
- **Semantic Compression** — High detail for faces, aggressive simplification for backgrounds
- **Hybrid Representation** — SVG framework + critical details + reusable patterns

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- ~10GB disk space for models

### Setup

```bash
# Clone repository
git clone https://github.com/Furinaaa-Cancan/JPG-to-SVG.git
cd JPG-to-SVG

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download SAM3 model (required)
# Place checkpoint in models/sam3/
```

---

## 🚀 Quick Start

### Basic Usage

```bash
# Run the complete pipeline
python tools/sam3_to_svg.py input.jpg -o output.svg

# Semantic segmentation only
python tools/sam3_semantic_segment.py input.jpg

# Style art generation (Low-poly, Cubism, Pop Art)
python 06_style_art/lowpoly_art.py input.jpg --style cubism
```

### Python API

```python
from tools.sam3_to_svg import ImageToSVG

converter = ImageToSVG(model_path="models/sam3/checkpoint.pt")
svg_content = converter.convert("photo.jpg", quality="high")
svg_content.save("output.svg")
```

---

## 🏗 Architecture

```
JPG-to-SVG/
├── 03_segmentation/          # SAM3-based semantic segmentation
│   ├── fine/                 # Fine-grained segmentation
│   ├── extreme/              # Ultra-detailed masks
│   └── thousand/             # 1000+ mask generation
├── 04_vectorization/         # Vectorization algorithms
│   ├── basic/                # Basic color-block vectorizer
│   └── semantic/             # Semantic-aware vectorizer
├── 05_diffusion/             # Differentiable vectorization
│   ├── diff_vectorizer/      # Gradient-based optimization
│   └── svg_dreamer/          # SVGDreamer + DiffVG
├── 06_style_art/             # Artistic style generation
│   └── lowpoly_art.py        # Low-poly / Cubism / Pop Art
└── tools/                    # Utility scripts
```

### Processing Pipeline

```
Input Image
    │
    ▼
┌─────────────────────────────────────┐
│  Module 0: Intelligent Analysis     │  ← Texture complexity, geometry detection
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Module 1: Semantic Segmentation    │  ← SAM3 + depth ordering + amodal completion
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Module 2: Adaptive Vectorization   │  ← Per-region method selection
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Module 3: Differentiable Refine    │  ← DiffVG optimization (optional)
└─────────────────────────────────────┘
    │
    ▼
Output SVG (Editable, Layered)
```

---

## 🎯 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| File Size | < 100KB | In Progress |
| Processing Time | < 5s | ~3s (segmentation) |
| Visual Similarity | > 85% | Testing |
| Editability | Full | ✅ |

---

## � Tech Stack

| Component | Technology |
|-----------|------------|
| Segmentation | **SAM3** (Segment Anything Model 3) |
| Vectorization | **Potrace**, Custom algorithms |
| Differentiable Rendering | **DiffVG** |
| Texture Enhancement | **SDXL** (optional) |
| Image Processing | **OpenCV**, **NumPy**, **Pillow** |

---

## 🗺 Roadmap

- [x] **Module 0**: Intelligent region analysis
- [x] **Module 1**: Semantic layer extraction with SAM3
- [ ] **Module 2**: Multi-method vectorization engine
- [ ] **Module 3**: DiffVG gradient optimization
- [ ] **Module 4**: Smart SVG fusion & compression
- [ ] Web UI for interactive editing
- [ ] Batch processing support

---

## 🆚 Why This Approach?

| Traditional Method | Problem | Our Solution |
|-------------------|---------|--------------|
| Super-pixel | Huge files, not editable | High detail only where needed |
| Color blocks | Cartoon-like, unrealistic | Hybrid gradients + textures |
| Single algorithm | One-size-fits-none | Semantic-aware method selection |

---

## � License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for the creative community**

*Star ⭐ this repo if you find it useful!*

</div>
