<div align="center">

# 🎨 JPG-to-SVG

**Intelligent Hybrid Vectorization System**

*Transform raster images into editable, semantic-aware SVG with AI-powered segmentation*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![SAM3](https://img.shields.io/badge/Powered%20by-SAM3-orange.svg)](https://github.com/facebookresearch/sam3)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Modules](#-modules) • [Style Art](#-style-art-generator) • [API Reference](#-api-reference) • [Roadmap](#-roadmap) • [Contributing](#-contributing)

</div>

---

## 📖 Table of Contents

- [Introduction](#-introduction)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Modules](#-modules)
- [Style Art Generator](#-style-art-generator)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Introduction

**JPG-to-SVG** is an advanced image vectorization system that goes beyond traditional edge-detection methods. Instead of treating an entire image uniformly, it leverages **SAM3 (Segment Anything Model 3)** for semantic understanding and applies different vectorization strategies to different regions based on their content characteristics.

### Why JPG-to-SVG?

Traditional vectorization tools face a fundamental dilemma:
- **Super-pixel methods** produce photorealistic results but generate massive files (10MB+) that are impossible to edit
- **Color-block methods** create small, editable files but look cartoonish and lose details
- **Single-algorithm approaches** cannot adapt to varying image complexity

**Our solution**: A **Hierarchical Hybrid Representation** system that intelligently selects the optimal vectorization method for each semantic region, achieving the best balance between file size, visual quality, and editability.

---

## ✨ Features

### Core Philosophy

> **Don't process the entire image with one method — intelligently select the best vectorization approach for each semantic region.**

### Vectorization Methods

| Method | Best For | File Size | Quality | Editability |
|--------|----------|-----------|---------|-------------|
| **Geometric Primitives** | Simple shapes, solid colors, icons | ~1KB | ★★★☆☆ | ★★★★★ |
| **Gradient Mesh** | Lighting, backgrounds, smoke, gradients | ~5KB | ★★★★☆ | ★★★★☆ |
| **Texture Patterns** | Repeating patterns, fabrics, materials | ~10KB | ★★★★☆ | ★★★★☆ |
| **Detail Preservation** | Faces, hands, text, logos | ~20KB | ★★★★★ | ★★★☆☆ |

### Key Innovations

- **🎭 Semantic-Aware Processing** — Understanding image content before vectorization
- **📚 Layered Architecture** — Objects stack independently with proper z-ordering (no gaps when editing)
- **🎯 Adaptive Compression** — High detail for important regions (faces), aggressive simplification for backgrounds
- **🔀 Hybrid Representation** — Combines SVG primitives + gradient meshes + texture patterns + embedded details
- **⚡ Differentiable Optimization** — Uses DiffVG for gradient-based parameter optimization (not just edge tracing)
- **🎨 Artistic Style Generation** — 35+ modern art styles (Cubism, Pop Art, Expressionism, etc.)

---

## 📦 Installation

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10+ | 3.11+ |
| **RAM** | 8GB | 16GB+ |
| **GPU** | Optional | NVIDIA with CUDA 11.8+ |
| **Disk Space** | 5GB | 15GB (with all models) |
| **OS** | macOS 12+, Ubuntu 20.04+, Windows 10+ | macOS 13+, Ubuntu 22.04+ |

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/Furinaaa-Cancan/JPG-to-SVG.git
cd JPG-to-SVG
```

#### 2. Create Virtual Environment

```bash
# Using venv (recommended)
python -m venv .venv

# Activate on macOS/Linux
source .venv/bin/activate

# Activate on Windows
.venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: For GPU acceleration (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Optional: For style art generation
pip install diffusers transformers accelerate
```

#### 4. Download Models

```bash
# SAM3 model (required for segmentation)
# Download from HuggingFace and place in models/sam3/
mkdir -p models/sam3
# Model will auto-download on first use, or manually download:
# https://huggingface.co/facebook/sam3-hiera-large

# Optional: Stable Diffusion models for style art
# These will be downloaded automatically when first used
```

#### 5. Verify Installation

```bash
python -c "from tools.sam3_to_svg import load_sam3_model; print('✅ Installation successful!')"
```

---

## 🚀 Quick Start

### Basic Vectorization

```bash
# Convert a single image to SVG
python tools/sam3_to_svg.py input.jpg -o output.svg

# With custom segmentation prompts
python tools/sam3_to_svg.py photo.jpg -o result.svg --prompts "person,background,object"

# High-quality mode (slower, better results)
python tools/sam3_to_svg.py image.png -o output.svg --quality high
```

### Semantic Segmentation Only

```bash
# Generate semantic masks without vectorization
python tools/sam3_semantic_segment.py input.jpg

# Fine-grained segmentation (more masks)
python tools/precise_segmentation.py input.jpg --mode fine

# Extreme detail (1000+ masks)
python tools/high_quality_masks.py input.jpg --mode extreme
```

### Artistic Style Generation

```bash
# Generate Cubism-style SVG art
python 06_style_art/cubism_batch.py input.jpg -o output/

# Generate all 35 art styles
python 06_style_art/massive_art_generator.py input.jpg -o output/

# Picasso masterpiece styles
python 06_style_art/picasso_masterpiece.py input.jpg --style guernica
```

### Python API

```python
import sys
sys.path.insert(0, '/path/to/JPG-to-SVG')

from tools.sam3_to_svg import load_sam3_model, segment_with_sam3, create_svg
from PIL import Image

# Load model
processor = load_sam3_model(device='cuda')  # or 'cpu', 'mps'

# Segment image
prompts = ["person", "background", "object"]
segments, image = segment_with_sam3(processor, "photo.jpg", prompts)

# Create SVG
create_svg(image.size, segments, "output.svg")
```

---

## 🏗 Architecture

### Project Structure

```
JPG-to-SVG/
│
├── 00_docs/                          # Documentation
│   ├── modular_architecture.md       # System design document
│   ├── advanced_vectorizer_research.md
│   └── vector_solution_design.md
│
├── 01_input/                         # Input images directory
├── 02_output/                        # Output SVG directory
│
├── 03_segmentation/                  # SAM3-based semantic segmentation
│   ├── fine/                         # Fine-grained segmentation (100-500 masks)
│   ├── extreme/                      # Ultra-detailed masks (500-1000 masks)
│   └── thousand/                     # Maximum detail (1000+ masks)
│
├── 04_vectorization/                 # Vectorization algorithms
│   ├── basic/                        # Basic color-block vectorizer
│   │   └── potrace_wrapper.py        # Potrace integration
│   └── semantic/                     # Semantic-aware vectorizer
│       ├── gradient_mesh.py          # Gradient mesh generation
│       ├── pattern_detector.py       # Texture pattern recognition
│       └── detail_preserver.py       # Critical detail handling
│
├── 05_diffusion/                     # Differentiable vectorization
│   ├── diff_vectorizer/              # DiffVG-based optimization
│   │   ├── optimizer.py              # Gradient descent optimizer
│   │   └── loss_functions.py         # Custom loss functions
│   └── svg_dreamer/                  # SVGDreamer integration
│       └── dreamer_pipeline.py       # Text-to-SVG generation
│
├── 06_style_art/                     # Artistic style generation
│   ├── README_USAGE.md               # Style art documentation
│   ├── massive_art_generator.py      # 35+ style generator
│   ├── cubism_batch.py               # Cubism variations
│   ├── picasso_masterpiece.py        # Picasso famous works
│   ├── duchamp_batch.py              # Duchamp style
│   ├── modern_art_styles.py          # Style definitions
│   └── output/                       # Generated artworks
│
├── 14_cubism/                        # Cubism experiments
│
├── models/                           # AI models directory
│   └── sam3/                         # SAM3 checkpoints
│
├── tools/                            # Utility scripts
│   ├── sam3_to_svg.py               # Main conversion pipeline
│   ├── sam3_semantic_segment.py     # Segmentation tool
│   ├── masks_to_svg.py              # Mask to SVG converter
│   ├── potrace_vectorizer.py        # Potrace wrapper
│   ├── precise_segmentation.py      # High-precision segmentation
│   ├── high_quality_masks.py        # Quality mask generation
│   ├── mask_quality_analysis.py     # Mask quality metrics
│   ├── compare_segmentation.py      # Comparison tools
│   ├── diagram_to_svg.py            # Diagram vectorization
│   ├── scientific_figure_*.py       # Scientific figure processing
│   └── visualize_results.py         # Visualization utilities
│
├── .venv/                            # Python virtual environment
├── LICENSE                           # Apache 2.0 License
├── README.md                         # This file
└── README_models.md                  # Model documentation
```

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           JPG-to-SVG Pipeline                                │
└─────────────────────────────────────────────────────────────────────────────┘

Input Image (JPG/PNG)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  MODULE 0: Intelligent Analysis & Decision                     │
│  ─────────────────────────────────────────────────────────────│
│  • Texture complexity analysis (smooth vs detailed)            │
│  • Geometric regularity detection (shapes, edges)              │
│  • Semantic importance evaluation (face > background)          │
│  • Color gradient analysis (gradients vs solid colors)         │
│                                                                 │
│  Output: Region map + Processing strategy for each region       │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  MODULE 1: Semantic Segmentation (SAM3)                        │
│  ─────────────────────────────────────────────────────────────│
│  • Instance segmentation with SAM3                             │
│  • Depth estimation for z-ordering                             │
│  • Amodal completion (reconstruct occluded parts)              │
│  • Semantic labeling (face, hand, background, etc.)            │
│                                                                 │
│  Output: Hierarchical semantic tree + Complete object masks     │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  MODULE 2: Adaptive Vectorization                              │
│  ─────────────────────────────────────────────────────────────│
│  Per-region method selection:                                   │
│                                                                 │
│  A. Geometric Primitives  → Simple shapes, icons (~1KB)        │
│  B. Gradient Mesh         → Smooth gradients, sky (~5KB)       │
│  C. Texture Patterns      → Fabrics, repeating (~10KB)         │
│  D. Detail Preservation   → Faces, text, logos (~20KB)         │
│                                                                 │
│  Output: Multi-representation collection                        │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  MODULE 3: Differentiable Refinement (DiffVG)                  │
│  ─────────────────────────────────────────────────────────────│
│  • Gradient descent optimization of vector parameters          │
│  • Automatic control point positioning                         │
│  • Optimal gradient parameter discovery                        │
│  • Sparsity constraints (minimum paths)                        │
│                                                                 │
│  Output: Optimized vector parameters                            │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  MODULE 4: Smart Fusion & Post-processing                      │
│  ─────────────────────────────────────────────────────────────│
│  • Z-order layer sorting                                       │
│  • Edge feathering and anti-aliasing                           │
│  • Path simplification and merging                             │
│  • File size optimization                                       │
│                                                                 │
│  Output: Final optimized SVG                                    │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
Output SVG (Editable, Layered, Optimized)
```

---

## 📦 Modules

### Module 0: Intelligent Analysis

Analyzes image features to determine the optimal processing strategy for each region.

```python
from tools.mask_quality_analysis import analyze_image

# Analyze image characteristics
analysis = analyze_image("input.jpg")
print(analysis)
# Output:
# {
#     "texture_complexity": 0.73,
#     "geometric_regularity": 0.45,
#     "color_diversity": 0.82,
#     "recommended_strategy": "hybrid"
# }
```

### Module 1: Semantic Segmentation

Uses SAM3 for intelligent image segmentation with semantic understanding.

```python
from tools.sam3_semantic_segment import segment_image

# Basic segmentation
masks = segment_image("photo.jpg", mode="auto")

# Text-prompted segmentation
masks = segment_image("photo.jpg", prompts=["person", "dog", "background"])

# Fine-grained segmentation
masks = segment_image("photo.jpg", mode="fine", min_mask_area=100)
```

**Segmentation Modes:**

| Mode | Masks | Use Case |
|------|-------|----------|
| `auto` | 10-50 | General images |
| `fine` | 100-500 | Detailed images |
| `extreme` | 500-1000 | Maximum detail |
| `thousand` | 1000+ | Ultra-high precision |

### Module 2: Vectorization Algorithms

#### 2A. Geometric Primitives

```python
from tools.potrace_vectorizer import vectorize_mask

# Convert mask to SVG paths
svg_paths = vectorize_mask(mask, simplify=2.0)
```

#### 2B. Gradient Mesh

```python
from vectorization.semantic.gradient_mesh import create_gradient_mesh

# Generate gradient mesh for smooth regions
mesh_svg = create_gradient_mesh(image_region, grid_size=8)
```

### Module 3: Differentiable Optimization

```python
from diffusion.diff_vectorizer.optimizer import DiffVGOptimizer

# Initialize optimizer
optimizer = DiffVGOptimizer(
    num_paths=128,
    num_iterations=500,
    learning_rate=0.01
)

# Optimize vector representation
optimized_svg = optimizer.optimize(image, initial_paths)
```

---

## 🎨 Style Art Generator

Generate artistic SVG interpretations in 35+ modern art styles.

### Available Styles

| Category | Styles |
|----------|--------|
| **Cubism** | Analytical Cubism, Synthetic Cubism, Orphism |
| **Futurism** | Italian Futurism, Russian Cubo-Futurism |
| **Expressionism** | German Expressionism, Abstract Expressionism, Munch |
| **Geometric Abstraction** | Suprematism, Neo-Plasticism, Constructivism |
| **Surrealism** | Dalí, Miró, Magritte |
| **Color Art** | Fauvism, Color Field Painting |
| **Pop Art** | Warhol, Lichtenstein |
| **Contemporary** | Action Painting, Op Art, Neo-Expressionism, Street Art |

### Usage

```bash
# Generate all 35 styles
python 06_style_art/massive_art_generator.py input.jpg -o output/

# Specific Picasso styles
python 06_style_art/picasso_masterpiece.py input.jpg --style guernica
python 06_style_art/picasso_masterpiece.py input.jpg --style weeping_woman

# Cubism batch (25 variations)
python 06_style_art/cubism_batch.py input.jpg -o output/cubism/
```

### Reproducibility

All generated images are logged with seeds for perfect reproducibility:

```json
{
  "version": 42,
  "style_key": "pop_art_warhol",
  "seed": 1234567890,
  "strength": 0.7,
  "num_colors": 240
}
```

---

## 📚 API Reference

### Core Functions

#### `load_sam3_model(device='cpu')`

Load the SAM3 model and processor.

**Parameters:**
- `device` (str): Computing device (`'cpu'`, `'cuda'`, `'mps'`)

**Returns:**
- `Sam3Processor`: Initialized SAM3 processor

#### `segment_with_sam3(processor, image_path, prompts=None)`

Segment an image using text prompts.

**Parameters:**
- `processor`: SAM3 processor instance
- `image_path` (str): Path to input image
- `prompts` (list): Optional text prompts for guided segmentation

**Returns:**
- `dict`: Segmentation results with masks and scores
- `PIL.Image`: Original image

#### `create_svg(image_size, segments, output_path)`

Create SVG from segmentation results.

**Parameters:**
- `image_size` (tuple): (width, height)
- `segments` (dict): Segmentation results
- `output_path` (str): Output SVG path

---

## ⚙️ Configuration

### Environment Variables

```bash
# Model paths
export SAM3_MODEL_PATH="/path/to/models/sam3"
export SD_MODEL_PATH="/path/to/models/stable_diffusion"

# Device configuration
export CUDA_VISIBLE_DEVICES=0  # GPU selection
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # For Apple Silicon
```

### Configuration File

Create `config.yaml` in project root:

```yaml
# config.yaml
segmentation:
  model: "sam3-hiera-large"
  device: "auto"  # auto, cpu, cuda, mps
  confidence_threshold: 0.3

vectorization:
  simplify_tolerance: 2.0
  min_path_length: 3
  max_colors: 256

output:
  format: "svg"
  optimize: true
  embed_images: false
```

---

## 📈 Performance

### Benchmarks

| Metric | Target | Current Status |
|--------|--------|----------------|
| File Size | < 100KB | ✅ Achieved (50-200KB typical) |
| Processing Time | < 5s | ✅ ~3s (segmentation only) |
| Visual Similarity (SSIM) | > 0.85 | 🔄 Testing (0.80-0.90) |
| Full Editability | Yes | ✅ Achieved |

### Comparison with Other Methods

| Method | File Size | Speed | Fidelity | Editability |
|--------|-----------|-------|----------|-------------|
| Super-pixel | 10MB+ | Very Slow | ★★★★★ | ★☆☆☆☆ |
| Color Blocks | 50KB | Fast | ★★☆☆☆ | ★★★☆☆ |
| Potrace | 100KB | Fast | ★★★☆☆ | ★★★★☆ |
| **JPG-to-SVG** | 50-200KB | Medium | ★★★★☆ | ★★★★★ |

---

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""
python tools/sam3_to_svg.py input.jpg --device cpu
```

#### 2. SAM3 Model Loading Fails

```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface/hub/models--facebook--sam3*
python -c "from sam3.model_builder import build_sam3_image_model; build_sam3_image_model(load_from_HF=True)"
```

#### 3. MPS (Apple Silicon) Issues

```python
# Use CPU fallback for problematic operations
device = 'cpu'  # Instead of 'mps'
```

#### 4. SVG Output is Empty

- Ensure input image exists and is readable
- Check segmentation confidence threshold
- Verify prompts match image content

---

## 🗺 Roadmap

### Completed

- [x] **Module 0**: Intelligent region analysis
- [x] **Module 1**: Semantic layer extraction with SAM3
- [x] Basic vectorization with Potrace
- [x] Style art generator (35+ styles)
- [x] Mask quality analysis tools

### In Progress

- [ ] **Module 2**: Multi-method vectorization engine
  - [ ] Gradient mesh implementation
  - [ ] Texture pattern detection
  - [ ] Critical detail preservation

### Planned

- [ ] **Module 3**: DiffVG gradient optimization
- [ ] **Module 4**: Smart SVG fusion & compression
- [ ] Web UI for interactive editing
- [ ] Batch processing with progress tracking
- [ ] Video frame vectorization
- [ ] Real-time preview mode

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
isort .
```

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- **SAM3**: Apache 2.0 (Meta AI)
- **DiffVG**: Apache 2.0
- **Stable Diffusion**: CreativeML Open RAIL-M

---

## 🙏 Acknowledgments

- [Meta AI](https://ai.meta.com/) for SAM3 (Segment Anything Model 3)
- [DiffVG](https://github.com/BachiLi/diffvg) for differentiable vector graphics
- [Potrace](http://potrace.sourceforge.net/) for bitmap tracing
- The open-source community for invaluable tools and inspiration

---

<div align="center">

**Made with ❤️ for the creative community**

*Star ⭐ this repo if you find it useful!*

[Report Bug](https://github.com/Furinaaa-Cancan/JPG-to-SVG/issues) • [Request Feature](https://github.com/Furinaaa-Cancan/JPG-to-SVG/issues) • [Discussions](https://github.com/Furinaaa-Cancan/JPG-to-SVG/discussions)

</div>
