# 🎨 Artistic Style SVG Generator

Transform photos into various modern art style SVG vector graphics.

## 📁 Core Scripts

| Script | Function |
|--------|----------|
| `picasso_masterpiece.py` | **Picasso Masterpiece Styles** (Guernica, Weeping Woman, etc. - 10 famous works) |
| `picasso_figurative.py` | Picasso Figurative Style (preserves human features) |
| `cubism_batch.py` | Cubism Batch Generation (25 variants) |
| `duchamp_batch.py` | Duchamp "Nude Descending a Staircase" Style |
| `massive_art_generator.py` | Massive Multi-Style Generator |
| `modern_art_styles.py` | Style Definition Library |
| `sd_to_svg.py` | SD Stylization + SVG Conversion |
| `svg_enhancer.py` | SVG Post-Processing Enhancement |

## 📂 Output Directory

```
output/
├── cubism/     # Cubism (30 images)
├── duchamp/    # Duchamp Style (28 images)
├── popart/     # Pop Art
└── lowpoly/    # Low Polygon
```

---

## 📋 Prerequisites

### 1. Ensure Python is Installed
Open terminal and check:
```bash
python --version
```
Should display Python 3.x.x

### 2. Install Required Libraries
Run in terminal:
```bash
pip install torch diffusers pillow opencv-python numpy svgwrite
```

### 3. Prepare Input Images
Place photos you want to convert in the `01_input` folder

---

## 🚀 How to Run

### Step 1: Open the Code File
Open `06_style_art/massive_art_generator.py` with any editor

### Step 2: Modify Configuration (in the main function at the bottom)

Find these lines (around line 380):

```python
def main():
    input_image = "/path/to/your/image.jpg"  # ← Change to your image path
    output_dir = "/path/to/output/directory"  # ← Change to your desired output location
    START_FROM = 1  # ← Start from which version (1 = from beginning)
```

**Parameters to modify:**
- `input_image`: Full path to your input image
- `output_dir`: Where to save generated SVGs
- `START_FROM`: Starting version number (can resume from checkpoint if interrupted)

### Step 3: Run the Program
In terminal, navigate to project directory and run:
```bash
cd /your/project/path/SAM3
python 06_style_art/massive_art_generator.py
```

---

## ⏱️ Processing Time

- Each image takes approximately 1-3 minutes
- 200 images total: 3-10 hours (depends on hardware)
- Progress is automatically saved; can resume after interruption

---

## 📁 Output Files Description

After completion, the output folder contains:

| File Type | Description |
|-----------|-------------|
| `art_v001_cubism_analytical.svg` | SVG Vector (primary output) |
| `art_v001_cubism_analytical_preview.png` | PNG Preview (for quick viewing) |
| `seed_log.json` | Seed Log (important for reproduction) |

---

## 🔄 Regenerating a Specific Image

### Method 1: Find Seed from Log
Open `seed_log.json`, locate the image you want to regenerate, note its parameters:
```json
{
  "version": 42,
  "style_key": "pop_art_warhol",
  "seed": 1234567890,
  "strength": 0.7,
  "num_colors": 240,
  ...
}
```

### Method 2: Regenerate Using Seed
Run in Python:
```python
from massive_art_generator import MassiveArtGenerator

generator = MassiveArtGenerator()
generator.load_sd()

# Regenerate using saved seed
generator.regenerate_single(
    image_path="your/image/path",
    output_dir="output/directory",
    version=42,           # Version number
    seed=1234567890,      # Seed from log
    style_key="pop_art_warhol",  # Style name
    strength=0.8,         # Adjustable
    num_colors=300,       # Increase for more colors
    simplify=0.0001       # Lower for more detail
)
```

---

## 🎭 Included Art Styles (35 Total)

### Cubism Series
- Analytical Cubism (Picasso, Braque)
- Synthetic Cubism (Picasso, Gris)
- Orphism (Delaunay)

### Futurism Series
- Italian Futurism (Boccioni, Balla)
- Russian Cubo-Futurism (Malevich)

### Expressionism Series
- German Expressionism (Kirchner, Nolde)
- Abstract Expressionism (de Kooning, Klein)
- Munch Expressionism (Munch)

### Geometric Abstraction Series
- Suprematism (Malevich)
- Neo-Plasticism (Mondrian)
- Constructivism (Lissitzky)

### Surrealism Series
- Dalí Surrealism
- Miró Surrealism
- Magritte Surrealism

### Color Series
- Fauvism (Matisse)
- Color Field Painting (Rothko)

### Pop Art Series
- Warhol Pop Art
- Lichtenstein Pop Art

### Contemporary Art Series
- Action Painting (Pollock)
- Op Art (Vasarely)
- Neo-Expressionism (Basquiat)
- Pointillism (Seurat)
- Art Nouveau (Mucha, Klimt)
- Bauhaus (Kandinsky, Klee)
- Minimalism (Judd)
- Conceptual Art (LeWitt)
- Digital Glitch Art
- Street Art (Banksy, KAWS)
- Photorealism (Close)
- Land Art (Smithson)
- Young British Artists (Hirst)
- Kinetic Art (Calder)
- Arte Povera
- Fluxus (Yoko Ono, Nam June Paik)

---

## ❓ FAQ

### Q: What if the program is interrupted?
A: Modify `START_FROM` to the next version number and rerun. For example, if stopped at v050, set `START_FROM = 51`

### Q: Output files are too small?
A: Adjust parameters:
- Increase `num_colors` (e.g., to 350)
- Lower `simplify` (e.g., to 0.00005)

### Q: Not enough memory?
A: The program automatically cleans memory. If still insufficient, reduce resolution (change 1536 to 1024 in code)

### Q: Want to generate different styles with the same seed?
A: Yes! Use `regenerate_single()` method, keep seed unchanged, just change style_key

---

## 📞 Need Help?

If you encounter issues, share the error message and we'll help you resolve it!
