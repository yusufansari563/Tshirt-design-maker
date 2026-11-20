# 👕 T-Shirt Design Maker & Vectorizer

A professional tool tailored for t-shirt businesses and garment decorators to transform low-resolution client raster images (PNG, JPG) into print-ready, infinitely scalable vector graphics (SVG, PDF, and 300 DPI Transparent PNG).

Designed for:
- **DTF (Direct-to-Film)**
- **DTG (Direct-to-Garment)**
- **Screen Printing (Film Positives / Spot Colors)**
- **Vinyl Cutters & Plotters (Stencils / Heat Transfer Vinyl)**

---

## Features

- **Local Neural AI Super-Resolution (Real-ESRGAN via ONNX Runtime):**
  - **Real-ESRGAN Anime 6B**: Specially tuned for vector art, logos, cartoons, typography, and graphic tees. Eliminates pixelation and staircasing while producing razor-sharp, anti-aliased curves without noisy halos.
  - **Real-ESRGAN x4plus**: Optimized for photorealistic apparel art, merchandise photos, and complex textures.
  - **Classical Lanczos + Bilateral Filter**: Lightweight non-AI fallback option for instantaneous processing.
  - **100% Local Inference**: Runs entirely offline on your device (CPU or GPU) with zero external API fees or data uploads.
  - **Tile-based Processing**: Automatically handles large images with padded tiles to prevent seams and memory limits.
  - **Alpha Bleed / Fringe Mitigation**: Telea inpainting across transparent boundaries prevents dark edge halos on garment prints.
- **AI Background Isolation (rembg / U2-Net):** Removes solid white or messy backgrounds with an optional toggle and fine-tuned edge alpha thresholding to prevent white ink bleed and halos.
- **Stacked Vector Curves (VTracer Rust Engine):** Converts images into clean Bezier paths without sub-pixel gaps between adjacent colors.
- **Dual Printing Modes:**
  - *Full Color (DTF / DTG / Digital)*: Preserves color palettes with stacked layers.
  - *Monochrome (Vinyl / Screen Printing)*: Generates clean 1-bit outlines for cutter plotters and silk screen separation.
- **Physical Garment Sizing & 300 DPI Export:** 
  - Standard presets (12"×16" Full Front, 10"×12" Chest, 4"×4" Pocket, or custom dimensions).
  - One-click downloads for **SVG**, **Vector PDF**, and **300 DPI Master PNG**.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```
This will automatically launch the interactive dashboard in your default web browser (default: `http://localhost:8501`).

### 3. Run Automated Tests
```bash
python test_pipeline.py
```

---

## File Structure

```
Tshirt-design-maker/
├── app.py                  # Streamlit Web User Interface with AI Engine controls
├── requirements.txt        # Python package dependencies
├── README.md               # User documentation & instructions
├── test_pipeline.py        # Automated test suite (pipeline, AI engines, edge cases)
├── models/                 # Cached ONNX neural model weights
│   ├── RealESRGAN_x4plus_anime_6B.onnx
│   └── RealESRGAN_x4plus.onnx
└── core/                   # Processing pipeline modules
    ├── __init__.py
    ├── bg_remover.py       # AI Background removal & alpha cutoff
    ├── enhancer.py         # Local AI Super-Resolution (Real-ESRGAN) & classical filters
    ├── vectorizer.py       # VTracer Rust-based vector curve generation
    └── exporter.py         # SVG, 300 DPI PNG & Vector PDF packaging
```
