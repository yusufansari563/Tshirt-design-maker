"""
Automated unit tests for T-Shirt Design Maker core pipeline,
including local AI Super-Resolution engines (Real-ESRGAN Anime & General)
and classical filters.
"""

import os
from PIL import Image, ImageDraw
import numpy as np
from core.bg_remover import remove_background
from core.enhancer import (
    enhance_raster_image,
    AVAILABLE_ENGINES,
    get_available_engines,
    is_model_downloaded
)
from core.vectorizer import vectorize_image
from core.exporter import export_print_package

def create_test_graphic():
    img = Image.new("RGBA", (300, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Circle badge
    draw.ellipse([20, 20, 280, 280], fill="#1E3A8A", outline="#F59E0B", width=6)
    draw.ellipse([50, 50, 250, 250], fill="#FFFFFF", outline="#1E3A8A", width=4)
    # Shapes
    draw.rectangle([90, 130, 210, 180], fill="#EF4444")
    draw.polygon([(150, 60), (190, 120), (110, 120)], fill="#F59E0B")
    return img

def run_test():
    print("[1/5] Creating synthetic test graphic...")
    test_img = create_test_graphic()
    test_img.save("test_input.png")

    print("[2/5] Testing background removal...")
    no_bg_img = remove_background(test_img, alpha_threshold=20)
    assert no_bg_img.mode == "RGBA"
    no_bg_img.save("test_no_bg.png")

    print("[3/5] Testing pre-vector enhancement...")
    enhanced = enhance_raster_image(no_bg_img, upscale_factor=2.0)
    assert enhanced.width == 600
    assert enhanced.height == 600

    print("[4/5] Testing VTracer vectorization...")
    svg_str = vectorize_image(enhanced, colormode="color", filter_speckle=4)
    assert "<svg" in svg_str and "</svg>" in svg_str
    print(f"      Generated SVG string length: {len(svg_str)} chars")

    print("[5/5] Testing print package export (SVG, 300 DPI PNG, Vector PDF)...")
    pkg = export_print_package(
        svg_content=svg_str,
        master_image=enhanced,
        target_width_inches=10.0,
        target_height_inches=12.0,
        dpi=300
    )
    
    assert len(pkg["svg"]) > 0
    assert len(pkg["png"]) > 0
    assert len(pkg["pdf"]) > 0

    with open("test_output.svg", "wb") as f:
        f.write(pkg["svg"])
    with open("test_output_300dpi.png", "wb") as f:
        f.write(pkg["png"])
    with open("test_output.pdf", "wb") as f:
        f.write(pkg["pdf"])

    print("\n[SUCCESS] ALL PIPELINE STAGES PASSED SUCCESSFULLY!")
    print("      - test_output.svg generated")
    print("      - test_output_300dpi.png generated (3000 x 3600 px)")
    print("      - test_output.pdf generated (10 x 12 inches)")


def test_ai_engines():
    print("\n--- Testing All Available Enhancement Engines ---")
    engines = get_available_engines()
    assert "realesrgan_anime" in engines
    assert "realesrgan_general" in engines
    assert "lanczos" in engines

    # Test small synthetic RGBA graphic
    small_img = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(small_img)
    draw.ellipse([10, 10, 70, 70], fill=(200, 30, 30, 255), outline=(30, 30, 30, 255), width=3)

    for eng in ["realesrgan_anime", "realesrgan_general", "lanczos"]:
        print(f"Testing engine: {eng} (upscale 2.0x)...")
        res = enhance_raster_image(small_img, engine=eng, upscale_factor=2.0)
        assert res.size == (160, 160), f"Expected (160, 160), got {res.size}"
        assert res.mode == "RGBA", f"Expected RGBA, got {res.mode}"
        # Verify transparency is retained
        alpha_np = np.array(res.split()[-1])
        assert np.any(alpha_np == 0), "Transparent pixels should remain transparent"
        assert np.any(alpha_np > 200), "Opaque graphic should remain opaque"
        print(f"  [OK] {eng} passed!")


def test_edge_cases():
    print("\n--- Testing Edge Cases & Robustness ---")
    
    # 1. Grayscale image (mode 'L')
    print("Edge Case 1: Grayscale 'L' input...")
    gray_img = Image.new("L", (64, 64), 128)
    res_gray = enhance_raster_image(gray_img, upscale_factor=1.5)
    assert res_gray.size == (96, 96)
    print("  [OK] Grayscale handled cleanly.")

    # 2. RGB image without alpha
    print("Edge Case 2: RGB input without alpha...")
    rgb_img = Image.new("RGB", (50, 70), (100, 150, 200))
    res_rgb = enhance_raster_image(rgb_img, upscale_factor=2.0)
    assert res_rgb.size == (100, 140)
    assert res_rgb.mode == "RGB"
    print("  [OK] Pure RGB handled cleanly.")

    # 3. Fractional upscale factor (e.g. 1.25x)
    print("Edge Case 3: Fractional upscale factor (1.25x)...")
    res_frac = enhance_raster_image(rgb_img, upscale_factor=1.25)
    assert res_frac.size == (int(round(50 * 1.25)), int(round(70 * 1.25)))
    print("  [OK] Fractional scaling handled cleanly.")

    # 4. Upscale factor 4.0x (native model scale)
    print("Edge Case 4: Native 4.0x upscale factor...")
    res_4x = enhance_raster_image(rgb_img, upscale_factor=4.0)
    assert res_4x.size == (200, 280)
    print("  [OK] Native 4.0x handled cleanly.")

    # 5. Tiled inference on large image (simulated with tile_size=64)
    print("Edge Case 5: Tiled inference with small tile_size (triggers tiling)...")
    tiled_res = enhance_raster_image(rgb_img, tile_size=32, tile_pad=8, upscale_factor=2.0)
    assert tiled_res.size == (100, 140)
    print("  [OK] Tiled inference executed and stitched cleanly.")

    # 6. Fallback with unknown engine key
    print("Edge Case 6: Fallback on unknown engine...")
    res_fallback = enhance_raster_image(rgb_img, engine="non_existent_engine", upscale_factor=2.0)
    assert res_fallback.size == (100, 140)
    print("  [OK] Graceful fallback handled cleanly.")

    print("\n[SUCCESS] ALL EDGE CASES PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_test()
    test_ai_engines()
    test_edge_cases()
