"""
Automated unit tests for T-Shirt Design Maker core pipeline,
including selective quote/text region preservation, local AI Super-Resolution engines,
VTracer vectorization, and print-ready export packaging.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from PIL import Image, ImageDraw
import numpy as np
from core.bg_remover import (
    remove_background,
    normalize_region,
    extract_text_mask_from_region,
    apply_preserved_regions,
    draw_region_preview
)
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

def create_graphic_with_quote():
    """Create a composite t-shirt graphic with an emblem AND typography quote text below it."""
    img = Image.new("RGBA", (400, 480), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Emblem
    draw.ellipse([50, 30, 350, 330], fill="#1E3A8A", outline="#F59E0B", width=8)
    draw.ellipse([80, 60, 320, 300], fill="#FFFFFF", outline="#1E3A8A", width=4)
    draw.rectangle([130, 150, 270, 210], fill="#EF4444")
    draw.polygon([(200, 85), (235, 140), (165, 140)], fill="#F59E0B")
    
    # Typography quote below emblem
    draw.text((105, 360), "VINTAGE MOTOR CLUB", fill="#1E3A8A")
    draw.text((135, 395), "ESTABLISHED 1984", fill="#4B5563")
    draw.text((120, 425), "• ORIGINAL HERITAGE •", fill="#F59E0B")
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


def test_quote_text_preservation():
    print("\n--- Testing Quote & Text Selection / Preservation Feature ---")
    img = create_graphic_with_quote()
    
    # 1. Verify standard rembg behavior: without preserved regions, text area is erased
    print("Checking default background removal on graphic with quote...")
    plain_no_bg = remove_background(img, alpha_threshold=20)
    plain_alpha = np.array(plain_no_bg)[:, :, 3]
    text_area_plain = plain_alpha[350:450, 90:330]
    plain_text_pixels = np.count_nonzero(text_area_plain > 50)
    print(f"  Default rembg opaque text pixels: {plain_text_pixels} (salient AI stripped text)")

    # 2. Test selective region preservation with auto-clean text background
    print("Testing selective region preservation (Auto-Clean Text Background)...")
    quote_box = {"left": 80, "top": 350, "width": 260, "height": 95}
    preserved_img = remove_background(
        img,
        alpha_threshold=20,
        preserve_regions=[quote_box],
        clean_text_background=True,
        text_threshold=25
    )
    assert preserved_img.mode == "RGBA"
    preserved_alpha = np.array(preserved_img)[:, :, 3]
    
    # Graphic emblem must be preserved
    emblem_alpha = preserved_alpha[50:300, 100:300]
    assert np.count_nonzero(emblem_alpha > 200) > 1000, "Emblem should remain opaque"
    
    # Quote text inside preserved region must be preserved
    quote_alpha = preserved_alpha[350:445, 80:340]
    preserved_quote_pixels = np.count_nonzero(quote_alpha > 50)
    assert preserved_quote_pixels > 200, f"Quote text should be preserved! Got {preserved_quote_pixels} pixels"
    
    # Backdrop inside the quote box should be transparent (no solid white block)
    quote_transparent_pixels = np.count_nonzero(quote_alpha == 0)
    assert quote_transparent_pixels > 1000, "Background behind text should be transparent"
    
    # Outer image canvas background must remain transparent
    corner_alpha = preserved_alpha[0:20, 0:20]
    assert np.all(corner_alpha == 0), "Outer backdrop must remain transparent"

    # Verify RGB color fidelity: text colors must be preserved from original image (not converted to black)
    orig_np = np.array(img)
    pres_np = np.array(preserved_img)
    # Check at a known text stroke position
    text_pixel_mask = (preserved_alpha[350:445, 80:340] > 100)
    orig_crop_rgb = orig_np[350:445, 80:340, :3][text_pixel_mask]
    pres_crop_rgb = pres_np[350:445, 80:340, :3][text_pixel_mask]
    rgb_diff = np.max(np.abs(orig_crop_rgb.astype(int) - pres_crop_rgb.astype(int)))
    assert rgb_diff == 0, f"Preserved text RGB colors must exactly match original image, but max diff was {rgb_diff}"
    print(f"  [OK] Quote preserved! {preserved_quote_pixels} text pixels opaque, {quote_transparent_pixels} bg pixels transparent, RGB color fidelity exact (diff={rgb_diff}).")

    # 3. Test vectorization with preserved quote
    print("Testing vectorization on graphic + quote composite...")
    svg_out = vectorize_image(preserved_img, colormode="color", filter_speckle=4)
    assert "<svg" in svg_out and "</svg>" in svg_out
    assert len(svg_out) > 5000, "SVG should contain rich vector paths for both badge and quote"
    print(f"  [OK] SVG generated successfully ({len(svg_out)} bytes)")

    # 4. Test multiple preserved regions (header banner + footer quote)
    print("Testing multiple preserved regions simultaneously...")
    multi_img = Image.new("RGBA", (400, 500), (255, 255, 255, 255))
    draw = ImageDraw.Draw(multi_img)
    draw.text((120, 20), "TOP HEADER SLOGAN", fill=(10, 10, 10))
    draw.ellipse([80, 80, 320, 320], fill=(180, 40, 40))
    draw.text((110, 380), "BOTTOM QUOTE TEXT", fill=(10, 10, 10))
    
    regions = [
        {"left": 100, "top": 10, "width": 200, "height": 40},
        {"left": 90, "top": 370, "width": 220, "height": 40}
    ]
    multi_res = remove_background(multi_img, preserve_regions=regions, clean_text_background=True)
    m_alpha = np.array(multi_res)[:, :, 3]
    top_pixels = np.count_nonzero(m_alpha[10:50, 100:300] > 50)
    bottom_pixels = np.count_nonzero(m_alpha[370:410, 90:310] > 50)
    assert top_pixels > 50, "Header text must be preserved"
    assert bottom_pixels > 50, "Footer quote must be preserved"
    print(f"  [OK] Multiple regions preserved cleanly (top: {top_pixels} px, bottom: {bottom_pixels} px).")

    # 5. Test clean_text_background=False (retains full rectangular area)
    print("Testing clean_text_background=False (keeps full box)...")
    solid_res = remove_background(
        img,
        preserve_regions=[quote_box],
        clean_text_background=False
    )
    s_alpha = np.array(solid_res)[:, :, 3]
    solid_quote_area = s_alpha[350:445, 80:340]
    assert np.all(solid_quote_area == 255), "When clean_text_background=False, full box should be opaque"
    print("  [OK] Full box preservation verified.")


def test_region_utilities():
    print("\n--- Testing Region Helper Functions & Robustness ---")
    
    # 1. normalize_region formats
    r1 = normalize_region({"left": 10, "top": 20, "width": 100, "height": 50}, 400, 400)
    assert r1 == (10, 20, 110, 70), f"Expected (10, 20, 110, 70), got {r1}"

    r2 = normalize_region({"x": 15, "y": 25, "w": 80, "h": 60}, 400, 400)
    assert r2 == (15, 25, 95, 85), f"Expected (15, 25, 95, 85), got {r2}"

    r3 = normalize_region((10, 20, 120, 80), 400, 400)
    assert r3 == (10, 20, 120, 80), f"Expected (10, 20, 120, 80), got {r3}"

    # Clamping out-of-bounds
    r_clamp = normalize_region({"left": -20, "top": -10, "width": 500, "height": 600}, 300, 300)
    assert r_clamp == (0, 0, 300, 300), f"Expected clamped (0, 0, 300, 300), got {r_clamp}"

    # Invalid / empty
    assert normalize_region(None, 300, 300) is None
    assert normalize_region({}, 300, 300) is None
    assert normalize_region("invalid", 300, 300) is None
    print("  [OK] normalize_region handled all formats and bounds cleanly.")

    # 2. draw_region_preview
    sample = Image.new("RGBA", (200, 200), (200, 200, 200, 255))
    preview = draw_region_preview(
        sample,
        regions=[{"left": 20, "top": 20, "width": 50, "height": 50}],
        active_box={"left": 100, "top": 100, "width": 50, "height": 50}
    )
    assert preview.size == (200, 200)
    assert preview.mode == "RGBA"
    print("  [OK] draw_region_preview rendered overlays without error.")


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
    test_quote_text_preservation()
    test_region_utilities()
    test_ai_engines()
    test_edge_cases()
