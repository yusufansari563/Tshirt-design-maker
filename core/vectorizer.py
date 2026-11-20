"""
Vectorization module using VTracer (Rust engine).
Converts raster pixel data into scalable vector curves (SVG).
"""

import os
import tempfile
from typing import Optional
from PIL import Image
import vtracer

def vectorize_image(
    image: Image.Image,
    output_svg_path: Optional[str] = None,
    colormode: str = "color",           # "color" or "binary"
    hierarchical: str = "stacked",       # "stacked" or "cutout"
    filter_speckle: int = 4,             # Minimum pixel area for shapes (filters noise)
    corner_threshold: int = 60,          # Angle threshold for turning corners into vertices
    color_precision: int = 6,            # Bits of color depth (1-8, default 6)
    layer_difference: int = 16,          # Color difference between consecutive layers
    max_iterations: int = 10
) -> str:
    """
    Vectorize a PIL Image into an SVG file or string.
    
    Args:
        image: Source PIL Image (RGB or RGBA)
        output_svg_path: Destination path for SVG (if None, a temp file is used)
        colormode: 'color' for multi-color DTF/DTG, 'binary' for 1-color screen/vinyl
        hierarchical: 'stacked' prevents sub-pixel gaps between colors; 'cutout' creates jigsaw shapes
        filter_speckle: Discard isolated speckles smaller than this pixel area
        corner_threshold: Sharpness of corners (higher = smoother curves)
        color_precision: Color depth / clustering resolution
        layer_difference: Color variance to merge into single layers
        max_iterations: Curve fitting refinement cycles
        
    Returns:
        SVG content string
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        temp_input_path = tmp_in.name

    try:
        # Handle alpha mode for VTracer
        if colormode == "binary":
            # For monochrome: convert transparent or white to pure white background,
            # and design content to black foreground
            if image.mode == "RGBA":
                # Create white canvas
                canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
                canvas.paste(image, mask=image.split()[-1])
                gray = canvas.convert("L")
            else:
                gray = image.convert("L")
            gray.save(temp_input_path, format="PNG")
        else:
            # Multi-color: preserve RGBA or RGB
            image.save(temp_input_path, format="PNG")

        if output_svg_path is None:
            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_out:
                output_svg_path = tmp_out.name
            cleanup_output = False
        else:
            cleanup_output = False

        vtracer.convert_image_to_svg_py(
            temp_input_path,
            output_svg_path,
            colormode=colormode,
            hierarchical=hierarchical,
            filter_speckle=filter_speckle,
            color_precision=color_precision,
            layer_difference=layer_difference,
            corner_threshold=corner_threshold,
            max_iterations=max_iterations
        )

        with open(output_svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()

        return svg_content

    finally:
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except Exception:
                pass
