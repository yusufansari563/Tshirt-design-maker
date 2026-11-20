"""
Print-Ready Exporter Module for T-Shirt Manufacturing.
Generates:
1. Scalable Vector Graphic (SVG)
2. 300 DPI Transparent Master PNG (for DTF/DTG RIP Software)
3. Print-Ready Vector PDF (calibrated to actual garment inches)
"""

import os
import io
from typing import Dict, Tuple, Optional
from PIL import Image

def calculate_pixel_dimensions(width_inches: float, height_inches: float, dpi: int = 300) -> Tuple[int, int]:
    """Calculate target pixel dimensions based on garment print size in inches."""
    return int(round(width_inches * dpi)), int(round(height_inches * dpi))

def generate_300dpi_png(
    image: Image.Image,
    target_width_inches: float,
    target_height_inches: float,
    dpi: int = 300
) -> bytes:
    """
    Generate a 300 DPI Master PNG at exact garment physical dimensions.
    """
    target_px_w, target_px_h = calculate_pixel_dimensions(target_width_inches, target_height_inches, dpi)
    
    # Resize preserving aspect ratio into the bounding box
    img = image.copy()
    img.thumbnail((target_px_w, target_px_h), Image.Resampling.LANCZOS)
    
    # Create target canvas (transparent RGBA)
    canvas = Image.new("RGBA", (target_px_w, target_px_h), (0, 0, 0, 0))
    offset_x = (target_px_w - img.width) // 2
    offset_y = (target_px_h - img.height) // 2
    canvas.paste(img, (offset_x, offset_y))
    
    buf = io.BytesIO()
    # Save with 300 DPI metadata header
    canvas.save(buf, format="PNG", dpi=(dpi, dpi))
    return buf.getvalue()

def generate_print_pdf(
    svg_content: str,
    target_width_inches: float,
    target_height_inches: float,
    fallback_image: Optional[Image.Image] = None
) -> bytes:
    """
    Generate a Vector PDF document sized to the specified physical inches (72 points/inch).
    """
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    
    pdf_width_pts = target_width_inches * inch
    pdf_height_pts = target_height_inches * inch
    
    pdf_buf = io.BytesIO()
    pdf_canvas = canvas.Canvas(pdf_buf, pagesize=(pdf_width_pts, pdf_height_pts))
    
    # Try vector rendering via svglib first
    rendered_vector = False
    try:
        from svglib.svglib import svg2rlg
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False, mode="w", encoding="utf-8") as tmp_svg:
            tmp_svg.write(svg_content)
            tmp_svg_path = tmp_svg.name
            
        drawing = svg2rlg(tmp_svg_path)
        try:
            os.remove(tmp_svg_path)
        except Exception:
            pass
            
        if drawing is not None and drawing.width > 0 and drawing.height > 0:
            from reportlab.graphics import renderPDF
            
            # Scale drawing to fit target page with padding
            scale_x = (pdf_width_pts * 0.95) / drawing.width
            scale_y = (pdf_height_pts * 0.95) / drawing.height
            scale = min(scale_x, scale_y)
            
            drawing.width = drawing.width * scale
            drawing.height = drawing.height * scale
            drawing.scale(scale, scale)
            
            x_offset = (pdf_width_pts - drawing.width) / 2
            y_offset = (pdf_height_pts - drawing.height) / 2
            
            renderPDF.draw(drawing, pdf_canvas, x_offset, y_offset)
            rendered_vector = True
    except Exception as e:
        print(f"[Notice] svglib vector rendering fallback: {e}")

    # Fallback to high-resolution raster draw if svglib encounters complex custom SVG elements
    if not rendered_vector and fallback_image is not None:
        from reportlab.lib.utils import ImageReader
        img_buf = io.BytesIO()
        fallback_image.save(img_buf, format="PNG")
        img_buf.seek(0)
        img_reader = ImageReader(img_buf)
        
        aspect = fallback_image.width / max(1, fallback_image.height)
        box_w = pdf_width_pts * 0.95
        box_h = pdf_height_pts * 0.95
        
        if box_w / box_h > aspect:
            draw_h = box_h
            draw_w = box_h * aspect
        else:
            draw_w = box_w
            draw_h = box_w / aspect
            
        x_offset = (pdf_width_pts - draw_w) / 2
        y_offset = (pdf_height_pts - draw_h) / 2
        pdf_canvas.drawImage(img_reader, x_offset, y_offset, width=draw_w, height=draw_h, mask='auto')

    pdf_canvas.save()
    return pdf_buf.getvalue()

def export_print_package(
    svg_content: str,
    master_image: Image.Image,
    target_width_inches: float = 12.0,
    target_height_inches: float = 16.0,
    dpi: int = 300
) -> Dict[str, bytes]:
    """
    Generate the complete print-ready export files.
    
    Returns:
        Dictionary mapping file extensions to file byte data:
        - "svg": UTF-8 encoded SVG bytes
        - "png": 300 DPI Transparent Master PNG bytes
        - "pdf": Vector/High-Res Print PDF bytes
    """
    svg_bytes = svg_content.encode("utf-8")
    png_bytes = generate_300dpi_png(master_image, target_width_inches, target_height_inches, dpi)
    pdf_bytes = generate_print_pdf(svg_content, target_width_inches, target_height_inches, master_image)
    
    return {
        "svg": svg_bytes,
        "png": png_bytes,
        "pdf": pdf_bytes
    }
