"""
T-Shirt Design Maker - Core Package
Converts low-res raster images (PNG, JPG) to print-ready vector graphics.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from .bg_remover import (
    remove_background,
    normalize_region,
    extract_text_mask_from_region,
    apply_preserved_regions,
    draw_region_preview,
)
from .enhancer import enhance_raster_image
from .vectorizer import vectorize_image
from .exporter import export_print_package

__all__ = [
    "remove_background",
    "normalize_region",
    "extract_text_mask_from_region",
    "apply_preserved_regions",
    "draw_region_preview",
    "enhance_raster_image",
    "vectorize_image",
    "export_print_package",
]
