"""
Background removal module using rembg with fine-tuned edge cleanup for garment printing,
and selective region preservation to retain quotes, slogans, and typography alongside graphics.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from typing import Optional, List, Dict, Any, Tuple, Union
from PIL import Image, ImageDraw
import numpy as np

# Cache session to avoid reloading model weights on each call
_rembg_session = None

def get_rembg_session():
    """
    Lazily initialize and cache the rembg ONNX session with CPU memory arena disabled
    to prevent memory allocator collisions on Windows.
    """
    global _rembg_session
    if _rembg_session is None:
        try:
            import onnxruntime as ort
            from rembg import new_session
            so = ort.SessionOptions()
            so.enable_cpu_mem_arena = False
            _rembg_session = new_session("u2net", sess_opts=so, providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"[Warning] Failed to initialize specific session: {e}")
            _rembg_session = None
    return _rembg_session


def normalize_region(region: Any, img_width: int, img_height: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Normalizes a region representation into (x1, y1, x2, y2) clamped to image boundaries.
    Supports:
      - dict with keys: 'left', 'top', 'width', 'height' (from streamlit-cropper)
      - dict with keys: 'x', 'y', 'w', 'h' or 'x1', 'y1', 'x2', 'y2'
      - tuple/list with 4 elements: (x1, y1, x2, y2) or (x, y, w, h)
    """
    if isinstance(region, dict):
        if "left" in region and "top" in region:
            l = int(round(region["left"]))
            t = int(round(region["top"]))
            w = int(round(region.get("width", 0)))
            h = int(round(region.get("height", 0)))
            x1, y1, x2, y2 = l, t, l + w, t + h
        elif "x" in region and "y" in region:
            l = int(round(region["x"]))
            t = int(round(region["y"]))
            w = int(round(region.get("w", region.get("width", 0))))
            h = int(round(region.get("h", region.get("height", 0))))
            x1, y1, x2, y2 = l, t, l + w, t + h
        elif "x1" in region and "y1" in region:
            x1 = int(round(region["x1"]))
            y1 = int(round(region["y1"]))
            x2 = int(round(region.get("x2", x1)))
            y2 = int(round(region.get("y2", y1)))
        else:
            return None
    elif isinstance(region, (tuple, list)) and len(region) == 4:
        r0, r1, r2, r3 = [int(round(v)) for v in region]
        if r2 > r0 and r3 > r1:
            x1, y1, x2, y2 = r0, r1, r2, r3
        else:
            x1, y1, x2, y2 = r0, r1, r0 + r2, r1 + r3
    else:
        return None

    # Clamp coordinates safely
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(x1 + 1, min(x2, img_width))
    y2 = max(y1 + 1, min(y2, img_height))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def extract_text_mask_from_region(
    image_np: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    clean_background: bool = True,
    text_threshold: int = 25
) -> np.ndarray:
    """
    Extract foreground alpha mask for a bounding box region.
    If clean_background is True, automatically separates text/quotes/slogans
    from the background within the box using color distance from the perimeter.
    If False, preserves the entire box region as opaque foreground.
    """
    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return np.zeros((0, 0), dtype=np.uint8)

    crop = image_np[y1:y2, x1:x2]
    crop_rgb = crop[:, :, :3]

    if not clean_background:
        mask = np.full((h, w), 255, dtype=np.uint8)
        if crop.shape[2] == 4:
            mask = np.minimum(mask, crop[:, :, 3])
        return mask

    # Sample perimeter border pixels to identify the local background color
    top_border = crop_rgb[0, :, :]
    bottom_border = crop_rgb[-1, :, :]
    left_border = crop_rgb[:, 0, :]
    right_border = crop_rgb[:, -1, :]
    border_pixels = np.concatenate([top_border, bottom_border, left_border, right_border], axis=0)

    if crop.shape[2] == 4:
        top_a = crop[0, :, 3]
        bot_a = crop[-1, :, 3]
        left_a = crop[:, 0, 3]
        right_a = crop[:, -1, 3]
        border_alpha = np.concatenate([top_a, bot_a, left_a, right_a], axis=0)
        valid_border = border_pixels[border_alpha > 50]
        if len(valid_border) > 0:
            bg_color = np.median(valid_border, axis=0)
        else:
            bg_color = np.median(border_pixels, axis=0)
    else:
        bg_color = np.median(border_pixels, axis=0)

    # Compute Euclidean distance in RGB color space from detected background
    diff = np.linalg.norm(crop_rgb.astype(np.float32) - bg_color.astype(np.float32), axis=-1)

    # Smooth anti-aliased alpha transition around the threshold
    low_cutoff = max(4.0, float(text_threshold) - 8.0)
    high_cutoff = float(text_threshold) + 8.0
    alpha_ramp = np.clip((diff - low_cutoff) / max(1.0, high_cutoff - low_cutoff), 0.0, 1.0) * 255.0
    text_mask = alpha_ramp.astype(np.uint8)

    # Respect any existing transparency in the source crop
    if crop.shape[2] == 4:
        text_mask = np.minimum(text_mask, crop[:, :, 3])

    return text_mask


def apply_preserved_regions(
    image: Image.Image,
    base_alpha: np.ndarray,
    preserve_regions: Optional[Any],
    clean_text_background: bool = True,
    text_threshold: int = 25
) -> np.ndarray:
    """
    Applies user-selected quote / text regions onto an existing alpha mask.
    Merges masks using np.maximum so salient objects and quote text coexist cleanly.
    """
    if preserve_regions is None:
        return base_alpha

    # Direct PIL Image or numpy array mask
    if isinstance(preserve_regions, Image.Image):
        mask_np = np.array(preserve_regions.convert("L").resize(image.size, Image.Resampling.NEAREST))
        return np.maximum(base_alpha, mask_np)
    elif isinstance(preserve_regions, np.ndarray) and preserve_regions.ndim == 2:
        if preserve_regions.shape[:2] == base_alpha.shape[:2]:
            return np.maximum(base_alpha, preserve_regions.astype(np.uint8))

    # Normalize into a list of regions
    if isinstance(preserve_regions, (dict, tuple)):
        region_list = [preserve_regions]
    elif isinstance(preserve_regions, list):
        region_list = preserve_regions
    else:
        return base_alpha

    if not region_list:
        return base_alpha

    img_np = np.array(image.convert("RGBA"))
    merged_alpha = base_alpha.copy()
    W, H = image.size

    for reg in region_list:
        coords = normalize_region(reg, W, H)
        if coords is None:
            continue
        x1, y1, x2, y2 = coords
        reg_mask = extract_text_mask_from_region(
            img_np, x1, y1, x2, y2,
            clean_background=clean_text_background,
            text_threshold=text_threshold
        )
        merged_alpha[y1:y2, x1:x2] = np.maximum(merged_alpha[y1:y2, x1:x2], reg_mask)

    return merged_alpha


def draw_region_preview(
    image: Image.Image,
    regions: Optional[List[Any]] = None,
    active_box: Optional[Any] = None
) -> Image.Image:
    """
    Draw an overlay visualization showing preserved quote/text regions
    and any currently active selection box for Streamlit UI display.
    """
    overlay_img = image.convert("RGBA").copy()
    W, H = image.size

    # Draw saved regions in bright emerald green
    if regions:
        for idx, reg in enumerate(regions, 1):
            coords = normalize_region(reg, W, H)
            if not coords:
                continue
            x1, y1, x2, y2 = coords
            overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([x1, y1, x2, y2], fill=(16, 185, 129, 65), outline=(16, 185, 129, 240), width=3)
            overlay_img = Image.alpha_composite(overlay_img, overlay)
            draw = ImageDraw.Draw(overlay_img)
            label = f"Preserved #{idx}"
            draw.rectangle([x1, max(0, y1 - 22), min(W, x1 + 105), y1], fill=(16, 185, 129, 230))
            draw.text((x1 + 6, max(0, y1 - 18)), label, fill=(255, 255, 255, 255))

    # Draw active selection in vibrant cyan
    if active_box:
        coords = normalize_region(active_box, W, H)
        if coords:
            x1, y1, x2, y2 = coords
            overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([x1, y1, x2, y2], fill=(6, 182, 212, 65), outline=(6, 182, 212, 240), width=3)
            overlay_img = Image.alpha_composite(overlay_img, overlay)
            draw = ImageDraw.Draw(overlay_img)
            draw.rectangle([x1, max(0, y1 - 22), min(W, x1 + 115), y1], fill=(6, 182, 212, 240))
            draw.text((x1 + 6, max(0, y1 - 18)), "Active Selection", fill=(255, 255, 255, 255))

    return overlay_img


def calculate_regions_bbox(
    regions: List[Any],
    img_width: int,
    img_height: int,
    padding: int = 5
) -> Optional[Tuple[int, int, int, int]]:
    """
    Computes the unified bounding box (x1, y1, x2, y2) enclosing all provided regions,
    with optional padding clamped to image bounds.
    """
    if not regions:
        return None

    min_x, min_y = img_width, img_height
    max_x, max_y = 0, 0
    valid_count = 0

    for reg in regions:
        coords = normalize_region(reg, img_width, img_height)
        if coords is not None:
            x1, y1, x2, y2 = coords
            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
            valid_count += 1

    if valid_count == 0 or max_x <= min_x or max_y <= min_y:
        return None

    # Apply padding
    x1 = max(0, min_x - padding)
    y1 = max(0, min_y - padding)
    x2 = min(img_width, max_x + padding)
    y2 = min(img_height, max_y + padding)

    return (x1, y1, x2, y2)


def extract_garment_design(
    image: Image.Image,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    method: str = "fabric_key",          # "fabric_key", "rembg", or "none"
    tolerance: int = 30,
    edge_feather: int = 8,
    alpha_threshold: int = 25
) -> Image.Image:
    """
    Extracts print design from a garment mockup or selected area.
    Crops to the specified bounding box (or uses full image if bbox is None),
    and removes the garment fabric background so only the printed artwork/text remains.
    
    Args:
        image: Source PIL Image
        bbox: Optional (x1, y1, x2, y2) crop coordinates
        method: 'fabric_key' (smart color keying based on garment cloth),
                'rembg' (AI neural segmentation on the crop),
                or 'none' (just crop without removing background)
        tolerance: Color distance threshold for fabric keying
        edge_feather: Feathering softness for anti-aliasing edges
        alpha_threshold: Cutoff for faint semi-transparent pixels
        
    Returns:
        PIL Image in RGBA mode containing only the extracted design.
    """
    if bbox is not None:
        W, H = image.size
        coords = normalize_region(bbox, W, H)
        if coords is not None:
            x1, y1, x2, y2 = coords
            crop = image.crop((x1, y1, x2, y2))
        else:
            crop = image.copy()
    else:
        crop = image.copy()

    if method == "none":
        return crop.convert("RGBA")

    if method == "rembg":
        from rembg import remove
        session = get_rembg_session()
        if session is not None:
            res = remove(crop, session=session)
        else:
            res = remove(crop)
        res_rgba = res.convert("RGBA")
        if alpha_threshold > 0:
            arr = np.array(res_rgba)
            arr[arr[:, :, 3] < alpha_threshold, 3] = 0
            res_rgba = Image.fromarray(arr, mode="RGBA")
        return res_rgba

    # Default: "fabric_key" (Smart Garment Fabric Keying)
    crop_rgba = crop.convert("RGBA")
    arr = np.array(crop_rgba)
    h, w = arr.shape[:2]
    if h < 4 or w < 4:
        return crop_rgba

    # Sample perimeter border pixels to identify the garment cloth color
    b_w = max(2, min(8, min(h, w) // 15))
    top_b = arr[:b_w, :, :3].reshape(-1, 3)
    bot_b = arr[-b_w:, :, :3].reshape(-1, 3)
    left_b = arr[:, :b_w, :3].reshape(-1, 3)
    right_b = arr[:, -b_w:, :3].reshape(-1, 3)
    border = np.concatenate([top_b, bot_b, left_b, right_b], axis=0)

    # Use median to avoid outliers from graphic elements touching the boundary
    fabric_color = np.median(border, axis=0)

    # Compute Euclidean distance in RGB color space from the fabric color
    diff = np.linalg.norm(arr[:, :, :3].astype(np.float32) - fabric_color.astype(np.float32), axis=-1)

    low_cutoff = max(2.0, float(tolerance) - float(edge_feather))
    high_cutoff = float(tolerance) + float(edge_feather)

    # Anti-aliased alpha ramp
    alpha = np.clip((diff - low_cutoff) / max(1.0, high_cutoff - low_cutoff) * 255.0, 0.0, 255.0).astype(np.uint8)

    if alpha_threshold > 0:
        alpha[alpha < alpha_threshold] = 0

    res_arr = arr.copy()
    res_arr[:, :, 3] = alpha
    return Image.fromarray(res_arr, mode="RGBA")


def remove_background(
    image: Image.Image,
    alpha_threshold: int = 25,
    smooth_mask: bool = True,
    preserve_regions: Optional[Any] = None,
    clean_text_background: bool = True,
    text_threshold: int = 25
) -> Image.Image:
    """
    Remove the background from a PIL Image using rembg with fine-tuned edge cleanup
    and optional selective region preservation (to preserve quotes, typography, or secondary objects).
    
    Args:
        image: Source PIL Image (RGB or RGBA)
        alpha_threshold: Cutoff (0-255) for faint semi-transparent pixels.
                         DTF/DTG printers lay white ink wherever alpha > 0,
                         so a cutoff prevents blurry halos around designs.
        smooth_mask: Whether to smooth out single-pixel artifacts.
        preserve_regions: List, tuple, dict, or mask defining regions to preserve.
        clean_text_background: Whether to automatically key out the background color inside
                               the preserved quote region so text glyphs remain transparent.
        text_threshold: Sensitivity threshold for separating text from background within regions.
        
    Returns:
        PIL Image in RGBA mode with background removed and indicated quote/text regions preserved.
    """
    from rembg import remove

    session = get_rembg_session()
    if session is not None:
        result_img = remove(image, session=session)
    else:
        result_img = remove(image)

    orig_rgba = image.convert("RGBA")
    orig_np = np.array(orig_rgba)

    if result_img.mode != "RGBA":
        result_img = result_img.convert("RGBA")

    np_img = np.array(result_img)
    alpha = np_img[:, :, 3].copy()

    # Merge user-selected quote / text regions if provided and restore original RGB
    if preserve_regions is not None:
        W, H = image.size
        # Direct PIL Image or numpy array mask
        if isinstance(preserve_regions, Image.Image):
            mask_np = np.array(preserve_regions.convert("L").resize(image.size, Image.Resampling.NEAREST))
            active = mask_np > 0
            np_img[active, :3] = orig_np[active, :3]
            alpha = np.maximum(alpha, mask_np)
        elif isinstance(preserve_regions, np.ndarray) and preserve_regions.ndim == 2:
            if preserve_regions.shape[:2] == alpha.shape[:2]:
                mask_np = preserve_regions.astype(np.uint8)
                active = mask_np > 0
                np_img[active, :3] = orig_np[active, :3]
                alpha = np.maximum(alpha, mask_np)
        else:
            region_list = [preserve_regions] if isinstance(preserve_regions, (dict, tuple)) else (preserve_regions if isinstance(preserve_regions, list) else [])
            for reg in region_list:
                coords = normalize_region(reg, W, H)
                if coords is None:
                    continue
                x1, y1, x2, y2 = coords
                reg_mask = extract_text_mask_from_region(
                    orig_np, x1, y1, x2, y2,
                    clean_background=clean_text_background,
                    text_threshold=text_threshold
                )
                active = reg_mask > 0
                sub_img = np_img[y1:y2, x1:x2, :3]
                sub_orig = orig_np[y1:y2, x1:x2, :3]
                sub_img[active] = sub_orig[active]
                alpha[y1:y2, x1:x2] = np.maximum(alpha[y1:y2, x1:x2], reg_mask)

        np_img[:, :, 3] = alpha

    if alpha_threshold > 0:
        # Clean up faint alpha pixels to prevent DTF white ink bleed/halos
        np_img[alpha < alpha_threshold, 3] = 0

    return Image.fromarray(np_img, mode="RGBA")
