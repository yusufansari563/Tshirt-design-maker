"""
Background removal module using rembg with fine-tuned edge cleanup for garment printing.
"""

from typing import Optional
from PIL import Image
import numpy as np

# Cache session to avoid reloading model weights on each call
_rembg_session = None

def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        try:
            from rembg import new_session
            # u2net or bria-rmbg
            _rembg_session = new_session("u2net")
        except Exception as e:
            print(f"[Warning] Failed to initialize specific session: {e}")
            _rembg_session = None
    return _rembg_session

def remove_background(
    image: Image.Image,
    alpha_threshold: int = 25,
    smooth_mask: bool = True
) -> Image.Image:
    """
    Remove the background from a PIL Image.
    
    Args:
        image: Source PIL Image (RGB or RGBA)
        alpha_threshold: Cutoff (0-255) for faint semi-transparent pixels.
                         DTF/DTG printers lay white ink wherever alpha > 0,
                         so a cutoff prevents blurry halos around designs.
        smooth_mask: Whether to smooth out single-pixel artifacts.
        
    Returns:
        PIL Image in RGBA mode with background removed.
    """
    from rembg import remove

    session = get_rembg_session()
    if session is not None:
        result_img = remove(image, session=session)
    else:
        result_img = remove(image)

    if result_img.mode != "RGBA":
        result_img = result_img.convert("RGBA")

    if alpha_threshold > 0:
        # Clean up faint alpha pixels to prevent DTF white ink bleed/halos
        np_img = np.array(result_img)
        alpha = np_img[:, :, 3]
        
        # Suppress low alpha values to pure 0
        np_img[alpha < alpha_threshold, 3] = 0
        
        result_img = Image.fromarray(np_img, mode="RGBA")

    return result_img
