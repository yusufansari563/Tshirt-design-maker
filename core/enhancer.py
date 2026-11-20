"""
Raster image enhancement and pre-vectorization preparation module.
Integrates local neural AI super-resolution (Real-ESRGAN via ONNX Runtime)
to eliminate pixelation, de-alias jagged edges, and sharpen low-res graphics
for crisp, professional vector tracing and garment printing.
"""

import os
import urllib.request
import logging
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ImageEnhance
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Base directory for storing AI model weights
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Registry of supported enhancement engines
AVAILABLE_ENGINES: Dict[str, Dict[str, Any]] = {
    "realesrgan_anime": {
        "name": "Real-ESRGAN Anime 6B (Vector & Line Art)",
        "model_file": "RealESRGAN_x4plus_anime_6B.onnx",
        "url": "https://huggingface.co/liangmu261004/real-esrgan-onnx/resolve/main/RealESRGAN_x4plus_anime_6B.onnx",
        "scale": 4,
        "description": "Specially trained for digital graphics, cartoons, logos, and t-shirt vector art. Produces ultra-clean edges and eliminates staircasing without noise.",
        "is_ai": True,
    },
    "realesrgan_general": {
        "name": "Real-ESRGAN x4plus (Photorealistic)",
        "model_file": "RealESRGAN_x4plus.onnx",
        "url": "https://huggingface.co/liangmu261004/real-esrgan-onnx/resolve/main/RealESRGAN_x4plus.onnx",
        "scale": 4,
        "description": "General-purpose super-resolution model for photorealistic apparel art, rich textures, and detailed artwork.",
        "is_ai": True,
    },
    "lanczos": {
        "name": "Classical Lanczos + Bilateral Filter (Fast / Non-AI)",
        "model_file": None,
        "url": None,
        "scale": 1,
        "description": "Traditional bicubic/lanczos interpolation with edge-preserving bilateral filtering. Instantaneous and requires minimal memory.",
        "is_ai": False,
    }
}

# In-memory session cache to avoid repeatedly loading model weights
_session_cache: Dict[str, Any] = {}


def get_available_engines() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of available engines with their configuration."""
    return AVAILABLE_ENGINES


def get_model_path(engine_key: str, models_dir: Optional[str] = None) -> Optional[str]:
    """Get absolute path to model file for an engine."""
    engine_cfg = AVAILABLE_ENGINES.get(engine_key)
    if not engine_cfg or not engine_cfg.get("model_file"):
        return None
    target_dir = models_dir or MODELS_DIR
    return os.path.join(target_dir, engine_cfg["model_file"])


def is_model_downloaded(engine_key: str, models_dir: Optional[str] = None) -> bool:
    """Check if model weights file exists locally."""
    model_path = get_model_path(engine_key, models_dir)
    return model_path is not None and os.path.isfile(model_path) and os.path.getsize(model_path) > 1000


def ensure_model_available(engine_key: str, models_dir: Optional[str] = None) -> Optional[str]:
    """
    Ensure the ONNX model file exists on disk.
    If missing, downloads automatically from the repository.
    """
    engine_cfg = AVAILABLE_ENGINES.get(engine_key)
    if not engine_cfg or not engine_cfg.get("model_file"):
        return None

    target_dir = models_dir or MODELS_DIR
    os.makedirs(target_dir, exist_ok=True)
    model_path = os.path.join(target_dir, engine_cfg["model_file"])

    if os.path.isfile(model_path) and os.path.getsize(model_path) > 1000:
        return model_path

    url = engine_cfg.get("url")
    if not url:
        return None

    logger.info(f"Downloading {engine_cfg['name']} model from {url}...")
    try:
        urllib.request.urlretrieve(url, model_path)
        logger.info(f"Successfully downloaded {model_path} ({os.path.getsize(model_path)} bytes).")
        return model_path
    except Exception as e:
        logger.warning(f"Failed to download model {engine_cfg['model_file']} from {url}: {e}")
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except Exception:
                pass
        return None


def get_onnx_session(engine_key: str, models_dir: Optional[str] = None):
    """
    Get or create a cached ONNX Runtime InferenceSession for the engine.
    """
    global _session_cache
    if engine_key in _session_cache:
        return _session_cache[engine_key]

    model_path = ensure_model_available(engine_key, models_dir)
    if not model_path:
        return None

    try:
        import onnxruntime as ort
        
        # Configure session options for optimal execution
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        available_providers = ort.get_available_providers()
        preferred_providers = []
        for prov in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]:
            if prov in available_providers:
                preferred_providers.append(prov)

        session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=preferred_providers)
        _session_cache[engine_key] = session
        return session
    except Exception as e:
        logger.warning(f"Failed to initialize ONNX Runtime session for {engine_key}: {e}")
        return None


def _prevent_alpha_bleed(bgr_image: np.ndarray, alpha_channel: np.ndarray) -> np.ndarray:
    """
    Extend foreground colors into transparent background regions using inpainting.
    Prevents dark/dirty border halos when upscaling images with transparency.
    """
    opaque_count = np.count_nonzero(alpha_channel > 0)
    total_pixels = alpha_channel.size

    # Only inpaint if there is a mix of transparent and opaque pixels
    if 0 < opaque_count < total_pixels:
        mask = (alpha_channel == 0).astype(np.uint8) * 255
        try:
            # Telea fast marching inpainting smoothly radiates color across transparent borders
            inpainted = cv2.inpaint(bgr_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            return inpainted
        except Exception:
            return bgr_image
    return bgr_image


def _run_tiled_inference(
    session,
    rgb_float: np.ndarray,
    tile_size: int = 512,
    tile_pad: int = 16,
    scale: int = 4
) -> np.ndarray:
    """
    Run neural super-resolution inference on RGB float array [H, W, 3] in range [0, 1].
    Uses tiled processing with overlapping padding to prevent seam artifacts on large images.
    """
    H, W, C = rgb_float.shape

    # If image fits in a single tile, infer directly without tiling overhead
    if H <= tile_size and W <= tile_size:
        inp_tensor = np.transpose(rgb_float, (2, 0, 1))[np.newaxis, ...]
        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: inp_tensor})[0][0]
        out = np.transpose(out, (1, 2, 0))
        return np.clip(out, 0.0, 1.0)

    # Tiled inference for large images
    out_full = np.zeros((H * scale, W * scale, C), dtype=np.float32)
    input_name = session.get_inputs()[0].name

    y = 0
    while y < H:
        y_end = min(y + tile_size, H)
        x = 0
        while x < W:
            x_end = min(x + tile_size, W)

            pad_top = min(y, tile_pad)
            pad_bottom = min(H - y_end, tile_pad)
            pad_left = min(x, tile_pad)
            pad_right = min(W - x_end, tile_pad)

            crop = rgb_float[y - pad_top : y_end + pad_bottom, x - pad_left : x_end + pad_right]
            crop_tensor = np.transpose(crop, (2, 0, 1))[np.newaxis, ...]

            tile_out = session.run(None, {input_name: crop_tensor})[0][0]
            tile_out = np.transpose(tile_out, (1, 2, 0))

            out_crop = tile_out[
                pad_top * scale : (pad_top + (y_end - y)) * scale,
                pad_left * scale : (pad_left + (x_end - x)) * scale
            ]

            out_full[y * scale : y_end * scale, x * scale : x_end * scale] = out_crop
            x = x_end
        y = y_end

    return np.clip(out_full, 0.0, 1.0)


def enhance_raster_image(
    image: Image.Image,
    engine: str = "realesrgan_anime",
    upscale_factor: float = 2.0,
    denoise_strength: int = 0,
    sharpen_amount: float = 1.0,
    contrast_boost: float = 1.0,
    tile_size: int = 512,
    tile_pad: int = 16,
    models_dir: Optional[str] = None,
    **kwargs
) -> Image.Image:
    """
    Enhance a raster image prior to vectorization using local AI super-resolution or classical filters.

    Args:
        image: Source PIL Image (RGB, RGBA, L, P)
        engine: Super-resolution engine key ('realesrgan_anime', 'realesrgan_general', or 'lanczos')
        upscale_factor: Scaling factor (e.g. 2.0x, 3.0x, 4.0x) for curve fitting & print resolution
        denoise_strength: Bilateral filter diameter for edge-preserving denoising (0 to skip)
        sharpen_amount: Sharpness multiplier (1.0 = unchanged)
        contrast_boost: Contrast multiplier (1.0 = unchanged)
        tile_size: Tile dimensions for large image processing
        tile_pad: Overlapping padding between tiles to eliminate edge seams
        models_dir: Directory containing ONNX models (defaults to project's models/ folder)

    Returns:
        Enhanced PIL Image (RGBA if source had alpha, else RGB)
    """
    # Defensive argument handling: if engine was passed as a numeric upscale factor positionally
    if isinstance(engine, (int, float)):
        upscale_factor = float(engine)
        engine = "realesrgan_anime"

    # Normalize image mode
    has_alpha = image.mode == "RGBA" or (image.mode == "P" and "transparency" in image.info)
    if has_alpha:
        rgba_img = image.convert("RGBA")
        alpha_channel = np.array(rgba_img.split()[-1])
        rgb_source = rgba_img.convert("RGB")
    else:
        alpha_channel = None
        rgb_source = image.convert("RGB")

    orig_w, orig_h = rgb_source.size
    target_w = max(1, int(round(orig_w * upscale_factor)))
    target_h = max(1, int(round(orig_h * upscale_factor)))

    used_ai = False

    # Attempt AI super-resolution if an AI engine is selected
    if engine in ["realesrgan_anime", "realesrgan_general"]:
        session = get_onnx_session(engine, models_dir=models_dir)
        if session is not None:
            try:
                np_rgb = np.array(rgb_source)

                # Pre-process transparency: inpaint to prevent dark halo artifacts
                if has_alpha and alpha_channel is not None:
                    bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
                    inpainted_bgr = _prevent_alpha_bleed(bgr, alpha_channel)
                    np_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

                # Neural 4x inference
                rgb_float = (np_rgb.astype(np.float32) / 255.0)
                out_sr = _run_tiled_inference(
                    session,
                    rgb_float,
                    tile_size=tile_size,
                    tile_pad=tile_pad,
                    scale=4
                )

                # Convert back to PIL
                sr_uint8 = (out_sr * 255.0).astype(np.uint8)
                result_rgb = Image.fromarray(sr_uint8)

                # Resample to exact requested upscale_factor (Lanczos downscale acts as supersampling anti-aliasing)
                if result_rgb.size != (target_w, target_h):
                    result_rgb = result_rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)

                used_ai = True
            except Exception as e:
                logger.warning(f"Error during AI inference with engine '{engine}': {e}. Falling back to Lanczos.")
                used_ai = False

    # Fallback to classical Lanczos scaling if AI was not used or failed
    if not used_ai:
        if (target_w, target_h) != (orig_w, orig_h):
            result_rgb = rgb_source.resize((target_w, target_h), Image.Resampling.LANCZOS)
        else:
            result_rgb = rgb_source.copy()

    # Denoising via bilateral filter (if requested or classical mode with denoise > 0)
    if denoise_strength > 0:
        np_rgb = np.array(result_rgb)
        bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
        d = denoise_strength
        sigma = denoise_strength * 10
        filtered = cv2.bilateralFilter(bgr, d=d, sigmaColor=sigma, sigmaSpace=sigma)
        result_rgb = Image.fromarray(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))

    # Contrast adjustment
    if contrast_boost != 1.0:
        enhancer = ImageEnhance.Contrast(result_rgb)
        result_rgb = enhancer.enhance(contrast_boost)

    # Sharpness adjustment
    if sharpen_amount != 1.0:
        enhancer = ImageEnhance.Sharpness(result_rgb)
        result_rgb = enhancer.enhance(sharpen_amount)

    # Recombine Alpha channel if present
    if has_alpha and alpha_channel is not None:
        pil_alpha = Image.fromarray(alpha_channel)
        if pil_alpha.size != (target_w, target_h):
            pil_alpha = pil_alpha.resize((target_w, target_h), Image.Resampling.LANCZOS)
        result_rgb.putalpha(pil_alpha)

    return result_rgb
