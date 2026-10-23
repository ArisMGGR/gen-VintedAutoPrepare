"""Loading and saving photos. Arrays are RGB uint8, shape (h, w, 3)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # iPhone photos
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover - optional
    pass


def load_rgb(path: Path) -> np.ndarray:
    """Load a photo, applying the EXIF orientation so sideways phone shots come out upright."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        return np.asarray(im.convert("RGB")).copy()


def save_jpeg(img: np.ndarray, path: Path, quality: int = 95) -> None:
    """Save without EXIF, so no GPS/location data ends up on Vinted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, "JPEG", quality=quality, optimize=True, subsampling=0)


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    im = Image.fromarray(img).resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return np.asarray(im)


def to_jpeg_b64(img: np.ndarray, max_side: int = 1024, quality: int = 85) -> str:
    buf = io.BytesIO()
    Image.fromarray(resize_max(img, max_side)).save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")
