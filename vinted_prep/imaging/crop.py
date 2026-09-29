"""Crop a photo around the item, centred, at a fixed aspect ratio (4:5 for Vinted)."""

from __future__ import annotations

import cv2
import numpy as np

Box = tuple[int, int, int, int]

VINTED_ASPECT = 4 / 5  # width / height


def _fit_aspect(w: float, h: float, aspect: float) -> tuple[float, float]:
    """Smallest w' >= w, h' >= h with w'/h' == aspect."""
    return (w, w / aspect) if w / h > aspect else (h * aspect, h)


def compute_crop_box(
    bbox: Box | None, img_w: int, img_h: int, aspect: float = VINTED_ASPECT, margin: float = 0.08
) -> Box:
    """Crop box (may extend past the photo edges when the item can't otherwise be centred
    without being cut; apply_crop fills that space)."""
    if bbox is None:  # item not found: largest centred 4:5 crop of the photo
        cw, ch = (img_h * aspect, img_h) if img_w / img_h > aspect else (img_w, img_w / aspect)
        left, top = (img_w - cw) / 2, (img_h - ch) / 2
        return int(round(left)), int(round(top)), int(round(left + cw)), int(round(top + ch))
    x0, y0, x1, y1 = bbox
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    tight_w, tight_h = _fit_aspect(bw, bh, aspect)
    pad = margin * max(bw, bh)
    cw, ch = _fit_aspect(bw + 2 * pad, bh + 2 * pad, aspect)

    # Prefer shrinking the margin over padding with made-up background,
    # but never cut into the item itself.
    fit_scale = min(img_w / cw, img_h / ch)
    min_scale = tight_w / cw
    scale = min(1.0, max(fit_scale, min_scale))
    cw, ch = cw * scale, ch * scale

    def place(center: float, size: float, limit: int) -> float:
        start = center - size / 2
        if size <= limit:  # fits: slide inside the photo instead of padding
            start = min(max(start, 0.0), limit - size)
        return start

    left = place(cx, cw, img_w)
    top = place(cy, ch, img_h)
    return int(round(left)), int(round(top)), int(round(left + cw)), int(round(top + ch))


def _blurred_fill(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """A soft, blurred version of the photo covering out_w x out_h (used as padding)."""
    # Work at 1/8 scale: heavy blur on a full-size canvas would be slow.
    h, w = img.shape[:2]
    src = cv2.resize(img, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    tw, th = max(1, out_w // 8), max(1, out_h // 8)
    scale = max(tw / src.shape[1], th / src.shape[0])  # "cover", like CSS background-size
    cover = cv2.resize(src, (int(np.ceil(src.shape[1] * scale)), int(np.ceil(src.shape[0] * scale))))
    ty, tx = (cover.shape[0] - th) // 2, (cover.shape[1] - tw) // 2
    cover = cv2.GaussianBlur(cover[ty : ty + th, tx : tx + tw], (0, 0), sigmaX=6)
    return cv2.resize(cover, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def apply_crop(img: np.ndarray, box: Box) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    if x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h:
        return img[y0:y1, x0:x1].copy()

    out_w, out_h = x1 - x0, y1 - y0
    canvas = _blurred_fill(img, out_w, out_h)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, w), min(y1, h)
    canvas[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return canvas
