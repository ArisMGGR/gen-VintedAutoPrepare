"""Mild, conservative photo enhancement: white balance, levels and brightness.

Everything here is deliberately gentle: the goal is a cleaner, brighter photo that
still shows the item's true colour (buyers complain when colours don't match).
"""

from __future__ import annotations

import cv2
import numpy as np


def white_balance(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Neutralise colour casts using near-grey pixels only.

    Plain grey-world would turn a red dress grey, so the illuminant is estimated
    from low-saturation, reasonably bright pixels (walls, white backgrounds, hangers).
    If there aren't enough of those, the photo is left alone.
    """
    if strength <= 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    neutral = (sat < 60) & (val > 90) & (val < 250)
    if neutral.mean() < 0.03:
        return img
    means = img[neutral].astype(np.float32).mean(axis=0)
    gains = means.mean() / np.maximum(means, 1.0)
    gains = np.clip(gains, 0.85, 1.18)
    gains = 1.0 + (gains - 1.0) * strength
    if np.all(np.abs(gains - 1.0) < 0.01):
        return img
    out = img.astype(np.float32) * gains
    return np.clip(out, 0, 255).astype(np.uint8)


def auto_levels(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Stretch lightness to the full range and lift dark midtones (phone photos indoors)."""
    if strength <= 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = lab[..., 0].astype(np.float32)
    lo, hi = np.percentile(L, (0.5, 99.5))
    lo = min(lo, 30.0)  # never clip real shadows away
    scale = min(255.0 / max(hi - lo, 1.0), 1.6)  # capped so noise isn't amplified much
    L2 = np.clip((L - lo) * scale, 0, 255)

    median = float(np.median(L2)) / 255.0
    if 0.02 < median < 0.45:  # dark photo: lift midtones with a gamma curve
        gamma = np.clip(np.log(0.47) / np.log(median), 0.7, 1.0)
        L2 = 255.0 * np.power(L2 / 255.0, gamma)

    L_final = L + (L2 - L) * strength
    lab[..., 0] = np.clip(L_final, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def enhance(img: np.ndarray, strength: float = 0.6) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    return auto_levels(white_balance(img, strength), strength)
