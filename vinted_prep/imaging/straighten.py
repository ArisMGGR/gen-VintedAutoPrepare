"""Detect and correct a slight camera tilt using long straight lines
(door frames, floorboards, table edges, hanger rails)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from vinted_prep.imaging.io import resize_max


def detect_tilt(img: np.ndarray, max_deg: float = 5.0) -> float:
    """Return the clockwise tilt of the photo in degrees (0 when unsure).

    A positive result means the scene appears rotated clockwise and should be
    rotated counter-clockwise by that amount.
    """
    small = resize_max(img, 1000)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge thresholds adapt to the photo's contrast (dim indoor shots have weak edges).
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    high = max(float(np.percentile(cv2.magnitude(gx, gy), 92)), 40.0)
    edges = cv2.Canny(gray, 0.4 * high, high, L2gradient=True)
    h, w = gray.shape
    min_len = 0.2 * min(h, w)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=80, minLineLength=min_len, maxLineGap=8)
    if lines is None:
        return 0.0

    devs, weights = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):  # (N,1,4) in OpenCV 4, (N,4) in 5
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle > 90:
            angle -= 180
        elif angle <= -90:
            angle += 180
        if abs(angle) <= max_deg:
            dev = angle
        elif abs(angle) >= 90 - max_deg:
            dev = angle - math.copysign(90, angle)
        else:
            continue
        devs.append(dev)
        weights.append(math.hypot(x2 - x1, y2 - y1))

    if len(devs) < 3:
        return 0.0
    devs_arr, w_arr = np.array(devs), np.array(weights)
    if w_arr.sum() < 1.5 * min(h, w):  # not enough straight structure to trust
        return 0.0

    order = np.argsort(devs_arr)
    cum = np.cumsum(w_arr[order])
    median = float(devs_arr[order][np.searchsorted(cum, cum[-1] / 2)])

    agree = w_arr[np.abs(devs_arr - median) <= 1.0].sum() / w_arr.sum()
    if agree < 0.5 or abs(median) < 0.3:
        return 0.0
    return median


def _max_inner_rect(w: float, h: float, angle_rad: float) -> tuple[float, float]:
    """Largest axis-aligned rectangle inside a w x h rectangle rotated by angle."""
    if w <= 0 or h <= 0:
        return 0.0, 0.0
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)
    sin_a, cos_a = abs(math.sin(angle_rad)), abs(math.cos(angle_rad))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


def rotate_and_crop(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate counter-clockwise by angle_deg and trim the empty corners."""
    if abs(angle_deg) < 1e-3:
        return img
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    wr, hr = _max_inner_rect(w, h, math.radians(angle_deg))
    x0, y0 = int(math.ceil((w - wr) / 2)), int(math.ceil((h - hr) / 2))
    x1, y1 = int(math.floor((w + wr) / 2)), int(math.floor((h + hr) / 2))
    if x1 - x0 < w * 0.5 or y1 - y0 < h * 0.5:
        return img
    return rotated[y0:y1, x0:x1].copy()
