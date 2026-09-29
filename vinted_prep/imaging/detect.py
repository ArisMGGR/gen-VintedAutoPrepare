"""Find the item in a photo and return its bounding box.

Primary method: U^2-Net salient-object segmentation (ONNX, runs on the GPU via
DirectML on Windows, else CPU). The model (~170 MB) is downloaded once on first use.
Fallback: a classic OpenCV "differs from the background" detector, used when the
model is unavailable (offline on first run, onnxruntime missing, ...).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import requests

from vinted_prep.imaging.io import resize_max

log = logging.getLogger(__name__)

BBox = tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)

MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
MODEL_FILE = "u2net.onnx"
MODEL_MD5 = "60024c5c889badc19c04ad937298a77b"
INPUT_SIZE = 320


def _union_of_main_components(mask: np.ndarray, min_rel: float = 0.15, min_abs: float = 0.002) -> BBox | None:
    """Bounding box of the largest blob plus any blob at least min_rel of its size
    (keeps both shoes of a pair, both parts of a set)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = areas.max()
    total = mask.shape[0] * mask.shape[1]
    keep = [i + 1 for i, a in enumerate(areas) if a >= largest * min_rel and a >= total * min_abs]
    if not keep:
        return None
    x0 = min(stats[i, cv2.CC_STAT_LEFT] for i in keep)
    y0 = min(stats[i, cv2.CC_STAT_TOP] for i in keep)
    x1 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in keep)
    y1 = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in keep)
    return int(x0), int(y0), int(x1), int(y1)


def _scale_bbox(bbox: BBox, sx: float, sy: float, w: int, h: int) -> BBox:
    x0, y0, x1, y1 = bbox
    return (
        max(0, int(x0 * sx)),
        max(0, int(y0 * sy)),
        min(w, int(round(x1 * sx))),
        min(h, int(round(y1 * sy))),
    )


def detect_classic(img: np.ndarray) -> BBox | None:
    """Item = the region that differs from the photo's border colour, plus strong edges."""
    h, w = img.shape[:2]
    small = resize_max(img, 600)
    sh, sw = small.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)

    b = max(2, int(0.04 * min(sh, sw)))
    border = np.concatenate(
        [lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3), lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)]
    )
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    dist = np.clip(dist * (255.0 / max(dist.max(), 1.0)), 0, 255).astype(np.uint8)
    _, fg = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 60, 160), np.ones((3, 3), np.uint8))
    mask = cv2.bitwise_or(fg, edges)

    k = max(3, int(0.03 * min(sh, sw)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bbox = _union_of_main_components(mask > 0, min_rel=0.25, min_abs=0.01)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    if (x1 - x0) * (y1 - y0) > 0.97 * sw * sh:  # "everything" = nothing found
        return None
    return _scale_bbox(bbox, w / sw, h / sh, w, h)


class ItemDetector:
    def __init__(self, model_dir: Path, allow_download: bool = True):
        self.model_path = Path(model_dir) / MODEL_FILE
        self.allow_download = allow_download
        self._session = None
        self._failed = False
        self._lock = threading.Lock()

    # --- model management ---
    def _download(self, progress: Callable[[str], None] | None) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.model_path.with_suffix(".part")
        md5 = hashlib.md5()
        with requests.get(MODEL_URL, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            done = 0
            last_pct = -1
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    md5.update(chunk)
                    done += len(chunk)
                    pct = int(done * 100 / total) if total else -1
                    if progress and pct != last_pct and pct % 5 == 0:
                        progress(f"Downloading item-detection model (one time): {pct}%")
                        last_pct = pct
        if md5.hexdigest() != MODEL_MD5:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("Downloaded model file is corrupt or incomplete")
        tmp.replace(self.model_path)

    def _get_session(self, progress: Callable[[str], None] | None = None):
        with self._lock:
            if self._session is not None or self._failed:
                return self._session
            try:
                import onnxruntime as ort

                if not self.model_path.exists():
                    if not self.allow_download:
                        raise FileNotFoundError(self.model_path)
                    self._download(progress)
                available = ort.get_available_providers()
                preferred = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in available]
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                self._session = ort.InferenceSession(
                    str(self.model_path), sess_options=opts, providers=preferred + ["CPUExecutionProvider"]
                )
                log.info("Detector ready (%s)", self._session.get_providers()[0])
            except Exception as exc:  # any failure -> classic fallback for this session
                log.warning("AI item detector unavailable, using classic detector: %s", exc)
                self._failed = True
            return self._session

    # --- detection ---
    def saliency_mask(self, img: np.ndarray, progress: Callable[[str], None] | None = None) -> np.ndarray | None:
        """Probability map (INPUT_SIZE x INPUT_SIZE, 0..1) of the main object, or None."""
        session = self._get_session(progress)
        if session is None:
            return None
        x = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
        x /= max(float(x.max()), 1e-6)
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        out = session.run(None, {session.get_inputs()[0].name: x})[0][0, 0]
        lo, hi = float(out.min()), float(out.max())
        return (out - lo) / (hi - lo + 1e-8)

    def detect(self, img: np.ndarray, progress: Callable[[str], None] | None = None) -> tuple[BBox | None, str]:
        h, w = img.shape[:2]
        try:
            prob = self.saliency_mask(img, progress)
        except Exception as exc:
            log.warning("Saliency detection failed: %s", exc)
            prob = None
        if prob is not None:
            mask = prob > 0.5
            if mask.mean() > 0.005:
                bbox = _union_of_main_components(mask)
                if bbox is not None:
                    return _scale_bbox(bbox, w / INPUT_SIZE, h / INPUT_SIZE, w, h), "ai"
        bbox = detect_classic(img)
        return bbox, ("classic" if bbox else "none")
