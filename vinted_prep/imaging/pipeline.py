"""Per-photo processing: orient -> straighten -> enhance -> find item -> crop 4:5."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from vinted_prep.config import Settings
from vinted_prep.imaging.crop import apply_crop, compute_crop_box
from vinted_prep.imaging.detect import ItemDetector
from vinted_prep.imaging.enhance import enhance
from vinted_prep.imaging.io import load_rgb
from vinted_prep.imaging.straighten import detect_tilt, rotate_and_crop
from vinted_prep.models import PhotoResult


def process_photo(
    path: Path,
    settings: Settings,
    detector: ItemDetector | None,
    uncropped: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, PhotoResult]:
    result = PhotoResult(source=path, uncropped=uncropped)
    img = load_rgb(path)  # EXIF orientation applied here

    if settings.auto_straighten:
        angle = detect_tilt(img, settings.max_straighten_deg)
        if angle:
            img = rotate_and_crop(img, angle)
            result.rotation_deg = angle

    if settings.auto_enhance:
        img = enhance(img, settings.enhance_strength)

    if uncropped:
        return img, result

    bbox, method = detector.detect(img, progress) if detector else (None, "none")
    result.detector = method
    h, w = img.shape[:2]
    box = compute_crop_box(bbox, w, h, margin=settings.crop_margin)
    result.crop_box = box
    return apply_crop(img, box), result
