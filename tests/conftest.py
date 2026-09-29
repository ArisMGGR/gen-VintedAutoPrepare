import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def make_photo(w=1200, h=900, box=(700, 200, 1000, 700), bg=(225, 222, 218), fg=(170, 30, 40), noise=True):
    """A 'garment' (coloured rectangle with some texture) on a plain wall-like background."""
    rng = np.random.default_rng(0)
    img = np.full((h, w, 3), bg, np.uint8)
    x0, y0, x1, y1 = box
    img[y0:y1, x0:x1] = fg
    cv2.line(img, (x0 + 20, y0 + 40), (x1 - 20, y0 + 40), (240, 240, 240), 6)  # a stripe/detail
    if noise:
        img = np.clip(img.astype(np.int16) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def photo_factory():
    return make_photo


@pytest.fixture
def item_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "red top"
    folder.mkdir()
    Image.fromarray(make_photo()).save(folder / "IMG_2.jpg", quality=92)
    Image.fromarray(make_photo(box=(100, 150, 500, 800))).save(folder / "IMG_10.jpg", quality=92)
    Image.fromarray(make_photo(w=900, h=1200, box=(300, 300, 600, 900))).save(folder / "IMG_1.png")
    (folder / "notes.txt").write_text("not an image")
    return folder
