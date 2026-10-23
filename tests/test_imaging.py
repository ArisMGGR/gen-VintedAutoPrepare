import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms

from vinted_prep.imaging.crop import VINTED_ASPECT, apply_crop, compute_crop_box
from vinted_prep.imaging.detect import ItemDetector, detect_classic
from vinted_prep.imaging.enhance import auto_levels, enhance, white_balance
from vinted_prep.imaging.io import load_rgb
from vinted_prep.imaging.straighten import detect_tilt, rotate_and_crop


def _aspect(box):
    return (box[2] - box[0]) / (box[3] - box[1])


def _contains(outer, inner):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


# --- crop ---
def test_crop_is_4_5_centered_and_contains_item():
    bbox = (500, 300, 700, 600)
    box = compute_crop_box(bbox, 1200, 900, margin=0.08)
    assert _aspect(box) == pytest.approx(VINTED_ASPECT, abs=0.01)
    assert _contains(box, bbox)
    assert (box[0] + box[2]) / 2 == pytest.approx(600, abs=1)
    assert (box[1] + box[3]) / 2 == pytest.approx(450, abs=1)
    assert box[0] >= 0 and box[1] >= 0 and box[2] <= 1200 and box[3] <= 900


def test_crop_near_edge_slides_inside_photo_without_cutting_item():
    bbox = (10, 100, 210, 400)
    box = compute_crop_box(bbox, 1200, 900)
    assert box[0] == 0
    assert _contains(box, bbox)
    assert _aspect(box) == pytest.approx(VINTED_ASPECT, abs=0.01)


def test_crop_shrinks_margin_before_padding():
    # Item is 800 tall in a 900-tall photo: 8% margin would need 928px, so margin shrinks.
    bbox = (400, 50, 800, 850)
    box = compute_crop_box(bbox, 1200, 900, margin=0.08)
    assert box[1] >= 0 and box[3] <= 900
    assert _contains(box, bbox)


def test_wide_item_in_landscape_photo_gets_padded_not_cut():
    img = np.full((600, 1000, 3), 200, np.uint8)
    bbox = (0, 100, 1000, 500)
    box = compute_crop_box(bbox, 1000, 600)
    assert _contains(box, bbox)
    out = apply_crop(img, box)
    assert out.shape[1] / out.shape[0] == pytest.approx(VINTED_ASPECT, abs=0.01)
    assert out.shape[:2] == (box[3] - box[1], box[2] - box[0])


def test_no_detection_falls_back_to_centered_4_5():
    box = compute_crop_box(None, 1200, 900)
    assert _aspect(box) == pytest.approx(VINTED_ASPECT, abs=0.01)
    assert box[3] - box[1] == 900


# --- detection ---
def test_classic_detector_finds_item(photo_factory):
    img = photo_factory(box=(700, 200, 1000, 700))
    bbox = detect_classic(img)
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert abs(x0 - 700) < 30 and abs(y0 - 200) < 30 and abs(x1 - 1000) < 30 and abs(y1 - 700) < 30


def test_classic_detector_keeps_both_shoes(photo_factory):
    img = photo_factory(box=(200, 300, 450, 600))
    img[300:600, 600:850] = (170, 30, 40)
    bbox = detect_classic(img)
    assert bbox[0] < 230 and bbox[2] > 820


def test_detector_falls_back_when_model_missing(tmp_path, photo_factory):
    detector = ItemDetector(tmp_path / "none", allow_download=False)
    bbox, method = detector.detect(photo_factory())
    assert method == "classic" and bbox is not None


MODEL_DIR = os.environ.get("VINTED_MODEL_DIR")


@pytest.mark.skipif(not MODEL_DIR or not (Path(MODEL_DIR) / "u2net.onnx").exists(), reason="set VINTED_MODEL_DIR")
def test_ai_detector_finds_item(photo_factory):
    img = photo_factory(box=(700, 200, 1000, 700))
    bbox, method = ItemDetector(Path(MODEL_DIR), allow_download=False).detect(img)
    assert method == "ai"
    x0, y0, x1, y1 = bbox
    assert abs(x0 - 700) < 60 and abs(y0 - 200) < 60 and abs(x1 - 1000) < 60 and abs(y1 - 700) < 60


# --- straighten ---
def _lined_scene(angle_deg: float) -> np.ndarray:
    img = np.full((900, 1200, 3), 210, np.uint8)
    for y in range(100, 900, 120):  # floorboards / shelves
        cv2.line(img, (0, y), (1199, y), (60, 50, 40), 4)
    for x in (150, 1050):  # door frame
        cv2.line(img, (x, 0), (x, 899), (60, 50, 40), 4)
    m = cv2.getRotationMatrix2D((600, 450), -angle_deg, 1.0)  # negative = clockwise tilt
    return cv2.warpAffine(img, m, (1200, 900), borderMode=cv2.BORDER_REPLICATE)


@pytest.mark.parametrize("tilt", [2.0, -3.0])
def test_detect_and_fix_tilt(tilt):
    img = _lined_scene(tilt)
    found = detect_tilt(img, max_deg=5)
    assert found == pytest.approx(tilt, abs=0.4)
    fixed = rotate_and_crop(img, found)
    assert abs(detect_tilt(fixed, max_deg=5)) < 0.5
    assert fixed.shape[0] > 700 and fixed.shape[1] > 1000


def test_no_tilt_for_straight_or_featureless_photos():
    assert detect_tilt(_lined_scene(0.0)) == 0.0
    assert detect_tilt(np.full((600, 800, 3), 128, np.uint8)) == 0.0


def test_large_tilts_are_ignored():
    assert detect_tilt(_lined_scene(12.0), max_deg=5) == 0.0


# --- enhance ---
def test_dark_photo_gets_brighter(photo_factory):
    dark = (photo_factory().astype(np.float32) * 0.45).astype(np.uint8)
    out = auto_levels(dark, 1.0)
    assert out.mean() > dark.mean() + 8  # brighter, but deliberately gentle
    assert out.max() <= 255 and np.percentile(out, 99) < 250


def test_colour_cast_on_neutral_background_is_reduced():
    img = np.full((400, 400, 3), (170, 178, 205), np.uint8)  # bluish white wall
    img[100:300, 100:300] = (40, 40, 40)
    out = white_balance(img, 1.0)
    wall = out[10, 10].astype(int)
    assert (wall.max() - wall.min()) < (205 - 170) / 2


def test_colourful_item_without_neutral_pixels_is_not_greyed():
    img = np.full((300, 300, 3), (200, 20, 30), np.uint8)
    assert np.array_equal(white_balance(img, 1.0), img)


def test_enhance_zero_strength_is_identity(photo_factory):
    img = photo_factory()
    assert np.array_equal(enhance(img, 0.0), img)


# --- io ---
def test_load_applies_exif_orientation(tmp_path):
    img = Image.new("RGB", (400, 200), (255, 0, 0))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90 CW on display
    img.save(tmp_path / "rot.jpg", exif=exif)
    arr = load_rgb(tmp_path / "rot.jpg")
    assert arr.shape[:2] == (400, 200)


def _display_p3_profile() -> bytes:
    """The built-in sRGB profile with its primaries replaced by Display P3's (D50, as in Apple's profile)."""
    data = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    primaries = {
        b"rXYZ": (0.5151, 0.2412, -0.0011),
        b"gXYZ": (0.2920, 0.6922, 0.0419),
        b"bXYZ": (0.1571, 0.0666, 0.7841),
    }
    for i in range(int.from_bytes(data[128:132], "big")):  # tag table: signature, offset, size
        entry = 132 + 12 * i
        signature, offset = bytes(data[entry : entry + 4]), int.from_bytes(data[entry + 4 : entry + 8], "big")
        for j, value in enumerate(primaries.get(signature, ())):
            start = offset + 8 + 4 * j  # XYZType: 'XYZ ', reserved, then s15Fixed16 numbers
            data[start : start + 4] = round(value * 65536).to_bytes(4, "big", signed=True)
    return bytes(data)


def test_display_p3_photo_is_converted_to_srgb(tmp_path):
    Image.new("RGB", (8, 8), (200, 60, 60)).save(tmp_path / "p3.png", icc_profile=_display_p3_profile())
    Image.new("RGB", (8, 8), (200, 60, 60)).save(tmp_path / "plain.png")
    r, g, b = load_rgb(tmp_path / "p3.png")[0, 0].astype(int)
    assert r > 210 and g < 50  # the same numbers mean a more saturated red in P3
    assert load_rgb(tmp_path / "plain.png")[0, 0].tolist() == [200, 60, 60]  # untagged: unchanged


def test_tilt_found_in_dim_low_contrast_photo():
    img = _lined_scene(2.5).astype(np.float32)
    img = (120 + (img - 120) * 0.25) * 0.6  # faint lines, underexposed
    assert detect_tilt(img.astype(np.uint8)) == pytest.approx(2.5, abs=0.4)
