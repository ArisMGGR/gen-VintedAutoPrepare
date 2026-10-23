import json
from pathlib import Path

import pytest
from PIL import Image

from vinted_prep import processor as processor_mod
from vinted_prep.ai import AIError
from vinted_prep.ai.base import Provider
from vinted_prep.config import Settings
from vinted_prep.export import LISTING_JSON, LISTING_TXT, assign_output_dir, listing_text, load_saved
from vinted_prep.imaging.detect import ItemDetector
from vinted_prep.models import Item, Listing, Status, expand_dropped_folder, list_images
from vinted_prep.processor import Processor, pick_evenly


class FakeProvider(Provider):
    name = "Fake"

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def analyze(self, images_b64, notes, language):
        self.calls.append((len(images_b64), notes, language))
        if self.fail:
            raise AIError("quota")
        return Listing(title="Red top", description="Nice red top.", brand="Zara", hashtags=["top"], price=6)

    def list_models(self):
        return []


@pytest.fixture
def fake_provider(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(processor_mod, "get_provider", lambda settings: provider)
    return provider


@pytest.fixture
def proc(tmp_path):
    settings = Settings(output_dir=str(tmp_path / "out"), language="English", max_ai_photos=2)
    return Processor(settings, ItemDetector(tmp_path / "no-model", allow_download=False))


def test_list_images_natural_order(item_folder):
    assert [p.name for p in list_images(item_folder)] == ["IMG_1.png", "IMG_2.jpg", "IMG_10.jpg"]


def test_expand_parent_folder(tmp_path, item_folder):
    assert expand_dropped_folder(item_folder) == [item_folder]
    assert expand_dropped_folder(item_folder.parent) == [item_folder]
    (tmp_path / "empty").mkdir()
    assert expand_dropped_folder(tmp_path / "empty") == []


def test_pick_evenly():
    assert pick_evenly([1, 2, 3], 5) == [1, 2, 3]
    assert pick_evenly(list(range(10)), 3) == [0, 4, 9]  # first and last kept
    assert pick_evenly(list(range(10)), 1) == [0]


def test_full_item_run(proc, item_folder, fake_provider):
    item = Item(item_folder, assign_output_dir(proc.settings.resolved_output_dir(), item_folder), notes="worn once")
    msgs = []
    proc.process(item, msgs.append)
    assert item.status == Status.DONE, item.message
    outputs = sorted(p.name for p in item.output_dir.glob("*.jpg"))
    assert outputs == ["01.jpg", "02.jpg", "03.jpg"]
    for name in outputs:
        w, h = Image.open(item.output_dir / name).size
        assert w / h == pytest.approx(0.8, abs=0.01)
    assert fake_provider.calls == [(2, "worn once", "English")]
    data = json.loads((item.output_dir / LISTING_JSON).read_text(encoding="utf-8"))
    assert data["listing"]["brand"] == "Zara" and data["notes"] == "worn once"
    assert "#top" in (item.output_dir / LISTING_TXT).read_text(encoding="utf-8")
    assert any("Photo 3/3" in m for m in msgs)

    # Reopening the same folder restores the result without reprocessing.
    again = Item(item_folder, assign_output_dir(proc.settings.resolved_output_dir(), item_folder))
    assert again.output_dir == item.output_dir
    assert load_saved(again) and again.status == Status.DONE
    assert again.listing.title == "Red top" and len(again.photos) == 3


def test_uncropped_photo_keeps_full_frame(proc, item_folder, fake_provider):
    item = Item(item_folder, proc.settings.resolved_output_dir() / "x")
    proc.process(item, lambda m: None)
    proc.reprocess_photo(item, 1, uncropped=True)  # IMG_2.jpg, 1200x900
    assert Image.open(item.photos[1].output).size == (1200, 900)
    assert item.photos[1].uncropped
    proc.process(item, lambda m: None, ai=False)  # full reprocess keeps the choice
    assert Image.open(item.photos[1].output).size == (1200, 900)


def test_ai_failure_keeps_photos_and_reports(proc, item_folder, monkeypatch):
    monkeypatch.setattr(processor_mod, "get_provider", lambda s: FakeProvider(fail=True))
    item = Item(item_folder, proc.settings.resolved_output_dir() / "x")
    proc.process(item, lambda m: None)
    assert item.status == Status.ERROR and item.message == "quota"
    assert len(list(item.output_dir.glob("*.jpg"))) == 3


def test_cancel(proc, item_folder, fake_provider):
    item = Item(item_folder, proc.settings.resolved_output_dir() / "x")
    proc.cancel_event.set()
    proc.process(item, lambda m: None)
    assert item.status == Status.PENDING and item.message == "Stopped"


def test_empty_folder_is_an_error(proc, tmp_path):
    (tmp_path / "nothing").mkdir()
    item = Item(tmp_path / "nothing", tmp_path / "o")
    proc.process(item, lambda m: None)
    assert item.status == Status.ERROR and "No photos" in item.message


def test_output_dir_name_collision(tmp_path):
    root = tmp_path / "out"
    a, b = tmp_path / "a" / "shirt", tmp_path / "b" / "shirt"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    first = assign_output_dir(root, a)
    first.mkdir(parents=True)
    (first / LISTING_JSON).write_text(json.dumps({"source_dir": str(a.resolve())}))
    assert assign_output_dir(root, a) == first
    assert assign_output_dir(root, b) == root / "shirt (2)"


def test_listing_text():
    text = listing_text(Listing(title="T", description="D", brand="B", hashtags=["x"], price=7.5))
    assert "TITLE\nT" in text and "D\n\n#x" in text and "Brand: B" in text and "7.50 €" in text
    assert "Size:" not in text


def test_settings_roundtrip(tmp_path):
    path = tmp_path / "s.json"
    Settings(provider="ollama", max_ai_photos=3).save(path)
    loaded = Settings.load(path)
    assert loaded.provider == "ollama" and loaded.max_ai_photos == 3
    path.write_text('{"max_ai_photos": "oops", "unknown": 1}')
    assert Settings.load(path).max_ai_photos == Settings().max_ai_photos
    assert Settings.load(tmp_path / "missing.json") == Settings()
