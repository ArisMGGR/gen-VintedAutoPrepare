import json
import threading
import time
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


def test_unreadable_subfolder_is_skipped(tmp_path, item_folder, monkeypatch):
    (tmp_path / "locked").mkdir()  # like the hidden 'My Music' link inside Documents
    real_iterdir = Path.iterdir

    def iterdir(self):
        if self.name == "locked":
            raise PermissionError(13, "Access is denied", str(self))
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    assert expand_dropped_folder(tmp_path) == [item_folder]


def test_natural_order_with_superscript_digits(tmp_path):
    for name in ("IMG_3.jpg", "IMG_1²2.jpg"):
        Image.new("RGB", (8, 8)).save(tmp_path / name)
    assert [p.name for p in list_images(tmp_path)] == ["IMG_1²2.jpg", "IMG_3.jpg"]


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


def test_stop_cancels_the_running_job_but_not_later_ones(proc, item_folder, fake_provider):
    item = Item(item_folder, proc.settings.resolved_output_dir() / "x")
    stopped = []

    def progress(msg):
        if not stopped:  # press Stop while the first photo is being processed
            stopped.append(msg)
            proc.stop()

    proc.process(item, progress)
    assert item.status == Status.PENDING and item.message == "Stopped"
    later = Item(item_folder, proc.settings.resolved_output_dir() / "y")
    proc.process(later, lambda m: None)
    assert later.status == Status.DONE, later.message


def test_stop_does_not_wait_for_a_slow_ai_reply(proc, item_folder, monkeypatch):
    release = threading.Event()

    class Slow(FakeProvider):
        def analyze(self, images_b64, notes, language):
            proc.stop()  # Stop pressed while the reply is still on its way
            release.wait(30)
            return super().analyze(images_b64, notes, language)

    monkeypatch.setattr(processor_mod, "get_provider", lambda s: Slow())
    item = Item(item_folder, proc.settings.resolved_output_dir() / "x")
    start = time.monotonic()
    try:
        proc.process(item, lambda m: None)
    finally:
        release.set()
    assert item.status == Status.PENDING and item.message == "Stopped"
    assert item.listing is None
    assert time.monotonic() - start < 15  # photos only, not the 30 s reply


def test_fallback_note_gives_the_reason(proc, item_folder, monkeypatch):
    class FellBack(FakeProvider):
        model, used_model, skip_reason = "gemini-typo", "gemini-flash-latest", "was not found"

    monkeypatch.setattr(processor_mod, "get_provider", lambda s: FellBack())
    item = Item(item_folder, proc.settings.resolved_output_dir() / "x")
    proc.process(item, lambda m: None)
    assert item.message == "Ready (written by gemini-flash-latest: gemini-typo was not found)"


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
    # Nothing written yet, but another item already has the folder.
    assert assign_output_dir(tmp_path / "new", b, taken={tmp_path / "new" / "shirt"}) == tmp_path / "new" / "shirt (2)"


def test_drive_root_gets_its_own_output_folder(tmp_path):
    drive = Path(tmp_path.anchor)  # e.g. a USB stick's E:\ holding photos directly
    assert Item(drive, tmp_path).name
    out = assign_output_dir(tmp_path / "out", drive)
    assert out.parent == tmp_path / "out" and out.name == Item(drive, tmp_path).name


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        '"text"',
        '{"photos": [{"file": "01.jpg"}, "junk"], "notes": 5, "listing": "x"}',
        '{"photos": {"a": 1}, "listing": {"title": 7, "hashtags": "abc"}}',
    ],
)
def test_damaged_listing_json_does_not_stop_the_folder_being_added(tmp_path, item_folder, content):
    root = tmp_path / "out"
    (root / item_folder.name).mkdir(parents=True)
    (root / item_folder.name / LISTING_JSON).write_text(content, encoding="utf-8")
    assert assign_output_dir(root, item_folder) == root / f"{item_folder.name} (2)"  # not ours: must not raise
    item = Item(item_folder, root / item_folder.name)
    load_saved(item)  # must not raise
    assert item.photos == [] and isinstance(item.notes, str)
    assert item.listing is None or (item.listing.title == "7" and item.listing.hashtags == [])


def test_listing_json_with_bom_is_reloaded(proc, item_folder, fake_provider):
    item = Item(item_folder, assign_output_dir(proc.settings.resolved_output_dir(), item_folder))
    proc.process(item, lambda m: None)
    path = item.output_dir / LISTING_JSON
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())  # saved again by an editor that adds a BOM
    again = Item(item_folder, assign_output_dir(proc.settings.resolved_output_dir(), item_folder))
    assert again.output_dir == item.output_dir
    assert load_saved(again) and again.listing.title == "Red top"


def test_listing_json_in_utf16_is_reloaded(proc, item_folder, fake_provider):
    item = Item(item_folder, assign_output_dir(proc.settings.resolved_output_dir(), item_folder))
    proc.process(item, lambda m: None)
    path = item.output_dir / LISTING_JSON
    path.write_bytes(path.read_text(encoding="utf-8").encode("utf-16"))  # re-saved as "Unicode" in Notepad
    again = Item(item_folder, assign_output_dir(proc.settings.resolved_output_dir(), item_folder))
    assert again.output_dir == item.output_dir  # not a new '(2)' folder
    assert load_saved(again) and again.listing.title == "Red top"


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
    path.write_text("[]")
    assert Settings.load(path) == Settings()
    # 0 and "" would become False through bool(), so these only stay True if non-bools are skipped.
    path.write_text('{"output_dir": null, "auto_enhance": 0, "auto_straighten": "", "max_ai_photos": 0, "crop_margin": 5}')
    loaded = Settings.load(path)
    assert loaded.output_dir == "" and loaded.auto_enhance is True and loaded.auto_straighten is True
    assert loaded.max_ai_photos == 1 and loaded.crop_margin == 0.4  # clamped to what Settings offers
    path.write_text('{"enhance_strength": NaN, "crop_margin": Infinity, "max_straighten_deg": -Infinity}')
    assert Settings.load(path) == Settings()  # NaN would pass min/max and turn photos black
    path.write_text('{"max_ai_photos": Infinity, "jpeg_quality": 1e999}')  # int(inf) raises OverflowError
    assert Settings.load(path) == Settings()


def test_settings_with_bom_load(tmp_path):
    path = tmp_path / "s.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"gemini_api_key": "abc", "language": "French"}).encode())
    loaded = Settings.load(path)
    assert loaded.gemini_api_key == "abc" and loaded.language == "French"


def test_unknown_provider_falls_back_to_gemini(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"provider": " Ollama "}')  # hand-edited
    assert Settings.load(path).provider == "ollama"
    path.write_text('{"provider": "chatgpt"}')  # the toolbar would say Ollama while Gemini ran
    assert Settings.load(path).provider == "gemini"


def test_settings_in_other_encodings_load_or_fall_back(tmp_path):
    path = tmp_path / "s.json"
    # Windows PowerShell 5.1:  '{...}' > settings.json  writes UTF-16 with a BOM
    path.write_bytes(json.dumps({"gemini_api_key": "abc"}).encode("utf-16"))
    assert Settings.load(path).gemini_api_key == "abc"
    path.write_bytes('{"output_dir": "C:\\\\Φωτογραφίες"}'.encode("cp1253"))  # ANSI save on Greek Windows
    assert Settings.load(path) == Settings()  # defaults, not a crash on start-up
    assert (tmp_path / "s.json.unreadable").exists()


def test_unreadable_settings_are_kept_before_defaults_replace_them(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"gemini_api_key": "abc",}')  # hand-edited, trailing comma
    assert Settings.load(path) == Settings()
    Settings().save(path)
    assert "abc" in (tmp_path / "s.json.unreadable").read_text(encoding="utf-8")
