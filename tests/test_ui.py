"""Smoke test: drive the real window offscreen with a fake AI provider."""

import copy
import json
import threading

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QCoreApplication, QElapsedTimer  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton  # noqa: E402

from vinted_prep import processor as processor_mod  # noqa: E402
from vinted_prep.config import Settings  # noqa: E402
from vinted_prep.imaging.detect import ItemDetector  # noqa: E402
from vinted_prep.models import Status  # noqa: E402
from tests.conftest import make_photo  # noqa: E402
from tests.test_processor import FakeProvider  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(cond, ms=20000):
    timer = QElapsedTimer()
    timer.start()
    while not cond() and timer.elapsed() < ms:
        QCoreApplication.processEvents()
    return cond()


def make_window(tmp_path, monkeypatch, provider):
    from vinted_prep.ui.main_window import MainWindow

    monkeypatch.setattr(Settings, "path", staticmethod(lambda: tmp_path / "settings.json"))
    monkeypatch.setattr(processor_mod, "get_provider", lambda s: provider)
    win = MainWindow(Settings(output_dir=str(tmp_path / "out"), gemini_api_key="k"))
    win.processor.detector = ItemDetector(tmp_path / "none", allow_download=False)
    return win


def item_dir(path):
    path.mkdir(parents=True)
    Image.fromarray(make_photo()).save(path / "IMG_1.jpg")
    return path


def test_main_window_flow(qapp, tmp_path, item_folder, monkeypatch):
    win = make_window(tmp_path, monkeypatch, FakeProvider())
    win.show()

    win.add_folders([item_folder.parent])  # parent folder expands to the item folder
    assert len(win.items) == 1
    item = next(iter(win.items.values()))
    assert win.panel.item is item and win.panel.photos.count() == 3  # original previews

    win.panel.notes.setPlainText("worn once")
    assert item.notes == "worn once"

    win.process_all()
    assert wait_until(lambda: not win.queued)
    assert item.status == Status.DONE, item.message
    assert win.panel.title.text() == "Red top"
    assert win.panel.photos.count() == 3
    assert "6 €" in win.queue.item(0).text()

    # Edits are written back to disk.
    win.panel.price.setValue(9.5)
    win._flush_saves()
    assert "9.50 €" in (item.output_dir / "listing.txt").read_text(encoding="utf-8")

    # The description's Copy button puts the description + hashtags on the clipboard.
    win.panel.description.setPlainText("Lovely top.")
    copy_button = next(b for b in win.panel.findChildren(QPushButton) if b.toolTip().startswith("Copies the description"))
    copy_button.click()
    assert QGuiApplication.clipboard().text() == "Lovely top.\n\n#top"
    win.close()


def test_same_named_folders_get_separate_output_folders(qapp, tmp_path, monkeypatch):
    win = make_window(tmp_path, monkeypatch, FakeProvider())
    win.add_folders([item_dir(tmp_path / "spring" / "shirt"), item_dir(tmp_path / "summer" / "shirt")])
    first, second = win.items.values()
    assert first.output_dir != second.output_dir
    win.close()


class HoldingProvider(FakeProvider):
    """Keeps the AI step of an item whose notes are 'hold' busy until release is set."""

    def __init__(self):
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()

    def analyze(self, images_b64, notes, language):
        if notes == "hold":
            self.entered.set()
            self.release.wait(20)
        return super().analyze(images_b64, notes, language)


def test_stop_keeps_queued_edits_and_lets_new_jobs_run(qapp, tmp_path, monkeypatch):
    provider = HoldingProvider()
    win = make_window(tmp_path, monkeypatch, provider)
    try:
        item_dir(tmp_path / "src" / "A")
        item_dir(tmp_path / "src" / "B")
        win.add_folders([tmp_path / "src"])
        a, b = (next(i for i in win.items.values() if i.name == name) for name in "AB")
        win.process_all()
        assert wait_until(lambda: not win.queued)  # both saved once

        a.notes = "hold"
        win.process([a.id, b.id])  # A waits on the AI, B waits in the queue
        assert provider.entered.wait(20) and wait_until(lambda: win.running == a.id)
        win.queue.setCurrentItem(win._entry(b.id))
        win.panel.notes.setPlainText("small mark on left sleeve")
        win._flush_saves()

        win.stop()  # A is still inside the AI call; B leaves the queue
        win._flush_saves()
        saved = json.loads((b.output_dir / "listing.json").read_text(encoding="utf-8"))
        assert saved["notes"] == "small mark on left sleeve"

        win.process([b.id])  # queued while the stopped job is still finishing
        provider.release.set()
        assert wait_until(lambda: not win.queued)
        assert b.status == Status.DONE, b.message
    finally:
        provider.release.set()
        wait_until(lambda: not win.queued)  # closing with jobs queued would open a "Quit?" dialog
        win.close()


def test_photo_toggle_queued_behind_a_job_keeps_that_jobs_status(qapp, tmp_path, monkeypatch):
    provider = HoldingProvider()
    win = make_window(tmp_path, monkeypatch, provider)
    try:
        item_dir(tmp_path / "src" / "A")
        item_dir(tmp_path / "src" / "B")
        win.add_folders([tmp_path / "src"])
        a, b = (next(i for i in win.items.values() if i.name == name) for name in "AB")
        provider.fail = True  # e.g. out of free quota
        win.process([b.id])
        assert wait_until(lambda: not win.queued) and b.status == Status.ERROR

        provider.fail = False
        a.notes = "hold"
        win.process([a.id, b.id])  # A waits on the AI, B waits in the queue
        assert provider.entered.wait(20) and wait_until(lambda: win.running == a.id)
        win._toggle_crop(b.id, 0, True)  # right-click → "Use full photo" on B while it waits
        provider.release.set()
        assert win.runner.wait(20000)
        QCoreApplication.processEvents()
        assert b.status == Status.DONE, b.message  # not the error B had when the photo was clicked
        assert b.photos[0].uncropped
    finally:
        provider.release.set()
        win.runner.wait(20000)
        wait_until(lambda: not win.queued)  # closing with jobs queued would open a "Quit?" dialog
        win.close()


def test_remove_saves_an_edit_made_just_before(qapp, tmp_path, monkeypatch):
    win = make_window(tmp_path, monkeypatch, FakeProvider())
    win.add_folders([item_dir(tmp_path / "src" / "A")])
    item = next(iter(win.items.values()))
    win.process_all()
    assert wait_until(lambda: not win.queued)
    win.panel.title.setText("Edited")  # the 700 ms save timer has not fired yet
    win.queue.setCurrentItem(win._entry(item.id))
    win.remove_selected()
    assert not win.items
    saved = json.loads((item.output_dir / "listing.json").read_text(encoding="utf-8"))
    assert saved["listing"]["title"] == "Edited"
    win.close()


def test_item_stays_busy_until_its_last_photo_toggle_is_done(qapp, tmp_path, monkeypatch):
    win = make_window(tmp_path, monkeypatch, FakeProvider())
    gate, calls, real = threading.Event(), [], win.processor.reprocess_photo
    try:
        win.add_folders([item_dir(tmp_path / "src" / "A")])
        a = next(iter(win.items.values()))
        win.process_all()
        assert wait_until(lambda: not win.queued)


        def reprocess(item, index, uncropped):
            calls.append(uncropped)
            if len(calls) == 2:
                gate.wait(20)  # the second toggle is still running
            return real(item, index, uncropped)

        monkeypatch.setattr(win.processor, "reprocess_photo", reprocess)
        win._toggle_crop(a.id, 0, True)  # two right-click toggles in a row
        win._toggle_crop(a.id, 0, False)
        assert wait_until(lambda: len(calls) == 2 and win.running == a.id)
        assert a.id in win.queued and win.stop_action.isEnabled()
        win.queue.setCurrentItem(win._entry(a.id))
        win.remove_selected()
        assert a.id in win.items  # can't remove while one of its jobs runs
        gate.set()
        assert wait_until(lambda: not win.queued)
        assert not a.photos[0].uncropped  # the last toggle wins
    finally:
        gate.set()
        win.runner.wait(20000)
        wait_until(lambda: not win.queued)
        win.close()


def test_new_output_folder_applies_to_items_not_saved_yet(qapp, tmp_path, monkeypatch, item_folder):
    from vinted_prep.ui import main_window

    class Dialog:  # stands in for the Settings dialog: the user picks another folder and saves
        def __init__(self, settings, parent):
            self.settings = copy.deepcopy(settings)
            self.settings.output_dir = str(tmp_path / "elsewhere")

        def exec(self):
            return True

    win = make_window(tmp_path, monkeypatch, FakeProvider())
    win.add_folders([item_folder])
    item = next(iter(win.items.values()))
    monkeypatch.setattr(main_window, "SettingsDialog", Dialog)
    win.open_settings()
    assert item.output_dir == tmp_path / "elsewhere" / item_folder.name
    win.close()


def test_choosing_ollama_instead_of_a_gemini_key_processes_at_once(qapp, tmp_path, monkeypatch, item_folder):
    from vinted_prep.ui import main_window

    class Dialog:  # stands in for the Settings dialog: the user picks Ollama instead of pasting a key
        def __init__(self, settings, parent):
            self.settings = copy.deepcopy(settings)
            self.settings.provider = "ollama"

        def exec(self):
            return True

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    win = make_window(tmp_path, monkeypatch, FakeProvider())
    win.settings.gemini_api_key = ""
    monkeypatch.setattr(main_window, "SettingsDialog", Dialog)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))  # "Open Settings now?"
    win.add_folders([item_folder])
    item = next(iter(win.items.values()))
    win.process_all()
    assert wait_until(lambda: not win.queued)
    assert item.status == Status.DONE, item.message
    win.close()


def test_refresh_uses_gemini_key_from_environment(qapp, monkeypatch):
    from vinted_prep.ui import settings_dialog

    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    keys = []
    monkeypatch.setattr(settings_dialog.GeminiProvider, "list_models", lambda self: keys.append(self.api_key) or ["m"])
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    settings_dialog.SettingsDialog(Settings())._refresh_gemini()
    assert keys == ["env-key"]


def test_refresh_strips_the_ollama_url(qapp, monkeypatch):
    from vinted_prep.ui import settings_dialog

    urls = []
    monkeypatch.setattr(settings_dialog.OllamaProvider, "list_models", lambda self: urls.append(self.base_url) or ["m"])
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    dialog = settings_dialog.SettingsDialog(Settings())
    dialog.ollama_url.setText("http://localhost:11434 ")  # pasted with a trailing space
    dialog._refresh_ollama()
    assert urls == ["http://localhost:11434"]  # what Save stores


def test_failed_refresh_does_not_leave_a_busy_cursor(qapp):
    from vinted_prep.ui.settings_dialog import SettingsDialog

    class Broken:
        def list_models(self):
            raise ValueError("reply was not JSON")

    dialog = SettingsDialog(Settings())
    with pytest.raises(ValueError):
        dialog._fill_models(dialog.ollama_model, Broken())
    assert QGuiApplication.overrideCursor() is None
