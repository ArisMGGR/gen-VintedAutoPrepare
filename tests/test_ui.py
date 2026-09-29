"""Smoke test: drive the real window offscreen with a fake AI provider."""

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QCoreApplication, QElapsedTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from vinted_prep import processor as processor_mod  # noqa: E402
from vinted_prep.config import Settings  # noqa: E402
from vinted_prep.imaging.detect import ItemDetector  # noqa: E402
from vinted_prep.models import Status  # noqa: E402
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


def test_main_window_flow(qapp, tmp_path, item_folder, monkeypatch):
    from vinted_prep.ui.main_window import MainWindow

    monkeypatch.setattr(Settings, "path", staticmethod(lambda: tmp_path / "settings.json"))
    monkeypatch.setattr(processor_mod, "get_provider", lambda s: FakeProvider())
    settings = Settings(output_dir=str(tmp_path / "out"), gemini_api_key="k")
    win = MainWindow(settings)
    win.processor.detector = ItemDetector(tmp_path / "none", allow_download=False)
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

    # Copy button puts the description + hashtags on the clipboard.
    win.panel.description.setPlainText("Lovely top.")
    assert win.panel._full_description() == "Lovely top.\n\n#top"
    win.close()
