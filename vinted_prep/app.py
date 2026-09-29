from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

from vinted_prep import APP_NAME, APP_USER_MODEL_ID
from vinted_prep.config import Settings, app_data_dir


def _setup_logging() -> None:
    handlers: list[logging.Handler] = [logging.FileHandler(app_data_dir() / "app.log", encoding="utf-8")]
    if sys.stderr:  # None in the windowed .exe
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)


def resource_path(relative: str) -> Path:
    """Files bundled by PyInstaller live under sys._MEIPASS; in development, next to the package."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / relative


def set_windows_app_id() -> bool:
    """Make Windows show our icon on the taskbar instead of python's.

    Without an explicit AppUserModelID the window is grouped under pythonw.exe.
    Must run before any window is created. Returns True if the ID is in effect."""
    if sys.platform != "win32":
        return True
    import ctypes

    try:
        shell32 = ctypes.windll.shell32
        shell32.SetCurrentProcessExplicitAppUserModelID(ctypes.c_wchar_p(APP_USER_MODEL_ID))
        current = ctypes.c_wchar_p()
        shell32.GetCurrentProcessExplicitAppUserModelID(ctypes.byref(current))
        ok = current.value == APP_USER_MODEL_ID
        ctypes.windll.ole32.CoTaskMemFree(current)
        return ok
    except (AttributeError, OSError) as exc:
        logging.warning("Could not set the taskbar app ID: %s", exc)
        return False


def app_icon_path() -> Path:
    # .ico holds every size Windows needs (16px taskbar, 32px alt-tab, 256px...)
    ico = resource_path("assets/icon.ico")
    return ico if ico.exists() else resource_path("assets/icon.png")


def selftest() -> int:
    """Checks the bundled libraries load and the photo pipeline runs (used by CI on the built .exe)."""
    import numpy as np
    import onnxruntime
    import pillow_heif  # noqa: F401
    from PySide6.QtWidgets import QApplication

    from vinted_prep.imaging.crop import apply_crop, compute_crop_box
    from vinted_prep.imaging.detect import detect_classic
    from vinted_prep.imaging.enhance import enhance

    app = QApplication.instance() or QApplication(["selftest", "-platform", "offscreen"])
    img = np.full((600, 800, 3), 210, np.uint8)
    img[150:450, 450:650] = (170, 30, 40)
    img = enhance(img)
    box = compute_crop_box(detect_classic(img), 800, 600)
    out = apply_crop(img, box)
    ok = abs(out.shape[1] / out.shape[0] - 0.8) < 0.01 and app_icon_path().exists() and set_windows_app_id()
    logging.info("selftest %s; onnxruntime providers: %s", "ok" if ok else "FAILED", onnxruntime.get_available_providers())
    del app
    return 0 if ok else 1


def main() -> int:
    _setup_logging()
    if "--selftest" in sys.argv:
        return selftest()
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication, QMessageBox

    from vinted_prep.ui.main_window import MainWindow

    set_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(str(app_icon_path())))

    def excepthook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        logging.error("Unhandled error:\n%s", text)
        QMessageBox.critical(None, "Unexpected error", f"{exc}\n\nDetails were written to {app_data_dir() / 'app.log'}")

    sys.excepthook = excepthook

    window = MainWindow(Settings.load())
    window.show()
    return app.exec()
