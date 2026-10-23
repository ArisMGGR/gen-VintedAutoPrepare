"""Runs the full job for one item: photos -> AI listing -> saved output. UI-independent."""

from __future__ import annotations

import threading
from typing import Callable

from vinted_prep.ai import AIError, get_provider
from vinted_prep.config import Settings, app_data_dir
from vinted_prep.export import photo_filename, remove_stale_photos, save_item
from vinted_prep.imaging.detect import ItemDetector
from vinted_prep.imaging.io import load_rgb, save_jpeg, to_jpeg_b64
from vinted_prep.imaging.pipeline import process_photo
from vinted_prep.models import Item, Status, list_images

Progress = Callable[[str], None]


class Cancelled(Exception):
    pass


def pick_evenly(items: list, n: int) -> list:
    """Up to n items spread over the list, always keeping the first and last
    (first is usually the front view, last ones are often labels)."""
    if n <= 0:
        return []
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[0]]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


class Processor:
    def __init__(self, settings: Settings, detector: ItemDetector | None = None):
        self.settings = settings
        self.detector = detector or ItemDetector(app_data_dir() / "models")
        self.cancel_event = threading.Event()

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    def process_photos(self, item: Item, progress: Progress) -> None:
        sources = list_images(item.source_dir)
        if not sources:
            raise RuntimeError("No photos found in this folder.")
        uncropped = {p.source: p.uncropped for p in item.photos}
        results = []
        for i, src in enumerate(sources):
            self._check_cancel()
            progress(f"Photo {i + 1}/{len(sources)}: {src.name}")
            img, result = process_photo(src, self.settings, self.detector, uncropped.get(src, False), progress)
            result.output = item.output_dir / photo_filename(i)
            save_jpeg(img, result.output, self.settings.jpeg_quality)
            results.append(result)
        item.photos = results
        remove_stale_photos(item.output_dir, {r.output.name for r in results})

    def reprocess_photo(self, item: Item, index: int, uncropped: bool) -> None:
        photo = item.photos[index]
        img, result = process_photo(photo.source, self.settings, self.detector, uncropped)
        result.output = photo.output or item.output_dir / photo_filename(index)
        save_jpeg(img, result.output, self.settings.jpeg_quality)
        item.photos[index] = result
        save_item(item)

    def run_ai(self, item: Item, progress: Progress) -> str:
        """Writes item.listing; returns a note for the status line ('' if nothing to add)."""
        photos = [p.output for p in item.photos if p.output and p.output.exists()]
        if not photos:
            raise RuntimeError("Process the photos first.")
        chosen = pick_evenly(photos, self.settings.max_ai_photos)
        max_side = 1024 if self.settings.provider == "gemini" else 896
        images = [to_jpeg_b64(load_rgb(p), max_side=max_side) for p in chosen]
        self._check_cancel()
        provider = get_provider(self.settings)
        provider.progress = progress
        progress(f"Asking {provider.name} to write the listing ({len(images)} photos)...")
        item.listing = provider.analyze(images, item.notes, self.settings.language)
        used = getattr(provider, "used_model", "")
        if used and used != getattr(provider, "model", used):
            return f" (written by {used}: {provider.model} was busy)"
        return ""

    def process(self, item: Item, progress: Progress, photos: bool = True, ai: bool = True) -> None:
        """Runs the requested steps and always leaves item.status/message set."""
        item.status = Status.PROCESSING
        try:
            item.output_dir.mkdir(parents=True, exist_ok=True)
            if photos:
                self.process_photos(item, progress)
                save_item(item)
            note = ""
            if ai:
                note = self.run_ai(item, progress)
                save_item(item)
            item.status, item.message = Status.DONE, "Ready" + note
        except Cancelled:
            item.status, item.message = Status.PENDING, "Stopped"
        except AIError as exc:
            item.status, item.message = Status.ERROR, str(exc)
        except Exception as exc:  # show anything unexpected in the UI instead of crashing
            item.status, item.message = Status.ERROR, f"{type(exc).__name__}: {exc}"
