"""Background execution: one job at a time on a worker thread, so the UI stays responsive
and the GPU/AI provider isn't hit by parallel requests."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class JobSignals(QObject):
    started = Signal(str)  # item id
    progress = Signal(str, str)  # item id, message
    finished = Signal(str)  # item id


class _Job(QRunnable):
    def __init__(self, item_id: str, fn: Callable[[Callable[[str], None]], None], signals: JobSignals):
        super().__init__()
        self.item_id = item_id
        self.fn = fn
        self.signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        self.signals.started.emit(self.item_id)
        try:
            self.fn(lambda msg: self.signals.progress.emit(self.item_id, msg))
        finally:
            self.signals.finished.emit(self.item_id)


class JobRunner(QObject):
    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.signals = JobSignals()

    def submit(self, item_id: str, fn: Callable[[Callable[[str], None]], None]) -> None:
        self.pool.start(_Job(item_id, fn, self.signals))

    def clear_queue(self) -> None:
        """Drop jobs that have not started yet."""
        self.pool.clear()

    def wait(self, msecs: int = -1) -> bool:
        return self.pool.waitForDone(msecs)
