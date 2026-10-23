from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from vinted_prep import APP_NAME, __version__
from vinted_prep.config import LANGUAGES, Settings
from vinted_prep.export import LISTING_JSON, assign_output_dir, load_saved, save_item
from vinted_prep.models import Item, Status, expand_dropped_folder
from vinted_prep.processor import Processor
from vinted_prep.ui.item_panel import ItemPanel
from vinted_prep.ui.jobs import JobRunner
from vinted_prep.ui.settings_dialog import SettingsDialog

STATUS_ICON = {Status.PENDING: "○", Status.PROCESSING: "⏳", Status.DONE: "✔", Status.ERROR: "⚠"}
STATUS_COLOR = {Status.DONE: "#1e8449", Status.ERROR: "#c0392b", Status.PROCESSING: "#2e86c1"}


class DropHint(QLabel):
    def __init__(self):
        super().__init__(
            "<div style='font-size:20px; font-weight:600;'>Drop item folders here</div>"
            "<p style='font-size:13px; color:gray;'>One folder = one item (front, back, labels…).<br>"
            "You can drop several folders at once, or a parent folder that contains item folders.</p>"
        )
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { border: 2px dashed #bbb; border-radius: 12px; margin: 24px; }")


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.processor = Processor(settings)
        self.runner = JobRunner(self)
        self.runner.signals.started.connect(self._job_started)
        self.runner.signals.progress.connect(self._job_progress)
        self.runner.signals.finished.connect(self._job_finished)
        self.items: dict[str, Item] = {}
        self.queued: set[str] = set()
        self.running: str | None = None
        self._dirty: set[str] = set()
        self._save_timer = QTimer(self, singleShot=True, interval=700)
        self._save_timer.timeout.connect(self._flush_saves)

        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1250, 860)
        self.setAcceptDrops(True)
        self._build_toolbar()

        # Left: queue
        self.queue = QListWidget()
        self.queue.setMinimumWidth(240)
        self.queue.setSelectionMode(QListWidget.ExtendedSelection)
        self.queue.currentItemChanged.connect(self._select)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self.remove_selected)
        clear_btn = QPushButton("Clear list")
        clear_btn.clicked.connect(self.clear_all)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 0, 6)
        ll.addWidget(QLabel("Items"))
        ll.addWidget(self.queue, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(remove_btn)
        buttons.addWidget(clear_btn)
        ll.addLayout(buttons)

        # Right: drop hint or item details
        self.panel = ItemPanel()
        self.panel.process_requested.connect(lambda item_id: self.process([item_id]))
        self.panel.rewrite_requested.connect(lambda item_id: self.process([item_id], photos=False))
        self.panel.photo_crop_toggled.connect(self._toggle_crop)
        self.panel.changed.connect(self._mark_dirty)
        self.stack = QStackedWidget()
        self.stack.addWidget(DropHint())
        self.stack.addWidget(self.panel)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 970])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Add item folders to begin.")

    # --- toolbar ---
    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)

        add = QAction("➕  Add folder…", self, shortcut=QKeySequence.Open, triggered=self.choose_folder)
        self.process_all_action = QAction("▶  Process all", self, triggered=self.process_all)
        self.process_all_action.setToolTip("Process every item that isn't done yet")
        self.stop_action = QAction("■  Stop", self, triggered=self.stop, enabled=False)
        tb.addAction(add)
        tb.addAction(self.process_all_action)
        tb.addAction(self.stop_action)
        tb.addSeparator()

        tb.addWidget(QLabel("  Listing language: "))
        self.language = QComboBox()
        self.language.addItems(LANGUAGES)
        if self.settings.language not in LANGUAGES:
            self.language.addItem(self.settings.language)
        self.language.setCurrentText(self.settings.language)
        self.language.currentTextChanged.connect(self._language_changed)
        tb.addWidget(self.language)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.provider_label = QLabel()
        self.provider_label.setStyleSheet("color: gray; padding-right: 8px;")
        tb.addWidget(self.provider_label)
        tb.addAction(QAction("📂  Output", self, triggered=self.open_output_root))
        tb.addAction(QAction("⚙  Settings", self, triggered=self.open_settings))
        self._update_provider_label()

    def _update_provider_label(self) -> None:
        s = self.settings
        model = s.gemini_model if s.provider == "gemini" else s.ollama_model
        self.provider_label.setText(f"AI: {'Gemini' if s.provider == 'gemini' else 'Ollama'} · {model}")

    def _language_changed(self, language: str) -> None:
        self.settings.language = language
        self.settings.save()

    # --- adding items ---
    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose an item folder (or a folder of item folders)")
        if path:
            self.add_folders([Path(path)])

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls() and any(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = []
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            paths.append(p if p.is_dir() else p.parent)  # dropped photos -> their folder
        self.add_folders(list(dict.fromkeys(paths)))

    def add_folders(self, folders: list[Path]) -> None:
        known = {i.source_dir.resolve() for i in self.items.values()}
        root = self.settings.resolved_output_dir()
        added, first = 0, None
        for folder in folders:
            for src in expand_dropped_folder(folder):
                if src.resolve() in known:
                    continue
                item = Item(source_dir=src, output_dir=assign_output_dir(root, src))
                load_saved(item)
                self.items[item.id] = item
                known.add(src.resolve())
                entry = QListWidgetItem()
                entry.setData(Qt.UserRole, item.id)
                self.queue.addItem(entry)
                self._refresh_entry(item.id)
                first = first or entry
                added += 1
        if added:
            self.stack.setCurrentIndex(1)
            if self.queue.currentItem() is None:
                self.queue.setCurrentItem(first)
            self.statusBar().showMessage(f"Added {added} item(s).", 5000)
        else:
            QMessageBox.information(
                self, "Nothing added", "No new folders with photos were found (already in the list, or no images)."
            )

    def _entry(self, item_id: str) -> QListWidgetItem | None:
        for row in range(self.queue.count()):
            entry = self.queue.item(row)
            if entry.data(Qt.UserRole) == item_id:
                return entry
        return None

    def _refresh_entry(self, item_id: str) -> None:
        entry, item = self._entry(item_id), self.items.get(item_id)
        if not entry or not item:
            return
        queued = item_id in self.queued and item.status != Status.PROCESSING
        icon = "…" if queued else STATUS_ICON[item.status]
        subtitle = ""
        if item.listing and item.status == Status.DONE:
            subtitle = f"\n    {item.listing.price:g} € · {item.listing.title[:34]}"
        elif queued:
            subtitle = "\n    Queued"
        elif item.status == Status.ERROR:
            subtitle = "\n    Error — click for details"
        entry.setText(f"{icon}  {item.name}{subtitle}")
        entry.setForeground(QColor(STATUS_COLOR.get(item.status, "#555")) if not queued else QColor("#555"))
        entry.setToolTip(f"{item.source_dir}\n{item.message}")

    def _select(self, current: QListWidgetItem | None, _previous=None) -> None:
        self._flush_saves()
        item = self.items.get(current.data(Qt.UserRole)) if current else None
        self.panel.show_item(item)
        self.stack.setCurrentIndex(1 if item else 0)

    def remove_selected(self) -> None:
        for entry in self.queue.selectedItems():
            item_id = entry.data(Qt.UserRole)
            if item_id in self.queued:
                continue  # can't remove while queued/processing
            self.items.pop(item_id, None)
            self.queue.takeItem(self.queue.row(entry))
        if not self.items:
            self.stack.setCurrentIndex(0)

    def clear_all(self) -> None:
        if self.queued:
            QMessageBox.information(self, "Busy", "Stop processing before clearing the list.")
            return
        self._flush_saves()
        self.items.clear()
        self.queue.clear()
        self.stack.setCurrentIndex(0)

    # --- processing ---
    def _ai_ready(self) -> bool:
        if self.settings.provider == "gemini" and not self.settings.effective_gemini_key():
            answer = QMessageBox.question(
                self,
                "Gemini API key needed",
                "To write listings with Google Gemini you need a free API key.\n\nOpen Settings now?",
            )
            if answer == QMessageBox.Yes:
                self.open_settings()
            return bool(self.settings.effective_gemini_key())
        return True

    def process(self, item_ids: list[str], photos: bool = True, ai: bool = True) -> None:
        if ai and not self._ai_ready():
            return
        self._flush_saves()
        if not self.queued:
            self.processor.cancel_event.clear()
        for item_id in item_ids:
            item = self.items.get(item_id)
            if not item or item_id in self.queued:
                continue
            self.queued.add(item_id)
            self._refresh_entry(item_id)
            self.runner.submit(item_id, lambda progress, it=item: self.processor.process(it, progress, photos, ai))
        self._update_busy()

    def process_all(self) -> None:
        todo = [i for i, item in self.items.items() if item.status in (Status.PENDING, Status.ERROR)]
        if not todo:
            self.statusBar().showMessage("Nothing to process: every item is done.", 5000)
            return
        self.process(todo)

    def _toggle_crop(self, item_id: str, index: int, uncropped: bool) -> None:
        item = self.items[item_id]
        previous = item.status, item.message

        def job(progress, item=item):
            item.status, item.message = Status.PROCESSING, "Updating photo…"
            try:
                self.processor.reprocess_photo(item, index, uncropped)
                item.status, item.message = previous
            except Exception as exc:
                item.status, item.message = Status.ERROR, f"Could not update photo: {exc}"

        self.queued.add(item_id)
        self.runner.submit(item_id, job)
        self._update_busy()

    def stop(self) -> None:
        self.processor.cancel_event.set()
        self.runner.clear_queue()
        for item_id in list(self.queued):
            if item_id != self.running:
                self.queued.discard(item_id)
                self._refresh_entry(item_id)
        self.statusBar().showMessage("Stopping after the current photo…")
        self._update_busy()

    def _update_busy(self) -> None:
        self.stop_action.setEnabled(bool(self.queued))

    def _job_started(self, item_id: str) -> None:
        self.running = item_id
        self._refresh_entry(item_id)
        if self.panel.item and self.panel.item.id == item_id:
            self.panel.refresh_status()

    def _job_progress(self, item_id: str, message: str) -> None:
        item = self.items.get(item_id)
        if not item:
            return
        item.message = message
        remaining = len(self.queued)
        self.statusBar().showMessage(f"{item.name}: {message}" + (f"   ({remaining} in queue)" if remaining > 1 else ""))
        self._refresh_entry(item_id)
        if self.panel.item is item:
            self.panel.refresh_status()

    def _job_finished(self, item_id: str) -> None:
        self.queued.discard(item_id)
        self.running = None
        self._refresh_entry(item_id)
        if self.panel.item and self.panel.item.id == item_id:
            self.panel.refresh_item()
        self._update_busy()
        if not self.queued:
            done = sum(i.status == Status.DONE for i in self.items.values())
            errors = sum(i.status == Status.ERROR for i in self.items.values())
            msg = f"Finished. {done} ready" + (f", {errors} with errors (click them for details)" if errors else "")
            self.statusBar().showMessage(msg + ".")

    # --- saving edits ---
    def _mark_dirty(self, item_id: str) -> None:
        self._dirty.add(item_id)
        self._save_timer.start()

    def _flush_saves(self) -> None:
        for item_id in list(self._dirty):
            item = self.items.get(item_id)
            if item and item_id not in self.queued and (item.output_dir / LISTING_JSON).exists():
                try:
                    save_item(item)
                except OSError as exc:
                    self.statusBar().showMessage(f"Could not save {item.name}: {exc}", 8000)
            self._refresh_entry(item_id)
        self._dirty.clear()

    # --- misc ---
    def open_output_root(self) -> None:
        root = self.settings.resolved_output_dir()
        root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            new = dialog.settings
            new.language = self.language.currentText()
            for field in vars(new):
                setattr(self.settings, field, getattr(new, field))
            self.settings.save()
            self._update_provider_label()

    def closeEvent(self, event) -> None:
        self._flush_saves()
        if self.queued:
            if QMessageBox.question(self, "Quit?", "Items are still being processed. Stop and quit?") != QMessageBox.Yes:
                event.ignore()
                return
            self.stop()
            self.runner.wait(10000)
        event.accept()
