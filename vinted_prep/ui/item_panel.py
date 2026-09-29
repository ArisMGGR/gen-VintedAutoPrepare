"""Right-hand panel: one item's notes, processed photos and the editable listing."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication, QIcon, QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from vinted_prep.ai.prompt import TITLE_MAX
from vinted_prep.imaging.io import load_rgb, resize_max
from vinted_prep.models import Item, Listing, Status, list_images, normalize_hashtag

THUMB = QSize(144, 180)  # 4:5


def load_thumbnail(path: Path, size: QSize = THUMB) -> QPixmap:
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    src = reader.size()
    if src.isValid():
        reader.setScaledSize(src.scaled(size, Qt.KeepAspectRatio))
    image = reader.read()
    if image.isNull():  # formats Qt can't read (e.g. iPhone HEIC)
        try:
            thumb = resize_max(load_rgb(path), max(size.width(), size.height()))
        except Exception:
            return QPixmap()
        h, w = thumb.shape[:2]
        image = QImage(thumb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class CopyButton(QPushButton):
    def __init__(self, get_text, parent=None):
        super().__init__("Copy", parent)
        self._get_text = get_text
        self.setFixedWidth(72)
        self.setToolTip("Copy to clipboard")
        self.clicked.connect(self._copy)

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self._get_text())
        self.setText("Copied ✓")
        QTimer.singleShot(1200, lambda: self.setText("Copy"))


class ItemPanel(QWidget):
    process_requested = Signal(str)  # item id
    rewrite_requested = Signal(str)  # item id (AI only)
    photo_crop_toggled = Signal(str, int, bool)  # item id, photo index, uncropped
    changed = Signal(str)  # item id, listing or notes edited by the user

    def __init__(self, parent=None):
        super().__init__(parent)
        self.item: Item | None = None
        self._loading = False

        # Header
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.process_btn = QPushButton("▶  Process item")
        self.process_btn.clicked.connect(lambda: self.item and self.process_requested.emit(self.item.id))
        self.rewrite_btn = QPushButton("↻  Rewrite listing")
        self.rewrite_btn.setToolTip("Ask the AI again using the current photos and notes (photos are not reprocessed)")
        self.rewrite_btn.clicked.connect(lambda: self.item and self.rewrite_requested.emit(self.item.id))
        self.open_btn = QPushButton("📂  Output folder")
        self.open_btn.clicked.connect(self._open_output)

        header = QHBoxLayout()
        header.addWidget(self.title_label, 1)
        header.addWidget(self.process_btn)
        header.addWidget(self.rewrite_btn)
        header.addWidget(self.open_btn)

        # Notes
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText(
            "Optional notes for the AI, e.g. 'worn twice', 'small mark on the left sleeve', "
            "'size runs small', 'bought for 60€'…"
        )
        self.notes.setFixedHeight(64)
        self.notes.textChanged.connect(self._notes_changed)

        # Photos
        self.photos = QListWidget()
        self.photos.setViewMode(QListWidget.IconMode)
        self.photos.setFlow(QListWidget.LeftToRight)
        self.photos.setWrapping(False)
        self.photos.setIconSize(THUMB)
        self.photos.setMovement(QListWidget.Static)
        self.photos.setSpacing(6)
        self.photos.setFixedHeight(THUMB.height() + 48)
        self.photos.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.photos.setContextMenuPolicy(Qt.CustomContextMenu)
        self.photos.customContextMenuRequested.connect(self._photo_menu)
        self.photos.itemDoubleClicked.connect(self._open_photo)
        self.photos.setToolTip("Double-click to open. Right-click to keep a photo uncropped.")

        # Listing form
        form_box = QGroupBox("Listing")
        form = QFormLayout(form_box)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)

        self.title = QLineEdit()
        self.title_count = QLabel()
        self.title_count.setFixedWidth(48)
        self.title.textChanged.connect(self._update_title_count)
        form.addRow("Title:", self._row(self.title, self.title_count, CopyButton(self.title.text)))

        self.description = QPlainTextEdit()
        self.description.setMinimumHeight(120)
        self.description.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        desc_copy = CopyButton(self._full_description)
        desc_copy.setToolTip("Copies the description with the hashtags appended")
        form.addRow("Description:", self._row(self.description, desc_copy, align_top=True))

        self.hashtags = QLineEdit()
        self.hashtags.setPlaceholderText("#keyword #keyword …")
        form.addRow("Hashtags:", self._row(self.hashtags, CopyButton(self.hashtags.text)))

        self.brand, self.size, self.color, self.material = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        form.addRow("Brand:", self._row(self.brand, CopyButton(self.brand.text)))
        form.addRow("Size:", self._row(self.size, CopyButton(self.size.text)))
        form.addRow("Colour:", self._row(self.color, CopyButton(self.color.text)))
        form.addRow("Material:", self._row(self.material, CopyButton(self.material.text)))

        self.price = QDoubleSpinBox()
        self.price.setRange(0, 100000)
        self.price.setDecimals(2)
        self.price.setSingleStep(0.5)
        self.price.setSuffix(" €")
        self.price.setFixedWidth(130)
        self.price.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.price_note = QLabel()
        self.price_note.setWordWrap(True)
        self.price_note.setStyleSheet("color: gray; font-style: italic;")
        price_copy = CopyButton(lambda: f"{self.price.value():g}")
        form.addRow("Price:", self._row(self.price, price_copy, self.price_note, stretch=2))

        for w in (self.title, self.hashtags, self.brand, self.size, self.color, self.material):
            w.textChanged.connect(self._listing_changed)
        self.description.textChanged.connect(self._listing_changed)
        self.price.valueChanged.connect(self._listing_changed)
        self.form_box = form_box

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.addLayout(header)
        lay.addWidget(self.status_label)
        lay.addWidget(QLabel("Notes for the AI (optional):"))
        lay.addWidget(self.notes)
        lay.addWidget(QLabel("Photos (saved as 01.jpg, 02.jpg… in the output folder):"))
        lay.addWidget(self.photos)
        lay.addWidget(form_box, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    @staticmethod
    def _row(*widgets, stretch: int = 0, align_top: bool = False) -> QWidget:
        """Widgets side by side; widgets[stretch] takes the spare width."""
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        for i, widget in enumerate(widgets):
            lay.addWidget(widget, 1 if i == stretch else 0)
            if align_top and i:
                lay.setAlignment(widget, Qt.AlignTop)
        return w

    # --- display ---
    def show_item(self, item: Item | None) -> None:
        self.item = item
        self._loading = True
        try:
            self.photos.clear()
            if item is None:
                return
            self.title_label.setText(item.name)
            self.notes.setPlainText(item.notes)
            self._fill_listing(item.listing)
            self._fill_photos(item)
            self.refresh_status()
        finally:
            self._loading = False

    def refresh_status(self) -> None:
        item = self.item
        if item is None:
            return
        busy = item.status == Status.PROCESSING
        colors = {Status.ERROR: "#c0392b", Status.DONE: "#1e8449", Status.PROCESSING: "#2e86c1"}
        text = item.message or {Status.PENDING: "Not processed yet"}.get(item.status, "")
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {colors.get(item.status, 'gray')};")
        has_photos = any(p.output for p in item.photos)
        self.process_btn.setEnabled(not busy)
        self.process_btn.setText("↻  Reprocess item" if has_photos else "▶  Process item")
        self.rewrite_btn.setEnabled(not busy and has_photos)
        self.open_btn.setEnabled(item.output_dir.exists())
        self.notes.setReadOnly(busy)
        self.photos.setEnabled(not busy)
        self.form_box.setEnabled(not busy and item.listing is not None)

    def refresh_item(self) -> None:
        """Reload everything after a background job changed the item."""
        self.show_item(self.item)

    def _fill_listing(self, listing: Listing | None) -> None:
        listing = listing or Listing()
        self.title.setText(listing.title)
        self.description.setPlainText(listing.description)
        self.hashtags.setText(listing.hashtag_line())
        self.brand.setText(listing.brand)
        self.size.setText(listing.size)
        self.color.setText(listing.color)
        self.material.setText(listing.material)
        self.price.setValue(listing.price)
        self.price_note.setText(listing.price_note)
        self._update_title_count()

    def _fill_photos(self, item: Item) -> None:
        for i, photo in enumerate(item.photos):
            path = photo.output if photo.output and photo.output.exists() else None
            entry = QListWidgetItem(f"{i + 1:02d}" + ("  (full)" if photo.uncropped else ""))
            if path:
                entry.setIcon(QIcon(load_thumbnail(path)))
            entry.setData(Qt.UserRole, i)
            entry.setToolTip(f"{photo.source.name}" + (f"\nDetector: {photo.detector}" if photo.detector else ""))
            self.photos.addItem(entry)
        if not item.photos:  # not processed yet: preview the originals
            for src in list_images(item.source_dir):
                entry = QListWidgetItem(src.name)
                entry.setIcon(QIcon(load_thumbnail(src)))
                entry.setFlags(Qt.ItemIsEnabled)
                self.photos.addItem(entry)

    def _update_title_count(self) -> None:
        n = len(self.title.text())
        self.title_count.setText(f"{n}/{TITLE_MAX}")
        self.title_count.setStyleSheet("color: #c0392b;" if n > TITLE_MAX else "color: gray;")

    # --- editing ---
    def _full_description(self) -> str:
        return self._listing_from_form().full_description()

    def _listing_from_form(self) -> Listing:
        base = (self.item.listing if self.item and self.item.listing else Listing())
        tags = [normalize_hashtag(t)[1:] for t in self.hashtags.text().split() if normalize_hashtag(t)]
        return Listing(
            title=self.title.text().strip(),
            description=self.description.toPlainText(),
            brand=self.brand.text().strip(),
            size=self.size.text().strip(),
            color=self.color.text().strip(),
            material=self.material.text().strip(),
            hashtags=tags,
            price=self.price.value(),
            price_note=base.price_note,
            language=base.language,
        )

    def _listing_changed(self) -> None:
        if self._loading or not self.item or self.item.listing is None:
            return
        self.item.listing = self._listing_from_form()
        self.changed.emit(self.item.id)

    def _notes_changed(self) -> None:
        if self._loading or not self.item:
            return
        self.item.notes = self.notes.toPlainText()
        self.changed.emit(self.item.id)

    # --- photos ---
    def _photo_menu(self, pos) -> None:
        entry = self.photos.itemAt(pos)
        if not entry or entry.data(Qt.UserRole) is None or not self.item:
            return
        index = entry.data(Qt.UserRole)
        photo = self.item.photos[index]
        menu = QMenu(self)
        toggle = menu.addAction("Use full photo (no crop)")
        toggle.setCheckable(True)
        toggle.setChecked(photo.uncropped)
        open_out = menu.addAction("Open processed photo")
        open_src = menu.addAction("Open original photo")
        chosen = menu.exec(self.photos.mapToGlobal(pos))
        if chosen == toggle:
            self.photo_crop_toggled.emit(self.item.id, index, not photo.uncropped)
        elif chosen == open_out and photo.output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(photo.output)))
        elif chosen == open_src:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(photo.source)))

    def _open_photo(self, entry: QListWidgetItem) -> None:
        index = entry.data(Qt.UserRole)
        if self.item and index is not None and self.item.photos[index].output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.item.photos[index].output)))

    def _open_output(self) -> None:
        if self.item and self.item.output_dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.item.output_dir)))
