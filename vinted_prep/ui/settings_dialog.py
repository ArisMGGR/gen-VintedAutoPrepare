from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from vinted_prep.ai import AIError
from vinted_prep.ai.gemini import KEY_URL, GeminiProvider
from vinted_prep.ai.ollama import DOWNLOAD_URL, OllamaProvider
from vinted_prep.config import Settings, default_output_dir

PROVIDERS = [("gemini", "Google Gemini — free tier, online"), ("ollama", "Ollama — free, offline on your PC")]


def _model_row(combo: QComboBox, button: QPushButton) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(combo, 1)
    lay.addWidget(button)
    return w


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(560)
        self.settings = copy.deepcopy(settings)
        s = self.settings

        # --- AI ---
        ai_box = QGroupBox("AI (writes title, description, details and price)")
        ai_form = QFormLayout(ai_box)
        self.provider = QComboBox()
        for key, label in PROVIDERS:
            self.provider.addItem(label, key)
        self.provider.setCurrentIndex(max(0, self.provider.findData(s.provider)))
        ai_form.addRow("Provider:", self.provider)

        self.pages = QStackedWidget()
        # Gemini page
        gem = QWidget()
        gem_form = QFormLayout(gem)
        gem_form.setContentsMargins(0, 0, 0, 0)
        key_row = QWidget()
        key_lay = QHBoxLayout(key_row)
        key_lay.setContentsMargins(0, 0, 0, 0)
        self.gemini_key = QLineEdit(s.gemini_api_key)
        self.gemini_key.setEchoMode(QLineEdit.Password)
        self.gemini_key.setPlaceholderText("Paste your free API key (or set GEMINI_API_KEY)")
        show = QCheckBox("Show")
        show.toggled.connect(lambda on: self.gemini_key.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
        key_lay.addWidget(self.gemini_key, 1)
        key_lay.addWidget(show)
        gem_form.addRow("API key:", key_row)
        link = QLabel(f'<a href="{KEY_URL}">Get a free Gemini API key (Google account needed)</a>')
        link.setOpenExternalLinks(True)
        gem_form.addRow("", link)
        self.gemini_model = QComboBox()
        self.gemini_model.setEditable(True)
        self.gemini_model.addItem(s.gemini_model)
        gem_refresh = QPushButton("Refresh")
        gem_refresh.setToolTip("Check the key and list available models")
        gem_refresh.clicked.connect(self._refresh_gemini)
        gem_form.addRow("Model:", _model_row(self.gemini_model, gem_refresh))
        note = QLabel("Free tier: limited requests per minute/day; Google may use free-tier data to improve its models.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        gem_form.addRow("", note)
        self.pages.addWidget(gem)

        # Ollama page
        oll = QWidget()
        oll_form = QFormLayout(oll)
        oll_form.setContentsMargins(0, 0, 0, 0)
        self.ollama_url = QLineEdit(s.ollama_url)
        oll_form.addRow("Ollama URL:", self.ollama_url)
        self.ollama_model = QComboBox()
        self.ollama_model.setEditable(True)
        self.ollama_model.addItem(s.ollama_model)
        oll_refresh = QPushButton("Refresh")
        oll_refresh.setToolTip("Check Ollama is running and list installed models")
        oll_refresh.clicked.connect(self._refresh_ollama)
        oll_form.addRow("Model:", _model_row(self.ollama_model, oll_refresh))
        hint = QLabel(
            f'Install <a href="{DOWNLOAD_URL}">Ollama</a>, then in a terminal run '
            "<code>ollama pull qwen2.5vl:7b</code> (a vision model is required)."
        )
        hint.setOpenExternalLinks(True)
        hint.setWordWrap(True)
        oll_form.addRow("", hint)
        self.pages.addWidget(oll)

        ai_form.addRow(self.pages)
        self.max_photos = QSpinBox()
        self.max_photos.setRange(1, 12)
        self.max_photos.setValue(s.max_ai_photos)
        self.max_photos.setToolTip("More photos = better label reading, but slower (especially with Ollama)")
        ai_form.addRow("Photos sent to AI (max):", self.max_photos)
        self.provider.currentIndexChanged.connect(self.pages.setCurrentIndex)
        self.pages.setCurrentIndex(self.provider.currentIndex())

        # --- Photos ---
        photo_box = QGroupBox("Photos")
        photo_form = QFormLayout(photo_box)
        self.straighten = QCheckBox("Straighten slightly tilted photos")
        self.straighten.setChecked(s.auto_straighten)
        self.max_angle = QDoubleSpinBox()
        self.max_angle.setRange(1, 15)
        self.max_angle.setSuffix("°")
        self.max_angle.setValue(s.max_straighten_deg)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.straighten)
        rl.addStretch()
        rl.addWidget(QLabel("up to"))
        rl.addWidget(self.max_angle)
        photo_form.addRow(row)

        self.enhance = QCheckBox("Auto brightness / contrast / white balance")
        self.enhance.setChecked(s.auto_enhance)
        self.strength = QSlider(Qt.Horizontal)
        self.strength.setRange(0, 100)
        self.strength.setValue(round(s.enhance_strength * 100))
        photo_form.addRow(self.enhance)
        photo_form.addRow("Enhance strength:", self.strength)

        self.margin = QSpinBox()
        self.margin.setRange(0, 40)
        self.margin.setSuffix(" %")
        self.margin.setValue(round(s.crop_margin * 100))
        self.margin.setToolTip("Space kept around the item when cropping")
        photo_form.addRow("Space around item:", self.margin)

        # --- Output ---
        out_box = QGroupBox("Output")
        out_form = QFormLayout(out_box)
        out_row = QWidget()
        ol = QHBoxLayout(out_row)
        ol.setContentsMargins(0, 0, 0, 0)
        self.output_dir = QLineEdit(str(s.resolved_output_dir()))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        ol.addWidget(self.output_dir, 1)
        ol.addWidget(browse)
        out_form.addRow("Save results to:", out_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(ai_box)
        layout.addWidget(photo_box)
        layout.addWidget(out_box)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output folder", self.output_dir.text())
        if path:
            self.output_dir.setText(path)

    def _fill_models(self, combo: QComboBox, provider) -> None:
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            models = provider.list_models()
        except AIError as exc:
            QGuiApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Connection problem", str(exc))
            return
        except Exception:
            QGuiApplication.restoreOverrideCursor()  # don't leave a busy cursor behind the error dialog
            raise
        QGuiApplication.restoreOverrideCursor()
        current = combo.currentText()
        combo.clear()
        combo.addItems(models)
        if current in models:
            combo.setCurrentText(current)
        if not models:
            combo.setEditText(current)
            QMessageBox.information(self, "No models", "Connected, but no models were found.")
        else:
            QMessageBox.information(self, "Connected", f"Connected. {len(models)} model(s) available.")

    def _refresh_gemini(self) -> None:
        key = Settings(gemini_api_key=self.gemini_key.text().strip()).effective_gemini_key()  # field, else GEMINI_API_KEY
        self._fill_models(self.gemini_model, GeminiProvider(key, self.gemini_model.currentText()))

    def _refresh_ollama(self) -> None:
        url = self.ollama_url.text().strip()  # as accept() saves it: a trailing space breaks the URL
        self._fill_models(self.ollama_model, OllamaProvider(url, self.ollama_model.currentText()))

    def accept(self) -> None:
        s = self.settings
        s.provider = self.provider.currentData()
        s.gemini_api_key = self.gemini_key.text().strip()
        s.gemini_model = self.gemini_model.currentText().strip() or Settings.gemini_model
        s.ollama_url = self.ollama_url.text().strip() or Settings.ollama_url
        s.ollama_model = self.ollama_model.currentText().strip() or Settings.ollama_model
        s.max_ai_photos = self.max_photos.value()
        s.auto_straighten = self.straighten.isChecked()
        s.max_straighten_deg = self.max_angle.value()
        s.auto_enhance = self.enhance.isChecked()
        s.enhance_strength = self.strength.value() / 100
        s.crop_margin = self.margin.value() / 100
        out = self.output_dir.text().strip()
        s.output_dir = "" if out == str(default_output_dir()) else out  # "" keeps following the default
        super().accept()
