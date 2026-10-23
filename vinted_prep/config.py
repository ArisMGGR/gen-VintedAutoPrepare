"""User settings, stored as JSON in the per-user app data folder."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from vinted_prep import APP_NAME

log = logging.getLogger(__name__)

# Languages offered for listing text. Each name is shown in the UI and sent to the AI as is.
LANGUAGES = [
    "Greek",
    "English",
    "French",
    "Italian",
    "Spanish",
    "German",
    "Dutch",
    "Portuguese",
    "Polish",
]

# AI providers: get_provider, the toolbar label and the key check all assume one of these.
PROVIDERS = ("gemini", "ollama")


def app_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    path = base / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_dir() -> Path:
    docs = Path.home() / "Documents"
    return (docs if docs.exists() else Path.home()) / APP_NAME


# Allowed ranges for numeric settings (the same ones the Settings dialog offers).
_LIMITS = {
    "max_ai_photos": (1, 12),
    "max_straighten_deg": (1.0, 15.0),
    "enhance_strength": (0.0, 1.0),
    "crop_margin": (0.0, 0.4),
    "jpeg_quality": (1, 100),
}


def _keep_unreadable(path: Path) -> None:
    """Copy a settings file that can't be read aside, since the next save replaces it with defaults."""
    backup = path.with_name(path.name + ".unreadable")
    try:
        shutil.copyfile(path, backup)
        log.warning("Could not read %s; using default settings. The old file was kept as %s", path, backup)
    except OSError as exc:
        log.warning("Could not read %s (%s); using default settings", path, exc)


@dataclass
class Settings:
    # AI
    provider: str = "gemini"  # "gemini" | "ollama"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-latest"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5vl:7b"
    max_ai_photos: int = 6
    language: str = "Greek"

    # Photos
    output_dir: str = ""
    auto_straighten: bool = True
    max_straighten_deg: float = 5.0
    auto_enhance: bool = True
    enhance_strength: float = 0.6  # 0..1
    crop_margin: float = 0.08  # fraction of the item's size added around it
    jpeg_quality: int = 95

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir) if self.output_dir else default_output_dir()

    def effective_gemini_key(self) -> str:
        return self.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")

    # --- persistence ---
    @staticmethod
    def path() -> Path:
        return app_data_dir() / "settings.json"

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or cls.path()
        try:
            # From bytes, json detects UTF-8 (with or without BOM) and UTF-16/32,
            # e.g. the UTF-16 that Windows PowerShell 5.1's '>' writes.
            data = json.loads(path.read_bytes())
        except OSError:
            return cls()  # no settings yet
        except ValueError:  # not JSON, or in another encoding (e.g. ANSI)
            data = None
        if not isinstance(data, dict):
            _keep_unreadable(path)
            return cls()
        defaults = cls()
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for key, value in data.items():
            if key not in known or value is None:
                continue
            default = getattr(defaults, key)
            if isinstance(default, bool) != isinstance(value, bool):
                continue  # bool("false") would be True; 1/0 aren't a valid on/off either
            try:
                value = type(default)(value)
            except (TypeError, ValueError, OverflowError):  # OverflowError: int(Infinity)
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue  # NaN would slip through min/max below
            if key == "provider":
                value = value.strip().lower()  # a hand-edited "Ollama" still means Ollama
                if value not in PROVIDERS:
                    continue
            if key in _LIMITS:
                lo, hi = _LIMITS[key]
                value = min(max(value, lo), hi)
            kwargs[key] = value
        return cls(**kwargs)

    def save(self, path: Path | None = None) -> None:
        path = path or self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
