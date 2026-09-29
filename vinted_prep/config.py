"""User settings, stored as JSON in the per-user app data folder."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from vinted_prep import APP_NAME

# Languages offered for listing text. Key = value sent to the AI, label = shown in the UI.
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
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f.name: f for f in fields(cls)}
        kwargs = {}
        for key, value in data.items():
            if key not in known:
                continue
            default = getattr(cls(), key)
            try:
                kwargs[key] = type(default)(value)
            except (TypeError, ValueError):
                pass
        return cls(**kwargs)

    def save(self, path: Path | None = None) -> None:
        path = path or self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
