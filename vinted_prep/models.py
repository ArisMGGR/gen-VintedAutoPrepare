"""Plain data objects shared by the pipeline, AI, export and UI layers."""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"}


class Status(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


@dataclass
class Listing:
    title: str = ""
    description: str = ""
    brand: str = ""
    size: str = ""
    color: str = ""
    material: str = ""
    hashtags: list[str] = field(default_factory=list)
    price: float = 0.0
    price_note: str = ""
    language: str = ""

    def hashtag_line(self) -> str:
        return " ".join(normalize_hashtag(h) for h in self.hashtags if normalize_hashtag(h))

    def full_description(self) -> str:
        """Description as it should be pasted into Vinted (hashtags appended)."""
        tags = self.hashtag_line()
        return f"{self.description.rstrip()}\n\n{tags}" if tags else self.description.rstrip()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Listing":
        listing = cls()
        for key in asdict(listing):
            if key in data and data[key] is not None:
                setattr(listing, key, data[key])
        listing.hashtags = [str(h) for h in (listing.hashtags or [])]
        try:
            listing.price = float(listing.price)
        except (TypeError, ValueError):
            listing.price = 0.0
        return listing


def normalize_hashtag(tag: str) -> str:
    tag = re.sub(r"[\s#]+", "", str(tag))
    return f"#{tag}" if tag else ""


@dataclass
class PhotoResult:
    source: Path
    output: Path | None = None
    uncropped: bool = False  # user chose to keep the full frame for this photo
    crop_box: tuple[int, int, int, int] | None = None
    rotation_deg: float = 0.0
    detector: str = ""


@dataclass
class Item:
    source_dir: Path
    output_dir: Path
    notes: str = ""
    status: Status = Status.PENDING
    message: str = ""
    photos: list[PhotoResult] = field(default_factory=list)
    listing: Listing | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def name(self) -> str:
        return self.source_dir.name


def natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=natural_key)


def expand_dropped_folder(folder: Path) -> list[Path]:
    """One folder = one item. If a folder holds no photos but its subfolders do,
    treat each subfolder as an item (so a parent folder can be dropped too)."""
    if list_images(folder):
        return [folder]
    subs = sorted((p for p in folder.iterdir() if p.is_dir()), key=natural_key) if folder.is_dir() else []
    return [s for s in subs if list_images(s)]
