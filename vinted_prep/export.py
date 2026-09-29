"""Output folder layout, per item:

    <output root>/<item folder name>/
        01.jpg, 02.jpg, ...   processed photos, in the original order
        listing.txt           ready-to-paste text
        listing.json          everything, used to reopen an item later
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from vinted_prep.models import Item, Listing, PhotoResult, Status

LISTING_JSON = "listing.json"
LISTING_TXT = "listing.txt"
_PHOTO_RE = re.compile(r"^\d{2,3}\.jpg$")


def photo_filename(index: int) -> str:
    return f"{index + 1:02d}.jpg"


def _saved_source(folder: Path) -> str | None:
    try:
        return json.loads((folder / LISTING_JSON).read_text(encoding="utf-8")).get("source_dir")
    except (OSError, ValueError):
        return None


def assign_output_dir(root: Path, source_dir: Path) -> Path:
    """<root>/<folder name>, or '<name> (2)' etc. if that belongs to a different source folder."""
    source = str(source_dir.resolve())
    candidate, n = root / source_dir.name, 2
    while candidate.exists() and any(candidate.iterdir()) and _saved_source(candidate) != source:
        candidate = root / f"{source_dir.name} ({n})"
        n += 1
    return candidate


def remove_stale_photos(output_dir: Path, keep: set[str]) -> None:
    if not output_dir.is_dir():
        return
    for p in output_dir.iterdir():
        if _PHOTO_RE.match(p.name) and p.name not in keep:
            p.unlink(missing_ok=True)


def listing_text(listing: Listing) -> str:
    rows = [
        ("Brand", listing.brand),
        ("Size", listing.size),
        ("Colour", listing.color),
        ("Material", listing.material),
    ]
    lines = ["TITLE", listing.title, "", "DESCRIPTION", listing.full_description(), ""]
    lines += [f"{label}: {value}" for label, value in rows if value]
    lines.append(f"Price: {listing.price:.2f} €")
    if listing.price_note:
        lines.append(f"({listing.price_note})")
    return "\n".join(lines) + "\n"


def save_item(item: Item) -> None:
    item.output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "source_dir": str(item.source_dir.resolve()),
        "notes": item.notes,
        "photos": [
            {"source": str(p.source), "file": p.output.name if p.output else None, "uncropped": p.uncropped}
            for p in item.photos
        ],
        "listing": item.listing.to_dict() if item.listing else None,
    }
    (item.output_dir / LISTING_JSON).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    txt = item.output_dir / LISTING_TXT
    if item.listing:
        txt.write_text(listing_text(item.listing), encoding="utf-8")
    else:
        txt.unlink(missing_ok=True)


def load_saved(item: Item) -> bool:
    """Restore a previously processed item from its output folder. Returns True if found."""
    path = item.output_dir / LISTING_JSON
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    item.notes = data.get("notes", "")
    item.photos = []
    for p in data.get("photos", []):
        out = item.output_dir / p["file"] if p.get("file") else None
        item.photos.append(
            PhotoResult(source=Path(p["source"]), output=out if out and out.exists() else None, uncropped=p.get("uncropped", False))
        )
    item.listing = Listing.from_dict(data["listing"]) if data.get("listing") else None
    has_photos = any(p.output for p in item.photos)
    item.status = Status.DONE if (item.listing and has_photos) else Status.PENDING
    item.message = "Loaded previous result" if item.status == Status.DONE else ""
    return True
