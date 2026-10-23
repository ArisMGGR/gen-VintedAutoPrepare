"""The listing prompt, the JSON shape we ask for, and tolerant parsing of the reply."""

from __future__ import annotations

import json
import math
import re

from vinted_prep.ai.base import AIError
from vinted_prep.models import Listing

TITLE_MAX = 60

SYSTEM_PROMPT = """\
You are an experienced reseller who writes listings for Vinted Greece (vinted.gr).
You receive several photos of ONE second-hand item (front, back, details, labels) and
return the listing as JSON. You are precise: you read labels carefully and you never
invent facts that are not visible in the photos or stated in the seller's notes."""

USER_PROMPT = """\
All photos show the same single item. Write its Vinted listing.

Language: write title, description, colour, material and hashtags in {language}.
Brand names stay as printed on the item.

Fields:
- brand: exactly as on the logo/label. Empty string if no brand is visible. Never guess.
- size: as on the size label (e.g. "M", "EU 38", "W32 L34", "42", "12-18 months").
  Empty string if not visible.
- color: main colour(s) of the item, simple words.
- material: for clothing/shoes copy the composition from the care label if readable
  (e.g. "100% cotton", "80% wool, 20% polyamide"). If there is no readable label, give
  your best guess from the texture and end it with "(estimated)". For other items give
  the main material (wood, ceramic, plastic, metal...). For books leave it empty.
- title: at most {title_max} characters, what buyers search for:
  brand + item type + key feature/colour + size. No emojis, no ALL CAPS.
- description: short and friendly, 3-5 short lines. Say what it is, the brand, size,
  material and 1-2 appealing details (fit, style, features, what to wear it with).
  Do NOT mention the item's condition or wear unless the seller's notes mention it.
  Do NOT put hashtags, prices or measurements you cannot see in the description.
  At most one emoji.
- hashtags: 5 to 10 search keywords buyers would type (item type, brand, style,
  colour, occasion), without the # sign, lowercase, no spaces inside a keyword.
- price: ONE realistic asking price in EUR for a quick sale on Vinted Greece. Greek
  Vinted prices are low: fast-fashion tops/t-shirts 3-8, fast-fashion dresses/jeans
  6-15, mid-range brands 10-30, premium/designer brands higher depending on the item,
  sneakers from popular brands 20-60, books 2-6, kids' clothes 2-8. Adjust for brand,
  item type, material quality and anything in the seller's notes. Whole euros or .50.
- price_note: one short sentence (in {language}) explaining the price.

Seller's notes (facts from the seller, always trust and use them): {notes}"""

RESPONSE_FIELDS = {
    "title": "string",
    "description": "string",
    "brand": "string",
    "size": "string",
    "color": "string",
    "material": "string",
    "hashtags": "array",
    "price": "number",
    "price_note": "string",
}


def json_schema() -> dict:
    """Plain JSON schema (Ollama structured outputs)."""
    props = {}
    for name, kind in RESPONSE_FIELDS.items():
        props[name] = {"type": "array", "items": {"type": "string"}} if kind == "array" else {"type": kind}
    return {"type": "object", "properties": props, "required": list(RESPONSE_FIELDS)}


def gemini_schema() -> dict:
    """Same schema in Gemini's OpenAPI dialect."""
    props = {}
    for name, kind in RESPONSE_FIELDS.items():
        if kind == "array":
            props[name] = {"type": "ARRAY", "items": {"type": "STRING"}}
        else:
            props[name] = {"type": kind.upper()}
    return {"type": "OBJECT", "properties": props, "required": list(RESPONSE_FIELDS)}


def build_user_prompt(notes: str, language: str) -> str:
    notes = notes.strip() or "(none)"
    return USER_PROMPT.format(language=language, notes=notes, title_max=TITLE_MAX)


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise AIError("The AI reply did not contain a listing (no JSON found).") from None
        try:
            data = json.loads(text[start : end + 1])
        except ValueError as exc:
            raise AIError(f"The AI reply could not be read as JSON: {exc}") from None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        raise AIError("The AI reply had an unexpected shape.")
    return data


def _to_price(value) -> float:
    if isinstance(value, str):
        match = re.search(r"\d+(?:[.,]\d+)?", value)
        value = match.group(0).replace(",", ".") if match else 0
    try:
        price = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(price * 2) or price <= 0:  # NaN, Infinity or too big to round
        return 0.0
    return max(1.0, round(price * 2) / 2)  # whole euros or .50


def _clean_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    text = str(value).strip()
    return "" if text.lower() in {"unknown", "n/a", "none", "null", "-"} else text


def _shorten_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    if len(title) <= TITLE_MAX:
        return title
    cut = title[:TITLE_MAX].rsplit(" ", 1)[0]
    return cut.rstrip(" ,-–|")


def parse_listing(text: str, language: str) -> Listing:
    data = _extract_json(text)
    tags = data.get("hashtags") or []
    if isinstance(tags, str):
        tags = re.split(r"[,\s]+", tags)
    elif not isinstance(tags, list):  # e.g. a number: not something to iterate over
        tags = []
    seen, hashtags = set(), []
    for tag in tags:
        tag = re.sub(r"[\s#]+", "", str(tag)).lower()
        if tag and tag not in seen:
            seen.add(tag)
            hashtags.append(tag)

    listing = Listing(
        title=_shorten_title(_clean_str(data.get("title"))),
        description=_clean_str(data.get("description")),
        brand=_clean_str(data.get("brand")),
        size=_clean_str(data.get("size")),
        color=_clean_str(data.get("color") or data.get("colour")),
        material=_clean_str(data.get("material")),
        hashtags=hashtags[:10],
        price=_to_price(data.get("price")),
        price_note=_clean_str(data.get("price_note")),
        language=language,
    )
    if not listing.title and not listing.description:
        raise AIError("The AI reply had no title or description.")
    return listing
