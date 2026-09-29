from __future__ import annotations

from abc import ABC, abstractmethod

from vinted_prep.models import Listing


class AIError(RuntimeError):
    """A user-presentable AI failure (bad key, quota, model missing, unreadable reply...)."""


class Provider(ABC):
    name = "provider"

    @abstractmethod
    def analyze(self, images_b64: list[str], notes: str, language: str) -> Listing:
        """images_b64: JPEG images of ONE item, base64 encoded."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Models usable for image analysis (also serves as a connection test)."""
