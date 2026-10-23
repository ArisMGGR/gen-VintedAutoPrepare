"""Ollama: free, offline vision models running on your own GPU."""

from __future__ import annotations

import requests

from vinted_prep.ai.base import AIError, Provider
from vinted_prep.ai.prompt import SYSTEM_PROMPT, build_user_prompt, json_schema, parse_listing
from vinted_prep.models import Listing

DOWNLOAD_URL = "https://ollama.com/download"


class OllamaProvider(Provider):
    name = "Ollama"

    def __init__(self, base_url: str, model: str, session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.http = session or requests.Session()

    def _not_running(self, exc: Exception) -> AIError:
        return AIError(
            f"Could not reach Ollama at {self.base_url}. Is Ollama installed and running? "
            f"({DOWNLOAD_URL}) [{exc}]"
        )

    def analyze(self, images_b64: list[str], notes: str, language: str) -> Listing:
        body = {
            "model": self.model,
            "stream": False,
            "format": json_schema(),
            "options": {"temperature": 0.4, "num_ctx": 8192},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(notes, language), "images": images_b64},
            ],
        }
        try:
            resp = self.http.post(f"{self.base_url}/api/chat", json=body, timeout=900)
        except requests.RequestException as exc:
            raise self._not_running(exc) from None
        if resp.status_code == 404:
            raise AIError(
                f"Ollama model '{self.model}' is not installed. Run:  ollama pull {self.model}"
            )
        if resp.status_code != 200:
            try:
                message = resp.json().get("error", resp.text)
            except ValueError:
                message = resp.text
            raise AIError(f"Ollama error {resp.status_code}: {message[:300]}")
        text = resp.json().get("message", {}).get("content", "")
        return parse_listing(text, language)

    def list_models(self) -> list[str]:
        try:
            resp = self.http.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise self._not_running(exc) from None
        return sorted(m.get("name", "") for m in resp.json().get("models", []) if m.get("name"))
