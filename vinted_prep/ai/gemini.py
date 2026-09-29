"""Google Gemini (free tier via an AI Studio API key)."""

from __future__ import annotations

import time

import requests

from vinted_prep.ai.base import AIError, Provider
from vinted_prep.ai.prompt import SYSTEM_PROMPT, build_user_prompt, gemini_schema, parse_listing
from vinted_prep.models import Listing

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
KEY_URL = "https://aistudio.google.com/apikey"


class GeminiProvider(Provider):
    name = "Gemini"

    def __init__(self, api_key: str, model: str, session: requests.Session | None = None, retry_wait: float = 5.0):
        self.api_key = api_key.strip()
        self.model = model.strip().removeprefix("models/")
        self.http = session or requests.Session()
        self.retry_wait = retry_wait

    def _check_key(self) -> None:
        if not self.api_key:
            raise AIError(f"No Gemini API key set. Get a free key at {KEY_URL} and paste it in Settings.")

    def _raise_for(self, resp: requests.Response) -> None:
        try:
            message = resp.json().get("error", {}).get("message", "")
        except ValueError:
            message = resp.text[:300]
        if resp.status_code in (400, 401, 403) and "key" in message.lower():
            raise AIError(f"Gemini rejected the API key. Check it in Settings. ({message})")
        if resp.status_code == 404:
            raise AIError(
                f"Gemini model '{self.model}' was not found. Open Settings and press 'Refresh' "
                f"to pick an available model. ({message})"
            )
        if resp.status_code == 429:
            raise AIError(f"Gemini free-tier limit reached; wait a minute (or until tomorrow) and retry. ({message})")
        raise AIError(f"Gemini error {resp.status_code}: {message}")

    def analyze(self, images_b64: list[str], notes: str, language: str) -> Listing:
        self._check_key()
        parts = [{"text": build_user_prompt(notes, language)}]
        parts += [{"inline_data": {"mime_type": "image/jpeg", "data": b64}} for b64 in images_b64]
        body = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.4,
                "responseMimeType": "application/json",
                "responseSchema": gemini_schema(),
            },
        }
        url = f"{API_BASE}/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key}

        for attempt in range(3):
            try:
                resp = self.http.post(url, json=body, headers=headers, timeout=180)
            except requests.RequestException as exc:
                raise AIError(f"Could not reach Gemini (check your internet connection): {exc}") from None
            if resp.status_code in (429, 500, 503) and attempt < 2:
                time.sleep(self.retry_wait * (attempt + 1))
                continue
            break
        if resp.status_code != 200:
            self._raise_for(resp)

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            reason = data.get("promptFeedback", {}).get("blockReason", "no answer")
            raise AIError(f"Gemini returned no listing ({reason}).")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text:
            raise AIError(f"Gemini returned an empty answer ({candidates[0].get('finishReason', '?')}).")
        return parse_listing(text, language)

    def list_models(self) -> list[str]:
        self._check_key()
        models, token = [], None
        while True:
            params = {"pageSize": 200}
            if token:
                params["pageToken"] = token
            try:
                resp = self.http.get(
                    f"{API_BASE}/models", params=params, headers={"x-goog-api-key": self.api_key}, timeout=30
                )
            except requests.RequestException as exc:
                raise AIError(f"Could not reach Gemini: {exc}") from None
            if resp.status_code != 200:
                self._raise_for(resp)
            data = resp.json()
            for m in data.get("models", []):
                name = m.get("name", "").removeprefix("models/")
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in methods and "gemini" in name and "embedding" not in name:
                    models.append(name)
            token = data.get("nextPageToken")
            if not token:
                break
        return sorted(set(models), key=lambda n: ("flash" not in n, "latest" not in n, n))
