"""Google Gemini (free tier via an AI Studio API key)."""

from __future__ import annotations

import time

import requests

from vinted_prep.ai.base import AIError, Provider
from vinted_prep.ai.prompt import SYSTEM_PROMPT, build_user_prompt, gemini_schema, parse_listing
from vinted_prep.models import Listing

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
KEY_URL = "https://aistudio.google.com/apikey"

# Tried in order after the chosen model when it is overloaded (503) or out of free quota (429).
# Models that don't exist (anymore) are skipped.
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
STALLED = 408  # stand-in status for a request Gemini accepted but never answered
BUSY = (STALLED, 500, 502, 503, 504)


class GeminiProvider(Provider):
    name = "Gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        session: requests.Session | None = None,
        retry_wait: float = 5.0,
        fallback_models: list[str] | None = None,
    ):
        self.api_key = api_key.strip()
        self.model = model.strip().removeprefix("models/")
        self.http = session or requests.Session()
        self.retry_wait = retry_wait
        self.fallback_models = FALLBACK_MODELS if fallback_models is None else fallback_models
        self.used_model = ""
        self.skip_reason = ""  # why the chosen model was passed over, for the status line
        self.progress = lambda msg: None  # set by the caller to show retry status

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
        resp = self._post_with_fallback(body)
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

    def _post(self, model: str, body: dict) -> requests.Response:
        try:
            return self.http.post(
                f"{API_BASE}/models/{model}:generateContent",
                json=body,
                headers={"x-goog-api-key": self.api_key},
                timeout=(15, 90),
            )
        except requests.ReadTimeout:
            # Connected fine, but the model hung (it happens when it's overloaded): treat it as busy.
            resp = requests.Response()
            resp.status_code = STALLED
            resp._content = b'{"error": {"message": "no answer within 90s"}}'
            return resp
        except requests.RequestException as exc:
            raise AIError(f"Could not reach Gemini (check your internet connection): {exc}") from None

    def _post_with_fallback(self, body: dict) -> requests.Response:
        """Retry an overloaded model with growing waits, then move on to the next model."""
        chain = [self.model] + [m for m in self.fallback_models if m != self.model]
        last_busy = None
        tried = []
        for model in chain:
            attempts = 3 if model == self.model else 2
            for attempt in range(attempts):
                resp = self._post(model, body)
                if resp.status_code == 200:
                    self.used_model = model
                    return resp
                if resp.status_code not in BUSY + (429,):
                    break
                last_busy = resp
                if resp.status_code in (429, STALLED) or attempt == attempts - 1:
                    break  # quota is per model, and waiting out another hang is too slow: try the next one
                wait = self.retry_wait * 2**attempt
                self.progress(f"Gemini '{model}' is busy, retrying in {wait:g}s…")
                time.sleep(wait)
            if model == self.model:
                self.skip_reason = {404: "was not found", 429: "is out of free quota"}.get(resp.status_code, "was busy")
            if resp.status_code == 404:
                if model == self.model and not self.fallback_models:
                    self._raise_for(resp)
                continue  # model doesn't exist: skip it
            if resp.status_code not in BUSY + (429,):
                self._raise_for(resp)
            tried.append(model)
            self.progress(f"Gemini '{model}' unavailable, trying another model…")
        if last_busy is None:
            raise AIError(f"Gemini model '{self.model}' was not found. Open Settings and press 'Refresh'.")
        if last_busy.status_code == 429:
            self._raise_for(last_busy)
        raise AIError(
            f"Gemini is overloaded right now (tried {', '.join(tried)}). This is on Google's side and "
            "usually passes within minutes: press 'Rewrite listing' later, or switch to Ollama in Settings."
        )

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
