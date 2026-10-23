import json

import pytest

from vinted_prep.ai import AIError
from vinted_prep.ai.gemini import GeminiProvider
from vinted_prep.ai.ollama import OllamaProvider
from vinted_prep.ai.prompt import TITLE_MAX, build_user_prompt, parse_listing

GOOD = {
    "title": "Zara μαύρο σατέν midi φόρεμα, M",
    "description": "Κομψό midi φόρεμα από τη Zara.\nΣατέν ύφασμα που πέφτει όμορφα.",
    "brand": "Zara",
    "size": "M",
    "color": "μαύρο",
    "material": "100% polyester",
    "hashtags": ["#zara", "φόρεμα", "midi dress", "zara"],
    "price": 11.3,
    "price_note": "Fast fashion, καλή κατάσταση.",
}


# --- parsing ---
def test_parse_plain_json():
    listing = parse_listing(json.dumps(GOOD), "Greek")
    assert listing.brand == "Zara"
    assert listing.price == 11.5  # rounded to .50
    assert listing.hashtags == ["zara", "φόρεμα", "mididress"]  # deduped, cleaned
    assert listing.language == "Greek"
    assert listing.full_description().endswith("#zara #φόρεμα #mididress")


def test_parse_fenced_json_with_chatter():
    text = "Sure! Here it is:\n```json\n" + json.dumps(GOOD) + "\n```\nHope it helps"
    assert parse_listing(text, "Greek").title == GOOD["title"]


def test_parse_price_strings_and_unknowns():
    data = dict(GOOD, price="12,5 €", brand="Unknown", size=None, hashtags="a, b c")
    listing = parse_listing(json.dumps(data), "English")
    assert listing.price == 12.5
    assert listing.brand == "" and listing.size == ""
    assert listing.hashtags == ["a", "b", "c"]


def test_parse_long_title_is_cut_on_a_word():
    data = dict(GOOD, title="Very " * 30 + "long title")
    title = parse_listing(json.dumps(data), "English").title
    assert len(title) <= TITLE_MAX and not title.endswith(" ")


@pytest.mark.parametrize("text", ["not json at all", "{broken", json.dumps({"brand": "x"})])
def test_parse_errors(text):
    with pytest.raises(AIError):
        parse_listing(text, "English")


def test_prompt_includes_language_and_notes():
    prompt = build_user_prompt("worn twice", "French")
    assert "French" in prompt and "worn twice" in prompt
    assert "(none)" in build_user_prompt("  ", "French")


# --- fake HTTP ---
class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        return self.responses.pop(0)

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        return self.responses.pop(0)


def gemini_ok(text):
    return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": text}]}}]})


def test_gemini_request_and_response():
    session = FakeSession(gemini_ok(json.dumps(GOOD)))
    listing = GeminiProvider("KEY", "models/gemini-flash-latest", session).analyze(["AAA", "BBB"], "notes", "Greek")
    assert listing.brand == "Zara"
    method, url, kw = session.calls[0]
    assert url.endswith("/models/gemini-flash-latest:generateContent")
    assert kw["headers"]["x-goog-api-key"] == "KEY"
    parts = kw["json"]["contents"][0]["parts"]
    assert [p["inline_data"]["data"] for p in parts[1:]] == ["AAA", "BBB"]
    assert kw["json"]["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_ignores_thought_parts():
    resp = FakeResponse(
        200, {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": json.dumps(GOOD)}]}}]}
    )
    assert GeminiProvider("K", "m", FakeSession(resp)).analyze(["A"], "", "Greek").title == GOOD["title"]


def _models_called(session):
    return [url.split("/models/")[1].split(":")[0] for _, url, _ in session.calls]


def test_gemini_retry_succeeds():
    busy = FakeResponse(503, {"error": {"message": "overloaded"}})
    session = FakeSession(busy, gemini_ok(json.dumps(GOOD)))
    provider = GeminiProvider("K", "m", session, retry_wait=0)
    messages = []
    provider.progress = messages.append
    assert provider.analyze(["A"], "", "Greek").brand == "Zara"
    assert _models_called(session) == ["m", "m"] and provider.used_model == "m"
    assert any("busy" in m for m in messages)


def test_gemini_falls_back_to_next_model_when_overloaded():
    busy = FakeResponse(503, {"error": {"message": "This model is currently experiencing high demand."}})
    missing = FakeResponse(404, {"error": {"message": "not found"}})
    session = FakeSession(busy, busy, busy, missing, gemini_ok(json.dumps(GOOD)))
    provider = GeminiProvider("K", "main", session, retry_wait=0, fallback_models=["gone", "backup"])
    assert provider.analyze(["A"], "", "Greek").brand == "Zara"
    assert _models_called(session) == ["main", "main", "main", "gone", "backup"]
    assert provider.used_model == "backup"


def test_gemini_all_overloaded_gives_friendly_error():
    busy = FakeResponse(503, {"error": {"message": "high demand"}})
    session = FakeSession(*[busy] * 5)
    with pytest.raises(AIError, match="overloaded right now.*Ollama"):
        GeminiProvider("K", "main", session, retry_wait=0, fallback_models=["backup"]).analyze(["A"], "", "Greek")
    assert _models_called(session) == ["main"] * 3 + ["backup"] * 2


def test_gemini_quota_skips_to_next_model_then_reports():
    quota = FakeResponse(429, {"error": {"message": "Resource exhausted"}})
    session = FakeSession(quota, quota)
    with pytest.raises(AIError, match="limit"):
        GeminiProvider("K", "main", session, retry_wait=0, fallback_models=["backup"]).analyze(["A"], "", "Greek")
    assert _models_called(session) == ["main", "backup"]


def test_gemini_errors():
    with pytest.raises(AIError, match="No Gemini API key"):
        GeminiProvider("", "m", FakeSession()).analyze(["A"], "", "Greek")
    bad_key = FakeResponse(400, {"error": {"message": "API key not valid"}})
    with pytest.raises(AIError, match="rejected the API key"):
        GeminiProvider("K", "m", FakeSession(bad_key)).analyze(["A"], "", "Greek")
    missing = FakeResponse(404, {"error": {"message": "not found"}})
    with pytest.raises(AIError, match="Refresh"):
        GeminiProvider("K", "m", FakeSession(missing), fallback_models=[]).analyze(["A"], "", "Greek")
    with pytest.raises(AIError, match="Refresh"):
        GeminiProvider("K", "m", FakeSession(missing, missing), fallback_models=["x"]).analyze(["A"], "", "Greek")


def test_gemini_list_models_filters_and_sorts():
    payload = {
        "models": [
            {"name": "models/gemini-pro-x", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/gemini-flash-latest", "supportedGenerationMethods": ["generateContent"]},
        ]
    }
    models = GeminiProvider("K", "m", FakeSession(FakeResponse(200, payload))).list_models()
    assert models == ["gemini-flash-latest", "gemini-pro-x"]


def test_ollama_request_and_response():
    session = FakeSession(FakeResponse(200, {"message": {"content": json.dumps(GOOD)}}))
    listing = OllamaProvider("http://localhost:11434/", "qwen2.5vl:7b", session).analyze(["AAA"], "", "Greek")
    assert listing.size == "M"
    _, url, kw = session.calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert kw["json"]["messages"][1]["images"] == ["AAA"]
    assert kw["json"]["format"]["type"] == "object"


def test_ollama_errors():
    import requests

    class Down(FakeSession):
        def post(self, url, **kw):
            raise requests.ConnectionError("refused")

    with pytest.raises(AIError, match="Is Ollama installed"):
        OllamaProvider("http://x", "m", Down()).analyze(["A"], "", "Greek")
    with pytest.raises(AIError, match="ollama pull m"):
        OllamaProvider("http://x", "m", FakeSession(FakeResponse(404, {"error": "model not found"}))).analyze(
            ["A"], "", "Greek"
        )


def test_ollama_list_models():
    payload = {"models": [{"name": "qwen2.5vl:7b"}, {"name": "gemma3:12b"}]}
    assert OllamaProvider("http://x", "m", FakeSession(FakeResponse(200, payload))).list_models() == [
        "gemma3:12b",
        "qwen2.5vl:7b",
    ]
