from vinted_prep.ai.base import AIError, Provider
from vinted_prep.config import Settings


def get_provider(settings: Settings) -> Provider:
    if settings.provider == "ollama":
        from vinted_prep.ai.ollama import OllamaProvider

        return OllamaProvider(settings.ollama_url, settings.ollama_model)
    from vinted_prep.ai.gemini import GeminiProvider

    return GeminiProvider(settings.effective_gemini_key(), settings.gemini_model)


__all__ = ["AIError", "Provider", "get_provider"]
