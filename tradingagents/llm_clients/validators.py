from .model_catalog import get_known_models


VALID_MODELS = {
    provider: models
    for provider, models in get_known_models().items()
    if provider not in ("ollama", "openrouter", "azure")
}


def validate_model(provider: str, model: str) -> bool:
    """Accept non-empty model IDs so new provider releases work immediately.

    ``VALID_MODELS`` remains useful for discovery and tests, but it must not be
    an API gate: every supported provider exposes either a models endpoint or
    stable model-ID convention, and those lists change faster than releases of
    this application.
    """
    provider_lower = (provider or "").lower()
    if not provider_lower:
        return False
    return bool(str(model or "").strip())
