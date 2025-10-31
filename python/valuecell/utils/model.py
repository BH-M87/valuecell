import os

from agno.models.google import Gemini
from agno.models.openrouter import OpenRouter


def get_model(env_key: str):
    """
    Get a model instance based on environment configuration.

    If USE_AUGGIE=true is set in environment, returns an auggie adapter.
    Otherwise, returns the original model (Gemini or OpenRouter).

    Args:
        env_key: Environment variable key for model ID

    Returns:
        Model instance (Gemini, OpenRouter, or AuggieAgnoAdapter)
    """
    model_id = os.getenv(env_key)
    use_auggie = os.getenv("USE_AUGGIE", "false").lower() == "true"

    if use_auggie:
        # Use auggie adapter when USE_AUGGIE is enabled
        from valuecell.utils.auggie_adapter import AuggieAgnoAdapter
        return AuggieAgnoAdapter(
            id=model_id or "google/gemini-2.5-flash",
            workspace_root=os.getenv("WORKSPACE_ROOT", os.getcwd())
        )

    # Original behavior
    if os.getenv("GOOGLE_API_KEY"):
        return Gemini(id=model_id or "gemini-2.5-flash")
    return OpenRouter(id=model_id or "google/gemini-2.5-flash", max_tokens=None)
