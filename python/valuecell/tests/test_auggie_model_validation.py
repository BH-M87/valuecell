"""
Tests for Auggie model validation and normalization.

These tests verify that model names are properly validated and normalized.
"""

import pytest
from valuecell.utils.auggie_client import (
    normalize_model_name,
    get_available_models,
    list_available_models,
    AUGGIE_SUPPORTED_MODELS,
    AUGGIE_MODEL_ALIASES,
)


class TestModelNormalization:
    """Tests for model name normalization."""

    def test_valid_auggie_models(self):
        """Test that valid auggie model IDs are accepted."""
        for model_id in AUGGIE_SUPPORTED_MODELS.keys():
            result = normalize_model_name(model_id)
            assert result == model_id

    def test_model_aliases(self):
        """Test that model aliases are properly mapped."""
        # Test some common aliases
        assert normalize_model_name("anthropic/claude-3-5-sonnet") == "sonnet4.5"
        assert normalize_model_name("anthropic/claude-haiku-4.5") == "haiku4.5"
        assert normalize_model_name("openai/gpt-5") == "gpt5"
        assert normalize_model_name("gpt-4o") == "gpt5"
        assert normalize_model_name("gpt-4o-mini") == "haiku4.5"

    def test_google_aliases(self):
        """Test that Google model aliases are mapped."""
        assert normalize_model_name("google/gemini-2.5-flash") == "sonnet4.5"
        assert normalize_model_name("google/gemini-1.5-pro") == "sonnet4.5"

    def test_deepseek_aliases(self):
        """Test that DeepSeek model aliases are mapped."""
        assert normalize_model_name("deepseek/deepseek-chat-v3-0324") == "sonnet4.5"

    def test_case_insensitive(self):
        """Test that model names are case-insensitive."""
        assert normalize_model_name("SONNET4.5") == "sonnet4.5"
        assert normalize_model_name("Haiku4.5") == "haiku4.5"
        assert normalize_model_name("GPT5") == "gpt5"

    def test_default_model(self):
        """Test that None returns default model."""
        result = normalize_model_name(None)
        assert result == "sonnet4.5"

    def test_invalid_model(self):
        """Test that invalid model names raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            normalize_model_name("invalid-model-name")
        
        error_msg = str(exc_info.value)
        assert "not supported" in error_msg
        assert "Supported models" in error_msg

    def test_all_aliases_valid(self):
        """Test that all defined aliases map to valid models."""
        for alias, target in AUGGIE_MODEL_ALIASES.items():
            assert target in AUGGIE_SUPPORTED_MODELS, f"Alias '{alias}' maps to invalid model '{target}'"


class TestModelUtilities:
    """Tests for model utility functions."""

    def test_get_available_models(self):
        """Test getting available models dictionary."""
        models = get_available_models()
        assert isinstance(models, dict)
        assert len(models) > 0
        assert "sonnet4.5" in models
        assert "haiku4.5" in models
        assert "gpt5" in models

    def test_list_available_models(self):
        """Test listing available models as string."""
        models_str = list_available_models()
        assert isinstance(models_str, str)
        assert "sonnet4.5" in models_str
        assert "haiku4.5" in models_str
        assert "gpt5" in models_str
        assert "Claude" in models_str
        assert "GPT" in models_str


class TestAuggieClientValidation:
    """Tests for AuggieClient model validation."""

    def test_client_with_valid_model(self):
        """Test creating client with valid model."""
        from valuecell.utils.auggie_client import AuggieClient
        
        client = AuggieClient(model="sonnet4.5")
        assert client.model == "sonnet4.5"

    def test_client_with_alias(self):
        """Test creating client with model alias."""
        from valuecell.utils.auggie_client import AuggieClient
        
        client = AuggieClient(model="anthropic/claude-3-5-sonnet")
        assert client.model == "sonnet4.5"

    def test_client_with_invalid_model(self):
        """Test that invalid model raises error."""
        from valuecell.utils.auggie_client import AuggieClient
        
        with pytest.raises(ValueError) as exc_info:
            AuggieClient(model="invalid-model")
        
        assert "not supported" in str(exc_info.value)

    def test_client_without_validation(self):
        """Test creating client without validation."""
        from valuecell.utils.auggie_client import AuggieClient
        
        # Should not raise error when validation is disabled
        client = AuggieClient(model="any-model", validate_model=False)
        assert client.model == "any-model"

    def test_client_default_model(self):
        """Test that client uses default model when None provided."""
        from valuecell.utils.auggie_client import AuggieClient
        
        client = AuggieClient(model=None)
        assert client.model == "sonnet4.5"


class TestModelAliasCompleteness:
    """Tests to ensure model alias mappings are complete."""

    def test_anthropic_models_covered(self):
        """Test that common Anthropic model names are covered."""
        anthropic_models = [
            "anthropic/claude-3-5-sonnet-20241022",
            "anthropic/claude-3-5-sonnet",
            "anthropic/claude-haiku-4.5",
            "claude-3-5-sonnet",
            "claude-haiku-4.5",
        ]
        
        for model in anthropic_models:
            result = normalize_model_name(model)
            assert result in AUGGIE_SUPPORTED_MODELS

    def test_openai_models_covered(self):
        """Test that common OpenAI model names are covered."""
        openai_models = [
            "openai/gpt-5",
            "openai/gpt-4o",
            "gpt-5",
            "gpt-4o",
            "gpt-4o-mini",
        ]
        
        for model in openai_models:
            result = normalize_model_name(model)
            assert result in AUGGIE_SUPPORTED_MODELS

    def test_google_models_covered(self):
        """Test that common Google model names are covered."""
        google_models = [
            "google/gemini-2.5-flash",
            "google/gemini-1.5-pro",
            "gemini-2.5-flash",
        ]
        
        for model in google_models:
            result = normalize_model_name(model)
            assert result in AUGGIE_SUPPORTED_MODELS


class TestErrorMessages:
    """Tests for error message quality."""

    def test_error_message_helpful(self):
        """Test that error messages are helpful."""
        with pytest.raises(ValueError) as exc_info:
            normalize_model_name("unknown-model")
        
        error_msg = str(exc_info.value)
        
        # Should mention the invalid model
        assert "unknown-model" in error_msg
        
        # Should list supported models
        assert "haiku4.5" in error_msg
        assert "sonnet4.5" in error_msg
        assert "gpt5" in error_msg
        
        # Should have clear structure
        assert "Supported models:" in error_msg or "Available models:" in error_msg


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing configurations."""

    def test_env_example_models(self):
        """Test that models from .env.example are supported."""
        # These are the models mentioned in .env.example
        example_models = [
            "google/gemini-2.5-flash",
            "openai/gpt-4o-mini",
            "deepseek/deepseek-chat-v3-0324",
            "anthropic/claude-haiku-4.5",
        ]
        
        for model in example_models:
            result = normalize_model_name(model)
            assert result in AUGGIE_SUPPORTED_MODELS, f"Model '{model}' from .env.example not supported"

    def test_readme_models(self):
        """Test that models mentioned in README are supported."""
        readme_models = [
            "google/gemini-2.5-flash",
            "openai/gpt-4o-mini",
            "anthropic/claude-3-5-sonnet-20241022",
        ]
        
        for model in readme_models:
            result = normalize_model_name(model)
            assert result in AUGGIE_SUPPORTED_MODELS, f"Model '{model}' from README not supported"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

