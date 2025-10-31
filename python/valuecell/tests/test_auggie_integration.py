"""
Tests for Auggie CLI integration.

These tests verify that the auggie client and adapters work correctly.
Note: These tests require auggie to be installed and authenticated.
"""

import os
import pytest
from pydantic import BaseModel
from typing import Optional

# Import auggie components
from valuecell.utils.auggie_client import AuggieClient, create_auggie_client
from valuecell.utils.auggie_adapter import (
    AuggieLangChainAdapter,
    AuggieAgnoAdapter,
    AuggieOpenAIAdapter,
    get_auggie_model,
)


# Test Pydantic models
class SimpleResponse(BaseModel):
    """Simple response model for testing."""
    answer: str
    confidence: float


class AnalysisResult(BaseModel):
    """Analysis result model for testing."""
    sentiment: str
    score: float
    reasoning: str


# Skip tests if auggie is not available
def check_auggie_available():
    """Check if auggie CLI is available."""
    try:
        import subprocess
        result = subprocess.run(
            ["auggie", "--help"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


auggie_available = check_auggie_available()
skip_if_no_auggie = pytest.mark.skipif(
    not auggie_available,
    reason="auggie CLI not available"
)


class TestAuggieClient:
    """Tests for AuggieClient."""

    @skip_if_no_auggie
    def test_client_creation(self):
        """Test creating an auggie client."""
        client = AuggieClient(model="google/gemini-2.5-flash")
        assert client is not None
        assert client.model == "google/gemini-2.5-flash"

    @skip_if_no_auggie
    def test_factory_function(self):
        """Test the factory function."""
        client = create_auggie_client(model="google/gemini-2.5-flash")
        assert client is not None
        assert isinstance(client, AuggieClient)

    @skip_if_no_auggie
    def test_simple_invoke(self):
        """Test simple text invocation."""
        client = AuggieClient(
            model="google/gemini-2.5-flash",
            max_turns=1
        )
        
        result = client.invoke("What is 2+2? Answer with just the number.")
        assert result is not None
        assert isinstance(result, str)
        assert "4" in result

    @skip_if_no_auggie
    def test_structured_output(self):
        """Test structured output with Pydantic model."""
        client = AuggieClient(
            model="google/gemini-2.5-flash",
            max_turns=1
        )
        
        result = client.invoke(
            prompt="What is the capital of France? Provide your answer and confidence (0-1).",
            output_schema=SimpleResponse
        )
        
        assert result is not None
        assert isinstance(result, SimpleResponse)
        assert "paris" in result.answer.lower()
        assert 0 <= result.confidence <= 1

    @skip_if_no_auggie
    @pytest.mark.asyncio
    async def test_async_invoke(self):
        """Test async invocation."""
        client = AuggieClient(
            model="google/gemini-2.5-flash",
            max_turns=1
        )
        
        result = await client.ainvoke("What is 2+2? Answer with just the number.")
        assert result is not None
        assert isinstance(result, str)
        assert "4" in result


class TestAuggieLangChainAdapter:
    """Tests for LangChain adapter."""

    @skip_if_no_auggie
    def test_adapter_creation(self):
        """Test creating a LangChain adapter."""
        adapter = AuggieLangChainAdapter(model="google/gemini-2.5-flash")
        assert adapter is not None

    @skip_if_no_auggie
    def test_simple_invoke(self):
        """Test simple invocation."""
        adapter = AuggieLangChainAdapter(
            model="google/gemini-2.5-flash",
            max_turns=1
        )
        
        result = adapter.invoke("What is 2+2? Answer with just the number.")
        assert result is not None
        assert isinstance(result, str)

    @skip_if_no_auggie
    def test_structured_output(self):
        """Test structured output."""
        adapter = AuggieLangChainAdapter(
            model="google/gemini-2.5-flash",
            max_turns=1
        )
        
        structured_adapter = adapter.with_structured_output(SimpleResponse)
        result = structured_adapter.invoke(
            "What is the capital of France? Provide your answer and confidence (0-1)."
        )
        
        assert result is not None
        assert isinstance(result, SimpleResponse)

    @skip_if_no_auggie
    def test_message_format(self):
        """Test with message format."""
        adapter = AuggieLangChainAdapter(
            model="google/gemini-2.5-flash",
            max_turns=1
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        result = adapter.invoke(messages)
        assert result is not None


class TestAuggieAgnoAdapter:
    """Tests for agno adapter."""

    @skip_if_no_auggie
    def test_adapter_creation(self):
        """Test creating an agno adapter."""
        adapter = AuggieAgnoAdapter(id="google/gemini-2.5-flash")
        assert adapter is not None
        assert adapter.id == "google/gemini-2.5-flash"

    @skip_if_no_auggie
    def test_response(self):
        """Test response method."""
        adapter = AuggieAgnoAdapter(
            id="google/gemini-2.5-flash",
            max_turns=1
        )
        
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        result = adapter.response(messages)
        assert result is not None
        assert isinstance(result, dict)
        assert "content" in result

    @skip_if_no_auggie
    @pytest.mark.asyncio
    async def test_async_response(self):
        """Test async response method."""
        adapter = AuggieAgnoAdapter(
            id="google/gemini-2.5-flash",
            max_turns=1
        )
        
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        result = await adapter.aresponse(messages)
        assert result is not None
        assert isinstance(result, dict)


class TestAuggieOpenAIAdapter:
    """Tests for OpenAI adapter."""

    @skip_if_no_auggie
    def test_adapter_creation(self):
        """Test creating an OpenAI adapter."""
        adapter = AuggieOpenAIAdapter(model="google/gemini-2.5-flash")
        assert adapter is not None

    @skip_if_no_auggie
    def test_create_completion(self):
        """Test create completion method."""
        adapter = AuggieOpenAIAdapter(
            model="google/gemini-2.5-flash",
            max_turns=1
        )
        
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        result = adapter.create(messages=messages)
        assert result is not None


class TestAdapterFactory:
    """Tests for adapter factory function."""

    @skip_if_no_auggie
    def test_langchain_factory(self):
        """Test creating LangChain adapter via factory."""
        adapter = get_auggie_model(
            model_name="google/gemini-2.5-flash",
            adapter_type="langchain"
        )
        assert isinstance(adapter, AuggieLangChainAdapter)

    @skip_if_no_auggie
    def test_agno_factory(self):
        """Test creating agno adapter via factory."""
        adapter = get_auggie_model(
            model_name="google/gemini-2.5-flash",
            adapter_type="agno"
        )
        assert isinstance(adapter, AuggieAgnoAdapter)

    @skip_if_no_auggie
    def test_openai_factory(self):
        """Test creating OpenAI adapter via factory."""
        adapter = get_auggie_model(
            model_name="google/gemini-2.5-flash",
            adapter_type="openai"
        )
        assert isinstance(adapter, AuggieOpenAIAdapter)

    def test_invalid_adapter_type(self):
        """Test that invalid adapter type raises error."""
        with pytest.raises(ValueError):
            get_auggie_model(
                model_name="google/gemini-2.5-flash",
                adapter_type="invalid"
            )


class TestIntegration:
    """Integration tests."""

    @skip_if_no_auggie
    def test_model_factory_with_auggie(self):
        """Test that model factory returns auggie adapter when USE_AUGGIE=true."""
        # Save original env
        original_use_auggie = os.getenv("USE_AUGGIE")
        
        try:
            # Set USE_AUGGIE
            os.environ["USE_AUGGIE"] = "true"
            
            from valuecell.utils.model import get_model
            
            # This should return an auggie adapter
            model = get_model("RESEARCH_AGENT_MODEL_ID")
            
            # Check that it's an auggie adapter
            assert hasattr(model, 'client')
            assert isinstance(model.client, AuggieClient)
            
        finally:
            # Restore original env
            if original_use_auggie is not None:
                os.environ["USE_AUGGIE"] = original_use_auggie
            else:
                os.environ.pop("USE_AUGGIE", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

