#!/usr/bin/env python3
"""
Example script demonstrating Auggie CLI integration.

This script shows how to use auggie for LLM calls without direct API keys.

Prerequisites:
1. Install auggie: Follow instructions at https://www.augmentcode.com/changelog/auggie-cli
2. Authenticate: Run `auggie login`
3. Set environment: export USE_AUGGIE=true

Usage:
    python examples/auggie_example.py
"""

import os
import sys
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from valuecell.utils.auggie_client import (
    AuggieClient,
    normalize_model_name,
    get_available_models,
    list_available_models,
)
from valuecell.utils.auggie_adapter import get_auggie_model


# Define response models
class StockAnalysis(BaseModel):
    """Stock analysis result."""
    ticker: str = Field(description="Stock ticker symbol")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score between 0 and 1")
    key_points: list[str] = Field(description="Key points from the analysis")
    recommendation: str = Field(description="Investment recommendation")


class MarketSummary(BaseModel):
    """Market summary."""
    market_trend: str = Field(description="Overall market trend")
    key_movers: list[str] = Field(description="Key market movers")
    risk_level: str = Field(description="Current risk level: low, medium, or high")


def example_0_model_validation():
    """Example 0: Model validation and normalization."""
    print("\n" + "="*60)
    print("Example 0: Model Validation and Normalization")
    print("="*60)

    # Show available models
    print("\n" + list_available_models())

    # Test model normalization
    print("\n" + "-"*60)
    print("Model Name Normalization Examples:")
    print("-"*60)

    test_models = [
        "sonnet4.5",
        "anthropic/claude-3-5-sonnet",
        "google/gemini-2.5-flash",
        "gpt-4o",
        "openai/gpt-4o-mini",
    ]

    for model in test_models:
        try:
            normalized = normalize_model_name(model)
            print(f"  '{model}' → '{normalized}'")
        except ValueError as e:
            print(f"  '{model}' → ERROR: {e}")

    # Test invalid model
    print("\n" + "-"*60)
    print("Invalid Model Example:")
    print("-"*60)
    try:
        normalize_model_name("invalid-model-name")
    except ValueError as e:
        print(f"  Error (expected): {str(e)[:100]}...")


def example_1_basic_usage():
    """Example 1: Basic text generation."""
    print("\n" + "="*60)
    print("Example 1: Basic Text Generation")
    print("="*60)

    # Note: Model names are automatically normalized
    # "google/gemini-2.5-flash" → "sonnet4.5"
    client = AuggieClient(
        model="google/gemini-2.5-flash",  # Will be mapped to sonnet4.5
        max_turns=1,
        quiet=True
    )

    print(f"Using model: {client.model}")  # Will show: sonnet4.5

    prompt = "Explain what a stock market index is in 2-3 sentences."
    print(f"\nPrompt: {prompt}")

    response = client.invoke(prompt)
    print(f"\nResponse:\n{response}")


def example_2_structured_output():
    """Example 2: Structured output with Pydantic model."""
    print("\n" + "="*60)
    print("Example 2: Structured Output")
    print("="*60)
    
    client = AuggieClient(
        model="google/gemini-2.5-flash",
        max_turns=1,
        quiet=True
    )
    
    prompt = """
    Analyze Apple Inc. (AAPL) stock based on the following information:
    - Recent earnings beat expectations
    - New product launches announced
    - Strong iPhone sales in Q4
    - Concerns about supply chain issues
    
    Provide a structured analysis.
    """
    
    print(f"\nPrompt: {prompt.strip()}")
    
    result = client.invoke(
        prompt=prompt,
        output_schema=StockAnalysis
    )
    
    print(f"\nStructured Result:")
    print(f"  Ticker: {result.ticker}")
    print(f"  Sentiment: {result.sentiment}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Key Points:")
    for point in result.key_points:
        print(f"    - {point}")
    print(f"  Recommendation: {result.recommendation}")


def example_3_langchain_adapter():
    """Example 3: Using LangChain adapter."""
    print("\n" + "="*60)
    print("Example 3: LangChain Adapter")
    print("="*60)
    
    # Create adapter
    adapter = get_auggie_model(
        model_name="google/gemini-2.5-flash",
        adapter_type="langchain",
        max_turns=1
    )
    
    # Use with structured output
    structured_adapter = adapter.with_structured_output(MarketSummary)
    
    prompt = """
    Provide a brief market summary for today:
    - S&P 500 up 1.2%
    - Tech stocks leading gains
    - Energy sector down
    - Fed meeting next week
    """
    
    print(f"\nPrompt: {prompt.strip()}")
    
    result = structured_adapter.invoke(prompt)
    
    print(f"\nMarket Summary:")
    print(f"  Trend: {result.market_trend}")
    print(f"  Key Movers: {', '.join(result.key_movers)}")
    print(f"  Risk Level: {result.risk_level}")


def example_4_agno_adapter():
    """Example 4: Using agno adapter."""
    print("\n" + "="*60)
    print("Example 4: agno Adapter")
    print("="*60)
    
    # Create adapter
    adapter = get_auggie_model(
        model_name="google/gemini-2.5-flash",
        adapter_type="agno",
        max_turns=1
    )
    
    messages = [
        {
            "role": "system",
            "content": "You are a financial advisor assistant."
        },
        {
            "role": "user",
            "content": "What are the key factors to consider when investing in tech stocks?"
        }
    ]
    
    print("\nMessages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    
    result = adapter.response(messages)
    
    print(f"\nResponse:")
    print(f"  {result['content']}")


async def example_5_async_usage():
    """Example 5: Async usage."""
    print("\n" + "="*60)
    print("Example 5: Async Usage")
    print("="*60)
    
    client = AuggieClient(
        model="google/gemini-2.5-flash",
        max_turns=1,
        quiet=True
    )
    
    # Multiple concurrent requests
    prompts = [
        "What is diversification in investing?",
        "Explain what a P/E ratio is.",
        "What is dollar-cost averaging?"
    ]
    
    print("\nSending multiple concurrent requests...")
    
    tasks = [client.ainvoke(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    
    for prompt, result in zip(prompts, results):
        print(f"\nQ: {prompt}")
        print(f"A: {result[:100]}...")  # Show first 100 chars


def example_6_model_factory():
    """Example 6: Using the model factory with USE_AUGGIE."""
    print("\n" + "="*60)
    print("Example 6: Model Factory Integration")
    print("="*60)
    
    # Save original env
    original_use_auggie = os.getenv("USE_AUGGIE")
    
    try:
        # Enable auggie
        os.environ["USE_AUGGIE"] = "true"
        os.environ["RESEARCH_AGENT_MODEL_ID"] = "google/gemini-2.5-flash"
        
        # Import after setting env
        from valuecell.utils.model import get_model
        
        # Get model (will use auggie)
        model = get_model("RESEARCH_AGENT_MODEL_ID")
        
        print("\nUsing model factory with USE_AUGGIE=true")
        print(f"Model type: {type(model).__name__}")
        
        # Use the model
        messages = [
            {"role": "user", "content": "What is a mutual fund?"}
        ]
        
        result = model.response(messages)
        print(f"\nResponse: {result['content'][:150]}...")
        
    finally:
        # Restore original env
        if original_use_auggie is not None:
            os.environ["USE_AUGGIE"] = original_use_auggie
        else:
            os.environ.pop("USE_AUGGIE", None)


def check_auggie_available():
    """Check if auggie is available."""
    import subprocess
    try:
        result = subprocess.run(
            ["auggie", "--help"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Auggie CLI Integration Examples")
    print("="*60)
    
    # Check if auggie is available
    if not check_auggie_available():
        print("\n❌ Error: auggie CLI is not available!")
        print("\nPlease install auggie:")
        print("  Visit: https://www.augmentcode.com/changelog/auggie-cli")
        print("\nThen authenticate:")
        print("  Run: auggie login")
        return
    
    print("\n✓ auggie CLI is available")
    
    try:
        # Run examples
        example_0_model_validation()
        example_1_basic_usage()
        example_2_structured_output()
        example_3_langchain_adapter()
        example_4_agno_adapter()
        
        # Async example
        asyncio.run(example_5_async_usage())
        
        example_6_model_factory()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

