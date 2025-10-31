# Auggie Integration Examples

This directory contains examples demonstrating how to use the Auggie CLI integration in ValueCell.

## What is Auggie?

Auggie is a command-line AI assistant provided by Augment Code. It allows you to interact with various LLM models without needing direct API keys from each provider. Instead, you authenticate once with Augment and can access multiple models through a unified interface.

## Why Use Auggie Integration?

1. **No API Keys Required**: Don't need separate API keys for OpenAI, Anthropic, Google, etc.
2. **Unified Billing**: All usage billed through your Augment account
3. **Built-in Features**: Automatic caching, rate limiting, and retry logic
4. **Easy Switching**: Toggle between direct API calls and auggie with one environment variable

## Prerequisites

### 1. Install Auggie CLI

Follow the installation instructions at: https://www.augmentcode.com/changelog/auggie-cli

**macOS/Linux:**
```bash
curl -fsSL https://install.augmentcode.com | sh
```

**Windows:**
```powershell
irm https://install.augmentcode.com/windows | iex
```

### 2. Authenticate

```bash
auggie login
```

This will open a browser window for authentication.

### 3. Verify Installation

```bash
auggie --help
auggie account
```

## Quick Start

### 1. Enable Auggie in Your Environment

Edit your `.env` file:

```bash
# Enable Auggie integration
USE_AUGGIE=true

# Optional: Set workspace root
WORKSPACE_ROOT=/path/to/valuecell
```

### 2. Run the Example Script

```bash
cd valuecell
python examples/auggie_example.py
```

This will run through several examples demonstrating different use cases.

## Examples Overview

### Example 1: Basic Text Generation

Simple text generation without structured output:

```python
from valuecell.utils.auggie_client import AuggieClient

client = AuggieClient(model="google/gemini-2.5-flash")
response = client.invoke("Explain what a stock market index is.")
print(response)
```

### Example 2: Structured Output

Get structured responses using Pydantic models:

```python
from pydantic import BaseModel

class StockAnalysis(BaseModel):
    ticker: str
    sentiment: str
    confidence: float
    recommendation: str

client = AuggieClient(model="google/gemini-2.5-flash")
result = client.invoke(
    prompt="Analyze AAPL stock...",
    output_schema=StockAnalysis
)
print(f"Sentiment: {result.sentiment}")
```

### Example 3: LangChain Adapter

Use auggie with LangChain-style interface:

```python
from valuecell.utils.auggie_adapter import get_auggie_model

adapter = get_auggie_model(
    model_name="google/gemini-2.5-flash",
    adapter_type="langchain"
)

structured_adapter = adapter.with_structured_output(StockAnalysis)
result = structured_adapter.invoke("Analyze AAPL...")
```

### Example 4: agno Adapter

Use auggie with agno-style interface:

```python
adapter = get_auggie_model(
    model_name="google/gemini-2.5-flash",
    adapter_type="agno"
)

messages = [
    {"role": "user", "content": "What is diversification?"}
]
result = adapter.response(messages)
```

### Example 5: Async Usage

Make concurrent requests efficiently:

```python
import asyncio

async def analyze_multiple():
    client = AuggieClient(model="google/gemini-2.5-flash")
    
    prompts = [
        "Analyze AAPL",
        "Analyze GOOGL",
        "Analyze MSFT"
    ]
    
    tasks = [client.ainvoke(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results

results = asyncio.run(analyze_multiple())
```

### Example 6: Model Factory Integration

Use auggie transparently through the existing model factory:

```python
import os
os.environ["USE_AUGGIE"] = "true"

from valuecell.utils.model import get_model

# This will automatically use auggie
model = get_model("RESEARCH_AGENT_MODEL_ID")

# Use as normal
result = model.response([
    {"role": "user", "content": "Analyze this stock..."}
])
```

## Available Models

When using auggie, you can access various models:

### OpenAI Models
- `openai/gpt-4o`
- `openai/gpt-4o-mini`
- `openai/gpt-4-turbo`

### Anthropic Models
- `anthropic/claude-3-5-sonnet-20241022`
- `anthropic/claude-haiku-4.5`

### Google Models
- `google/gemini-2.5-flash`
- `google/gemini-1.5-pro`

### DeepSeek Models
- `deepseek/deepseek-chat-v3-0324`

### And many more...

Check available models:
```bash
auggie model list
```

## Configuration Options

### Environment Variables

```bash
# Enable/disable auggie
USE_AUGGIE=true

# Workspace root (optional)
WORKSPACE_ROOT=/path/to/workspace

# Model IDs (used when USE_AUGGIE=true)
PLANNER_MODEL_ID=google/gemini-2.5-flash
RESEARCH_AGENT_MODEL_ID=google/gemini-2.5-flash
SEC_PARSER_MODEL_ID=openai/gpt-4o-mini
```

### Client Options

```python
client = AuggieClient(
    model="google/gemini-2.5-flash",  # Model to use
    workspace_root="/path/to/workspace",  # Workspace directory
    max_turns=1,  # Max agentic turns (1 for simple completion)
    quiet=True,  # Only show final output
)
```

## Troubleshooting

### "auggie: command not found"

Make sure auggie is installed and in your PATH:
```bash
which auggie
auggie --version
```

### "Authentication required"

Run the login command:
```bash
auggie login
```

### "Model not found"

Check available models:
```bash
auggie model list
```

### Timeout Errors

Increase the timeout:
```python
client.invoke(prompt, timeout=600)  # 10 minutes
```

### JSON Parsing Errors

The model might not be following the schema. Try:
1. Using a more capable model
2. Adjusting your prompt to be more specific
3. Checking the raw response in error messages

## Performance Tips

1. **Use Async for Multiple Requests**: When making multiple requests, use async methods for better performance.

2. **Set Appropriate Timeouts**: Adjust timeouts based on expected response time.

3. **Cache Results**: Auggie has built-in caching, but you can also implement your own caching layer.

4. **Choose the Right Model**: Faster models like `gemini-2.5-flash` for simple tasks, more capable models for complex analysis.

## Switching Between Auggie and Direct API

You can easily switch between auggie and direct API calls:

```python
import os

# Use auggie
os.environ["USE_AUGGIE"] = "true"
model = get_model("RESEARCH_AGENT_MODEL_ID")  # Uses auggie

# Use direct API
os.environ["USE_AUGGIE"] = "false"
model = get_model("RESEARCH_AGENT_MODEL_ID")  # Uses OpenRouter/Google
```

## Best Practices

1. **Always Define Schemas**: Use Pydantic models for structured output
2. **Handle Errors**: Implement proper error handling and retries
3. **Monitor Usage**: Check your Augment account for usage and costs
4. **Test Both Modes**: Ensure your code works with and without auggie
5. **Use Appropriate Models**: Choose models based on task complexity and cost

## Additional Resources

- **Auggie Documentation**: https://docs.augmentcode.com
- **Auggie CLI Changelog**: https://www.augmentcode.com/changelog/auggie-cli
- **ValueCell Integration Guide**: ../docs/AUGGIE_INTEGRATION.md
- **ValueCell Issues**: https://github.com/ValueCell-ai/valuecell/issues

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the full integration guide: `docs/AUGGIE_INTEGRATION.md`
3. Check auggie logs: `auggie session list`
4. Open an issue on GitHub with details about your setup and error messages

