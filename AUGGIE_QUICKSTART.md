# Auggie Integration - Quick Start Guide

## What is This?

This integration allows you to use LLM models in ValueCell through the `auggie` CLI tool, **without needing API keys** from OpenAI, Anthropic, Google, or other providers.

## 3-Step Setup

### Step 1: Install Auggie

**macOS/Linux:**
```bash
curl -fsSL https://install.augmentcode.com | sh
```

**Windows (PowerShell):**
```powershell
irm https://install.augmentcode.com/windows | iex
```

Or follow instructions at: https://www.augmentcode.com/changelog/auggie-cli

### Step 2: Authenticate

```bash
auggie login
```

This opens a browser for authentication. Follow the prompts.

### Step 3: Enable in ValueCell

Edit your `.env` file:

```bash
# Add this line
USE_AUGGIE=true
```

That's it! 🎉

## Verify It Works

Run the example script:

```bash
python examples/auggie_example.py
```

Or test in Python:

```python
import os
os.environ["USE_AUGGIE"] = "true"

from valuecell.utils.model import get_model

model = get_model("RESEARCH_AGENT_MODEL_ID")
result = model.response([
    {"role": "user", "content": "What is a stock market index?"}
])

print(result["content"])
```

## What Changed?

**Before (requires API keys):**
```bash
# .env
OPENROUTER_API_KEY=sk-or-v1-xxxxx
GOOGLE_API_KEY=AIzaSyxxxxx
```

**After (no API keys needed):**
```bash
# .env
USE_AUGGIE=true
# No API keys required!
```

## How It Works

```
Your Code
    ↓
Model Factory (detects USE_AUGGIE=true)
    ↓
Auggie Adapter
    ↓
auggie CLI (subprocess)
    ↓
LLM Model (OpenAI, Anthropic, Google, etc.)
```

## Available Models

Auggie supports the following models:

- **Claude Haiku 4.5** - `haiku4.5` - Fast and efficient
- **Claude Sonnet 4** - `sonnet4` - Balanced performance
- **Claude Sonnet 4.5** - `sonnet4.5` - Most capable (default)
- **GPT-5** - `gpt5` - OpenAI's latest

### Model Aliases

For backward compatibility, the integration automatically maps common model names:

```python
# These all map to auggie's supported models:
"anthropic/claude-3-5-sonnet" → "sonnet4.5"
"anthropic/claude-haiku-4.5" → "haiku4.5"
"openai/gpt-5" → "gpt5"
"google/gemini-2.5-flash" → "sonnet4.5"
"gpt-4o" → "gpt5"
"gpt-4o-mini" → "haiku4.5"
```

Check available models:
```bash
auggie model list
```

## Configuration

Your existing model configuration works with auggie (model names are automatically mapped):

```bash
# .env
USE_AUGGIE=true

# These model IDs will be automatically mapped to auggie models:
PLANNER_MODEL_ID=sonnet4.5              # Or use: google/gemini-2.5-flash
RESEARCH_AGENT_MODEL_ID=sonnet4.5       # Or use: anthropic/claude-3-5-sonnet
SEC_PARSER_MODEL_ID=haiku4.5            # Or use: openai/gpt-4o-mini
```

**Recommended Configuration:**
```bash
# Use auggie's native model IDs for best performance
PLANNER_MODEL_ID=sonnet4.5
RESEARCH_AGENT_MODEL_ID=sonnet4.5
SEC_PARSER_MODEL_ID=haiku4.5
AI_HEDGE_FUND_PARSER_MODEL_ID=sonnet4.5
PRODUCT_MODEL_ID=haiku4.5
```

## Switching Back

To use direct API calls again:

```bash
# .env
USE_AUGGIE=false

# Now you need API keys again
OPENROUTER_API_KEY=sk-or-v1-xxxxx
```

## Common Issues

### "auggie: command not found"

Auggie is not installed or not in PATH.

**Fix:**
```bash
# Check if installed
which auggie

# If not found, install it (see Step 1)
```

### "Authentication required"

Not logged in to auggie.

**Fix:**
```bash
auggie login
```

### "Model not found"

Invalid model ID.

**Fix:**
```bash
# List available models
auggie model list

# Use correct format: provider/model-name
# Example: google/gemini-2.5-flash
```

## Examples

### Basic Usage

```python
from valuecell.utils.auggie_client import AuggieClient

client = AuggieClient(model="google/gemini-2.5-flash")
response = client.invoke("Explain what a P/E ratio is.")
print(response)
```

### Structured Output

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float

client = AuggieClient(model="google/gemini-2.5-flash")
result = client.invoke(
    prompt="Analyze AAPL stock sentiment",
    output_schema=Analysis
)
print(f"Sentiment: {result.sentiment}, Confidence: {result.confidence}")
```

### Async Usage

```python
import asyncio

async def analyze():
    client = AuggieClient(model="google/gemini-2.5-flash")
    result = await client.ainvoke("What is diversification?")
    return result

result = asyncio.run(analyze())
```

## Benefits

✅ **No API Key Management** - Single authentication with Auggie  
✅ **Multi-Provider Access** - OpenAI, Anthropic, Google, DeepSeek, etc.  
✅ **Built-in Features** - Caching, rate limiting, retry logic  
✅ **Easy Toggle** - Switch with one environment variable  
✅ **Backward Compatible** - Existing code works without changes  

## Next Steps

1. **Read Full Documentation**: `docs/AUGGIE_INTEGRATION.md`
2. **Run Examples**: `python examples/auggie_example.py`
3. **Run Tests**: `pytest python/valuecell/tests/test_auggie_integration.py`
4. **Check Usage**: `auggie account`

## Need Help?

- **Auggie Docs**: https://docs.augmentcode.com
- **Auggie CLI**: https://www.augmentcode.com/changelog/auggie-cli
- **ValueCell Issues**: https://github.com/ValueCell-ai/valuecell/issues

## Summary

```bash
# 1. Install auggie
curl -fsSL https://install.augmentcode.com | sh

# 2. Login
auggie login

# 3. Enable in .env
echo "USE_AUGGIE=true" >> .env

# 4. Run your code - it just works!
python your_script.py
```

That's all you need! 🚀

