# Auggie CLI Integration Guide

This guide explains how to use the Auggie CLI tool as an alternative to direct LLM API calls in ValueCell.

## Overview

The Auggie integration allows you to use the `auggie` command-line tool to interact with LLM models without needing direct API keys. This is particularly useful when:

- You don't have API keys for specific LLM providers
- You want to use Augment's authentication and billing
- You want to leverage Auggie's built-in features like caching and rate limiting

## Prerequisites

1. **Install Auggie CLI**: Follow the installation instructions at https://www.augmentcode.com/changelog/auggie-cli

2. **Authenticate with Auggie**:
   ```bash
   auggie login
   ```

3. **Verify Installation**:
   ```bash
   auggie --help
   ```

## Supported Models

Auggie currently supports the following models:

| Model ID | Full Name | Description |
|----------|-----------|-------------|
| `haiku4.5` | Claude Haiku 4.5 | Fast and efficient, best for simple tasks |
| `sonnet4` | Claude Sonnet 4 | Balanced performance and capability |
| `sonnet4.5` | Claude Sonnet 4.5 | Most capable, recommended (default) |
| `gpt5` | GPT-5 | OpenAI's latest model |

### Model Aliases

For backward compatibility, common model names are automatically mapped to auggie's supported models:

```python
# Anthropic aliases
"anthropic/claude-3-5-sonnet-20241022" → "sonnet4.5"
"anthropic/claude-3-5-sonnet" → "sonnet4.5"
"anthropic/claude-haiku-4.5" → "haiku4.5"
"claude-3-5-sonnet" → "sonnet4.5"

# OpenAI aliases
"openai/gpt-5" → "gpt5"
"openai/gpt-4o" → "gpt5"
"gpt-4o" → "gpt5"
"gpt-4o-mini" → "haiku4.5"

# Google aliases (mapped to best available)
"google/gemini-2.5-flash" → "sonnet4.5"
"google/gemini-1.5-pro" → "sonnet4.5"

# DeepSeek aliases (mapped to best available)
"deepseek/deepseek-chat-v3-0324" → "sonnet4.5"
```

**Note**: If you specify an unsupported model, you'll get a clear error message listing available options.

## Configuration

### Enable Auggie Integration

Add the following to your `.env` file:

```bash
# Enable Auggie integration
USE_AUGGIE=true

# Optional: Specify workspace root (defaults to current directory)
WORKSPACE_ROOT=/path/to/your/workspace
```

### Model Configuration

When `USE_AUGGIE=true`, the system will use auggie to call models. You can use either auggie's native model IDs or common aliases (which are automatically mapped):

**Option 1: Use auggie's native model IDs (recommended)**
```bash
PLANNER_MODEL_ID=sonnet4.5
SEC_PARSER_MODEL_ID=haiku4.5
SEC_ANALYSIS_MODEL_ID=sonnet4.5
AI_HEDGE_FUND_PARSER_MODEL_ID=sonnet4.5
RESEARCH_AGENT_MODEL_ID=sonnet4.5
PRODUCT_MODEL_ID=haiku4.5
```

**Option 2: Use familiar model names (automatically mapped)**
```bash
# These will be automatically converted to auggie model IDs
PLANNER_MODEL_ID=google/gemini-2.5-flash        # → sonnet4.5
SEC_PARSER_MODEL_ID=openai/gpt-4o-mini          # → haiku4.5
SEC_ANALYSIS_MODEL_ID=deepseek/deepseek-chat    # → sonnet4.5
AI_HEDGE_FUND_PARSER_MODEL_ID=anthropic/claude-3-5-sonnet  # → sonnet4.5
RESEARCH_AGENT_MODEL_ID=google/gemini-2.5-flash # → sonnet4.5
PRODUCT_MODEL_ID=anthropic/claude-haiku-4.5     # → haiku4.5
```

## Architecture

### Components

1. **AuggieClient** (`python/valuecell/utils/auggie_client.py`)
   - Core client for interacting with auggie CLI
   - Handles command execution and response parsing
   - Supports both sync and async operations

2. **Adapters** (`python/valuecell/utils/auggie_adapter.py`)
   - `AuggieLangChainAdapter`: Compatible with LangChain's ChatModel interface
   - `AuggieAgnoAdapter`: Compatible with agno's Model interface
   - `AuggieOpenAIAdapter`: Compatible with OpenAI client interface

3. **Integration Points**
   - `valuecell/utils/model.py`: Main model factory with auggie support
   - `third_party/ai-hedge-fund/src/utils/auggie_llm.py`: Auggie support for ai-hedge-fund

### How It Works

```
┌─────────────────┐
│  Your Code      │
│  (Agent/Tool)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Factory  │
│  get_model()    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌──────────────┐
│ Direct │  │   Auggie     │
│  API   │  │   Adapter    │
└────────┘  └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │  auggie CLI  │
            │  subprocess  │
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │  LLM Model   │
            └──────────────┘
```

## Usage Examples

### Basic Usage

```python
from valuecell.utils.model import get_model

# This will automatically use auggie if USE_AUGGIE=true
model = get_model("RESEARCH_AGENT_MODEL_ID")

# Use the model as normal
response = model.response([
    {"role": "user", "content": "Analyze this stock..."}
])
```

### Direct Auggie Client Usage

```python
from valuecell.utils.auggie_client import AuggieClient
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    sentiment: str
    confidence: float
    reasoning: str

# Create client
client = AuggieClient(
    model="google/gemini-2.5-flash",
    workspace_root="/path/to/workspace"
)

# Get structured output
result = client.invoke(
    prompt="Analyze the sentiment of this text: ...",
    output_schema=AnalysisResult
)

print(f"Sentiment: {result.sentiment}")
print(f"Confidence: {result.confidence}")
```

### Using Adapters

```python
from valuecell.utils.auggie_adapter import get_auggie_model

# LangChain-style adapter
langchain_model = get_auggie_model(
    model_name="google/gemini-2.5-flash",
    adapter_type="langchain"
)

# Use with structured output
from pydantic import BaseModel

class Response(BaseModel):
    answer: str
    confidence: float

structured_model = langchain_model.with_structured_output(Response)
result = structured_model.invoke("What is the capital of France?")

# agno-style adapter
agno_model = get_auggie_model(
    model_name="google/gemini-2.5-flash",
    adapter_type="agno"
)

response = agno_model.response([
    {"role": "user", "content": "Hello!"}
])
```

### Async Usage

```python
import asyncio
from valuecell.utils.auggie_client import AuggieClient

async def analyze():
    client = AuggieClient(model="google/gemini-2.5-flash")
    
    result = await client.ainvoke(
        prompt="Analyze this data...",
        output_schema=AnalysisResult
    )
    
    return result

# Run async
result = asyncio.run(analyze())
```

## Migration Guide

### Migrating Existing Code

1. **For valuecell agents**: Simply set `USE_AUGGIE=true` in your `.env` file. No code changes needed.

2. **For ai-hedge-fund**: Update imports to use the auggie-aware version:

   ```python
   # Before
   from src.utils.llm import call_llm
   
   # After (automatic switching based on USE_AUGGIE)
   from src.utils.auggie_llm import call_llm_auto as call_llm
   ```

3. **For custom code**: Use the adapter pattern:

   ```python
   # Before
   from langchain_openai import ChatOpenAI
   model = ChatOpenAI(model="gpt-4", api_key=api_key)
   
   # After
   import os
   if os.getenv("USE_AUGGIE") == "true":
       from valuecell.utils.auggie_adapter import AuggieLangChainAdapter
       model = AuggieLangChainAdapter(model="openai/gpt-4")
   else:
       from langchain_openai import ChatOpenAI
       model = ChatOpenAI(model="gpt-4", api_key=api_key)
   ```

## Troubleshooting

### Common Issues

1. **"auggie: command not found"**
   - Make sure auggie is installed and in your PATH
   - Run `auggie --help` to verify installation

2. **"Auggie command failed with code 1"**
   - Check if you're authenticated: `auggie login`
   - Verify the model ID is correct
   - Check auggie logs for more details

3. **"Failed to parse JSON response"**
   - The model might not be following the schema
   - Try a different model or adjust your prompt
   - Check the raw response in error messages

4. **Timeout errors**
   - Increase the timeout parameter: `client.invoke(prompt, timeout=600)`
   - Check if the model is responding slowly
   - Consider using a faster model

### Debug Mode

Enable debug output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your code here
```

### Checking Auggie Status

```bash
# Check auggie version
auggie --version

# List available models
auggie model list

# Check account status
auggie account
```

## Performance Considerations

1. **Latency**: Auggie adds subprocess overhead (~100-200ms). For high-throughput scenarios, consider batching requests.

2. **Caching**: Auggie has built-in caching. Identical requests may return faster.

3. **Rate Limiting**: Auggie handles rate limiting automatically.

4. **Parallel Requests**: Use async methods for parallel requests:

   ```python
   import asyncio
   
   async def process_batch(prompts):
       client = AuggieClient()
       tasks = [client.ainvoke(p) for p in prompts]
       return await asyncio.gather(*tasks)
   ```

## Advanced Configuration

### Custom Auggie Options

```python
from valuecell.utils.auggie_client import AuggieClient

client = AuggieClient(
    model="google/gemini-2.5-flash",
    workspace_root="/custom/path",
    max_turns=5,  # Allow multiple agentic turns
    quiet=False,  # Show intermediate steps
)
```

### Using with Images

```python
client = AuggieClient(model="google/gemini-2.5-flash")

result = client.invoke(
    prompt="Describe this image",
    image_paths=["/path/to/image.png"]
)
```

## Best Practices

1. **Use structured output**: Always define Pydantic models for predictable responses
2. **Handle errors gracefully**: Implement retry logic and fallbacks
3. **Set appropriate timeouts**: Adjust based on expected response time
4. **Monitor usage**: Check auggie account for usage and costs
5. **Test both modes**: Ensure your code works with and without auggie

## Support

- Auggie Documentation: https://docs.augmentcode.com
- Auggie CLI Changelog: https://www.augmentcode.com/changelog/auggie-cli
- ValueCell Issues: https://github.com/ValueCell-ai/valuecell/issues

