# Auggie CLI Integration - Implementation Summary

## Overview

This document summarizes the implementation of Auggie CLI integration in ValueCell, which allows the system to use LLM models through the `auggie` command-line tool instead of requiring direct API keys.

## Motivation

The integration addresses several key needs:

1. **No API Keys Required**: Users can leverage Auggie's authentication instead of managing multiple API keys
2. **Unified Access**: Access to multiple LLM providers (OpenAI, Anthropic, Google, DeepSeek, etc.) through a single interface
3. **Built-in Features**: Automatic caching, rate limiting, and retry logic provided by Auggie
4. **Easy Migration**: Minimal code changes required to switch between direct API calls and Auggie

## Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (Agents, Tools, Services using LLM models)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Factory Layer                        │
│  - valuecell/utils/model.py (get_model)                    │
│  - ai-hedge-fund/src/utils/auggie_llm.py                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                    ┌────┴────┐
                    │         │
                    ▼         ▼
        ┌──────────────┐  ┌──────────────────┐
        │  Direct API  │  │  Auggie Layer    │
        │  (Original)  │  │  (New)           │
        └──────────────┘  └────────┬─────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
            ┌──────────┐  ┌──────────┐  ┌──────────┐
            │LangChain │  │   agno   │  │  OpenAI  │
            │ Adapter  │  │ Adapter  │  │ Adapter  │
            └────┬─────┘  └────┬─────┘  └────┬─────┘
                 │             │             │
                 └─────────────┼─────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  AuggieClient    │
                    │  (Core Client)   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  auggie CLI      │
                    │  (subprocess)    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   LLM Models     │
                    └──────────────────┘
```

## Implementation Details

### 1. Core Client (`python/valuecell/utils/auggie_client.py`)

**Purpose**: Provides the core functionality to interact with auggie CLI.

**Key Features**:
- Executes auggie commands via subprocess
- Supports both sync (`invoke`) and async (`ainvoke`) operations
- Handles structured output with Pydantic models
- Parses JSON responses from auggie
- Configurable timeout, model selection, and workspace root

**Main Class**: `AuggieClient`

**Key Methods**:
- `invoke(prompt, output_schema, ...)`: Synchronous LLM call
- `ainvoke(prompt, output_schema, ...)`: Asynchronous LLM call
- `_build_command(...)`: Constructs auggie CLI command
- `_extract_json_from_text(...)`: Extracts JSON from markdown responses

### 2. Adapters (`python/valuecell/utils/auggie_adapter.py`)

**Purpose**: Provides compatibility layers for different LLM interfaces.

**Adapters Implemented**:

1. **AuggieLangChainAdapter**
   - Compatible with LangChain's ChatModel interface
   - Supports `with_structured_output()` method
   - Methods: `invoke()`, `ainvoke()`

2. **AuggieAgnoAdapter**
   - Compatible with agno's Model interface
   - Methods: `response()`, `aresponse()`
   - Returns responses in agno's expected format

3. **AuggieOpenAIAdapter**
   - Compatible with OpenAI client interface
   - Supports OpenAI Responses API style
   - Method: `create()`

**Factory Function**: `get_auggie_model(model_name, adapter_type, ...)`

### 3. Integration Points

#### A. ValueCell Model Factory (`python/valuecell/utils/model.py`)

**Changes**:
- Added check for `USE_AUGGIE` environment variable
- Returns `AuggieAgnoAdapter` when auggie is enabled
- Falls back to original behavior (Gemini/OpenRouter) when disabled

**Usage**:
```python
# Automatically uses auggie if USE_AUGGIE=true
model = get_model("RESEARCH_AGENT_MODEL_ID")
```

#### B. AI-Hedge-Fund Integration (`python/third_party/ai-hedge-fund/src/utils/auggie_llm.py`)

**New Functions**:
- `call_llm_with_auggie()`: Auggie-powered LLM call
- `call_llm_auto()`: Automatically chooses between auggie and standard LLM

**Usage**:
```python
from src.utils.auggie_llm import call_llm_auto as call_llm

result = call_llm(prompt, pydantic_model, agent_name, state)
```

### 4. Configuration

#### Environment Variables (`.env.example`)

New variables:
```bash
# Enable Auggie integration
USE_AUGGIE=false

# Optional: Set workspace root
WORKSPACE_ROOT=
```

Existing variables (used with auggie):
```bash
# Model IDs - used by auggie when USE_AUGGIE=true
PLANNER_MODEL_ID=google/gemini-2.5-flash
SEC_PARSER_MODEL_ID=openai/gpt-4o-mini
RESEARCH_AGENT_MODEL_ID=google/gemini-2.5-flash
# ... etc
```

### 5. Documentation

#### Main Documentation (`docs/AUGGIE_INTEGRATION.md`)

Comprehensive guide covering:
- Overview and prerequisites
- Configuration instructions
- Architecture explanation
- Usage examples
- Migration guide
- Troubleshooting
- Performance considerations
- Best practices

#### Examples (`examples/`)

- `auggie_example.py`: Runnable examples demonstrating all features
- `README_AUGGIE.md`: Quick start guide for examples

### 6. Testing (`python/valuecell/tests/test_auggie_integration.py`)

**Test Coverage**:
- AuggieClient creation and basic operations
- Structured output with Pydantic models
- Async operations
- All three adapters (LangChain, agno, OpenAI)
- Adapter factory function
- Integration with model factory

**Test Strategy**:
- Tests are skipped if auggie is not available
- Uses `@skip_if_no_auggie` decorator
- Tests both sync and async operations

## Files Created/Modified

### New Files

1. `python/valuecell/utils/auggie_client.py` - Core auggie client
2. `python/valuecell/utils/auggie_adapter.py` - Compatibility adapters
3. `python/third_party/ai-hedge-fund/src/utils/auggie_llm.py` - AI-hedge-fund integration
4. `docs/AUGGIE_INTEGRATION.md` - Comprehensive documentation
5. `examples/auggie_example.py` - Example script
6. `examples/README_AUGGIE.md` - Examples documentation
7. `python/valuecell/tests/test_auggie_integration.py` - Test suite
8. `AUGGIE_INTEGRATION_SUMMARY.md` - This file

### Modified Files

1. `python/valuecell/utils/model.py` - Added auggie support
2. `.env.example` - Added auggie configuration

## Usage Patterns

### Pattern 1: Transparent Integration (Recommended)

```python
import os
os.environ["USE_AUGGIE"] = "true"

from valuecell.utils.model import get_model

# Automatically uses auggie
model = get_model("RESEARCH_AGENT_MODEL_ID")
result = model.response([{"role": "user", "content": "..."}])
```

### Pattern 2: Direct Client Usage

```python
from valuecell.utils.auggie_client import AuggieClient
from pydantic import BaseModel

class Response(BaseModel):
    answer: str
    confidence: float

client = AuggieClient(model="google/gemini-2.5-flash")
result = client.invoke(prompt="...", output_schema=Response)
```

### Pattern 3: Adapter Usage

```python
from valuecell.utils.auggie_adapter import get_auggie_model

# LangChain style
adapter = get_auggie_model(
    model_name="google/gemini-2.5-flash",
    adapter_type="langchain"
)
result = adapter.invoke("...")

# agno style
adapter = get_auggie_model(
    model_name="google/gemini-2.5-flash",
    adapter_type="agno"
)
result = adapter.response([{"role": "user", "content": "..."}])
```

## Migration Guide

### For Existing Code

1. **No changes required** for code using `valuecell.utils.model.get_model()`
   - Just set `USE_AUGGIE=true` in `.env`

2. **For ai-hedge-fund code**:
   ```python
   # Change import
   from src.utils.auggie_llm import call_llm_auto as call_llm
   ```

3. **For custom code**:
   - Use adapter pattern or direct client
   - See examples in `examples/auggie_example.py`

## Benefits

1. **No API Key Management**: Single authentication with Auggie
2. **Multi-Provider Access**: Access to OpenAI, Anthropic, Google, DeepSeek, etc.
3. **Built-in Features**: Caching, rate limiting, retry logic
4. **Easy Toggle**: Switch between auggie and direct API with one env var
5. **Backward Compatible**: Existing code continues to work
6. **Type Safe**: Full Pydantic support for structured output
7. **Async Support**: Efficient concurrent requests

## Limitations

1. **Subprocess Overhead**: ~100-200ms latency from subprocess execution
2. **Requires Auggie**: Must have auggie CLI installed and authenticated
3. **Limited Streaming**: Current implementation doesn't support streaming responses
4. **Command-line Only**: Requires shell access to run auggie commands

## Future Enhancements

1. **Streaming Support**: Add support for streaming responses
2. **Batch Processing**: Optimize for batch requests
3. **Caching Layer**: Add application-level caching
4. **Monitoring**: Add usage tracking and metrics
5. **Error Recovery**: Enhanced error handling and retry strategies
6. **Tool Support**: Better integration with auggie's tool calling features

## Testing

### Running Tests

```bash
# Run all auggie tests
pytest python/valuecell/tests/test_auggie_integration.py -v

# Run specific test
pytest python/valuecell/tests/test_auggie_integration.py::TestAuggieClient::test_simple_invoke -v

# Run with coverage
pytest python/valuecell/tests/test_auggie_integration.py --cov=valuecell.utils
```

### Running Examples

```bash
# Run all examples
python examples/auggie_example.py

# Or run individual examples by modifying the script
```

## Troubleshooting

### Common Issues

1. **"auggie: command not found"**
   - Install auggie: https://www.augmentcode.com/changelog/auggie-cli
   - Verify: `which auggie`

2. **"Authentication required"**
   - Run: `auggie login`

3. **"Model not found"**
   - Check available models: `auggie model list`
   - Verify model ID format

4. **Timeout errors**
   - Increase timeout: `client.invoke(prompt, timeout=600)`

5. **JSON parsing errors**
   - Use more capable model
   - Adjust prompt for clarity
   - Check raw response in error messages

## Conclusion

The Auggie CLI integration provides a flexible, powerful alternative to direct API calls while maintaining full backward compatibility. Users can easily switch between auggie and direct API calls based on their needs, and the implementation is designed to be extensible for future enhancements.

## References

- Auggie CLI Documentation: https://docs.augmentcode.com
- Auggie CLI Changelog: https://www.augmentcode.com/changelog/auggie-cli
- ValueCell Repository: https://github.com/ValueCell-ai/valuecell

