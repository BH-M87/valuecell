# Auggie Model Validation - Update Summary

## Overview

This document describes the model validation and normalization feature added to the Auggie CLI integration. This ensures that only supported auggie models are used and provides automatic mapping of common model names.

## Supported Models

Auggie currently supports **4 models**:

| Model ID | Full Name | Description | Use Case |
|----------|-----------|-------------|----------|
| `haiku4.5` | Claude Haiku 4.5 | Fast and efficient | Simple tasks, quick responses |
| `sonnet4` | Claude Sonnet 4 | Balanced | General purpose |
| `sonnet4.5` | Claude Sonnet 4.5 | Most capable (default) | Complex analysis, research |
| `gpt5` | GPT-5 | OpenAI's latest | Advanced reasoning |

## Model Aliases

For backward compatibility and ease of use, common model names are automatically mapped to auggie's supported models:

### Anthropic Aliases
```
anthropic/claude-3-5-sonnet-20241022 → sonnet4.5
anthropic/claude-3-5-sonnet → sonnet4.5
anthropic/claude-haiku-4.5 → haiku4.5
anthropic/claude-3-5-haiku → haiku4.5
claude-3-5-sonnet → sonnet4.5
claude-haiku-4.5 → haiku4.5
```

### OpenAI Aliases
```
openai/gpt-5 → gpt5
openai/gpt-4o → gpt5
openai/gpt-4o-mini → haiku4.5
gpt-5 → gpt5
gpt-4o → gpt5
gpt-4o-mini → haiku4.5
gpt-4 → gpt5
```

### Google Aliases
```
google/gemini-2.5-flash → sonnet4.5
google/gemini-1.5-pro → sonnet4.5
gemini-2.5-flash → sonnet4.5
gemini-1.5-pro → sonnet4.5
```

### DeepSeek Aliases
```
deepseek/deepseek-chat-v3-0324 → sonnet4.5
deepseek-chat → sonnet4.5
```

## Features

### 1. Automatic Model Normalization

When you create an `AuggieClient`, model names are automatically validated and normalized:

```python
from valuecell.utils.auggie_client import AuggieClient

# All of these work and map to the same model:
client1 = AuggieClient(model="sonnet4.5")
client2 = AuggieClient(model="anthropic/claude-3-5-sonnet")
client3 = AuggieClient(model="google/gemini-2.5-flash")

# All will have client.model == "sonnet4.5"
```

### 2. Clear Error Messages

If you specify an unsupported model, you get a helpful error:

```python
try:
    client = AuggieClient(model="unsupported-model")
except ValueError as e:
    print(e)
    # Output:
    # Model 'unsupported-model' is not supported by auggie.
    # Supported models: haiku4.5, sonnet4, sonnet4.5, gpt5.
    # Available models:
    #   - Claude Haiku 4.5 [haiku4.5]
    #   - Claude Sonnet 4 [sonnet4]
    #   - Claude Sonnet 4.5 [sonnet4.5]
    #   - GPT-5 [gpt5]
```

### 3. Utility Functions

Three new utility functions are available:

```python
from valuecell.utils.auggie_client import (
    normalize_model_name,
    get_available_models,
    list_available_models,
)

# Normalize a model name
normalized = normalize_model_name("gpt-4o")  # Returns: "gpt5"

# Get available models as dict
models = get_available_models()
# Returns: {"haiku4.5": "Claude Haiku 4.5", ...}

# Get formatted list of models
print(list_available_models())
# Prints:
# Available auggie models:
#   - Claude Haiku 4.5 [haiku4.5]
#   - Claude Sonnet 4 [sonnet4]
#   - Claude Sonnet 4.5 [sonnet4.5]
#   - GPT-5 [gpt5]
```

### 4. Case-Insensitive Matching

Model names are case-insensitive:

```python
normalize_model_name("SONNET4.5")  # → "sonnet4.5"
normalize_model_name("Haiku4.5")   # → "haiku4.5"
normalize_model_name("GPT5")       # → "gpt5"
```

### 5. Default Model

If no model is specified, `sonnet4.5` is used as the default:

```python
client = AuggieClient()  # Uses sonnet4.5
client = AuggieClient(model=None)  # Uses sonnet4.5
```

### 6. Optional Validation

You can disable validation if needed:

```python
# Skip validation (not recommended)
client = AuggieClient(model="any-model", validate_model=False)
```

## Configuration Examples

### Recommended Configuration (Native Model IDs)

```bash
# .env
USE_AUGGIE=true

# Use auggie's native model IDs
PLANNER_MODEL_ID=sonnet4.5
RESEARCH_AGENT_MODEL_ID=sonnet4.5
SEC_PARSER_MODEL_ID=haiku4.5
AI_HEDGE_FUND_PARSER_MODEL_ID=sonnet4.5
SEC_ANALYSIS_MODEL_ID=sonnet4.5
PRODUCT_MODEL_ID=haiku4.5
```

### Alternative Configuration (Using Aliases)

```bash
# .env
USE_AUGGIE=true

# Use familiar model names (automatically mapped)
PLANNER_MODEL_ID=anthropic/claude-3-5-sonnet
RESEARCH_AGENT_MODEL_ID=google/gemini-2.5-flash
SEC_PARSER_MODEL_ID=gpt-4o-mini
AI_HEDGE_FUND_PARSER_MODEL_ID=anthropic/claude-3-5-sonnet
SEC_ANALYSIS_MODEL_ID=deepseek/deepseek-chat-v3-0324
PRODUCT_MODEL_ID=anthropic/claude-haiku-4.5
```

Both configurations work identically - aliases are automatically mapped to auggie's native model IDs.

## Migration Guide

### If You're Already Using Auggie Integration

No changes needed! Your existing configuration will continue to work. Model names will be automatically normalized.

### If You're Using Unsupported Models

If your configuration uses models that aren't supported by auggie, you'll need to update them:

1. **Check your current configuration**:
   ```bash
   grep "MODEL_ID" .env
   ```

2. **Map to supported models**:
   - For fast/cheap models → use `haiku4.5`
   - For balanced models → use `sonnet4` or `sonnet4.5`
   - For most capable models → use `sonnet4.5` or `gpt5`

3. **Update your `.env` file** with the new model IDs

4. **Test the configuration**:
   ```bash
   python scripts/test_auggie.py
   ```

## Testing

### Run Model Validation Tests

```bash
# Run all model validation tests
pytest python/valuecell/tests/test_auggie_model_validation.py -v

# Run specific test
pytest python/valuecell/tests/test_auggie_model_validation.py::TestModelNormalization::test_model_aliases -v
```

### Test in Your Code

```python
from valuecell.utils.auggie_client import normalize_model_name

# Test your model names
test_models = [
    "your-model-1",
    "your-model-2",
]

for model in test_models:
    try:
        normalized = normalize_model_name(model)
        print(f"✓ {model} → {normalized}")
    except ValueError as e:
        print(f"✗ {model} - {e}")
```

### Run Example Script

```bash
python examples/auggie_example.py
```

This will run Example 0 which demonstrates model validation.

## Benefits

1. **Prevents Runtime Errors**: Catch unsupported models at initialization time
2. **Automatic Mapping**: Use familiar model names without worrying about auggie's format
3. **Clear Feedback**: Helpful error messages when something goes wrong
4. **Backward Compatible**: Existing configurations continue to work
5. **Type Safe**: Ensures only valid models are used
6. **Easy Migration**: Smooth transition from other LLM providers

## Implementation Details

### Files Modified

1. **python/valuecell/utils/auggie_client.py**
   - Added `AUGGIE_SUPPORTED_MODELS` constant
   - Added `AUGGIE_MODEL_ALIASES` constant
   - Added `normalize_model_name()` function
   - Added `get_available_models()` function
   - Added `list_available_models()` function
   - Updated `AuggieClient.__init__()` to validate models

2. **Documentation**
   - Updated `AUGGIE_QUICKSTART.md` with model information
   - Updated `docs/AUGGIE_INTEGRATION.md` with detailed model section
   - Updated `examples/README_AUGGIE.md` with model examples

3. **Tests**
   - Added `test_auggie_model_validation.py` with comprehensive tests

4. **Examples**
   - Added Example 0 to `auggie_example.py` demonstrating validation

### Code Structure

```python
# Constants
AUGGIE_SUPPORTED_MODELS = {
    "haiku4.5": "Claude Haiku 4.5",
    "sonnet4": "Claude Sonnet 4",
    "sonnet4.5": "Claude Sonnet 4.5",
    "gpt5": "GPT-5",
}

AUGGIE_MODEL_ALIASES = {
    # Mapping of common names to auggie model IDs
    ...
}

# Functions
def normalize_model_name(model: Optional[str]) -> str:
    """Validate and normalize model name."""
    ...

def get_available_models() -> Dict[str, str]:
    """Get dict of available models."""
    ...

def list_available_models() -> str:
    """Get formatted string of models."""
    ...

# Updated class
class AuggieClient:
    def __init__(self, model=None, validate_model=True, ...):
        if validate_model:
            self.model = normalize_model_name(model)
        ...
```

## Future Enhancements

1. **Dynamic Model Discovery**: Query auggie for available models at runtime
2. **Model Capabilities**: Add metadata about model capabilities (context length, etc.)
3. **Cost Information**: Include pricing information for each model
4. **Performance Metrics**: Track and report model performance
5. **Smart Model Selection**: Automatically choose best model based on task

## Support

If you encounter issues with model validation:

1. Check available models: `python -c "from valuecell.utils.auggie_client import list_available_models; print(list_available_models())"`
2. Test your model name: `python -c "from valuecell.utils.auggie_client import normalize_model_name; print(normalize_model_name('your-model'))"`
3. Run validation tests: `pytest python/valuecell/tests/test_auggie_model_validation.py -v`
4. Check documentation: `docs/AUGGIE_INTEGRATION.md`

## Summary

The model validation feature ensures that only supported auggie models are used while maintaining backward compatibility through automatic model name mapping. This prevents runtime errors and provides a better user experience with clear error messages and helpful utilities.

**Key Points**:
- 4 supported models: `haiku4.5`, `sonnet4`, `sonnet4.5`, `gpt5`
- Automatic mapping of 20+ common model names
- Clear error messages for unsupported models
- Backward compatible with existing configurations
- Comprehensive test coverage

