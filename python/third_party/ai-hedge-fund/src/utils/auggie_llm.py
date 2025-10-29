"""
Auggie-based LLM utility functions for ai-hedge-fund.

This module provides auggie-powered alternatives to the standard LLM functions,
allowing the system to work without direct API keys.
"""

import os
import json
from pydantic import BaseModel
from typing import Optional

# Import the auggie client from valuecell utils
import sys
from pathlib import Path

# Add valuecell to path if not already there
valuecell_path = Path(__file__).parent.parent.parent.parent.parent / "valuecell"
if str(valuecell_path) not in sys.path:
    sys.path.insert(0, str(valuecell_path))

from valuecell.utils.auggie_client import AuggieClient
from src.utils.progress import progress
from src.graph.state import AgentState


def call_llm_with_auggie(
    prompt: any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: AgentState | None = None,
    max_retries: int = 3,
    default_factory=None,
    model_name: str | None = None,
) -> BaseModel:
    """
    Makes an LLM call using auggie CLI with retry logic.

    Args:
        prompt: The prompt to send to the LLM
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates
        state: Optional state object (for compatibility)
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure
        model_name: Optional model name to use (overrides environment)

    Returns:
        An instance of the specified Pydantic model
    """
    # Determine model to use
    if not model_name:
        # Try to get from environment or use default
        model_name = os.getenv("AI_HEDGE_FUND_PARSER_MODEL_ID", "google/gemini-2.5-flash")

    # Create auggie client
    client = AuggieClient(
        model=model_name,
        workspace_root=os.getenv("WORKSPACE_ROOT", os.getcwd()),
        max_turns=1,
        quiet=True,
    )

    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            if agent_name:
                progress.update_status(agent_name, None, f"Calling LLM (attempt {attempt + 1}/{max_retries})")

            # Call auggie
            result = client.invoke(
                prompt=prompt,
                output_schema=pydantic_model,
                timeout=300,
            )

            if agent_name:
                progress.update_status(agent_name, None, "LLM call successful")

            return result

        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")

            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)


def create_default_response(model_class: type[BaseModel]) -> BaseModel:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def get_agent_model_config(state, agent_name):
    """
    Get model configuration for a specific agent from the state.
    Falls back to global model configuration if agent-specific config is not available.
    Always returns valid model_name and model_provider values.
    """
    request = state.get("metadata", {}).get("request")
    
    if request and hasattr(request, 'get_agent_model_config'):
        # Get agent-specific model configuration
        model_name, model_provider = request.get_agent_model_config(agent_name)
        # Ensure we have valid values
        if model_name and model_provider:
            return model_name, model_provider.value if hasattr(model_provider, 'value') else str(model_provider)
    
    # Fall back to global configuration (system defaults)
    model_name = state.get("metadata", {}).get("model_name") or "google/gemini-2.5-flash"
    model_provider = state.get("metadata", {}).get("model_provider") or "OPENROUTER"
    
    # Convert enum to string if necessary
    if hasattr(model_provider, 'value'):
        model_provider = model_provider.value
    
    return model_name, model_provider


# Wrapper function to choose between auggie and standard LLM
def call_llm_auto(
    prompt: any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: AgentState | None = None,
    max_retries: int = 3,
    default_factory=None,
) -> BaseModel:
    """
    Automatically choose between auggie and standard LLM based on environment.
    
    If USE_AUGGIE=true is set, uses auggie. Otherwise, uses standard LLM.
    
    Args:
        prompt: The prompt to send to the LLM
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates
        state: Optional state object
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure
        
    Returns:
        An instance of the specified Pydantic model
    """
    use_auggie = os.getenv("USE_AUGGIE", "false").lower() == "true"
    
    if use_auggie:
        # Get model name from state if available
        model_name = None
        if state and agent_name:
            model_name, _ = get_agent_model_config(state, agent_name)
        
        return call_llm_with_auggie(
            prompt=prompt,
            pydantic_model=pydantic_model,
            agent_name=agent_name,
            state=state,
            max_retries=max_retries,
            default_factory=default_factory,
            model_name=model_name,
        )
    else:
        # Use standard LLM
        from src.utils.llm import call_llm
        return call_llm(
            prompt=prompt,
            pydantic_model=pydantic_model,
            agent_name=agent_name,
            state=state,
            max_retries=max_retries,
            default_factory=default_factory,
        )

