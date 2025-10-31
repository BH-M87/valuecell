"""
Auggie CLI client for LLM interactions.

This module provides a unified interface to call LLM models through the auggie CLI tool,
eliminating the need for direct API keys while maintaining compatibility with existing code.
"""

import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from pathlib import Path
import asyncio
from pydantic import BaseModel


# Supported Auggie models
AUGGIE_SUPPORTED_MODELS = {
    "haiku4.5": "Claude Haiku 4.5",
    "sonnet4": "Claude Sonnet 4",
    "sonnet4.5": "Claude Sonnet 4.5",
    "gpt5": "GPT-5",
}

# Model aliases for backward compatibility
AUGGIE_MODEL_ALIASES = {
    # Anthropic aliases
    "anthropic/claude-haiku-4.5": "haiku4.5",
    "anthropic/claude-3-5-haiku": "haiku4.5",
    "claude-haiku-4.5": "haiku4.5",
    "claude-3-5-haiku": "haiku4.5",

    "anthropic/claude-sonnet-4": "sonnet4",
    "anthropic/claude-4-sonnet": "sonnet4",
    "claude-sonnet-4": "sonnet4",
    "claude-4-sonnet": "sonnet4",

    "anthropic/claude-sonnet-4.5": "sonnet4.5",
    "anthropic/claude-3-5-sonnet-20241022": "sonnet4.5",
    "anthropic/claude-3-5-sonnet": "sonnet4.5",
    "claude-sonnet-4.5": "sonnet4.5",
    "claude-3-5-sonnet": "sonnet4.5",

    # OpenAI aliases
    "openai/gpt-5": "gpt5",
    "gpt-5": "gpt5",
    "openai/gpt5": "gpt5",

    # Google aliases (map to best available)
    "google/gemini-2.5-flash": "sonnet4.5",
    "google/gemini-1.5-pro": "sonnet4.5",
    "gemini-2.5-flash": "sonnet4.5",
    "gemini-1.5-pro": "sonnet4.5",

    # DeepSeek aliases (map to best available)
    "deepseek/deepseek-chat-v3-0324": "sonnet4.5",
    "deepseek-chat": "sonnet4.5",

    # Generic aliases
    "gpt-4o": "gpt5",
    "gpt-4o-mini": "haiku4.5",
    "gpt-4": "gpt5",
}


def normalize_model_name(model: Optional[str]) -> str:
    """
    Normalize model name to auggie's supported format.

    Args:
        model: Model name in various formats

    Returns:
        Normalized model name supported by auggie

    Raises:
        ValueError: If model is not supported
    """
    if not model:
        # Default to sonnet4.5
        return "sonnet4.5"

    # Check if already a valid auggie model
    if model in AUGGIE_SUPPORTED_MODELS:
        return model

    # Check aliases
    if model in AUGGIE_MODEL_ALIASES:
        return AUGGIE_MODEL_ALIASES[model]

    # Try case-insensitive match
    model_lower = model.lower()
    if model_lower in AUGGIE_SUPPORTED_MODELS:
        return model_lower

    for alias, auggie_model in AUGGIE_MODEL_ALIASES.items():
        if alias.lower() == model_lower:
            return auggie_model

    # Model not found - provide helpful error message
    raise ValueError(
        f"Model '{model}' is not supported by auggie. "
        f"Supported models: {', '.join(AUGGIE_SUPPORTED_MODELS.keys())}. "
        f"Available models:\n" +
        "\n".join([f"  - {name} [{key}]" for key, name in AUGGIE_SUPPORTED_MODELS.items()])
    )


def get_available_models() -> Dict[str, str]:
    """
    Get dictionary of available auggie models.

    Returns:
        Dictionary mapping model IDs to their full names
    """
    return AUGGIE_SUPPORTED_MODELS.copy()


def list_available_models() -> str:
    """
    Get a formatted string of available models.

    Returns:
        Formatted string listing all available models
    """
    lines = ["Available auggie models:"]
    for key, name in AUGGIE_SUPPORTED_MODELS.items():
        lines.append(f"  - {name} [{key}]")
    return "\n".join(lines)


class AuggieClient:
    """Client for interacting with LLM models through auggie CLI."""

    def __init__(
        self,
        model: Optional[str] = None,
        workspace_root: Optional[str] = None,
        max_turns: Optional[int] = None,
        quiet: bool = True,
        validate_model: bool = True,
    ):
        """
        Initialize the Auggie client.

        Args:
            model: Model ID to use (e.g., "sonnet4.5", "haiku4.5", "gpt5")
                   Also accepts common aliases like "anthropic/claude-3-5-sonnet"
            workspace_root: Workspace root directory
            max_turns: Maximum number of agentic turns
            quiet: Only show final assistant message
            validate_model: Whether to validate and normalize model name (default: True)
        """
        if validate_model:
            self.model = normalize_model_name(model)
        else:
            self.model = model

        self.workspace_root = workspace_root or os.getcwd()
        self.max_turns = max_turns
        self.quiet = quiet

    def _build_command(
        self,
        instruction: str,
        print_mode: bool = True,
        output_format: str = "json",
        image_paths: Optional[List[str]] = None,
    ) -> List[str]:
        """Build the auggie command with arguments."""
        cmd = ["auggie"]

        # Add instruction
        cmd.extend(["-i", instruction])

        # Add print mode for one-shot execution
        if print_mode:
            cmd.append("--print")

        # Add output format
        if output_format:
            cmd.extend(["--output-format", output_format])

        # Add model if specified
        if self.model:
            cmd.extend(["-m", self.model])

        # Add workspace root
        cmd.extend(["-w", self.workspace_root])

        # Add max turns if specified
        if self.max_turns:
            cmd.extend(["--max-turns", str(self.max_turns)])

        # Add quiet mode
        if self.quiet:
            cmd.append("--quiet")

        # Add images if provided
        if image_paths:
            for img_path in image_paths:
                cmd.extend(["--image", img_path])

        return cmd

    def invoke(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        output_schema: Optional[type[BaseModel]] = None,
        image_paths: Optional[List[str]] = None,
        timeout: Optional[int] = 300,
    ) -> Union[str, BaseModel, Dict]:
        """
        Invoke auggie with a prompt and get the response.

        Args:
            prompt: The prompt string or list of messages
            output_schema: Optional Pydantic model for structured output
            image_paths: Optional list of image file paths to attach
            timeout: Timeout in seconds (default: 300)

        Returns:
            Response as string, Pydantic model instance, or dict
        """
        # Convert prompt to string if it's a list of messages
        if isinstance(prompt, list):
            prompt_str = self._format_messages(prompt)
        else:
            prompt_str = prompt

        # Add schema instruction if output_schema is provided
        if output_schema:
            schema_json = output_schema.model_json_schema()
            prompt_str += f"\n\nPlease respond with a JSON object that matches this schema:\n{json.dumps(schema_json, indent=2)}"

        # Build command
        cmd = self._build_command(
            instruction=prompt_str,
            print_mode=True,
            output_format="json" if output_schema else "text",
            image_paths=image_paths,
        )

        try:
            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace_root,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Auggie command failed with code {result.returncode}: {result.stderr}"
                )

            response_text = result.stdout.strip()

            # Parse response based on output format
            if output_schema:
                # Try to parse as JSON
                try:
                    response_data = json.loads(response_text)
                    return output_schema(**response_data)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract JSON from text
                    response_data = self._extract_json_from_text(response_text)
                    if response_data:
                        return output_schema(**response_data)
                    raise ValueError(f"Failed to parse JSON response: {response_text}")
            else:
                return response_text

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Auggie command timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Error executing auggie command: {e}")

    async def ainvoke(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        output_schema: Optional[type[BaseModel]] = None,
        image_paths: Optional[List[str]] = None,
        timeout: Optional[int] = 300,
    ) -> Union[str, BaseModel, Dict]:
        """
        Async version of invoke.

        Args:
            prompt: The prompt string or list of messages
            output_schema: Optional Pydantic model for structured output
            image_paths: Optional list of image file paths to attach
            timeout: Timeout in seconds (default: 300)

        Returns:
            Response as string, Pydantic model instance, or dict
        """
        # Convert prompt to string if it's a list of messages
        if isinstance(prompt, list):
            prompt_str = self._format_messages(prompt)
        else:
            prompt_str = prompt

        # Add schema instruction if output_schema is provided
        if output_schema:
            schema_json = output_schema.model_json_schema()
            prompt_str += f"\n\nPlease respond with a JSON object that matches this schema:\n{json.dumps(schema_json, indent=2)}"

        # Build command
        cmd = self._build_command(
            instruction=prompt_str,
            print_mode=True,
            output_format="json" if output_schema else "text",
            image_paths=image_paths,
        )

        try:
            # Execute command asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_root,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Auggie command timed out after {timeout} seconds")

            if process.returncode != 0:
                raise RuntimeError(
                    f"Auggie command failed with code {process.returncode}: {stderr.decode()}"
                )

            response_text = stdout.decode().strip()

            # Parse response based on output format
            if output_schema:
                # Try to parse as JSON
                try:
                    response_data = json.loads(response_text)
                    return output_schema(**response_data)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract JSON from text
                    response_data = self._extract_json_from_text(response_text)
                    if response_data:
                        return output_schema(**response_data)
                    raise ValueError(f"Failed to parse JSON response: {response_text}")
            else:
                return response_text

        except Exception as e:
            if not isinstance(e, (TimeoutError, RuntimeError, ValueError)):
                raise RuntimeError(f"Error executing auggie command: {e}")
            raise

    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Format a list of messages into a single prompt string."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multi-part content
                text_parts = [
                    part.get("text", "") for part in content if part.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            formatted.append(f"{role.upper()}: {content}")
        return "\n\n".join(formatted)

    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """Extract JSON object from text that may contain markdown code blocks."""
        # Try to find JSON in markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                json_str = text[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # Try to find JSON in plain code blocks
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                json_str = text[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # Try to parse the entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        return None


def create_auggie_client(
    model: Optional[str] = None,
    workspace_root: Optional[str] = None,
    **kwargs
) -> AuggieClient:
    """
    Factory function to create an AuggieClient instance.

    Args:
        model: Model ID to use
        workspace_root: Workspace root directory
        **kwargs: Additional arguments for AuggieClient

    Returns:
        AuggieClient instance
    """
    return AuggieClient(model=model, workspace_root=workspace_root, **kwargs)

