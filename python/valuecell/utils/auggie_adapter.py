"""
Adapters to make AuggieClient compatible with existing LLM interfaces.

This module provides compatibility layers for:
- LangChain ChatModel interface
- agno Model interface
- OpenAI client interface
"""

import os
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from pydantic import BaseModel

from valuecell.utils.auggie_client import AuggieClient


class AuggieLangChainAdapter:
    """
    Adapter to make AuggieClient compatible with LangChain's ChatModel interface.
    
    This allows using auggie as a drop-in replacement for LangChain models.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        workspace_root: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the adapter.

        Args:
            model: Model ID to use
            workspace_root: Workspace root directory
            **kwargs: Additional arguments for AuggieClient
        """
        self.client = AuggieClient(
            model=model,
            workspace_root=workspace_root,
            **kwargs
        )
        self._structured_output_schema = None

    def with_structured_output(
        self,
        schema: type[BaseModel],
        method: str = "json_mode",
        **kwargs
    ):
        """
        Configure the model to return structured output.

        Args:
            schema: Pydantic model class for output structure
            method: Method to use (e.g., "json_mode")
            **kwargs: Additional arguments

        Returns:
            Self for method chaining
        """
        self._structured_output_schema = schema
        return self

    def invoke(self, prompt: Union[str, List[Dict[str, Any]]], **kwargs) -> Union[str, BaseModel]:
        """
        Invoke the model with a prompt.

        Args:
            prompt: The prompt string or list of messages
            **kwargs: Additional arguments

        Returns:
            Response as string or Pydantic model instance
        """
        return self.client.invoke(
            prompt=prompt,
            output_schema=self._structured_output_schema,
            **kwargs
        )

    async def ainvoke(self, prompt: Union[str, List[Dict[str, Any]]], **kwargs) -> Union[str, BaseModel]:
        """
        Async invoke the model with a prompt.

        Args:
            prompt: The prompt string or list of messages
            **kwargs: Additional arguments

        Returns:
            Response as string or Pydantic model instance
        """
        return await self.client.ainvoke(
            prompt=prompt,
            output_schema=self._structured_output_schema,
            **kwargs
        )


class AuggieAgnoAdapter:
    """
    Adapter to make AuggieClient compatible with agno's Model interface.
    
    This allows using auggie with agno agents.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        model: Optional[str] = None,
        workspace_root: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the adapter.

        Args:
            id: Model ID (agno style parameter)
            model: Model ID (alternative parameter)
            workspace_root: Workspace root directory
            **kwargs: Additional arguments for AuggieClient
        """
        model_id = id or model
        self.client = AuggieClient(
            model=model_id,
            workspace_root=workspace_root,
            **kwargs
        )
        self.id = model_id

    def response(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Generate a response (synchronous).

        Args:
            messages: List of message dictionaries
            **kwargs: Additional arguments

        Returns:
            Response dictionary
        """
        result = self.client.invoke(prompt=messages, **kwargs)
        
        # Format response to match agno's expected structure
        if isinstance(result, str):
            return {
                "content": result,
                "role": "assistant",
            }
        elif isinstance(result, BaseModel):
            return {
                "content": result.model_dump_json(),
                "role": "assistant",
                "parsed": result,
            }
        return result

    async def aresponse(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Generate a response (asynchronous).

        Args:
            messages: List of message dictionaries
            **kwargs: Additional arguments

        Returns:
            Response dictionary
        """
        result = await self.client.ainvoke(prompt=messages, **kwargs)
        
        # Format response to match agno's expected structure
        if isinstance(result, str):
            return {
                "content": result,
                "role": "assistant",
            }
        elif isinstance(result, BaseModel):
            return {
                "content": result.model_dump_json(),
                "role": "assistant",
                "parsed": result,
            }
        return result


class AuggieOpenAIAdapter:
    """
    Adapter to make AuggieClient compatible with OpenAI client interface.
    
    This allows using auggie as a drop-in replacement for OpenAI client.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        workspace_root: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the adapter.

        Args:
            model: Model ID to use
            base_url: Base URL (ignored, for compatibility)
            api_key: API key (ignored, for compatibility)
            workspace_root: Workspace root directory
            **kwargs: Additional arguments for AuggieClient
        """
        self.client = AuggieClient(
            model=model,
            workspace_root=workspace_root,
            **kwargs
        )
        self.model = model
        self.responses = self  # For compatibility with OpenAI client structure

    def create(
        self,
        model: Optional[str] = None,
        input: Optional[List[Dict[str, Any]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        text: Optional[Dict] = None,
        reasoning: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        store: Optional[bool] = None,
        **kwargs
    ) -> Any:
        """
        Create a completion (OpenAI-style).

        Args:
            model: Model to use
            input: Input messages (OpenAI Responses API style)
            messages: Messages (standard chat completion style)
            text: Text format configuration
            reasoning: Reasoning configuration
            tools: Tools to use
            temperature: Temperature parameter
            max_output_tokens: Maximum output tokens
            top_p: Top-p parameter
            store: Whether to store the conversation
            **kwargs: Additional arguments

        Returns:
            Response object
        """
        # Use input or messages
        prompt = input or messages
        
        # Add tool instructions if tools are provided
        if tools:
            tool_descriptions = []
            for tool in tools:
                tool_type = tool.get("type", "")
                tool_descriptions.append(f"- {tool_type}")
            
            if isinstance(prompt, list):
                # Add tool instruction to the last message
                if prompt:
                    last_msg = prompt[-1]
                    if isinstance(last_msg.get("content"), list):
                        last_msg["content"].append({
                            "type": "text",
                            "text": f"\n\nAvailable tools:\n" + "\n".join(tool_descriptions)
                        })
                    else:
                        last_msg["content"] += f"\n\nAvailable tools:\n" + "\n".join(tool_descriptions)
        
        result = self.client.invoke(prompt=prompt, **kwargs)
        
        # Format response to match OpenAI structure
        class OpenAIResponse:
            def __init__(self, content):
                self.output = [
                    type('obj', (object,), {'content': [type('obj', (object,), {'text': ''})()]})(),
                    type('obj', (object,), {'content': [type('obj', (object,), {'text': content})()]})()
                ]
        
        return OpenAIResponse(result if isinstance(result, str) else str(result))


def get_auggie_model(
    model_name: Optional[str] = None,
    adapter_type: str = "langchain",
    workspace_root: Optional[str] = None,
    **kwargs
) -> Union[AuggieLangChainAdapter, AuggieAgnoAdapter, AuggieOpenAIAdapter]:
    """
    Factory function to create an auggie adapter of the specified type.

    Args:
        model_name: Model ID to use
        adapter_type: Type of adapter ("langchain", "agno", or "openai")
        workspace_root: Workspace root directory
        **kwargs: Additional arguments for the adapter

    Returns:
        Adapter instance

    Raises:
        ValueError: If adapter_type is not supported
    """
    if adapter_type == "langchain":
        return AuggieLangChainAdapter(
            model=model_name,
            workspace_root=workspace_root,
            **kwargs
        )
    elif adapter_type == "agno":
        return AuggieAgnoAdapter(
            id=model_name,
            workspace_root=workspace_root,
            **kwargs
        )
    elif adapter_type == "openai":
        return AuggieOpenAIAdapter(
            model=model_name,
            workspace_root=workspace_root,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported adapter type: {adapter_type}. "
            f"Supported types: langchain, agno, openai"
        )

