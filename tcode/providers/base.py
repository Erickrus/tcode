from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncIterator, Optional
from dataclasses import dataclass, field


class ProviderChunk(dict):
    """A streaming chunk from a provider.

    Chunk types:
      - {"type": "delta", "text": "..."}                           — incremental text
      - {"type": "reasoning_delta", "text": "..."}                 — reasoning/thinking text
      - {"type": "tool_call_start", "id": "...", "name": "..."}    — start of a tool call
      - {"type": "tool_call_delta", "id": "...", "arguments": "..."}  — tool call argument fragment
      - {"type": "tool_call_end", "id": "...", "name": "...", "arguments": {...}}  — complete tool call
      - {"type": "usage", "input_tokens": N, "output_tokens": N, ...}
      - {"type": "final", "data": {...}}                           — stream ended
      - {"type": "error", "error": "..."}                          — provider error
    """
    pass


@dataclass
class ModelDescriptor:
    """Describes a model's metadata."""
    id: str
    name: str = ""
    provider: str = ""
    supports_tools: bool = False
    supports_streaming: bool = True
    max_output_tokens: int = 4096
    extra: Dict[str, Any] = field(default_factory=dict)


class ProviderAdapter(ABC):
    """Base class for LLM provider adapters.

    All adapters use native Python httpx for HTTP calls.
    Tool definitions are passed via the `tools` parameter as a list of dicts
    in a provider-agnostic format:
        [{"name": "tool_name", "description": "...", "parameters": {json_schema}}]

    Each adapter is responsible for converting this to the provider's native format.
    """

    @abstractmethod
    async def chat(self, messages: List[Dict[str, Any]], model: str,
                   options: Dict[str, Any],
                   tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Single (non-streaming) chat completion."""
        raise NotImplementedError

    @abstractmethod
    async def chat_stream(self, messages: List[Dict[str, Any]], model: str,
                          options: Dict[str, Any],
                          tools: Optional[List[Dict[str, Any]]] = None) -> AsyncIterator[ProviderChunk]:
        """Streaming chat completion. Yields ProviderChunk dicts."""
        raise NotImplementedError
        # Make this an async generator
        yield  # pragma: no cover

    def supports_tools(self) -> bool:
        """Whether this provider supports tool/function calling."""
        return False

    def get_model(self, model_id: str) -> Dict[str, Any]:
        """Return model metadata."""
        raise NotImplementedError
