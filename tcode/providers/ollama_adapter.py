from __future__ import annotations
import os
import json
from typing import List, Dict, Any, AsyncIterator, Optional
from ollama import AsyncClient as OllamaAsyncClient
from .base import ProviderAdapter, ProviderChunk


class OllamaAdapter(ProviderAdapter):
    """Ollama adapter using the official ollama Python SDK.

    Uses AsyncClient for async chat. Tool calling uses non-streaming mode
    (streaming + tools is unreliable in ollama SDK as of v0.6).
    """

    def __init__(self, host: str | None = None, timeout: int = 120, **opts):
        self.host = host or os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_BASE_URL")
        self.timeout = timeout
        self.opts = opts or {}
        self._client: Optional[OllamaAsyncClient] = None

    def _get_client(self) -> OllamaAsyncClient:
        if self._client is None:
            kwargs: Dict[str, Any] = {}
            if self.host:
                kwargs["host"] = self.host
            self._client = OllamaAsyncClient(**kwargs)
        return self._client

    def _format_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert generic tool defs to Ollama tools format (OpenAI-compatible)."""
        if not tools:
            return None
        result = []
        for t in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        return result

    async def chat(self, messages: List[Dict[str, Any]], model: str,
                   options: Dict[str, Any],
                   tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        full_text = ""
        tool_calls = []
        async for chunk in self.chat_stream(messages, model, options, tools=tools):
            ctype = chunk.get("type")
            if ctype == "delta":
                full_text += chunk.get("text", "")
            elif ctype == "tool_call_end":
                tool_calls.append({
                    "id": chunk.get("id"),
                    "name": chunk.get("name"),
                    "arguments": chunk.get("arguments"),
                })
        result: Dict[str, Any] = {"text": full_text}
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    async def chat_stream(self, messages: List[Dict[str, Any]], model: str,
                          options: Dict[str, Any],
                          tools: Optional[List[Dict[str, Any]]] = None) -> AsyncIterator[ProviderChunk]:
        client = self._get_client()
        formatted_tools = self._format_tools(tools)

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        # Ollama options
        ollama_options: Dict[str, Any] = {}
        if "temperature" in options:
            ollama_options["temperature"] = options["temperature"]
        if "top_p" in options:
            ollama_options["top_p"] = options["top_p"]
        if ollama_options:
            kwargs["options"] = ollama_options

        if formatted_tools:
            kwargs["tools"] = formatted_tools

        full_text = ""

        try:
            if formatted_tools:
                # Non-streaming for tool calls (more reliable)
                response = await client.chat(**kwargs)
                msg = response.message
                content = msg.content or ""
                if content:
                    full_text = content
                    yield ProviderChunk({"type": "delta", "text": content})

                # Tool calls — ollama returns arguments as dicts already
                if msg.tool_calls:
                    for i, tc in enumerate(msg.tool_calls):
                        func = tc.function
                        args = func.arguments if isinstance(func.arguments, dict) else {}
                        yield ProviderChunk({
                            "type": "tool_call_end",
                            "id": f"ollama-{func.name}-{i}",
                            "name": func.name,
                            "arguments": args,
                        })

                # Usage
                if hasattr(response, 'prompt_eval_count'):
                    yield ProviderChunk({
                        "type": "usage",
                        "input_tokens": getattr(response, 'prompt_eval_count', 0) or 0,
                        "output_tokens": getattr(response, 'eval_count', 0) or 0,
                    })
            else:
                # Streaming for text-only responses
                kwargs["stream"] = True
                stream = await client.chat(**kwargs)
                async for part in stream:
                    msg = part.message
                    content = msg.content or ""
                    if content:
                        full_text += content
                        yield ProviderChunk({"type": "delta", "text": content})
                    if getattr(part, 'done', False):
                        if hasattr(part, 'prompt_eval_count'):
                            yield ProviderChunk({
                                "type": "usage",
                                "input_tokens": getattr(part, 'prompt_eval_count', 0) or 0,
                                "output_tokens": getattr(part, 'eval_count', 0) or 0,
                            })
                        break

            yield ProviderChunk({"type": "final", "data": {"text": full_text}})

        except Exception as e:
            from .errors import map_provider_error
            yield ProviderChunk({"type": "error", "error": map_provider_error(e)})

    def supports_tools(self) -> bool:
        return True

    def get_model(self, model_id: str) -> Dict[str, Any]:
        return {"id": model_id, "provider": "ollama", "supports_tools": True}
