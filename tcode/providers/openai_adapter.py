from __future__ import annotations
import os
import json
from typing import List, Dict, Any, AsyncIterator, Optional
from openai import AsyncOpenAI
from .base import ProviderAdapter, ProviderChunk


class OpenAIAdapter(ProviderAdapter):
    """OpenAI adapter using the official openai Python SDK.

    Uses AsyncOpenAI for async streaming with tool_calls support.
    Also serves as base for Azure and LiteLLM (same API shape).
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 timeout: int = 120, **opts):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1")
        self.timeout = timeout
        self.opts = opts or {}
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def _format_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert generic tool defs to OpenAI tools format."""
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
        usage = {}
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
            elif ctype == "usage":
                usage = chunk
            elif ctype == "final":
                break
        result: Dict[str, Any] = {"text": full_text}
        if tool_calls:
            result["tool_calls"] = tool_calls
        if usage:
            result["usage"] = usage
        return result

    async def chat_stream(self, messages: List[Dict[str, Any]], model: str,
                          options: Dict[str, Any],
                          tools: Optional[List[Dict[str, Any]]] = None) -> AsyncIterator[ProviderChunk]:
        client = self._get_client()

        # Build kwargs
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # Merge options
        for k in ("temperature", "top_p", "max_tokens", "max_completion_tokens"):
            if k in options:
                kwargs[k] = options[k]

        # Add tools
        formatted_tools = self._format_tools(tools)
        if formatted_tools:
            kwargs["tools"] = formatted_tools

        # Track tool calls being accumulated by index
        active_tool_calls: Dict[int, Dict[str, Any]] = {}
        full_text = ""

        try:
            stream = await client.chat.completions.create(**kwargs)

            async for chunk in stream:
                # Usage (arrives on final chunk with stream_options)
                if chunk.usage:
                    yield ProviderChunk({
                        "type": "usage",
                        "input_tokens": chunk.usage.prompt_tokens or 0,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                        "total_tokens": chunk.usage.total_tokens or 0,
                    })

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta
                finish_reason = choice.finish_reason

                # Text content
                if delta and delta.content:
                    full_text += delta.content
                    yield ProviderChunk({"type": "delta", "text": delta.content})

                # Tool calls (streamed incrementally)
                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in active_tool_calls:
                            active_tool_calls[idx] = {
                                "id": tc.id or "",
                                "name": "",
                                "arguments_fragments": [],
                            }
                        tc_state = active_tool_calls[idx]
                        if tc.id:
                            tc_state["id"] = tc.id
                        if tc.function and tc.function.name:
                            tc_state["name"] = tc.function.name
                            yield ProviderChunk({
                                "type": "tool_call_start",
                                "id": tc_state["id"],
                                "name": tc_state["name"],
                            })
                        if tc.function and tc.function.arguments:
                            tc_state["arguments_fragments"].append(tc.function.arguments)
                            yield ProviderChunk({
                                "type": "tool_call_delta",
                                "id": tc_state["id"],
                                "arguments": tc.function.arguments,
                            })

                # On finish, emit completed tool calls
                if finish_reason in ("tool_calls", "stop") and active_tool_calls:
                    for idx in sorted(active_tool_calls.keys()):
                        tc_state = active_tool_calls[idx]
                        args_text = "".join(tc_state["arguments_fragments"])
                        try:
                            args_obj = json.loads(args_text) if args_text else {}
                        except json.JSONDecodeError:
                            args_obj = {"_raw": args_text}
                        yield ProviderChunk({
                            "type": "tool_call_end",
                            "id": tc_state["id"],
                            "name": tc_state["name"],
                            "arguments": args_obj,
                        })
                    active_tool_calls.clear()

            yield ProviderChunk({"type": "final", "data": {"text": full_text}})

        except Exception as e:
            from .errors import map_provider_error
            yield ProviderChunk({"type": "error", "error": map_provider_error(e)})

    def supports_tools(self) -> bool:
        return True

    def get_model(self, model_id: str) -> Dict[str, Any]:
        return {"id": model_id, "provider": "openai", "supports_tools": True}
