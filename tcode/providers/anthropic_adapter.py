from __future__ import annotations
import os
import json
from typing import List, Dict, Any, AsyncIterator, Optional
from anthropic import AsyncAnthropic
from .base import ProviderAdapter, ProviderChunk


class AnthropicAdapter(ProviderAdapter):
    """Anthropic adapter using the official anthropic Python SDK.

    Uses AsyncAnthropic for async streaming with tool_use content blocks.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 timeout: int = 120, **opts):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL")
                         or os.environ.get("ANTHROPIC_API_BASE"))
        self.timeout = timeout
        self.opts = opts or {}
        self._client: Optional[AsyncAnthropic] = None

    def _get_client(self) -> AsyncAnthropic:
        if self._client is None:
            kwargs: Dict[str, Any] = {"timeout": self.timeout}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncAnthropic(**kwargs)
        return self._client

    def _format_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert generic tool defs to Anthropic tools format."""
        if not tools:
            return None
        result = []
        for t in tools:
            result.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    def _extract_system_and_messages(self, messages: List[Dict[str, Any]]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Extract system prompt and convert messages to Anthropic format.

        Anthropic requires system as a top-level param.
        Tool call/result messages need conversion to Anthropic content blocks.
        """
        system_prompt = None
        anthropic_msgs: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                system_prompt = msg.get("content", "")
                continue
            if role == "user":
                anthropic_msgs.append({"role": "user", "content": msg.get("content", "")})
            elif role == "assistant":
                content_blocks: List[Dict[str, Any]] = []
                text = msg.get("content")
                if text:
                    content_blocks.append({"type": "text", "text": text})
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        input_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        input_obj = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": input_obj,
                    })
                if content_blocks:
                    anthropic_msgs.append({"role": "assistant", "content": content_blocks})
            elif role == "tool":
                anthropic_msgs.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }],
                })

        return system_prompt, anthropic_msgs

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
        system_prompt, anthropic_msgs = self._extract_system_and_messages(messages)

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_msgs,
            "max_tokens": options.get("max_tokens", 4096),
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if "temperature" in options:
            kwargs["temperature"] = options["temperature"]
        if "top_p" in options:
            kwargs["top_p"] = options["top_p"]

        formatted_tools = self._format_tools(tools)
        if formatted_tools:
            kwargs["tools"] = formatted_tools

        # Track tool_use blocks by index
        active_blocks: Dict[int, Dict[str, Any]] = {}
        full_text = ""

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    etype = event.type

                    if etype == "message_start":
                        msg = getattr(event, "message", None)
                        if msg and hasattr(msg, "usage") and msg.usage:
                            yield ProviderChunk({
                                "type": "usage",
                                "input_tokens": getattr(msg.usage, "input_tokens", 0),
                                "output_tokens": getattr(msg.usage, "output_tokens", 0),
                            })

                    elif etype == "content_block_start":
                        idx = event.index
                        block = event.content_block
                        if block.type == "tool_use":
                            active_blocks[idx] = {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input_fragments": [],
                            }
                            yield ProviderChunk({
                                "type": "tool_call_start",
                                "id": block.id,
                                "name": block.name,
                            })
                        elif block.type == "text":
                            active_blocks[idx] = {"type": "text"}
                        elif block.type == "thinking":
                            active_blocks[idx] = {"type": "thinking"}

                    elif etype == "content_block_delta":
                        idx = event.index
                        delta = event.delta
                        if delta.type == "text_delta":
                            text = delta.text
                            if text:
                                full_text += text
                                yield ProviderChunk({"type": "delta", "text": text})
                        elif delta.type == "input_json_delta":
                            partial = delta.partial_json
                            if idx in active_blocks and active_blocks[idx]["type"] == "tool_use":
                                active_blocks[idx]["input_fragments"].append(partial)
                                yield ProviderChunk({
                                    "type": "tool_call_delta",
                                    "id": active_blocks[idx]["id"],
                                    "arguments": partial,
                                })
                        elif delta.type == "thinking_delta":
                            text = getattr(delta, "thinking", "")
                            if text:
                                yield ProviderChunk({"type": "reasoning_delta", "text": text})

                    elif etype == "content_block_stop":
                        idx = event.index
                        if idx in active_blocks and active_blocks[idx]["type"] == "tool_use":
                            block = active_blocks[idx]
                            input_text = "".join(block["input_fragments"])
                            try:
                                input_obj = json.loads(input_text) if input_text else {}
                            except json.JSONDecodeError:
                                input_obj = {"_raw": input_text}
                            yield ProviderChunk({
                                "type": "tool_call_end",
                                "id": block["id"],
                                "name": block["name"],
                                "arguments": input_obj,
                            })
                        if idx in active_blocks:
                            del active_blocks[idx]

                    elif etype == "message_delta":
                        u = getattr(event, "usage", None)
                        if u:
                            yield ProviderChunk({
                                "type": "usage",
                                "input_tokens": getattr(u, "input_tokens", 0),
                                "output_tokens": getattr(u, "output_tokens", 0),
                            })

            yield ProviderChunk({"type": "final", "data": {"text": full_text}})

        except Exception as e:
            from .errors import map_provider_error
            yield ProviderChunk({"type": "error", "error": map_provider_error(e)})

    def supports_tools(self) -> bool:
        return True

    def get_model(self, model_id: str) -> Dict[str, Any]:
        return {"id": model_id, "provider": "anthropic", "supports_tools": True}
