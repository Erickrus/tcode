from __future__ import annotations
import os
import json
from typing import List, Dict, Any, AsyncIterator, Optional
from google import genai
from google.genai import types
from .base import ProviderAdapter, ProviderChunk


class GeminiAdapter(ProviderAdapter):
    """Google Gemini adapter using the official google-genai Python SDK.

    Uses the async generate_content_stream for streaming.
    """

    def __init__(self, api_key: str | None = None, timeout: int = 120, **opts):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.timeout = timeout
        self.opts = opts or {}
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    # Fields that Gemini's Schema actually supports.
    _GEMINI_SCHEMA_KEYS = frozenset({
        "type", "description", "properties", "required", "items",
        "enum", "format", "nullable", "minimum", "maximum",
        "min_items", "max_items",
    })

    def _sanitize_schema(self, schema: Any) -> Any:
        """Recursively strip fields unsupported by Gemini (e.g. additional_properties, anyOf, $schema)."""
        if not isinstance(schema, dict):
            return schema

        cleaned: Dict[str, Any] = {}
        for key, value in schema.items():
            # Convert camelCase variants to snake_case used by Gemini
            normalized = key
            if key == "additionalProperties":
                normalized = "additional_properties"
            elif key == "minItems":
                normalized = "min_items"
            elif key == "maxItems":
                normalized = "max_items"

            # Drop keys Gemini doesn't understand
            if normalized not in self._GEMINI_SCHEMA_KEYS:
                # Special handling: flatten anyOf/oneOf with a single entry
                if key in ("anyOf", "oneOf") and isinstance(value, list) and len(value) == 1:
                    merged = self._sanitize_schema(value[0])
                    if isinstance(merged, dict):
                        for mk, mv in merged.items():
                            cleaned.setdefault(mk, mv)
                continue

            # Recurse into nested structures
            if normalized == "properties" and isinstance(value, dict):
                cleaned[normalized] = {
                    k: self._sanitize_schema(v) for k, v in value.items()
                }
            elif normalized == "items" and isinstance(value, dict):
                cleaned[normalized] = self._sanitize_schema(value)
            else:
                cleaned[normalized] = value

        # Ensure a type is always present
        if "type" not in cleaned:
            cleaned["type"] = "object"

        return cleaned

    def _format_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[types.Tool]]:
        """Convert generic tool defs to Gemini Tool objects."""
        if not tools:
            return None
        declarations = []
        for t in tools:
            raw_params = t.get("parameters", {"type": "object", "properties": {}})
            declarations.append(types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=self._sanitize_schema(raw_params),
            ))
        return [types.Tool(function_declarations=declarations)]

    def _format_contents(self, messages: List[Dict[str, Any]]) -> tuple[Optional[str], List[types.Content]]:
        """Convert OpenAI-format messages to Gemini contents.

        Returns (system_instruction, contents).
        """
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                system_instruction = msg.get("content", "")
                continue

            gemini_role = "user" if role in ("user", "tool") else "model"
            parts = []

            if role == "tool":
                parts.append(types.Part.from_function_response(
                    name=msg.get("tool_call_id", "unknown"),
                    response={"content": msg.get("content", "")},
                ))
            elif role == "assistant" and msg.get("tool_calls"):
                text = msg.get("content")
                if text:
                    parts.append(types.Part.from_text(text=text))
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        args_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args_obj = {}
                    parts.append(types.Part.from_function_call(
                        name=func.get("name", ""),
                        args=args_obj,
                    ))
            else:
                content = msg.get("content", "")
                if content:
                    parts.append(types.Part.from_text(text=content))

            if parts:
                contents.append(types.Content(role=gemini_role, parts=parts))

        return system_instruction, contents

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
        system_instruction, contents = self._format_contents(messages)

        # Build config
        config_kwargs: Dict[str, Any] = {}
        if "temperature" in options:
            config_kwargs["temperature"] = options["temperature"]
        if "max_tokens" in options:
            config_kwargs["max_output_tokens"] = options["max_tokens"]
        if "top_p" in options:
            config_kwargs["top_p"] = options["top_p"]
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        formatted_tools = self._format_tools(tools)
        config = types.GenerateContentConfig(**config_kwargs, tools=formatted_tools) if (config_kwargs or formatted_tools) else None

        full_text = ""

        try:
            async for response in await client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            ):
                if response.candidates:
                    for candidate in response.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if part.text:
                                    full_text += part.text
                                    yield ProviderChunk({"type": "delta", "text": part.text})
                                elif part.function_call:
                                    fc = part.function_call
                                    args = dict(fc.args) if fc.args else {}
                                    yield ProviderChunk({
                                        "type": "tool_call_end",
                                        "id": f"gemini-{fc.name}",
                                        "name": fc.name,
                                        "arguments": args,
                                    })

                # Usage metadata
                if response.usage_metadata:
                    um = response.usage_metadata
                    yield ProviderChunk({
                        "type": "usage",
                        "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
                        "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
                    })

            yield ProviderChunk({"type": "final", "data": {"text": full_text}})

        except Exception as e:
            from .errors import map_provider_error
            yield ProviderChunk({"type": "error", "error": map_provider_error(e)})

    def supports_tools(self) -> bool:
        return True

    def get_model(self, model_id: str) -> Dict[str, Any]:
        return {"id": model_id, "provider": "gemini", "supports_tools": True}
