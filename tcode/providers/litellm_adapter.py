from __future__ import annotations
import os
from typing import Dict, Any
from .openai_adapter import OpenAIAdapter


class LitellmAdapter(OpenAIAdapter):
    """LiteLLM proxy adapter. LiteLLM exposes an OpenAI-compatible API,
    so we use the openai SDK's AsyncOpenAI with a custom base_url.
    """

    def __init__(self, api_url: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, timeout: int = 120, **opts):
        resolved_url = api_url or base_url or os.environ.get("LITELLM_BASE_URL") or "http://localhost:4000/v1"
        key = api_key or os.environ.get("LITELLM_API_KEY") or "sk-123456"
        super().__init__(api_key=key, base_url=resolved_url, timeout=timeout, **opts)

    def get_model(self, model_id: str) -> Dict[str, Any]:
        return {"id": model_id, "provider": "litellm", "supports_tools": True}
