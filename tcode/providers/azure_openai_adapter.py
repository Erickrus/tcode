from __future__ import annotations
import os
from typing import Dict, Any, Optional
from openai import AsyncAzureOpenAI
from .openai_adapter import OpenAIAdapter


class AzureOpenAIAdapter(OpenAIAdapter):
    """Azure OpenAI adapter using the official openai SDK's AsyncAzureOpenAI.

    Same API as OpenAI, different client constructor.
    """

    def __init__(self, api_key: str | None = None, azure_endpoint: str | None = None,
                 deployment: str | None = None, api_version: str = "2024-06-01",
                 timeout: int = 120, **opts):
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = api_version
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = (azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
                               or os.environ.get("AZURE_OPENAI_API_BASE") or "")
        self.timeout = timeout
        self.opts = opts or {}
        # Don't set base_url — Azure uses azure_endpoint
        self.base_url = None
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
                timeout=self.timeout,
            )
        return self._client

    def get_model(self, model_id: str) -> Dict[str, Any]:
        return {"id": model_id, "provider": "azure-openai", "supports_tools": True}
