from __future__ import annotations
from typing import Dict, Callable, Any
from .base import ProviderAdapter

class ProviderFactory:
    def __init__(self):
        self._constructors: Dict[str, Callable[..., ProviderAdapter]] = {}
        self._instances: Dict[str, ProviderAdapter] = {}

    def register_adapter(self, name: str, constructor: Callable[..., ProviderAdapter]):
        self._constructors[name] = constructor

    def get_adapter(self, provider_id: str, config: Dict[str, Any]) -> ProviderAdapter:
        # provider_id could be used to cache instances by id
        if provider_id in self._instances:
            return self._instances[provider_id]
        adapter_name = config.get("type")
        constructor = self._constructors.get(adapter_name)
        if not constructor:
            raise KeyError(f"Provider adapter not registered: {adapter_name}")
        instance = constructor(**(config.get("options") or {}))
        self._instances[provider_id] = instance
        return instance
