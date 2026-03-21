"""Shared mutable state for the TUI, single-threaded (asyncio)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TuiState:
    screen: str = "home"  # "home" | "session"
    session_id: str | None = None
    status: str = "idle"  # "idle" | "running" | "waiting_permission"
    streaming_text: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    pending_permission: dict[str, Any] | None = None
    cost: float = 0.0
    tokens: dict[str, Any] = field(default_factory=dict)
    model_id: str = ""
    provider_id: str = ""
    agent_name: str = "build"
    mcp_status: dict[str, str] = field(default_factory=dict)
    queued_prompts: list[str] = field(default_factory=list)
    prompt_history: list[str] = field(default_factory=list)
    cancel_pending: bool = False
    cancel_pending_time: float = 0.0
