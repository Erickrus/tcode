from __future__ import annotations
import secrets

# Minimal helper for OAuth state storage used by MCP start/auth/finish flows
# For v1 store in-memory; in production store in persistent secure storage

_oauth_state: dict[str, str] = {}

async def set_oauth_state(mcp_name: str, state: str):
    _oauth_state[mcp_name] = state

async def get_oauth_state(mcp_name: str) -> str | None:
    return _oauth_state.get(mcp_name)

async def clear_oauth_state(mcp_name: str):
    _oauth_state.pop(mcp_name, None)

async def gen_state() -> str:
    return secrets.token_hex(32)
