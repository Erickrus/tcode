from __future__ import annotations
from .mcp import MCPManager
from .mcp_auth import get_oauth_state, clear_oauth_state
from .event import Event

async def finish_auth(manager: MCPManager, name: str, code: str):
    transport = manager.pending_oauth.get(name)
    if not transport:
        raise RuntimeError('no pending oauth flow')
    try:
        await transport.finish_auth(code)
    except Exception as e:
        if manager.events:
            ev = Event.create('mcp.auth.failed', {'mcp': name, 'error': str(e)})
            await manager.events.publish(ev)
        raise
    # clear state
    await clear_oauth_state(name)
    # reconnect
    await manager.add(name, {'url': transport.base_url, 'type': 'remote'})
    if manager.events:
        ev = Event.create('mcp.auth.finished', {'mcp': name})
        await manager.events.publish(ev)
    return manager.status.get(name)
