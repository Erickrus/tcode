from __future__ import annotations
import asyncio
from typing import Dict, Any, List, Optional
import httpx
import json
from .event import Event
from .tools import ToolInfo, ToolResult
from .util import next_id

# Minimal MCP implementation (HTTP-first) with transports compatibility helpers

class MCPClientWrapper:
    def __init__(self, transport: Any):
        self.transport = transport

    async def list_tools(self) -> List[Dict[str, Any]]:
        # prefer transport.callable list if available
        if hasattr(self.transport, 'list_tools'):
            return await self.transport.list_tools()
        if hasattr(self.transport, '_client') and self.transport._client:
            url = f"{self.transport.base_url}/tools"
            async with httpx.AsyncClient(timeout=self.transport.timeout) as client:
                r = await client.get(url, headers=self.transport.headers)
                r.raise_for_status()
                return r.json()
        raise RuntimeError('transport does not support list_tools')

    async def call_tool(self, name: str, args: Dict[str, Any], stream: bool = False):
        # prefer transport.call_tool if available
        if hasattr(self.transport, 'call_tool'):
            async for chunk in self.transport.call_tool(name, args, stream=stream):
                yield chunk
            return
        if hasattr(self.transport, '_client') and self.transport._client:
            url = f"{self.transport.base_url}/call"
            payload = {"tool": name, "args": args}
            async with httpx.AsyncClient(timeout=self.transport.timeout) as client:
                async with client.stream("POST", url, headers=self.transport.headers, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_text():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except Exception:
                            continue
                        yield data
            return
        raise RuntimeError('transport does not support call_tool')

    async def list_prompts(self) -> List[Dict[str, Any]]:
        if hasattr(self.transport, 'list_prompts'):
            return await self.transport.list_prompts()
        if hasattr(self.transport, '_client') and self.transport._client:
            url = f"{self.transport.base_url}/prompts"
            async with httpx.AsyncClient(timeout=self.transport.timeout) as client:
                r = await client.get(url, headers=self.transport.headers)
                r.raise_for_status()
                return r.json()
        raise RuntimeError('transport does not support list_prompts')

    async def list_resources(self) -> List[Dict[str, Any]]:
        if hasattr(self.transport, 'list_resources'):
            return await self.transport.list_resources()
        if hasattr(self.transport, '_client') and self.transport._client:
            url = f"{self.transport.base_url}/resources"
            async with httpx.AsyncClient(timeout=self.transport.timeout) as client:
                r = await client.get(url, headers=self.transport.headers)
                r.raise_for_status()
                return r.json()
        raise RuntimeError('transport does not support list_resources')

class MCPManager:
    def __init__(self, events: Optional["EventBus"] = None, tool_registry: Optional["ToolRegistry"] = None):
        self.clients: Dict[str, MCPClientWrapper] = {}
        self.status: Dict[str, str] = {}
        self.pending_oauth: Dict[str, Any] = {}
        self.transports: Dict[str, Any] = {}
        self.events = events
        self.tool_registry = tool_registry

    async def add(self, name: str, config: Dict[str, Any]):
        base_url = config.get("url")
        headers = config.get("headers") or {}
        timeout = config.get("timeout", 60)
        # pick transport
        transport = None
        if config.get('type') == 'local' and config.get('command'):
            from .mcp_transports import StdioTransport
            cmd = config['command'][0]
            args = config['command'][1:]
            transport = StdioTransport(cmd, args=args, cwd=config.get('cwd'))
        else:
            from .mcp_transports import HTTPStreamTransport
            transport = HTTPStreamTransport(base_url, headers=headers, timeout=timeout)

        try:
            await transport.connect()
        except Exception as e:
            # Unauthorized handling
            msg = str(e)
            if hasattr(e, 'authorization_url') or 'unauthorized' in msg.lower():
                self.pending_oauth[name] = transport
                self.status[name] = 'needs_auth'
                if self.events:
                    await self.events.publish(Event.create('mcp.auth.required', {'mcp': name, 'url': getattr(e, 'authorization_url', None)}))
                return
            if 'registration' in msg or 'client_id' in msg:
                self.status[name] = 'needs_client_registration'
                if self.events:
                    await self.events.publish(Event.create('mcp.client_registration_required', {'mcp': name, 'error': msg}))
                return
            self.status[name] = f'failed: {msg}'
            return

        # success
        self.transports[name] = transport
        client = MCPClientWrapper(transport)
        self.clients[name] = client
        self.status[name] = 'connected'

        # register notification handler
        if hasattr(transport, 'setNotificationHandler'):
            async def _handle(p):
                if self.events:
                    await self.events.publish(Event.create('mcp.tools.changed', {'server': name}))
                try:
                    tools = await client.list_tools()
                    for t in tools:
                        toolinfo = self.convert_mcp_tool(name, t)
                        if self.tool_registry:
                            self.tool_registry.register(toolinfo)
                except Exception:
                    pass
            transport.setNotificationHandler('tool_list_changed', _handle)
        elif hasattr(transport, 'subscribe_notification'):
            transport.subscribe_notification('tool_list_changed', _handle)

    async def remove(self, name: str):
        t = self.transports.get(name)
        if t:
            await t.close()
        self.clients.pop(name, None)
        self.transports.pop(name, None)
        self.status.pop(name, None)
        self.pending_oauth.pop(name, None)

    async def list_tools(self, name: str) -> List[str]:
        client = self.clients.get(name)
        if not client:
            return []
        try:
            tools = await client.list_tools()
            return [t.get('name') for t in tools]
        except Exception:
            self.status[name] = 'failed'
            return []

    async def call_tool(self, name: str, tool: str, args: Dict[str, Any], stream: bool = False):
        client = self.clients.get(name)
        if not client:
            raise KeyError('mcp client not found')
        async for chunk in client.call_tool(tool, args, stream=stream):
            if self.events:
                await self.events.publish(Event.create('mcp.tool.progress', {'mcp': name, 'tool': tool, 'chunk': chunk}))
            yield chunk

    def convert_mcp_tool(self, mcp_name: str, mcp_tool_def: Dict[str, Any]) -> ToolInfo:
        schema = mcp_tool_def.get('inputSchema') or {}
        if isinstance(schema, dict):
            schema = {**schema, 'type': 'object', 'additionalProperties': False}
        mcp_tool_def['inputSchema'] = schema
        # Use underscores instead of dots — LLM APIs (Anthropic, OpenAI) require
        # tool names to match ^[a-zA-Z0-9_-]{1,128}$
        tool_id = f"mcp_{mcp_name}_{mcp_tool_def.get('name') or mcp_tool_def.get('id') or next_id('tool')}"
        description = mcp_tool_def.get('description') or ''
        # permissive params model
        from pydantic import BaseModel
        Params = type('Params', (BaseModel,), {'model_config': {'extra': 'allow'}})

        async def execute(args: Dict[str, Any], ctx) -> ToolResult:
            """Execute an MCP tool and map streaming progress into message.part.updated events when session context is available.
            This is a lightweight mapping: it publishes running/progress/completed events to the EventBus so clients can reflect tool progress.
            """
            output_accum = ''
            part_id = getattr(ctx, 'call_id', None) or next_id('part')
            session_id = getattr(ctx, 'session_id', None)
            message_id = getattr(ctx, 'message_id', None)

            # If we have session context, emit an initial running part update
            if session_id and message_id:
                part = {
                    'id': part_id,
                    'session_id': session_id,
                    'message_id': message_id,
                    'type': 'tool',
                    'call_id': part_id,
                    'tool': mcp_tool_def.get('name'),
                    'state': {'status': 'running', 'input': args, 'time': {'start': int(asyncio.get_event_loop().time())}},
                }
                try:
                    if self.events:
                        await self.events.publish(Event.create('message.part.updated', {'part': part}, session_id=session_id))
                except Exception:
                    pass

            # Stream from MCP client and forward progress
            last_chunk = None
            async for chunk in self.call_tool(mcp_name, mcp_tool_def.get('name'), args, stream=True):
                last_chunk = chunk
                # try to extract text-like progress
                text = None
                if isinstance(chunk, dict):
                    text = chunk.get('text') or chunk.get('output') or chunk.get('partial')
                if text is None:
                    text = str(chunk)
                output_accum += (text or '')

                # emit delta and part updated for progress when session context available
                if session_id and message_id:
                    try:
                        # delta event
                        delta = Event.create('message.part.delta', {'sessionID': session_id, 'messageID': message_id, 'partID': part_id, 'field': 'output', 'delta': text}, session_id=session_id)
                        if self.events:
                            await self.events.publish(delta)
                        # updated part
                        part = {
                            'id': part_id,
                            'session_id': session_id,
                            'message_id': message_id,
                            'type': 'tool',
                            'call_id': part_id,
                            'tool': mcp_tool_def.get('name'),
                            'state': {'status': 'running', 'input': args, 'output': output_accum, 'time': {'start': int(asyncio.get_event_loop().time())}},
                        }
                        if self.events:
                            await self.events.publish(Event.create('message.part.updated', {'part': part}, session_id=session_id))
                    except Exception:
                        pass

            # finalize output
            if isinstance(last_chunk, dict) and 'text' in last_chunk:
                output_text = last_chunk.get('text')
            else:
                output_text = output_accum or (str(last_chunk) if last_chunk is not None else '')

            # emit completed state
            if session_id and message_id:
                try:
                    part = {
                        'id': part_id,
                        'session_id': session_id,
                        'message_id': message_id,
                        'type': 'tool',
                        'call_id': part_id,
                        'tool': mcp_tool_def.get('name'),
                        'state': {
                            'status': 'completed',
                            'input': args,
                            'output': output_text,
                            'metadata': {},
                            'time': {'end': int(asyncio.get_event_loop().time())},
                        },
                    }
                    if self.events:
                        await self.events.publish(Event.create('message.part.updated', {'part': part}, session_id=session_id))
                except Exception:
                    pass

            return ToolResult(title=mcp_tool_def.get('name'), metadata={}, output=output_text, attachments=None)

        return ToolInfo(id=tool_id, description=description, parameters=Params, execute=execute)

    async def finish_auth(self, name: str, code: str):
        transport = self.pending_oauth.get(name)
        if not transport:
            raise KeyError(f'no pending oauth for {name}')
        try:
            await transport.finish_auth(code)
        except Exception as e:
            if self.events:
                await self.events.publish(Event.create('mcp.auth.failed', {'mcp': name, 'error': str(e)}))
            raise
        self.pending_oauth.pop(name, None)
        try:
            await transport.connect()
        except Exception as e:
            self.status[name] = f'failed: {e}'
            if self.events:
                await self.events.publish(Event.create('mcp.auth.failed', {'mcp': name, 'error': str(e)}))
            raise
        client = MCPClientWrapper(transport)
        self.transports[name] = transport
        self.clients[name] = client
        self.status[name] = 'connected'
        try:
            tools = await client.list_tools()
            for t in tools:
                toolinfo = self.convert_mcp_tool(name, t)
                if self.tool_registry:
                    self.tool_registry.register(toolinfo)
        except Exception:
            pass
        if self.events:
            await self.events.publish(Event.create('mcp.auth.finished', {'mcp': name}))
            await self.events.publish(Event.create('mcp.tools.changed', {'server': name}))
        return self.status.get(name)
