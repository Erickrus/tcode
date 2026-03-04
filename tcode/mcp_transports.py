from __future__ import annotations
import httpx
import asyncio
import subprocess
from typing import Callable, Dict, Any, Optional, Awaitable
import json

class TransportError(Exception):
    pass

class UnauthorizedError(TransportError):
    def __init__(self, message: str, authorization_url: Optional[str] = None):
        super().__init__(message)
        self.authorization_url = authorization_url

class Transport:
    async def connect(self):
        raise NotImplementedError
    async def close(self):
        raise NotImplementedError
    async def finish_auth(self, code: str):
        raise NotImplementedError
    def subscribe_notification(self, notification: str, handler: Callable[[Dict[str,Any]], Awaitable[None]]):
        raise NotImplementedError
    # SDK-compatible name
    def setNotificationHandler(self, schema: str, handler: Callable[[Dict[str,Any]], Awaitable[None]]):
        return self.subscribe_notification(schema, handler)
    # Optional call_tool interface
    async def call_tool(self, name: str, args: Dict[str,Any], stream: bool = False):
        raise NotImplementedError

class HTTPStreamTransport(Transport):
    def __init__(self, base_url: str, headers: Optional[Dict[str,str]] = None, timeout: int = 30, auth_provider=None):
        self.base_url = base_url.rstrip('/')
        self.headers = headers or {}
        self.timeout = timeout
        self.auth_provider = auth_provider
        self._client: Optional[httpx.AsyncClient] = None
        self._notify_handlers: Dict[str, Callable] = {}

    async def connect(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        # Simple health check
        url = f"{self.base_url}/health" if self.base_url else None
        if url:
            try:
                r = await self._client.get(url, headers=self.headers)
                if r.status_code == 401:
                    # extract authorization url if present
                    raise UnauthorizedError("unauthorized", authorization_url=r.headers.get('WWW-Authenticate'))
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                # Some MCP servers don't have /health; ignore 404
                if e.response.status_code == 404:
                    return
                raise

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def finish_auth(self, code: str):
        # If auth_provider provided, call its finish method
        if self.auth_provider and hasattr(self.auth_provider, 'finish_auth'):
            await self.auth_provider.finish_auth(code)

    def subscribe_notification(self, notification: str, handler: Callable[[Dict[str,Any]], Awaitable[None]]):
        self._notify_handlers[notification] = handler
    def setNotificationHandler(self, schema: str, handler: Callable[[Dict[str,Any]], Awaitable[None]]):
        # SDK-compatible alias
        return self.subscribe_notification(schema, handler)

    async def call_tool(self, name: str, args: Dict[str,Any], stream: bool = False):
        """Call a remote MCP tool via HTTP. If stream=True, yields streaming chunks (parsed JSON per line).
        Expected endpoint: POST {base_url}/call with payload {"tool": name, "args": args} or POST {base_url}/tools/{name}/call
        """
        if not self._client:
            raise TransportError('client not connected')
        # try tools/{name}/call first
        urls = [f"{self.base_url}/tools/{name}/call", f"{self.base_url}/call"]
        payload = {"tool": name, "args": args}
        for url in urls:
            try:
                if stream:
                    async with self._client.stream("POST", url, headers=self.headers, json=payload) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            # try to parse each line as a JSON chunk
                            try:
                                data = json.loads(line)
                            except Exception:
                                # try to peel 'data:' prefixes
                                parts = [p.strip() for p in line.split('data:') if p.strip()]
                                for part in parts:
                                    try:
                                        data = json.loads(part)
                                    except Exception:
                                        continue
                                    yield data
                                continue
                            yield data
                    return
                else:
                    r = await self._client.post(url, headers=self.headers, json=payload)
                    r.raise_for_status()
                    try:
                        yield r.json()
                    except Exception:
                        yield {"text": await r.aread()}
                    return
            except Exception:
                # try next url
                continue
        raise TransportError('call_tool failed: no reachable endpoint')

class SSETransport(Transport):
    def __init__(self, base_url: str, headers: Optional[Dict[str,str]] = None, timeout: int = 30, auth_provider=None):
        self.base_url = base_url.rstrip('/')
        self.headers = headers or {}
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=self.timeout)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._handlers: Dict[str, Callable] = {}

    async def connect(self):
        # start background task that reads events from /events
        self._running = True
        async def run():
            url = f"{self.base_url}/events"
            async with self._client.stream('GET', url, headers=self.headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line: continue
                    parts = [p.strip() for p in line.split('data:') if p.strip()]
                    for part in parts:
                        try:
                            payload = json.loads(part)
                        except Exception:
                            continue
                        # dispatch to handlers by notification type if present
                        ntype = payload.get('type')
                        if ntype and ntype in self._handlers:
                            asyncio.create_task(self._handlers[ntype](payload))
        self._task = asyncio.create_task(run())

    async def close(self):
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        await self._client.aclose()

    async def finish_auth(self, code: str):
        # no-op for SSE
        return

    def subscribe_notification(self, notification: str, handler: Callable[[Dict[str,Any]], Awaitable[None]]):
        self._handlers[notification] = handler

class StdioTransport(Transport):
    """MCP stdio transport using JSON-RPC over stdin/stdout.

    Speaks the MCP protocol:
      - initialize handshake
      - tools/list to enumerate tools
      - tools/call to invoke a tool
    """

    def __init__(self, command: str, args: Optional[list] = None, cwd: Optional[str] = None, env: Optional[dict] = None):
        self.command = command
        self.args = args or []
        self.cwd = cwd
        self.env = env
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._handlers: Dict[str, Callable] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._next_id = 1
        self._read_buf = b""

    # ---- low-level JSON-RPC over stdio ----

    async def _send(self, msg: dict):
        """Send a JSON-RPC message as newline-delimited JSON."""
        if not self._proc or not self._proc.stdin:
            raise TransportError("stdio process not running")
        body = json.dumps(msg).encode("utf-8") + b"\n"
        self._proc.stdin.write(body)
        await self._proc.stdin.drain()

    async def _read_message(self) -> Optional[dict]:
        """Read one JSON-RPC message from stdout.

        Supports both newline-delimited JSON (MCP Python SDK / FastMCP)
        and Content-Length framed (LSP-style) messages.
        """
        if not self._proc or not self._proc.stdout:
            return None
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                return None  # EOF
            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue  # skip blank lines
            # Content-Length framed message (LSP-style)
            if line_str.startswith("Content-Length:"):
                length = int(line_str.split(":", 1)[1].strip())
                # consume blank line after headers
                while True:
                    sep = await self._proc.stdout.readline()
                    if sep.strip() == b"":
                        break
                body = await self._proc.stdout.readexactly(length)
                return json.loads(body)
            # Newline-delimited JSON
            try:
                return json.loads(line_str)
            except json.JSONDecodeError:
                continue  # skip non-JSON lines (e.g. server log output)

    async def _request(self, method: str, params: Optional[dict] = None) -> Any:
        """Send a JSON-RPC request and wait for the response."""
        rid = self._next_id
        self._next_id += 1
        msg = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[rid] = future
        await self._send(msg)

        # read messages until we get our response
        while not future.done():
            resp = await self._read_message()
            if resp is None:
                future.set_exception(TransportError("stdio process closed"))
                break
            resp_id = resp.get("id")
            if resp_id is not None and resp_id in self._pending:
                pending_future = self._pending.pop(resp_id)
                if "error" in resp:
                    pending_future.set_exception(
                        TransportError(f"JSON-RPC error: {resp['error']}")
                    )
                else:
                    pending_future.set_result(resp.get("result"))
            elif resp.get("method") == "notifications/message":
                # server notification — dispatch if handler registered
                ntype = (resp.get("params") or {}).get("type")
                if ntype and ntype in self._handlers:
                    asyncio.create_task(self._handlers[ntype](resp.get("params", {})))

        return future.result()

    async def _notify(self, method: str, params: Optional[dict] = None):
        """Send a JSON-RPC notification (no id, no response expected)."""
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._send(msg)

    # ---- Transport interface ----

    async def connect(self):
        self._proc = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=self.env,
        )
        # MCP initialize handshake (with timeout to avoid hanging forever)
        try:
            result = await asyncio.wait_for(self._request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "tcode", "version": "0.1.0"},
            }), timeout=30)
        except asyncio.TimeoutError:
            # collect any stderr output for diagnostics
            stderr_out = ""
            if self._proc and self._proc.stderr:
                try:
                    stderr_out = (await asyncio.wait_for(
                        self._proc.stderr.read(4096), timeout=1
                    )).decode("utf-8", errors="replace")
                except Exception:
                    pass
            await self.close()
            raise TransportError(
                f"Timeout waiting for MCP server to respond to initialize. "
                f"Command: {self.command} {' '.join(self.args)}"
                + (f"\nstderr: {stderr_out}" if stderr_out else "")
            )
        # send initialized notification
        await self._notify("notifications/initialized")

    async def close(self):
        if self._proc:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    async def finish_auth(self, code: str):
        pass  # stdio transports don't use OAuth

    def subscribe_notification(self, notification: str, handler: Callable[[Dict[str,Any]], Awaitable[None]]):
        self._handlers[notification] = handler

    # ---- MCP tool interface ----

    async def list_tools(self) -> list:
        """Call tools/list and return the tool definitions."""
        result = await self._request("tools/list")
        return (result or {}).get("tools", [])

    async def call_tool(self, name: str, args: Dict[str, Any], stream: bool = False):
        """Call tools/call and yield the result."""
        result = await self._request("tools/call", {"name": name, "arguments": args})
        # MCP tools/call returns {"content": [{"type":"text","text":"..."},...]}
        content = (result or {}).get("content", [])
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        output = "\n".join(text_parts) if text_parts else json.dumps(result)
        yield {"text": output}
