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
                        return r.json()
                    except Exception:
                        return {"text": await r.aread()}
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
    def __init__(self, command: str, args: Optional[list] = None, cwd: Optional[str] = None, env: Optional[dict] = None):
        self.command = command
        self.args = args or []
        self.cwd = cwd
        self.env = env
        self.proc: Optional[subprocess.Popen] = None
        self._handlers: Dict[str, Callable] = {}
        self._reader_task: Optional[asyncio.Task] = None

    async def connect(self):
        # spawn subprocess with pipes
        self.proc = subprocess.Popen([self.command] + self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.cwd, env=self.env, text=True)
        loop = asyncio.get_event_loop()
        def read_stdout():
            if not self.proc or not self.proc.stdout:
                return
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                ntype = obj.get('type')
                if ntype and ntype in self._handlers:
                    asyncio.run_coroutine_threadsafe(self._handlers[ntype](obj), loop)
        self._reader_task = asyncio.get_event_loop().run_in_executor(None, read_stdout)

    async def close(self):
        if self.proc:
            self.proc.terminate()
            self.proc = None
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None

    async def finish_auth(self, code: str):
        # send code via stdin if subprocess supports it
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.write(code + "\n")
                self.proc.stdin.flush()
            except Exception:
                pass

    def subscribe_notification(self, notification: str, handler: Callable[[Dict[str,Any]], Awaitable[None]]):
        self._handlers[notification] = handler
