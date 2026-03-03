"""Tests for tcode abort/cancellation mechanism."""
from __future__ import annotations
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tcode.agent import AgentRunner
from tcode.providers.base import ProviderChunk
from tcode.tools import ToolRegistry, ToolInfo, ToolResult
from tcode.storage import Storage
from tcode.event import EventBus
from tcode.session import SessionManager
from tcode.mcp import MCPManager
from tcode.attachments import AttachmentStore
from tcode.providers.factory import ProviderFactory
from pydantic import BaseModel


class EchoParams(BaseModel):
    text: str


async def echo_execute(args, ctx):
    return ToolResult(title="echo", output=f"Echo: {args.get('text', '')}", metadata={})


def _make_runner():
    storage = Storage()
    events = EventBus()
    sessions = SessionManager(storage=storage, events=events)
    tools = ToolRegistry()
    tools.register(ToolInfo(
        id="echo", description="Echo", parameters=EchoParams, execute=echo_execute
    ))
    factory = ProviderFactory()
    mcp = MCPManager()
    attachments = AttachmentStore()
    runner = AgentRunner(factory, tools, mcp, sessions, events, attachments)
    return runner, sessions, factory


def _make_chunks(chunks_list):
    """Create an async generator from a list of chunk dicts."""
    async def gen(*args, **kwargs):
        for c in chunks_list:
            yield ProviderChunk(c)
    return gen


@pytest.mark.asyncio
async def test_abort_stops_agent_loop():
    """If abort_event is set, agent loop should stop."""
    runner, sessions, factory = _make_runner()

    # Simulate: tool call, then text response
    call_chunks = [
        {"type": "tool_call_end", "id": "c1", "name": "echo", "arguments": {"text": "hi"}},
        {"type": "final", "data": {}},
    ]
    text_chunks = [
        {"type": "delta", "text": "Done"},
        {"type": "final", "data": {}},
    ]

    call_count = 0
    async def mock_stream(messages, model, options, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            for c in call_chunks:
                yield ProviderChunk(c)
        else:
            for c in text_chunks:
                yield ProviderChunk(c)

    mock_adapter = MagicMock()
    mock_adapter.chat_stream = mock_stream
    factory.register_adapter("mock", lambda **kw: mock_adapter)
    factory._instances["mock"] = mock_adapter

    sid = await sessions.create_session()
    await sessions.set_permission(sid, [{"permission": "*", "pattern": "*", "action": "allow"}])
    uid = await sessions.create_message(sid, "user")
    await sessions.append_text_part(sid, uid, "Say echo hi")

    # Create abort event and set it immediately
    abort = asyncio.Event()
    abort.set()

    result = await runner.run("mock", "test-model", sid, uid, abort_event=abort)
    # Should stop quickly — 0 or 1 steps
    assert result["steps"] <= 1


@pytest.mark.asyncio
async def test_abort_during_streaming():
    """Abort during streaming should break the loop."""
    runner, sessions, factory = _make_runner()
    abort = asyncio.Event()

    async def slow_stream(messages, model, options, tools=None):
        yield ProviderChunk({"type": "delta", "text": "Start "})
        # Simulate abort during streaming
        abort.set()
        yield ProviderChunk({"type": "delta", "text": "should stop"})
        yield ProviderChunk({"type": "final", "data": {}})

    mock_adapter = MagicMock()
    mock_adapter.chat_stream = slow_stream
    factory.register_adapter("mock", lambda **kw: mock_adapter)
    factory._instances["mock"] = mock_adapter

    sid = await sessions.create_session()
    await sessions.set_permission(sid, [{"permission": "*", "pattern": "*", "action": "allow"}])
    uid = await sessions.create_message(sid, "user")
    await sessions.append_text_part(sid, uid, "Test abort")

    result = await runner.run("mock", "test-model", sid, uid, abort_event=abort)
    assert result["steps"] == 1


@pytest.mark.asyncio
async def test_no_abort_runs_normally():
    """Without abort, agent loop runs to completion."""
    runner, sessions, factory = _make_runner()

    async def simple_stream(messages, model, options, tools=None):
        yield ProviderChunk({"type": "delta", "text": "Hello"})
        yield ProviderChunk({"type": "final", "data": {}})

    mock_adapter = MagicMock()
    mock_adapter.chat_stream = simple_stream
    factory.register_adapter("mock", lambda **kw: mock_adapter)
    factory._instances["mock"] = mock_adapter

    sid = await sessions.create_session()
    uid = await sessions.create_message(sid, "user")
    await sessions.append_text_part(sid, uid, "Say hello")

    result = await runner.run("mock", "test-model", sid, uid, abort_event=None)
    assert result["final_text"] == "Hello"
    assert result["steps"] == 1


@pytest.mark.asyncio
async def test_agent_with_named_agent():
    """Test that agent_name parameter loads agent config."""
    runner, sessions, factory = _make_runner()

    async def simple_stream(messages, model, options, tools=None):
        yield ProviderChunk({"type": "delta", "text": "Explored"})
        yield ProviderChunk({"type": "final", "data": {}})

    mock_adapter = MagicMock()
    mock_adapter.chat_stream = simple_stream
    factory.register_adapter("mock", lambda **kw: mock_adapter)
    factory._instances["mock"] = mock_adapter

    sid = await sessions.create_session()
    uid = await sessions.create_message(sid, "user")
    await sessions.append_text_part(sid, uid, "Explore something")

    # Use the "explore" agent — it has a specific prompt
    result = await runner.run("mock", "test-model", sid, uid, agent_name="explore")
    assert result["final_text"] == "Explored"
