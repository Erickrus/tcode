"""Tests for AgentRunner multi-turn tool loop."""
from __future__ import annotations
import asyncio
import json
import pytest
from typing import List, Dict, Any, Optional, AsyncIterator
from tcode.providers.base import ProviderAdapter, ProviderChunk
from tcode.providers.factory import ProviderFactory
from tcode.tools import ToolRegistry, ToolInfo, ToolResult, ToolContext
from tcode.mcp import MCPManager
from tcode.storage import Storage
from tcode.event import EventBus
from tcode.session import SessionManager
from tcode.attachments import AttachmentStore
from tcode.agent import AgentRunner
from pydantic import BaseModel


# --- Mock provider that simulates LLM behavior ---

class MockAdapter(ProviderAdapter):
    """A mock provider that returns scripted responses.

    `script` is a list of turn responses. Each turn is either:
      - {"text": "some text"}                    -> yields text delta + final
      - {"tool_calls": [{"id", "name", "arguments"}]}  -> yields tool_call_end chunks
      - {"text": "...", "tool_calls": [...]}     -> both
    """

    def __init__(self, script: List[Dict[str, Any]]):
        self.script = script
        self.call_count = 0

    async def chat(self, messages, model, options, tools=None):
        result = {"text": ""}
        async for chunk in self.chat_stream(messages, model, options, tools=tools):
            if chunk.get("type") == "delta":
                result["text"] += chunk.get("text", "")
        return result

    async def chat_stream(self, messages, model, options, tools=None) -> AsyncIterator[ProviderChunk]:
        if self.call_count >= len(self.script):
            # No more scripted responses — just finish
            yield ProviderChunk({"type": "final", "data": {"text": ""}})
            return

        turn = self.script[self.call_count]
        self.call_count += 1

        # Yield text
        text = turn.get("text", "")
        if text:
            yield ProviderChunk({"type": "delta", "text": text})

        # Yield tool calls
        for tc in turn.get("tool_calls", []):
            yield ProviderChunk({
                "type": "tool_call_end",
                "id": tc["id"],
                "name": tc["name"],
                "arguments": tc.get("arguments", {}),
            })

        yield ProviderChunk({"type": "final", "data": {"text": text}})

    def supports_tools(self):
        return True

    def get_model(self, model_id):
        return {"id": model_id}


# --- Test tools ---

class EchoParams(BaseModel):
    text: str

async def echo_execute(args, ctx):
    return ToolResult(title="echo", output=args.get("text", ""), metadata={})

class UpperParams(BaseModel):
    text: str

async def upper_execute(args, ctx):
    return ToolResult(title="upper", output=args.get("text", "").upper(), metadata={})


# --- Fixtures ---

@pytest.fixture
def setup():
    """Create all components for testing."""
    storage = Storage()
    events = EventBus()
    sessions = SessionManager(storage=storage, events=events)
    tools = ToolRegistry()
    tools.register(ToolInfo(
        id="echo", description="Echo text back",
        parameters=EchoParams, execute=echo_execute
    ))
    tools.register(ToolInfo(
        id="upper", description="Uppercase text",
        parameters=UpperParams, execute=upper_execute
    ))
    # Store allow-all rules helper
    sessions._default_allow_rules = [
        {"permission": "*", "pattern": "*", "action": "allow"},
    ]
    return storage, events, sessions, tools


def make_runner(sessions, events, tools, script):
    """Create an AgentRunner with a mock provider."""
    factory = ProviderFactory()
    mock = MockAdapter(script)
    factory._constructors["mock"] = lambda **kw: mock
    factory._instances["mock"] = mock
    mcp = MCPManager()
    attachments = AttachmentStore()
    runner = AgentRunner(factory, tools, mcp, sessions, events, attachments)
    return runner, mock


async def _allow_all(sessions, sid):
    """Set allow-all permission rules on a session."""
    await sessions.set_permission(sid, [
        {"permission": "*", "pattern": "*", "action": "allow"},
    ])


# --- Tests ---

@pytest.mark.asyncio
async def test_simple_text_response(setup):
    """LLM returns just text, no tool calls — single iteration."""
    storage, events, sessions, tools = setup
    sid = await sessions.create_session()
    uid = await sessions.create_message(sid, "user")
    await sessions.append_text_part(sid, uid, "Hello")

    runner, mock = make_runner(sessions, events, tools, [
        {"text": "Hi there!"},
    ])

    result = await runner.run("mock", "test-model", sid, uid)
    assert result["final_text"] == "Hi there!"
    assert result["steps"] == 1
    assert mock.call_count == 1


@pytest.mark.asyncio
async def test_tool_call_then_response(setup):
    """LLM calls a tool, gets result, then responds with text."""
    storage, events, sessions, tools = setup
    sid = await sessions.create_session()
    await _allow_all(sessions, sid)
    uid = await sessions.create_message(sid, "user")
    await sessions.append_text_part(sid, uid, "Echo hello")

    runner, mock = make_runner(sessions, events, tools, [
        # Turn 1: LLM calls echo tool
        {"tool_calls": [{"id": "call_1", "name": "echo", "arguments": {"text": "hello"}}]},
        # Turn 2: LLM sees tool result and responds
        {"text": "The echo returned: hello"},
    ])

    result = await runner.run("mock", "test-model", sid, uid)
    assert result["final_text"] == "The echo returned: hello"
    assert result["steps"] == 2
    assert mock.call_count == 2


@pytest.mark.asyncio
async def test_multiple_tool_calls_then_response(setup):
    """LLM calls multiple tools in sequence, then responds."""
    storage, events, sessions, tools = setup
    sid = await sessions.create_session()
    await _allow_all(sessions, sid)
    uid = await sessions.create_message(sid, "user")
    await sessions.append_text_part(sid, uid, "Process text")

    runner, mock = make_runner(sessions, events, tools, [
        # Turn 1: call echo
        {"tool_calls": [{"id": "call_1", "name": "echo", "arguments": {"text": "abc"}}]},
        # Turn 2: call upper on the result
        {"tool_calls": [{"id": "call_2", "name": "upper", "arguments": {"text": "abc"}}]},
        # Turn 3: respond
        {"text": "Done: ABC"},
    ])

    result = await runner.run("mock", "test-model", sid, uid)
    assert result["final_text"] == "Done: ABC"
    assert result["steps"] == 3


@pytest.mark.asyncio
async def test_parallel_tool_calls(setup):
    """LLM calls two tools in parallel in one turn."""
    storage, events, sessions, tools = setup
    sid = await sessions.create_session()
    await _allow_all(sessions, sid)
    uid = await sessions.create_message(sid, "user")
    await sessions.append_text_part(sid, uid, "Do two things")

    runner, mock = make_runner(sessions, events, tools, [
        # Turn 1: two parallel tool calls
        {"tool_calls": [
            {"id": "call_a", "name": "echo", "arguments": {"text": "one"}},
            {"id": "call_b", "name": "upper", "arguments": {"text": "two"}},
        ]},
        # Turn 2: respond
        {"text": "Got one and TWO"},
    ])

    result = await runner.run("mock", "test-model", sid, uid)
    assert result["final_text"] == "Got one and TWO"
    assert result["steps"] == 2


@pytest.mark.asyncio
async def test_max_steps_prevents_infinite_loop(setup):
    """Agent loop stops after max_steps even if LLM keeps calling tools."""
    storage, events, sessions, tools = setup
    sid = await sessions.create_session()
    await _allow_all(sessions, sid)
    uid = await sessions.create_message(sid, "user")
    await sessions.append_text_part(sid, uid, "Loop forever")

    # Create a script that always calls tools (more turns than max_steps)
    infinite_script = [
        {"tool_calls": [{"id": f"call_{i}", "name": "echo", "arguments": {"text": "x"}}]}
        for i in range(100)
    ]
    runner, mock = make_runner(sessions, events, tools, infinite_script)

    result = await runner.run("mock", "test-model", sid, uid, max_steps=3)
    assert result["steps"] == 3  # Stopped at max


@pytest.mark.asyncio
async def test_tool_schemas_passed_to_provider(setup):
    """Verify tool schemas are built and passed to the adapter."""
    storage, events, sessions, tools = setup
    sid = await sessions.create_session()
    uid = await sessions.create_message(sid, "user")
    await sessions.append_text_part(sid, uid, "Hello")

    # Use a custom adapter that captures the tools parameter
    captured_tools = []

    class CapturingAdapter(ProviderAdapter):
        async def chat(self, messages, model, options, tools=None):
            return {"text": "ok"}
        async def chat_stream(self, messages, model, options, tools=None):
            captured_tools.append(tools)
            yield ProviderChunk({"type": "delta", "text": "ok"})
            yield ProviderChunk({"type": "final", "data": {"text": "ok"}})
        def get_model(self, model_id):
            return {"id": model_id}

    factory = ProviderFactory()
    cap = CapturingAdapter()
    factory._constructors["cap"] = lambda **kw: cap
    factory._instances["cap"] = cap
    mcp = MCPManager()
    attachments = AttachmentStore()
    runner = AgentRunner(factory, tools, mcp, sessions, events, attachments)

    await runner.run("cap", "test-model", sid, uid)

    assert len(captured_tools) == 1
    tools_list = captured_tools[0]
    assert len(tools_list) == 2
    names = {t["name"] for t in tools_list}
    assert "echo" in names
    assert "upper" in names
    # Check schema structure
    for t in tools_list:
        assert "description" in t
        assert "parameters" in t
        assert t["parameters"].get("type") == "object"


@pytest.mark.asyncio
async def test_compose_messages_includes_tool_history(setup):
    """After a tool call, the next LLM call should see tool call + result in messages."""
    storage, events, sessions, tools = setup
    sid = await sessions.create_session()
    uid = await sessions.create_message(sid, "user")
    await sessions.append_text_part(sid, uid, "Echo test")

    captured_messages = []

    class CapturingAdapter(ProviderAdapter):
        def __init__(self):
            self.call_count = 0

        async def chat(self, messages, model, options, tools=None):
            return {"text": "ok"}

        async def chat_stream(self, messages, model, options, tools=None):
            self.call_count += 1
            captured_messages.append(list(messages))
            if self.call_count == 1:
                # First call: request tool
                yield ProviderChunk({
                    "type": "tool_call_end",
                    "id": "call_1", "name": "echo",
                    "arguments": {"text": "hello"},
                })
                yield ProviderChunk({"type": "final", "data": {"text": ""}})
            else:
                # Second call: just respond
                yield ProviderChunk({"type": "delta", "text": "Done"})
                yield ProviderChunk({"type": "final", "data": {"text": "Done"}})

        def get_model(self, model_id):
            return {"id": model_id}

    factory = ProviderFactory()
    cap = CapturingAdapter()
    factory._constructors["cap"] = lambda **kw: cap
    factory._instances["cap"] = cap
    mcp = MCPManager()
    attachments = AttachmentStore()
    runner = AgentRunner(factory, tools, mcp, sessions, events, attachments)

    # Set allow-all permissions so tools can execute
    await sessions.set_permission(sid, [{"permission": "*", "pattern": "*", "action": "allow"}])

    await runner.run("cap", "test-model", sid, uid)

    # Second call should have tool call + tool result in messages
    assert len(captured_messages) == 2
    second_call_msgs = captured_messages[1]

    # Find the assistant message with tool_calls
    assistant_with_tools = [m for m in second_call_msgs
                           if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(assistant_with_tools) >= 1

    # Find tool result message
    tool_results = [m for m in second_call_msgs if m.get("role") == "tool"]
    assert len(tool_results) >= 1
    assert tool_results[0]["tool_call_id"] == "call_1"
    # The echo tool returns the text as-is
    assert "hello" in tool_results[0]["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
