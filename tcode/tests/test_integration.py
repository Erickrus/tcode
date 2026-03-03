"""Integration test: Full agent loop with mock provider, real tools, session, permissions.

This test verifies the complete flow:
  1. Create session with permissions
  2. Send user message
  3. Agent calls provider -> provider requests tool use -> agent executes tool -> loops
  4. Agent returns final response with tool results in message history
  5. All parts stored correctly in session
"""
from __future__ import annotations
import json
import pytest
from typing import List, Dict, Any, AsyncIterator
from pydantic import BaseModel
from tcode.providers.base import ProviderAdapter, ProviderChunk
from tcode.providers.factory import ProviderFactory
from tcode.tools import ToolRegistry, ToolInfo, ToolResult, ToolContext
from tcode.mcp import MCPManager
from tcode.storage import Storage
from tcode.event import EventBus, Event
from tcode.session import SessionManager
from tcode.attachments import AttachmentStore
from tcode.agent import AgentRunner, _tool_to_schema


# --- Realistic mock provider ---

class RealisticMockAdapter(ProviderAdapter):
    """Simulates a real LLM that:
    1. First call: reads the user's question, calls read_file tool
    2. Second call: sees file content, calls grep tool
    3. Third call: has all info, gives final text answer
    """

    def __init__(self):
        self.call_count = 0
        self.received_messages: List[List[Dict]] = []
        self.received_tools: List = []

    async def chat(self, messages, model, options, tools=None):
        return {}

    async def chat_stream(self, messages, model, options, tools=None):
        self.call_count += 1
        self.received_messages.append(list(messages))
        self.received_tools.append(tools)

        if self.call_count == 1:
            # First call: LLM decides to read a file
            yield ProviderChunk({"type": "delta", "text": "I'll read the file for you. "})
            yield ProviderChunk({
                "type": "tool_call_end",
                "id": "call_read",
                "name": "read_file",
                "arguments": {"path": "/tmp/test_integration_file.txt"},
            })
            yield ProviderChunk({"type": "final", "data": {"text": ""}})

        elif self.call_count == 2:
            # Second call: LLM has file content, now searches it
            yield ProviderChunk({
                "type": "tool_call_end",
                "id": "call_grep",
                "name": "search",
                "arguments": {"text": "hello"},
            })
            yield ProviderChunk({"type": "final", "data": {"text": ""}})

        else:
            # Third call: LLM gives final answer
            yield ProviderChunk({
                "type": "delta",
                "text": "The file contains 'hello world' and I found a match for 'hello'.",
            })
            yield ProviderChunk({"type": "final", "data": {"text": ""}})

    def supports_tools(self):
        return True

    def get_model(self, model_id):
        return {"id": model_id}


# --- Test tools ---

class ReadFileParams(BaseModel):
    path: str

async def mock_read_file(args, ctx):
    path = args.get("path", "")
    if "test_integration" in path:
        return ToolResult(title="read_file", output="hello world\nfoo bar", metadata={"path": path})
    return ToolResult(title="read_file", output=f"Error: file not found: {path}", metadata={})

class SearchParams(BaseModel):
    text: str

async def mock_search(args, ctx):
    text = args.get("text", "")
    return ToolResult(title="search", output=f"Found 1 match for '{text}'", metadata={"matches": 1})


# --- Test ---

@pytest.mark.asyncio
async def test_full_agent_loop_integration():
    """End-to-end test: user asks question -> agent calls 2 tools -> gives answer."""
    # Setup
    storage = Storage()
    events = EventBus()
    sessions = SessionManager(storage=storage, events=events)
    tools = ToolRegistry()
    tools.register(ToolInfo(id="read_file", description="Read a file", parameters=ReadFileParams, execute=mock_read_file))
    tools.register(ToolInfo(id="search", description="Search text", parameters=SearchParams, execute=mock_search))

    factory = ProviderFactory()
    mock = RealisticMockAdapter()
    factory._constructors["mock"] = lambda **kw: mock
    factory._instances["mock"] = mock
    mcp = MCPManager()
    attachments = AttachmentStore()
    runner = AgentRunner(factory, tools, mcp, sessions, events, attachments)

    # Create session with allow-all permissions
    sid = await sessions.create_session(metadata={
        "system_prompt": "You are a helpful coding assistant.",
    })
    await sessions.set_permission(sid, [{"permission": "*", "pattern": "*", "action": "allow"}])

    # User message
    uid = await sessions.create_message(sid, "user")
    await sessions.append_text_part(sid, uid, "What's in /tmp/test_integration_file.txt?")

    # Collect events
    collected_events: List[Event] = []
    events.subscribe("*", lambda e: collected_events.append(e))

    # Run agent
    result = await runner.run(
        "mock", "test-model", sid, uid,
        system_prompt="You are a helpful coding assistant.",
        max_steps=10,
    )

    # --- Assertions ---

    # 1. Agent made 3 calls (read_file, search, final text)
    assert result["steps"] == 3
    assert mock.call_count == 3

    # 2. Final text contains the answer
    assert "hello world" in result["final_text"] or "match" in result["final_text"]

    # 3. Tool schemas were passed to the provider
    assert mock.received_tools[0] is not None
    tool_names = {t["name"] for t in mock.received_tools[0]}
    assert "read_file" in tool_names
    assert "search" in tool_names

    # 4. Second call saw tool results in messages
    second_msgs = mock.received_messages[1]
    # Should contain system, user, assistant(tool_call), tool(result)
    roles = [m["role"] for m in second_msgs]
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles
    assert "tool" in roles
    # Tool result should contain the file content
    tool_results = [m for m in second_msgs if m["role"] == "tool"]
    assert any("hello world" in m.get("content", "") for m in tool_results)

    # 5. Third call saw both tool results
    third_msgs = mock.received_messages[2]
    tool_results_3 = [m for m in third_msgs if m["role"] == "tool"]
    assert len(tool_results_3) >= 2
    contents = [m["content"] for m in tool_results_3]
    assert any("hello world" in c for c in contents)
    assert any("Found 1 match" in c for c in contents)

    # 6. Session messages stored correctly
    all_msgs = await sessions.compose_messages(sid)
    # All tool calls are on one assistant message, so structure is:
    # system, user, assistant(2 tool_calls + text), tool(read_file), tool(search)
    non_sys = [m for m in all_msgs if m["role"] != "system"]
    assert len(non_sys) >= 4  # user + assistant(with tool_calls) + 2 tool results

    # 7. Events were emitted
    event_types = {e.type for e in collected_events}
    assert "message.updated" in event_types or "message.part.updated" in event_types
    assert "session.status.changed" in event_types

    # 8. System prompt was in the messages
    assert second_msgs[0]["role"] == "system"
    assert "helpful coding assistant" in second_msgs[0]["content"]


@pytest.mark.asyncio
async def test_tool_schema_generation():
    """Verify _tool_to_schema produces correct JSON schemas from Pydantic models."""
    class MyParams(BaseModel):
        file_path: str
        max_lines: int = 100
        include_hidden: bool = False

    async def noop(args, ctx):
        return ToolResult(output="")

    tool = ToolInfo(id="my_tool", description="A test tool", parameters=MyParams, execute=noop)
    schema = _tool_to_schema(tool)

    assert schema["name"] == "my_tool"
    assert schema["description"] == "A test tool"
    props = schema["parameters"]["properties"]
    assert "file_path" in props
    assert "max_lines" in props
    assert "include_hidden" in props
    assert props["file_path"]["type"] == "string"
    assert props["max_lines"]["type"] == "integer"
    assert props["include_hidden"]["type"] == "boolean"


@pytest.mark.asyncio
async def test_permission_denied_stops_tool():
    """When permissions deny a tool, the tool part should show error."""
    storage = Storage()
    events = EventBus()
    sessions = SessionManager(storage=storage, events=events)
    tools = ToolRegistry()
    tools.register(ToolInfo(id="dangerous", description="Dangerous op",
                            parameters=BaseModel, execute=lambda a, c: ToolResult(output="should not run")))

    class DenyAdapter(ProviderAdapter):
        async def chat(self, messages, model, options, tools=None):
            return {}
        async def chat_stream(self, messages, model, options, tools=None):
            yield ProviderChunk({
                "type": "tool_call_end", "id": "call_1",
                "name": "dangerous", "arguments": {},
            })
            yield ProviderChunk({"type": "final", "data": {"text": ""}})
        def get_model(self, model_id):
            return {"id": model_id}

    factory = ProviderFactory()
    deny = DenyAdapter()
    factory._constructors["deny"] = lambda **kw: deny
    factory._instances["deny"] = deny
    mcp = MCPManager()
    attachments = AttachmentStore()
    runner = AgentRunner(factory, tools, mcp, sessions, events, attachments)

    sid = await sessions.create_session()
    # Set deny-all permissions
    await sessions.set_permission(sid, [{"permission": "*", "pattern": "*", "action": "deny"}])
    uid = await sessions.create_message(sid, "user")
    await sessions.append_text_part(sid, uid, "Do dangerous thing")

    result = await runner.run("deny", "test-model", sid, uid, max_steps=3)

    # The tool should have been denied — check that tool part has error state
    msgs = await sessions.compose_messages(sid)
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    # Tool result should show denied
    assert any("denied" in m.get("content", "").lower() or "error" in m.get("content", "").lower()
               for m in tool_msgs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
