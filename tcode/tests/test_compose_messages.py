"""Tests for SessionManager.compose_messages() — verifies tool calls and results
are properly included in provider-format messages."""
from __future__ import annotations
import asyncio
import json
import pytest
from tcode.storage import Storage
from tcode.event import EventBus
from tcode.session import SessionManager


@pytest.fixture
def sm():
    """Create a SessionManager with in-memory storage."""
    storage = Storage()
    events = EventBus()
    return SessionManager(storage=storage, events=events)


@pytest.mark.asyncio
async def test_basic_text_messages(sm):
    """User text + assistant text compose correctly."""
    sid = await sm.create_session()
    # User message
    uid = await sm.create_message(sid, "user")
    await sm.append_text_part(sid, uid, "Hello")
    # Assistant message
    aid = await sm.create_message(sid, "assistant")
    await sm.append_text_part(sid, aid, "Hi there!")

    msgs = await sm.compose_messages(sid)
    # Filter out system messages
    non_sys = [m for m in msgs if m["role"] != "system"]
    assert len(non_sys) == 2
    assert non_sys[0] == {"role": "user", "content": "Hello"}
    assert non_sys[1] == {"role": "assistant", "content": "Hi there!"}


@pytest.mark.asyncio
async def test_system_prompt(sm):
    """System prompt from session metadata is prepended."""
    sid = await sm.create_session(metadata={"system_prompt": "You are helpful."})
    uid = await sm.create_message(sid, "user")
    await sm.append_text_part(sid, uid, "Hi")

    msgs = await sm.compose_messages(sid)
    assert msgs[0] == {"role": "system", "content": "You are helpful."}
    assert msgs[1] == {"role": "user", "content": "Hi"}


@pytest.mark.asyncio
async def test_explicit_system_prompt_overrides(sm):
    """Explicit system_prompt param overrides session metadata."""
    sid = await sm.create_session(metadata={"system_prompt": "Old prompt"})
    uid = await sm.create_message(sid, "user")
    await sm.append_text_part(sid, uid, "Hi")

    msgs = await sm.compose_messages(sid, system_prompt="New prompt")
    assert msgs[0] == {"role": "system", "content": "New prompt"}


@pytest.mark.asyncio
async def test_tool_call_and_result(sm):
    """Assistant with tool call + completed result composes correctly."""
    sid = await sm.create_session()
    # User message
    uid = await sm.create_message(sid, "user")
    await sm.append_text_part(sid, uid, "Read test.txt")

    # Assistant message with tool call
    aid = await sm.create_message(sid, "assistant")
    part_id = await sm.insert_tool_part(sid, aid, "call_1", "read_file", {"path": "test.txt"})
    # Complete the tool
    await sm.update_part_state(sid, aid, part_id, {
        "status": "completed",
        "input": {"path": "test.txt"},
        "output": "file contents here",
        "title": "read_file",
        "metadata": {},
        "time": {"start": 100, "end": 101},
    })

    msgs = await sm.compose_messages(sid)
    non_sys = [m for m in msgs if m["role"] != "system"]

    # Should be: user, assistant (with tool_calls), tool (result)
    assert len(non_sys) == 3
    assert non_sys[0]["role"] == "user"

    # Assistant message has tool_calls
    assistant = non_sys[1]
    assert assistant["role"] == "assistant"
    assert "tool_calls" in assistant
    assert len(assistant["tool_calls"]) == 1
    tc = assistant["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["function"]["name"] == "read_file"
    assert json.loads(tc["function"]["arguments"]) == {"path": "test.txt"}

    # Tool result message
    tool_msg = non_sys[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"] == "file contents here"


@pytest.mark.asyncio
async def test_tool_error_result(sm):
    """Tool error produces error content in tool result message."""
    sid = await sm.create_session()
    uid = await sm.create_message(sid, "user")
    await sm.append_text_part(sid, uid, "Do something")

    aid = await sm.create_message(sid, "assistant")
    part_id = await sm.insert_tool_part(sid, aid, "call_2", "shell", {"command": "fail"})
    await sm.update_part_state(sid, aid, part_id, {
        "status": "error",
        "input": {"command": "fail"},
        "error": "command not found",
        "time": {"start": 100, "end": 101},
    })

    msgs = await sm.compose_messages(sid)
    non_sys = [m for m in msgs if m["role"] != "system"]
    tool_msg = [m for m in non_sys if m["role"] == "tool"][0]
    assert "Error: command not found" in tool_msg["content"]


@pytest.mark.asyncio
async def test_pending_tool_gets_interrupted_marker(sm):
    """Pending/running tools get '[Tool execution was interrupted]' marker."""
    sid = await sm.create_session()
    uid = await sm.create_message(sid, "user")
    await sm.append_text_part(sid, uid, "Do something")

    aid = await sm.create_message(sid, "assistant")
    # Insert tool part — it stays in "pending" status
    await sm.insert_tool_part(sid, aid, "call_3", "read_file", {"path": "x"})

    msgs = await sm.compose_messages(sid)
    non_sys = [m for m in msgs if m["role"] != "system"]
    tool_msg = [m for m in non_sys if m["role"] == "tool"][0]
    assert tool_msg["content"] == "[Tool execution was interrupted]"


@pytest.mark.asyncio
async def test_multiple_tool_calls_in_one_message(sm):
    """Multiple tool calls in one assistant message (parallel calls)."""
    sid = await sm.create_session()
    uid = await sm.create_message(sid, "user")
    await sm.append_text_part(sid, uid, "Read two files")

    aid = await sm.create_message(sid, "assistant")
    p1 = await sm.insert_tool_part(sid, aid, "call_a", "read_file", {"path": "a.txt"})
    p2 = await sm.insert_tool_part(sid, aid, "call_b", "read_file", {"path": "b.txt"})
    await sm.update_part_state(sid, aid, p1, {
        "status": "completed", "input": {"path": "a.txt"},
        "output": "content_a", "title": "read_file", "metadata": {},
        "time": {"start": 1, "end": 2},
    })
    await sm.update_part_state(sid, aid, p2, {
        "status": "completed", "input": {"path": "b.txt"},
        "output": "content_b", "title": "read_file", "metadata": {},
        "time": {"start": 1, "end": 2},
    })

    msgs = await sm.compose_messages(sid)
    non_sys = [m for m in msgs if m["role"] != "system"]

    assistant = [m for m in non_sys if m["role"] == "assistant"][0]
    assert len(assistant["tool_calls"]) == 2

    tool_msgs = [m for m in non_sys if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["content"] == "content_a"
    assert tool_msgs[1]["content"] == "content_b"


@pytest.mark.asyncio
async def test_multi_turn_conversation(sm):
    """Full multi-turn: user -> assistant+tool -> tool_result -> assistant text."""
    sid = await sm.create_session()

    # Turn 1: user asks
    u1 = await sm.create_message(sid, "user")
    await sm.append_text_part(sid, u1, "What's in test.txt?")

    # Turn 1: assistant calls tool
    a1 = await sm.create_message(sid, "assistant")
    p1 = await sm.insert_tool_part(sid, a1, "call_1", "read_file", {"path": "test.txt"})
    await sm.update_part_state(sid, a1, p1, {
        "status": "completed", "input": {"path": "test.txt"},
        "output": "hello world", "title": "read_file", "metadata": {},
        "time": {"start": 1, "end": 2},
    })

    # Turn 2: assistant responds with text
    a2 = await sm.create_message(sid, "assistant")
    await sm.append_text_part(sid, a2, "The file contains: hello world")

    msgs = await sm.compose_messages(sid)
    non_sys = [m for m in msgs if m["role"] != "system"]

    # user, assistant(tool_call), tool(result), assistant(text)
    assert len(non_sys) == 4
    assert non_sys[0]["role"] == "user"
    assert non_sys[1]["role"] == "assistant"
    assert "tool_calls" in non_sys[1]
    assert non_sys[2]["role"] == "tool"
    assert non_sys[3]["role"] == "assistant"
    assert non_sys[3]["content"] == "The file contains: hello world"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
