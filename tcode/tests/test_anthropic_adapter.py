"""Tests for Anthropic adapter using official SDK, with mocked AsyncAnthropic."""
from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from tcode.providers.anthropic_adapter import AnthropicAdapter


def _event(etype, **kwargs):
    """Create a mock streaming event."""
    e = MagicMock()
    e.type = etype
    for k, v in kwargs.items():
        setattr(e, k, v)
    return e


def _text_block(text=""):
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_use_block(id="", name=""):
    b = MagicMock()
    b.type = "tool_use"
    b.id = id
    b.name = name
    return b


def _text_delta(text):
    d = MagicMock()
    d.type = "text_delta"
    d.text = text
    return d


def _input_json_delta(partial):
    d = MagicMock()
    d.type = "input_json_delta"
    d.partial_json = partial
    return d


class FakeMessageStream:
    """Simulates async with client.messages.stream(...) as stream."""
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self._iter_events()

    async def _iter_events(self):
        for e in self._events:
            yield e


@pytest.mark.asyncio
async def test_text_streaming():
    events = [
        _event("message_start", message=MagicMock(usage=MagicMock(input_tokens=5, output_tokens=0))),
        _event("content_block_start", index=0, content_block=_text_block()),
        _event("content_block_delta", index=0, delta=_text_delta("Hello")),
        _event("content_block_delta", index=0, delta=_text_delta(" world")),
        _event("content_block_stop", index=0),
    ]

    adapter = AnthropicAdapter(api_key="test-key")
    mock_client = MagicMock()
    mock_client.messages.stream = MagicMock(return_value=FakeMessageStream(events))
    adapter._client = mock_client

    chunks = []
    async for c in adapter.chat_stream([{"role": "user", "content": "Hi"}], "claude-sonnet-4-20250514", {}):
        chunks.append(dict(c))

    deltas = [c for c in chunks if c["type"] == "delta"]
    assert len(deltas) == 2
    assert deltas[0]["text"] == "Hello"
    assert deltas[1]["text"] == " world"
    final = [c for c in chunks if c["type"] == "final"]
    assert final[0]["data"]["text"] == "Hello world"


@pytest.mark.asyncio
async def test_tool_use():
    events = [
        _event("message_start", message=MagicMock(usage=None)),
        _event("content_block_start", index=0, content_block=_tool_use_block(id="toolu_123", name="read_file")),
        _event("content_block_delta", index=0, delta=_input_json_delta('{"path":')),
        _event("content_block_delta", index=0, delta=_input_json_delta(' "test.txt"}')),
        _event("content_block_stop", index=0),
    ]

    adapter = AnthropicAdapter(api_key="test-key")
    mock_client = MagicMock()
    mock_client.messages.stream = MagicMock(return_value=FakeMessageStream(events))
    adapter._client = mock_client

    chunks = []
    async for c in adapter.chat_stream([], "claude-sonnet-4-20250514", {}, tools=[
        {"name": "read_file", "description": "Read", "parameters": {}}
    ]):
        chunks.append(dict(c))

    starts = [c for c in chunks if c["type"] == "tool_call_start"]
    assert len(starts) == 1
    assert starts[0]["name"] == "read_file"
    assert starts[0]["id"] == "toolu_123"

    ends = [c for c in chunks if c["type"] == "tool_call_end"]
    assert len(ends) == 1
    assert ends[0]["arguments"] == {"path": "test.txt"}


@pytest.mark.asyncio
async def test_text_and_tool_use():
    events = [
        _event("message_start", message=MagicMock(usage=None)),
        _event("content_block_start", index=0, content_block=_text_block()),
        _event("content_block_delta", index=0, delta=_text_delta("Let me read that.")),
        _event("content_block_stop", index=0),
        _event("content_block_start", index=1, content_block=_tool_use_block(id="toolu_1", name="read_file")),
        _event("content_block_delta", index=1, delta=_input_json_delta('{"path": "x.py"}')),
        _event("content_block_stop", index=1),
    ]

    adapter = AnthropicAdapter(api_key="test-key")
    mock_client = MagicMock()
    mock_client.messages.stream = MagicMock(return_value=FakeMessageStream(events))
    adapter._client = mock_client

    chunks = []
    async for c in adapter.chat_stream([], "claude-sonnet-4-20250514", {}):
        chunks.append(dict(c))

    deltas = [c for c in chunks if c["type"] == "delta"]
    assert deltas[0]["text"] == "Let me read that."
    ends = [c for c in chunks if c["type"] == "tool_call_end"]
    assert ends[0]["arguments"] == {"path": "x.py"}


@pytest.mark.asyncio
async def test_message_format_conversion():
    adapter = AnthropicAdapter(api_key="test-key")
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Read file"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
    ]

    system, msgs = adapter._extract_system_and_messages(messages)
    assert system == "You are helpful."
    assert len(msgs) == 3
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    blocks = msgs[1]["content"]
    assert blocks[0]["type"] == "tool_use"
    assert blocks[0]["name"] == "read_file"
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0]["type"] == "tool_result"


@pytest.mark.asyncio
async def test_tool_format():
    adapter = AnthropicAdapter(api_key="test-key")
    tools = [{"name": "grep", "description": "Search", "parameters": {
        "type": "object", "properties": {"pattern": {"type": "string"}}
    }}]
    formatted = adapter._format_tools(tools)
    assert len(formatted) == 1
    assert formatted[0]["name"] == "grep"
    assert formatted[0]["input_schema"]["properties"]["pattern"]["type"] == "string"


@pytest.mark.asyncio
async def test_usage_parsing():
    events = [
        _event("message_start", message=MagicMock(usage=MagicMock(input_tokens=100, output_tokens=0))),
        _event("content_block_start", index=0, content_block=_text_block()),
        _event("content_block_delta", index=0, delta=_text_delta("ok")),
        _event("content_block_stop", index=0),
        _event("message_delta", usage=MagicMock(input_tokens=0, output_tokens=50)),
    ]

    adapter = AnthropicAdapter(api_key="test-key")
    mock_client = MagicMock()
    mock_client.messages.stream = MagicMock(return_value=FakeMessageStream(events))
    adapter._client = mock_client

    chunks = []
    async for c in adapter.chat_stream([], "claude-sonnet-4-20250514", {}):
        chunks.append(dict(c))

    usage = [c for c in chunks if c["type"] == "usage"]
    assert len(usage) == 2
    assert usage[0]["input_tokens"] == 100
    assert usage[1]["output_tokens"] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
