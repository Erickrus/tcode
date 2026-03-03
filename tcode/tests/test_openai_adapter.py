"""Tests for OpenAI adapter using official SDK, with mocked AsyncOpenAI."""
from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tcode.providers.openai_adapter import OpenAIAdapter


def _make_chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    """Create a mock ChatCompletionChunk."""
    chunk = MagicMock()
    chunk.usage = None
    if usage:
        chunk.usage = MagicMock()
        chunk.usage.prompt_tokens = usage.get("prompt_tokens", 0)
        chunk.usage.completion_tokens = usage.get("completion_tokens", 0)
        chunk.usage.total_tokens = usage.get("total_tokens", 0)

    choice = MagicMock()
    choice.finish_reason = finish_reason
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None

    if tool_calls:
        tc_list = []
        for tc in tool_calls:
            tc_mock = MagicMock()
            tc_mock.index = tc.get("index", 0)
            tc_mock.id = tc.get("id")
            tc_mock.type = tc.get("type", "function")
            func_mock = MagicMock()
            func_mock.name = tc.get("name")
            func_mock.arguments = tc.get("arguments")
            tc_mock.function = func_mock
            tc_list.append(tc_mock)
        delta.tool_calls = tc_list

    choice.delta = delta
    chunk.choices = [choice]
    return chunk


class FakeAsyncStream:
    """Simulates the async iterator returned by client.chat.completions.create(stream=True)."""
    def __init__(self, chunks):
        self._chunks = chunks
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        c = self._chunks[self._idx]
        self._idx += 1
        return c


@pytest.mark.asyncio
async def test_text_streaming():
    chunks = [
        _make_chunk(content="Hello"),
        _make_chunk(content=" world"),
        _make_chunk(finish_reason="stop"),
    ]

    adapter = OpenAIAdapter(api_key="test-key")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=FakeAsyncStream(chunks))
    adapter._client = mock_client

    result_chunks = []
    async for c in adapter.chat_stream([{"role": "user", "content": "Hi"}], "gpt-4o", {}):
        result_chunks.append(dict(c))

    deltas = [c for c in result_chunks if c["type"] == "delta"]
    assert len(deltas) == 2
    assert deltas[0]["text"] == "Hello"
    assert deltas[1]["text"] == " world"
    final = [c for c in result_chunks if c["type"] == "final"]
    assert final[0]["data"]["text"] == "Hello world"


@pytest.mark.asyncio
async def test_single_tool_call():
    chunks = [
        _make_chunk(tool_calls=[{"index": 0, "id": "call_123", "name": "read_file", "arguments": ""}]),
        _make_chunk(tool_calls=[{"index": 0, "id": None, "name": None, "arguments": '{"path":'}]),
        _make_chunk(tool_calls=[{"index": 0, "id": None, "name": None, "arguments": ' "test.txt"}'}]),
        _make_chunk(finish_reason="tool_calls"),
    ]

    adapter = OpenAIAdapter(api_key="test-key")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=FakeAsyncStream(chunks))
    adapter._client = mock_client

    result_chunks = []
    async for c in adapter.chat_stream([], "gpt-4o", {}, tools=[
        {"name": "read_file", "description": "Read", "parameters": {}}
    ]):
        result_chunks.append(dict(c))

    starts = [c for c in result_chunks if c["type"] == "tool_call_start"]
    assert len(starts) == 1
    assert starts[0]["name"] == "read_file"

    ends = [c for c in result_chunks if c["type"] == "tool_call_end"]
    assert len(ends) == 1
    assert ends[0]["arguments"] == {"path": "test.txt"}


@pytest.mark.asyncio
async def test_parallel_tool_calls():
    chunks = [
        _make_chunk(tool_calls=[
            {"index": 0, "id": "call_a", "name": "read_file", "arguments": '{"path": "a.txt"}'},
            {"index": 1, "id": "call_b", "name": "grep", "arguments": '{"pattern": "foo"}'},
        ]),
        _make_chunk(finish_reason="tool_calls"),
    ]

    adapter = OpenAIAdapter(api_key="test-key")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=FakeAsyncStream(chunks))
    adapter._client = mock_client

    result_chunks = []
    async for c in adapter.chat_stream([], "gpt-4o", {}, tools=[
        {"name": "read_file", "description": "Read", "parameters": {}},
        {"name": "grep", "description": "Search", "parameters": {}},
    ]):
        result_chunks.append(dict(c))

    ends = [c for c in result_chunks if c["type"] == "tool_call_end"]
    assert len(ends) == 2
    assert ends[0]["name"] == "read_file"
    assert ends[0]["arguments"] == {"path": "a.txt"}
    assert ends[1]["name"] == "grep"
    assert ends[1]["arguments"] == {"pattern": "foo"}


@pytest.mark.asyncio
async def test_text_then_tool_call():
    chunks = [
        _make_chunk(content="Let me read that file."),
        _make_chunk(tool_calls=[{"index": 0, "id": "call_1", "name": "read_file", "arguments": '{"path": "x.py"}'}]),
        _make_chunk(finish_reason="tool_calls"),
    ]

    adapter = OpenAIAdapter(api_key="test-key")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=FakeAsyncStream(chunks))
    adapter._client = mock_client

    result_chunks = []
    async for c in adapter.chat_stream([], "gpt-4o", {}, tools=[
        {"name": "read_file", "description": "Read", "parameters": {}},
    ]):
        result_chunks.append(dict(c))

    deltas = [c for c in result_chunks if c["type"] == "delta"]
    assert deltas[0]["text"] == "Let me read that file."
    ends = [c for c in result_chunks if c["type"] == "tool_call_end"]
    assert ends[0]["arguments"] == {"path": "x.py"}


@pytest.mark.asyncio
async def test_usage_parsing():
    chunks = [
        _make_chunk(content="Hi"),
        _make_chunk(finish_reason="stop", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
    ]

    adapter = OpenAIAdapter(api_key="test-key")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=FakeAsyncStream(chunks))
    adapter._client = mock_client

    result_chunks = []
    async for c in adapter.chat_stream([], "gpt-4o", {}):
        result_chunks.append(dict(c))

    usage = [c for c in result_chunks if c["type"] == "usage"]
    assert len(usage) == 1
    assert usage[0]["input_tokens"] == 10
    assert usage[0]["output_tokens"] == 5


@pytest.mark.asyncio
async def test_chat_collects_tool_calls():
    chunks = [
        _make_chunk(tool_calls=[{"index": 0, "id": "call_99", "name": "echo", "arguments": '{"text": "hi"}'}]),
        _make_chunk(finish_reason="tool_calls"),
    ]

    adapter = OpenAIAdapter(api_key="test-key")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=FakeAsyncStream(chunks))
    adapter._client = mock_client

    result = await adapter.chat([], "gpt-4o", {}, tools=[
        {"name": "echo", "description": "Echo", "parameters": {}},
    ])
    assert "tool_calls" in result
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "echo"
    assert result["tool_calls"][0]["arguments"] == {"text": "hi"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
