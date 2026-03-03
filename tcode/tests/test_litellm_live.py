"""Live test against a real LiteLLM proxy endpoint.

Usage:
    bin/python3.12 -m pytest tcode/tests/test_litellm_live.py -v -s

This test hits a real LiteLLM proxy and requires network access.
Mark with @pytest.mark.live so it can be skipped in CI.
"""
from __future__ import annotations
import asyncio
import json
import pytest
from pydantic import BaseModel
from tcode.providers.litellm_adapter import LitellmAdapter
from tcode.providers.factory import ProviderFactory
from tcode.tools import ToolRegistry, ToolInfo, ToolResult
from tcode.storage import Storage
from tcode.event import EventBus
from tcode.session import SessionManager
from tcode.mcp import MCPManager
from tcode.attachments import AttachmentStore
from tcode.agent import AgentRunner


# ---- LiteLLM proxy config ----
LITELLM_URL = "https://39b6-34-125-156-6.ngrok-free.app/v1"
LITELLM_KEY = "sk-123456"
LITELLM_MODEL = "claude-haiku-4-5-20251001"


# ---- Test tools ----
class EchoParams(BaseModel):
    text: str

async def echo_execute(args, ctx):
    return ToolResult(title="echo", output=f"Echo: {args.get('text', '')}", metadata={})


class CalcParams(BaseModel):
    expression: str

async def calc_execute(args, ctx):
    expr = args.get("expression", "0")
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return ToolResult(title="calc", output=str(result), metadata={})
    except Exception as e:
        return ToolResult(title="calc", output=f"Error: {e}", metadata={})


# ---- Tests ----

@pytest.mark.asyncio
async def test_litellm_text_streaming():
    """Simple text streaming — no tools."""
    adapter = LitellmAdapter(api_url=LITELLM_URL, api_key=LITELLM_KEY)

    chunks = []
    async for chunk in adapter.chat_stream(
        [{"role": "user", "content": "Say exactly: hello world"}],
        LITELLM_MODEL,
        {"max_tokens": 50},
    ):
        chunks.append(dict(chunk))
        val = chunk.get('text') or str(chunk.get('data', ''))
        print(f"  chunk: {chunk.get('type')}: {val[:80]}")

    deltas = [c for c in chunks if c["type"] == "delta"]
    assert len(deltas) > 0, "Expected at least one text delta"
    full_text = "".join(c["text"] for c in deltas)
    print(f"\n  Full response: {full_text}")
    assert len(full_text) > 0


@pytest.mark.asyncio
async def test_litellm_tool_call():
    """LLM should call the echo tool when asked."""
    adapter = LitellmAdapter(api_url=LITELLM_URL, api_key=LITELLM_KEY)

    tools = [{
        "name": "echo",
        "description": "Echo back the given text",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to echo"}},
            "required": ["text"],
        },
    }]

    chunks = []
    async for chunk in adapter.chat_stream(
        [{"role": "user", "content": "Use the echo tool to echo 'hello from litellm'"}],
        LITELLM_MODEL,
        {"max_tokens": 200},
        tools=tools,
    ):
        chunks.append(dict(chunk))
        ctype = chunk.get("type")
        if ctype == "tool_call_end":
            print(f"  Tool call: {chunk.get('name')}({chunk.get('arguments')})")
        elif ctype == "delta":
            print(chunk.get("text", ""), end="", flush=True)

    ends = [c for c in chunks if c["type"] == "tool_call_end"]
    print(f"\n  Tool calls: {len(ends)}")
    assert len(ends) > 0, "Expected at least one tool call"
    assert ends[0]["name"] == "echo"


@pytest.mark.asyncio
async def test_litellm_full_agent_loop():
    """Full agent loop: LLM calls tool, gets result, responds."""
    storage = Storage()
    events = EventBus()
    sessions = SessionManager(storage=storage, events=events)
    tools = ToolRegistry()
    tools.register(ToolInfo(id="echo", description="Echo back text", parameters=EchoParams, execute=echo_execute))
    tools.register(ToolInfo(id="calc", description="Calculate a math expression", parameters=CalcParams, execute=calc_execute))

    factory = ProviderFactory()
    factory.register_adapter("litellm", lambda **kw: LitellmAdapter(
        api_url=LITELLM_URL, api_key=LITELLM_KEY, **kw
    ))
    mcp = MCPManager()
    attachments = AttachmentStore()
    runner = AgentRunner(factory, tools, mcp, sessions, events, attachments)

    sid = await sessions.create_session()
    await sessions.set_permission(sid, [{"permission": "*", "pattern": "*", "action": "allow"}])
    uid = await sessions.create_message(sid, "user")
    await sessions.append_text_part(sid, uid, "Use the calc tool to compute 2+3, then tell me the result.")

    print("\n  Running agent loop...")
    result = await runner.run(
        "litellm", LITELLM_MODEL, sid, uid,
        system_prompt="You are a helpful assistant. Use the provided tools when needed.",
        max_steps=5,
        options={"max_tokens": 300},
    )

    print(f"  Steps: {result['steps']}")
    print(f"  Final text: {result['final_text'][:200]}")

    # The agent should have called calc and gotten "5"
    msgs = await sessions.compose_messages(sid)
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    print(f"  Tool results: {[m['content'][:50] for m in tool_msgs]}")

    assert result["steps"] >= 2, "Expected at least 2 steps (tool call + response)"
    assert any("5" in m.get("content", "") for m in tool_msgs), "Expected calc result '5' in tool output"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
