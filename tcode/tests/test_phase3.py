"""Tests for Phase 3: Part model parity, parent chain, token tracking, doom loop."""
from __future__ import annotations
import asyncio
import json
import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel

from tcode.session import (
    SessionManager, TextPart, ReasoningPart, FilePart, ToolPart,
    StepStartPart, StepFinishPart, CompactionPart, SubtaskPart,
    AgentPart, RetryPart, PatchPart, Part,
)
from tcode.storage import Storage
from tcode.event import EventBus
from tcode.agent import AgentRunner, _MODEL_COSTS
from tcode.providers.base import ProviderChunk
from tcode.providers.factory import ProviderFactory
from tcode.tools import ToolRegistry, ToolInfo, ToolResult
from tcode.mcp import MCPManager
from tcode.attachments import AttachmentStore


# ---- Helpers ----

def _make_sessions():
    storage = Storage()
    events = EventBus()
    return SessionManager(storage=storage, events=events), events


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
    return runner, sessions, factory, events


# ============================================================
# 3.1 New Part Types
# ============================================================

class TestNewPartTypes:
    """Test that all new part types are properly defined and in the Part union."""

    def test_step_start_part_fields(self):
        p = StepStartPart(id="p1", session_id="s1", message_id="m1", snapshot="abc123")
        assert p.type == "step-start"
        assert p.snapshot == "abc123"

    def test_step_finish_part_fields(self):
        tokens = {"input": 100, "output": 50, "reasoning": 0, "cache": {"read": 10, "write": 5}}
        p = StepFinishPart(id="p1", session_id="s1", message_id="m1",
                           reason="stop", cost=0.005, tokens=tokens)
        assert p.type == "step-finish"
        assert p.cost == 0.005
        assert p.tokens["input"] == 100
        assert p.tokens["cache"]["read"] == 10

    def test_compaction_part_fields(self):
        p = CompactionPart(id="p1", session_id="s1", message_id="m1",
                          auto=True, text="Summary of conversation")
        assert p.type == "compaction"
        assert p.auto is True
        assert p.text == "Summary of conversation"

    def test_subtask_part_fields(self):
        p = SubtaskPart(id="p1", session_id="s1", message_id="m1",
                       prompt="Find the bug", description="Debug task",
                       agent="explore", model={"provider_id": "openai", "model_id": "gpt-4o"})
        assert p.type == "subtask"
        assert p.agent == "explore"

    def test_agent_part_fields(self):
        p = AgentPart(id="p1", session_id="s1", message_id="m1", name="build")
        assert p.type == "agent"
        assert p.name == "build"

    def test_retry_part_fields(self):
        p = RetryPart(id="p1", session_id="s1", message_id="m1",
                     attempt=2, error={"type": "rate_limit", "message": "throttled"})
        assert p.type == "retry"
        assert p.attempt == 2

    def test_patch_part_fields(self):
        p = PatchPart(id="p1", session_id="s1", message_id="m1",
                     hash="abc", files=["src/main.py", "tests/test.py"])
        assert p.type == "patch"
        assert len(p.files) == 2

    def test_part_union_includes_all_types(self):
        """Part union should include all part types."""
        import typing
        args = typing.get_args(Part)
        type_names = {t.__name__ for t in args}
        assert "StepStartPart" in type_names
        assert "StepFinishPart" in type_names
        assert "CompactionPart" in type_names
        assert "SubtaskPart" in type_names
        assert "AgentPart" in type_names
        assert "RetryPart" in type_names
        assert "PatchPart" in type_names


# ============================================================
# 3.1 Session methods for new part types
# ============================================================

class TestSessionPartMethods:

    @pytest.mark.asyncio
    async def test_insert_step_start_part(self):
        sessions, events = _make_sessions()
        sid = await sessions.create_session()
        mid = await sessions.create_message(sid, "assistant")
        pid = await sessions.insert_step_start_part(sid, mid, snapshot="snap1")
        assert pid.startswith("part")
        msg = await sessions.get_message(sid, mid)
        parts = msg.parts
        assert any(p.get("type") == "step-start" for p in parts)

    @pytest.mark.asyncio
    async def test_insert_step_finish_part(self):
        sessions, events = _make_sessions()
        sid = await sessions.create_session()
        mid = await sessions.create_message(sid, "assistant")
        tokens = {"input": 200, "output": 100, "reasoning": 0, "cache": {"read": 0, "write": 0}}
        pid = await sessions.insert_step_finish_part(sid, mid, reason="stop", cost=0.01, tokens=tokens)
        msg = await sessions.get_message(sid, mid)
        finish_parts = [p for p in msg.parts if p.get("type") == "step-finish"]
        assert len(finish_parts) == 1
        assert finish_parts[0]["cost"] == 0.01
        assert finish_parts[0]["tokens"]["input"] == 200

    @pytest.mark.asyncio
    async def test_insert_compaction_part(self):
        sessions, events = _make_sessions()
        sid = await sessions.create_session()
        mid = await sessions.create_message(sid, "user")
        pid = await sessions.insert_compaction_part(sid, mid, text="Summary text", auto=True)
        msg = await sessions.get_message(sid, mid)
        comp_parts = [p for p in msg.parts if p.get("type") == "compaction"]
        assert len(comp_parts) == 1
        assert comp_parts[0]["text"] == "Summary text"

    @pytest.mark.asyncio
    async def test_insert_retry_part(self):
        sessions, events = _make_sessions()
        sid = await sessions.create_session()
        mid = await sessions.create_message(sid, "assistant")
        pid = await sessions.insert_retry_part(sid, mid, attempt=1, error={"type": "rate_limit"})
        msg = await sessions.get_message(sid, mid)
        retry_parts = [p for p in msg.parts if p.get("type") == "retry"]
        assert len(retry_parts) == 1
        assert retry_parts[0]["attempt"] == 1


# ============================================================
# 3.1 CompactionPart in compose_messages
# ============================================================

class TestCompactionInCompose:

    @pytest.mark.asyncio
    async def test_compaction_part_included_in_user_message(self):
        """Compaction parts should appear as text in composed user messages."""
        sessions, events = _make_sessions()
        sid = await sessions.create_session()
        uid = await sessions.create_message(sid, "user")
        await sessions.insert_compaction_part(sid, uid, text="We discussed X, Y, Z.")
        await sessions.append_text_part(sid, uid, "Continue from where we left off")

        messages = await sessions.compose_messages(sid)
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert "We discussed X, Y, Z." in user_msgs[0]["content"]
        assert "Continue from where we left off" in user_msgs[0]["content"]


# ============================================================
# 3.2 Message parent chain
# ============================================================

class TestParentChain:

    @pytest.mark.asyncio
    async def test_create_message_with_parent_id(self):
        sessions, events = _make_sessions()
        sid = await sessions.create_session()
        uid = await sessions.create_message(sid, "user")
        aid = await sessions.create_message(sid, "assistant", parent_id=uid)
        msg = await sessions.get_message(sid, aid)
        assert msg.info["parent_id"] == uid

    @pytest.mark.asyncio
    async def test_create_message_without_parent_id(self):
        sessions, events = _make_sessions()
        sid = await sessions.create_session()
        uid = await sessions.create_message(sid, "user")
        msg = await sessions.get_message(sid, uid)
        assert msg.info["parent_id"] is None

    @pytest.mark.asyncio
    async def test_agent_sets_parent_id(self):
        """AgentRunner should link assistant message to user message via parent_id."""
        runner, sessions, factory, events = _make_runner()

        async def simple_stream(messages, model, options, tools=None):
            yield ProviderChunk({"type": "delta", "text": "Hi"})
            yield ProviderChunk({"type": "final", "data": {}})

        mock = MagicMock()
        mock.chat_stream = simple_stream
        factory.register_adapter("mock", lambda **kw: mock)
        factory._instances["mock"] = mock

        sid = await sessions.create_session()
        uid = await sessions.create_message(sid, "user")
        await sessions.append_text_part(sid, uid, "Hello")

        result = await runner.run("mock", "test", sid, uid)
        aid = result["message_id"]
        msg = await sessions.get_message(sid, aid)
        assert msg.info["parent_id"] == uid


# ============================================================
# 3.3 Token and cost tracking
# ============================================================

class TestTokenCostTracking:

    @pytest.mark.asyncio
    async def test_usage_captured_in_result(self):
        """Agent run should capture tokens and cost in result."""
        runner, sessions, factory, events = _make_runner()

        async def stream_with_usage(messages, model, options, tools=None):
            yield ProviderChunk({"type": "delta", "text": "Hello"})
            yield ProviderChunk({"type": "usage", "input_tokens": 50, "output_tokens": 20})
            yield ProviderChunk({"type": "final", "data": {}})

        mock = MagicMock()
        mock.chat_stream = stream_with_usage
        factory.register_adapter("mock", lambda **kw: mock)
        factory._instances["mock"] = mock

        sid = await sessions.create_session()
        uid = await sessions.create_message(sid, "user")
        await sessions.append_text_part(sid, uid, "Hi")

        result = await runner.run("mock", "test-model", sid, uid)
        assert result["tokens"]["input"] == 50
        assert result["tokens"]["output"] == 20

    @pytest.mark.asyncio
    async def test_step_finish_part_emitted(self):
        """Agent should emit step-finish parts with usage data."""
        runner, sessions, factory, events = _make_runner()

        async def stream(messages, model, options, tools=None):
            yield ProviderChunk({"type": "delta", "text": "Done"})
            yield ProviderChunk({"type": "usage", "input_tokens": 100, "output_tokens": 50})
            yield ProviderChunk({"type": "final", "data": {}})

        mock = MagicMock()
        mock.chat_stream = stream
        factory.register_adapter("mock", lambda **kw: mock)
        factory._instances["mock"] = mock

        sid = await sessions.create_session()
        uid = await sessions.create_message(sid, "user")
        await sessions.append_text_part(sid, uid, "Test")

        result = await runner.run("mock", "test-model", sid, uid)
        msg = await sessions.get_message(sid, result["message_id"])
        step_finishes = [p for p in msg.parts if p.get("type") == "step-finish"]
        assert len(step_finishes) >= 1
        assert step_finishes[0]["tokens"]["input"] == 100

    @pytest.mark.asyncio
    async def test_step_start_part_emitted(self):
        """Agent should emit step-start part at beginning of each step."""
        runner, sessions, factory, events = _make_runner()

        async def stream(messages, model, options, tools=None):
            yield ProviderChunk({"type": "delta", "text": "OK"})
            yield ProviderChunk({"type": "final", "data": {}})

        mock = MagicMock()
        mock.chat_stream = stream
        factory.register_adapter("mock", lambda **kw: mock)
        factory._instances["mock"] = mock

        sid = await sessions.create_session()
        uid = await sessions.create_message(sid, "user")
        await sessions.append_text_part(sid, uid, "Go")

        result = await runner.run("mock", "test-model", sid, uid)
        msg = await sessions.get_message(sid, result["message_id"])
        step_starts = [p for p in msg.parts if p.get("type") == "step-start"]
        assert len(step_starts) >= 1

    def test_cost_calculation_known_model(self):
        """Cost calculation for a model in the pricing table."""
        runner, _, _, _ = _make_runner()
        usage = {"input": 1000, "output": 500, "reasoning": 0, "cache": {"read": 0, "write": 0}}
        cost = runner._calculate_cost("gpt-4o", usage)
        # gpt-4o: input=2.50, output=10.0 per million
        expected = 1000 * 2.50 / 1_000_000 + 500 * 10.0 / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_cost_calculation_unknown_model(self):
        """Cost calculation for unknown model should return 0."""
        runner, _, _, _ = _make_runner()
        usage = {"input": 1000, "output": 500, "reasoning": 0, "cache": {"read": 0, "write": 0}}
        cost = runner._calculate_cost("unknown-model-xyz", usage)
        assert cost == 0.0

    def test_cost_calculation_empty_usage(self):
        runner, _, _, _ = _make_runner()
        assert runner._calculate_cost("gpt-4o", {}) == 0.0
        assert runner._calculate_cost("gpt-4o", None) == 0.0

    @pytest.mark.asyncio
    async def test_multi_step_cost_accumulation(self):
        """Cost should accumulate across multiple steps."""
        runner, sessions, factory, events = _make_runner()
        step_count = 0

        async def multi_step_stream(messages, model, options, tools=None):
            nonlocal step_count
            step_count += 1
            if step_count == 1:
                yield ProviderChunk({"type": "tool_call_end", "id": "c1", "name": "echo", "arguments": {"text": "hi"}})
                yield ProviderChunk({"type": "usage", "input_tokens": 100, "output_tokens": 50})
                yield ProviderChunk({"type": "final", "data": {}})
            else:
                yield ProviderChunk({"type": "delta", "text": "Done"})
                yield ProviderChunk({"type": "usage", "input_tokens": 200, "output_tokens": 80})
                yield ProviderChunk({"type": "final", "data": {}})

        mock = MagicMock()
        mock.chat_stream = multi_step_stream
        factory.register_adapter("mock", lambda **kw: mock)
        factory._instances["mock"] = mock

        sid = await sessions.create_session()
        await sessions.set_permission(sid, [{"permission": "*", "pattern": "*", "action": "allow"}])
        uid = await sessions.create_message(sid, "user")
        await sessions.append_text_part(sid, uid, "Echo hi")

        result = await runner.run("mock", "test-model", sid, uid)
        assert result["tokens"]["input"] == 300  # 100 + 200
        assert result["tokens"]["output"] == 130  # 50 + 80
        assert result["steps"] == 2


# ============================================================
# 3.4 Doom loop detection
# ============================================================

class TestDoomLoop:

    def test_no_doom_loop_with_different_calls(self):
        runner, _, _, _ = _make_runner()
        recent = [
            {"name": "echo", "arguments": {"text": "a"}},
            {"name": "echo", "arguments": {"text": "b"}},
            {"name": "echo", "arguments": {"text": "c"}},
        ]
        assert runner._detect_doom_loop(recent) is False

    def test_doom_loop_detected_with_identical_calls(self):
        runner, _, _, _ = _make_runner()
        recent = [
            {"name": "echo", "arguments": {"text": "same"}},
            {"name": "echo", "arguments": {"text": "same"}},
            {"name": "echo", "arguments": {"text": "same"}},
        ]
        assert runner._detect_doom_loop(recent) is True

    def test_no_doom_loop_below_threshold(self):
        runner, _, _, _ = _make_runner()
        recent = [
            {"name": "echo", "arguments": {"text": "same"}},
            {"name": "echo", "arguments": {"text": "same"}},
        ]
        assert runner._detect_doom_loop(recent) is False

    def test_doom_loop_checks_last_three_only(self):
        runner, _, _, _ = _make_runner()
        recent = [
            {"name": "echo", "arguments": {"text": "different"}},
            {"name": "echo", "arguments": {"text": "same"}},
            {"name": "echo", "arguments": {"text": "same"}},
            {"name": "echo", "arguments": {"text": "same"}},
        ]
        # Last 3 are identical
        assert runner._detect_doom_loop(recent) is True

    def test_doom_loop_allowed_by_agent_permission(self):
        """If agent permission allows doom_loop, don't stop."""
        from tcode.agent_defs import AgentInfo
        runner, _, _, _ = _make_runner()
        agent = AgentInfo(
            name="test",
            permission=[{"permission": "doom_loop", "action": "allow", "pattern": "*"}],
        )
        recent = [
            {"name": "echo", "arguments": {"text": "same"}},
            {"name": "echo", "arguments": {"text": "same"}},
            {"name": "echo", "arguments": {"text": "same"}},
        ]
        assert runner._detect_doom_loop(recent, agent_info=agent) is False

    @pytest.mark.asyncio
    async def test_doom_loop_stops_agent(self):
        """Agent should stop when doom loop is detected."""
        runner, sessions, factory, events = _make_runner()
        call_count = 0

        async def doom_stream(messages, model, options, tools=None):
            nonlocal call_count
            call_count += 1
            # Always call echo with same args
            yield ProviderChunk({"type": "tool_call_end", "id": f"c{call_count}", "name": "echo", "arguments": {"text": "same"}})
            yield ProviderChunk({"type": "final", "data": {}})

        mock = MagicMock()
        mock.chat_stream = doom_stream
        factory.register_adapter("mock", lambda **kw: mock)
        factory._instances["mock"] = mock

        sid = await sessions.create_session()
        await sessions.set_permission(sid, [{"permission": "*", "pattern": "*", "action": "allow"}])
        uid = await sessions.create_message(sid, "user")
        await sessions.append_text_part(sid, uid, "Loop forever")

        result = await runner.run("mock", "test-model", sid, uid, max_steps=10)
        # Should stop after 3 identical calls (doom loop detected on step 3)
        assert result["blocked"] is True
        assert result["steps"] <= 4  # At most 3 tool calls + detection


# ============================================================
# Model cost table
# ============================================================

class TestModelCosts:

    def test_known_models_in_table(self):
        assert "gpt-4o" in _MODEL_COSTS
        assert "claude-sonnet-4-20250514" in _MODEL_COSTS
        assert "claude-haiku-4-5-20251001" in _MODEL_COSTS

    def test_cost_fields_present(self):
        for model, costs in _MODEL_COSTS.items():
            assert "input" in costs, f"{model} missing input cost"
            assert "output" in costs, f"{model} missing output cost"
