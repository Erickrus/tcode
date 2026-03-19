from __future__ import annotations
import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from .providers.factory import ProviderFactory
from .providers.base import ProviderChunk
from .tools import ToolRegistry, ToolInfo
from .mcp import MCPManager
from .session import SessionManager
from .event import EventBus, Event
from .attachments import AttachmentStore
from .toolrunner import ToolRunner
from .util import next_id
from .agent_defs import AgentRegistry, AgentInfo, disabled_tools
from .providers.errors import (
    map_provider_error, is_retryable, retry_delay, retry_sleep,
    AbortedError, MAX_RETRIES,
)
from .permission_next import PermissionDeniedError, PermissionRejectedError


# Cost per million tokens for common models (USD)
_MODEL_COSTS: Dict[str, Dict[str, float]] = {
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0, "cache_read": 0.08, "cache_write": 1.0},
    "claude-opus-4-5-20251101": {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.0, "cache_read": 1.25, "cache_write": 0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cache_read": 0.075, "cache_write": 0},
    "gpt-4.1": {"input": 2.0, "output": 8.0, "cache_read": 0.50, "cache_write": 0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cache_read": 0.10, "cache_write": 0},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cache_read": 0.025, "cache_write": 0},
    # Gemini
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0, "cache_read": 0, "cache_write": 0},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60, "cache_read": 0, "cache_write": 0},
}


def _tool_to_schema(tool: ToolInfo) -> Dict[str, Any]:
    """Convert a ToolInfo to the generic tool schema format expected by adapters.

    Returns: {"name": str, "description": str, "parameters": {json_schema}}
    """
    schema: Dict[str, Any] = {"type": "object", "properties": {}}
    try:
        from pydantic import BaseModel
        if isinstance(tool.parameters, type) and issubclass(tool.parameters, BaseModel):
            schema = tool.parameters.model_json_schema()
            # Remove pydantic metadata keys that providers don't need
            schema.pop("title", None)
    except Exception:
        pass
    return {
        "name": tool.id,
        "description": tool.description,
        "parameters": schema,
    }


class AgentRunner:
    """Runs the agent loop: compose messages -> call LLM -> handle tool calls -> repeat.

    Follows opencode's processor.ts pattern:
      - while True loop
      - Stream from provider with tool definitions
      - Execute tool calls, store results as parts
      - Re-compose messages and loop until LLM finishes without tool calls
      - Compaction check before each iteration
    """

    DEFAULT_MAX_STEPS = 50

    def __init__(self, providers: ProviderFactory, tools: ToolRegistry, mcp: MCPManager,
                 sessions: SessionManager, events: EventBus, attachments: AttachmentStore,
                 agent_registry: Optional[AgentRegistry] = None):
        self.providers = providers
        self.tools = tools
        self.mcp = mcp
        self.sessions = sessions
        self.events = events
        self.attachments = attachments
        self.toolrunner = ToolRunner(tools, events, sessions)
        self.agent_registry = agent_registry or AgentRegistry()

    async def run(self, provider_id: str, model: str, session_id: str, message_id: str,
                  system_prompt: Optional[str] = None,
                  max_steps: int = DEFAULT_MAX_STEPS,
                  options: Optional[Dict[str, Any]] = None,
                  agent_name: Optional[str] = None,
                  abort_event: Optional[asyncio.Event] = None) -> Dict[str, Any]:
        """Run the agent loop for a session.

        Args:
            provider_id: Which provider adapter to use (e.g., "openai", "anthropic")
            model: Model ID (e.g., "gpt-4o", "claude-sonnet-4-20250514")
            session_id: Session to run in
            message_id: The user message that triggers this run
            system_prompt: Optional system prompt override
            max_steps: Maximum LLM call iterations to prevent infinite loops
            options: Provider options (temperature, top_p, max_tokens, etc.)
            agent_name: Named agent to use (loads prompt, permissions, model from registry)
            abort_event: asyncio.Event that when set signals the loop to abort
        """
        # Resolve agent config if specified
        agent_info = None
        if agent_name:
            agent_info = self.agent_registry.get(agent_name)

        if agent_info:
            # Apply agent config as defaults (explicit params override)
            if agent_info.model and provider_id == "openai" and model == "gpt-4o":
                # Only override if caller used defaults
                provider_id = agent_info.model.get("provider_id", provider_id)
                model = agent_info.model.get("model_id", model)
            if system_prompt is None and agent_info.prompt:
                system_prompt = agent_info.prompt
            if max_steps == self.DEFAULT_MAX_STEPS and agent_info.steps:
                max_steps = agent_info.steps
            options = options or {}
            if agent_info.temperature is not None and 'temperature' not in options:
                options['temperature'] = agent_info.temperature
            if agent_info.top_p is not None and 'top_p' not in options:
                options['top_p'] = agent_info.top_p
            if agent_info.options:
                for k, v in agent_info.options.items():
                    if k not in options:
                        options[k] = v

        adapter = self.providers.get_adapter(provider_id, {"type": provider_id})
        options = options or {}
        final_text = ""
        step = 0
        retry_attempt = 0
        total_cost = 0.0
        total_tokens = {"input": 0, "output": 0, "reasoning": 0, "cache": {"read": 0, "write": 0}}
        blocked = False

        # Emit session busy status
        await self.sessions.set_session_status(session_id, "busy")

        # Build tool schemas for the LLM, filtering by agent permissions
        tool_schemas = self._build_tool_schemas(agent_info=agent_info)

        # Create assistant message for this run, linked to user message
        assistant_msg_id = await self.sessions.create_message(
            session_id, "assistant",
            model={"provider_id": provider_id, "model_id": model},
            parent_id=message_id,
        )

        # Compaction check before starting
        await self._try_compact(session_id)

        # Track recent tool calls for doom loop detection
        recent_tool_calls: List[Dict[str, Any]] = []

        try:
            while step < max_steps:
                step += 1

                # Emit step-start part
                await self.sessions.insert_step_start_part(session_id, assistant_msg_id)

                # Compose full message history including tool calls and results
                messages = await self.sessions.compose_messages(
                    session_id, system_prompt=system_prompt
                )

                # Stream from provider
                tool_calls_this_turn: List[Dict[str, Any]] = []
                text_this_turn = ""
                had_error = False
                should_retry = False
                step_usage: Dict[str, Any] = {}
                finish_reason = ""

                # Track tool call accumulation (for streaming fragments)
                active_tool_calls: Dict[str, Dict[str, Any]] = {}

                async for chunk in adapter.chat_stream(messages, model, options, tools=tool_schemas or None):
                    # Check abort signal
                    if abort_event and abort_event.is_set():
                        break

                    ctype = chunk.get("type")

                    if ctype == "delta":
                        text = chunk.get("text", "")
                        if text:
                            text_this_turn += text
                            final_text += text

                    elif ctype == "tool_call_start":
                        tc_id = chunk.get("id", next_id("call"))
                        active_tool_calls[tc_id] = {
                            "id": tc_id,
                            "name": chunk.get("name", ""),
                            "arguments_fragments": [],
                        }

                    elif ctype == "tool_call_delta":
                        tc_id = chunk.get("id", "")
                        if tc_id in active_tool_calls:
                            active_tool_calls[tc_id]["arguments_fragments"].append(
                                chunk.get("arguments", "")
                            )

                    elif ctype == "tool_call_end":
                        tc_id = chunk.get("id", next_id("call"))
                        tool_calls_this_turn.append({
                            "id": tc_id,
                            "name": chunk.get("name", ""),
                            "arguments": chunk.get("arguments", {}),
                        })

                    elif ctype == "usage":
                        # Capture token usage from provider
                        step_usage = {
                            "input": chunk.get("input_tokens", 0) or 0,
                            "output": chunk.get("output_tokens", 0) or 0,
                            "reasoning": chunk.get("reasoning_tokens", 0) or 0,
                            "cache": {
                                "read": chunk.get("cache_read_tokens", 0) or 0,
                                "write": chunk.get("cache_creation_tokens", 0) or 0,
                            },
                        }

                    elif ctype == "error":
                        raw_error = chunk.get("error", "Unknown provider error")
                        if isinstance(raw_error, Exception):
                            error_dict = map_provider_error(raw_error)
                        elif isinstance(raw_error, dict):
                            error_dict = raw_error
                        else:
                            error_dict = {"type": "api_error", "message": str(raw_error), "retryable": False}

                        if is_retryable(error_dict) and retry_attempt < MAX_RETRIES:
                            retry_attempt += 1
                            delay = retry_delay(retry_attempt, error_dict)
                            # Record retry part (strip non-serializable exception object)
                            serializable_error = {
                                k: v for k, v in error_dict.items()
                                if k != "error"
                            }
                            await self.sessions.insert_retry_part(
                                session_id, assistant_msg_id,
                                attempt=retry_attempt, error=serializable_error,
                            )
                            await self.events.publish(Event.create(
                                "session.status.changed",
                                {"status": "retry", "attempt": retry_attempt,
                                 "message": error_dict.get("message", ""),
                                 "next_retry_in": delay},
                                session_id=session_id,
                            ))
                            try:
                                await retry_sleep(delay, abort_event)
                            except AbortedError:
                                had_error = True
                                break
                            should_retry = True
                            break
                        else:
                            error_msg = str(error_dict.get("message", "Unknown provider error"))
                            error_text = f"[Provider error: {error_msg}]"
                            await self.sessions.append_text_part(
                                session_id, assistant_msg_id,
                                error_text, synthetic=True
                            )
                            final_text += error_text
                            had_error = True
                            break

                    elif ctype == "final":
                        finish_reason = "stop" if not tool_calls_this_turn else "tool_calls"
                        break

                # If retrying, skip the rest and re-enter the loop
                if should_retry:
                    active_tool_calls.clear()
                    step -= 1  # Don't count retries as steps
                    continue

                # Accumulate token usage
                if step_usage:
                    total_tokens["input"] += step_usage.get("input", 0)
                    total_tokens["output"] += step_usage.get("output", 0)
                    total_tokens["reasoning"] += step_usage.get("reasoning", 0)
                    total_tokens["cache"]["read"] += step_usage.get("cache", {}).get("read", 0)
                    total_tokens["cache"]["write"] += step_usage.get("cache", {}).get("write", 0)

                # Calculate step cost
                step_cost = self._calculate_cost(model, step_usage)
                total_cost += step_cost

                # Emit step-finish part with usage data
                if not finish_reason:
                    finish_reason = "tool_calls" if tool_calls_this_turn else "stop"
                await self.sessions.insert_step_finish_part(
                    session_id, assistant_msg_id,
                    reason=finish_reason, cost=step_cost,
                    tokens=step_usage or {"input": 0, "output": 0, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                )

                # On error, discard any incomplete tool calls and break immediately
                if had_error:
                    active_tool_calls.clear()
                    if text_this_turn:
                        await self.sessions.append_text_part(
                            session_id, assistant_msg_id, text_this_turn
                        )
                    break

                # Flush any active_tool_calls that weren't finalized via tool_call_end
                for tc_id, tc_state in active_tool_calls.items():
                    if not any(tc["id"] == tc_id for tc in tool_calls_this_turn):
                        args_text = "".join(tc_state["arguments_fragments"])
                        try:
                            args_obj = json.loads(args_text) if args_text else {}
                        except json.JSONDecodeError:
                            args_obj = {"_raw": args_text}
                        tool_calls_this_turn.append({
                            "id": tc_id,
                            "name": tc_state["name"],
                            "arguments": args_obj,
                        })
                active_tool_calls.clear()

                # Store assistant text if any
                if text_this_turn:
                    await self.sessions.append_text_part(
                        session_id, assistant_msg_id, text_this_turn
                    )

                # If no tool calls, we're done
                if not tool_calls_this_turn:
                    break

                # Doom loop detection: check if last 3 tool calls are identical
                for tc in tool_calls_this_turn:
                    recent_tool_calls.append({"name": tc["name"], "arguments": tc["arguments"]})

                if self._detect_doom_loop(recent_tool_calls, agent_info):
                    blocked = True
                    await self.sessions.append_text_part(
                        session_id, assistant_msg_id,
                        "[Doom loop detected: repeated identical tool calls. Stopping.]",
                        synthetic=True,
                    )
                    break

                # Execute tool calls
                for tc in tool_calls_this_turn:
                    call_id = tc["id"]
                    tool_name = tc["name"]
                    tool_args = tc["arguments"]

                    part_id = await self.sessions.insert_tool_part(
                        session_id, assistant_msg_id, call_id, tool_name, tool_args
                    )

                    try:
                        result = await self.toolrunner.execute_tool(
                            session_id, assistant_msg_id, call_id, tool_name, tool_args,
                            timeout=120, part_id=part_id
                        )
                    except PermissionRejectedError:
                        # User rejected permission — set blocked and stop
                        blocked = True
                        await self.sessions.append_text_part(
                            session_id, assistant_msg_id,
                            f"[Permission rejected by user for tool: {tool_name}]",
                            synthetic=True,
                        )
                        break
                    except PermissionDeniedError:
                        # Rule denied permission — ToolRunner already updated part state
                        pass
                    except Exception as e:
                        # ToolRunner already updates part state to error
                        pass

                if blocked:
                    break

                # Check abort after tool execution
                if abort_event and abort_event.is_set():
                    break

                # Compaction check
                await self._try_compact(session_id)

        except Exception as e:
            # Unhandled error — record it so the UI can show something
            error_text = f"[Unexpected error: {e}]"
            try:
                await self.sessions.append_text_part(
                    session_id, assistant_msg_id, error_text, synthetic=True
                )
            except Exception:
                pass
            final_text += error_text
        finally:
            # If aborted, mark any pending tool parts as error
            if abort_event and abort_event.is_set():
                await self._cleanup_pending_tools(session_id, assistant_msg_id)

            # Always set session back to idle
            await self.sessions.set_session_status(session_id, "idle")

        return {
            "message_id": assistant_msg_id,
            "final_text": final_text,
            "steps": step,
            "cost": total_cost,
            "tokens": total_tokens,
            "blocked": blocked,
        }

    def _build_tool_schemas(self, agent_info: Optional[AgentInfo] = None) -> List[Dict[str, Any]]:
        """Build tool schemas for the LLM from the tool registry.

        If agent_info is provided, tools denied by the agent's permission ruleset
        are excluded from the schema list.
        """
        all_tool_ids = self.tools.list()

        # Filter out disabled tools based on agent permissions
        if agent_info and agent_info.permission:
            denied = disabled_tools(agent_info.permission, all_tool_ids)
            allowed_ids = [tid for tid in all_tool_ids if tid not in denied]
        else:
            allowed_ids = all_tool_ids

        schemas = []
        for tool_id in allowed_ids:
            tool = self.tools.get(tool_id)
            if tool:
                schemas.append(_tool_to_schema(tool))
        return schemas

    DOOM_LOOP_THRESHOLD = 3

    def _detect_doom_loop(self, recent_tool_calls: List[Dict[str, Any]],
                          agent_info: Optional[AgentInfo] = None) -> bool:
        """Check if the last N tool calls are identical (same name + same arguments).

        Returns True if doom loop detected and permission denies continuation.
        """
        threshold = self.DOOM_LOOP_THRESHOLD
        if len(recent_tool_calls) < threshold:
            return False

        last_n = recent_tool_calls[-threshold:]
        first = last_n[0]
        all_same = all(
            tc["name"] == first["name"] and json.dumps(tc["arguments"], sort_keys=True) == json.dumps(first["arguments"], sort_keys=True)
            for tc in last_n
        )
        if not all_same:
            return False

        # Check doom_loop permission from agent ruleset
        if agent_info and agent_info.permission:
            from .permission_next import evaluate_rules
            action = evaluate_rules(agent_info.permission, "doom_loop",
                                     metadata={"tool": first["name"]})
            if action == "allow":
                return False  # Explicitly allowed to continue
            # "deny" or "ask" -> stop
        return True

    def _calculate_cost(self, model: str, usage: Dict[str, Any]) -> float:
        """Calculate cost for a step based on model and token usage.

        Uses cost-per-million-tokens from config if available, otherwise returns 0.
        """
        if not usage:
            return 0.0
        try:
            from .config import get_config
            cfg = get_config()
            # Look for model cost in config (provider.*.options.costs.{model_id})
            # For now use a built-in pricing table for common models
            cost_table = _MODEL_COSTS.get(model)
            if not cost_table:
                return 0.0

            input_tokens = usage.get("input", 0)
            output_tokens = usage.get("output", 0)
            reasoning_tokens = usage.get("reasoning", 0)
            cache_read = usage.get("cache", {}).get("read", 0)
            cache_write = usage.get("cache", {}).get("write", 0)

            cost = (
                input_tokens * cost_table.get("input", 0) / 1_000_000
                + output_tokens * cost_table.get("output", 0) / 1_000_000
                + reasoning_tokens * cost_table.get("output", 0) / 1_000_000  # reasoning at output rate
                + cache_read * cost_table.get("cache_read", 0) / 1_000_000
                + cache_write * cost_table.get("cache_write", 0) / 1_000_000
            )
            return cost
        except Exception:
            return 0.0

    async def _cleanup_pending_tools(self, session_id: str, message_id: str):
        """Mark any pending/running tool parts as error on abort."""
        try:
            msg = await self.sessions.get_message(session_id, message_id)
            if msg and hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'state') and part.state:
                        status = part.state.get('status', '') if isinstance(part.state, dict) else ''
                        if status in ('pending', 'running'):
                            await self.sessions.update_part_state(
                                session_id, message_id, part.id,
                                {"status": "error", "error": "Tool execution aborted"}
                            )
        except Exception:
            pass

    async def _try_compact(self, session_id: str):
        """Check if compaction is needed and run it."""
        try:
            from .session_compaction import SessionCompaction
            compactor = SessionCompaction(self.sessions, self.providers, self.events)
            should = await compactor.should_compact(session_id)
            if should:
                await compactor.compact(session_id)
        except Exception:
            pass
