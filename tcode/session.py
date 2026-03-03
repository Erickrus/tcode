from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal
from typing import Optional, List, Dict, Any, AsyncIterator
from .event import EventBus, Event
from .storage import Storage
from .storage_sqlite import SQLiteStorage
from .util import next_id
import asyncio
import time

# Define Part and Message models reflecting MessageV2 shapes (simplified for v1)

class PartBase(BaseModel):
    id: str
    session_id: str
    message_id: str
    type: str

class TextPart(PartBase):
    type: Literal["text"] = Field("text")
    text: str
    synthetic: Optional[bool] = False
    ignored: Optional[bool] = False
    time: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class ReasoningPart(PartBase):
    type: Literal["reasoning"] = Field("reasoning")
    text: str
    metadata: Optional[Dict[str, Any]] = None
    time: Dict[str, Any]

class FilePart(PartBase):
    type: Literal["file"] = Field("file")
    mime: str
    filename: Optional[str] = None
    url: str
    source: Optional[Dict[str, Any]] = None

class ToolStatePending(BaseModel):
    status: Literal["pending"] = Field("pending")
    input: Dict[str, Any]
    raw: Optional[str] = None

class ToolStateRunning(BaseModel):
    status: Literal["running"] = Field("running")
    input: Dict[str, Any]
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    time: Dict[str, int]

class ToolStateCompleted(BaseModel):
    status: Literal["completed"] = Field("completed")
    input: Dict[str, Any]
    output: str
    title: str
    metadata: Dict[str, Any]
    time: Dict[str, int]
    attachments: Optional[List[FilePart]] = None

class ToolStateError(BaseModel):
    status: Literal["error"] = Field("error")
    input: Dict[str, Any]
    error: str
    metadata: Optional[Dict[str, Any]] = None
    time: Dict[str, int]

from typing import Union
ToolState = Union[ToolStatePending, ToolStateRunning, ToolStateCompleted, ToolStateError]

class ToolPart(PartBase):
    type: Literal["tool"] = Field("tool")
    call_id: str
    tool: str
    state: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class StepStartPart(PartBase):
    """Emitted at the start of each LLM turn."""
    type: Literal["step-start"] = Field("step-start")
    snapshot: Optional[str] = None


class StepFinishPart(PartBase):
    """Emitted at the end of each LLM turn with usage/cost data."""
    type: Literal["step-finish"] = Field("step-finish")
    reason: str = ""  # finish reason (e.g., "stop", "tool_calls", "max_tokens")
    snapshot: Optional[str] = None
    cost: float = 0.0  # cost in USD
    tokens: Dict[str, Any] = Field(default_factory=lambda: {
        "input": 0, "output": 0, "reasoning": 0,
        "cache": {"read": 0, "write": 0},
    })


class CompactionPart(PartBase):
    """Marks a compaction event. Summary text stored separately or in text field."""
    type: Literal["compaction"] = Field("compaction")
    auto: bool = True
    text: Optional[str] = None  # summary text for compose_messages


class SubtaskPart(PartBase):
    """Represents a delegated subtask."""
    type: Literal["subtask"] = Field("subtask")
    prompt: str = ""
    description: str = ""
    agent: str = ""
    model: Optional[Dict[str, str]] = None
    command: Optional[str] = None


class AgentPart(PartBase):
    """Marks an agent switch."""
    type: Literal["agent"] = Field("agent")
    name: str = ""
    source: Optional[Dict[str, Any]] = None


class RetryPart(PartBase):
    """Records a retry attempt."""
    type: Literal["retry"] = Field("retry")
    attempt: int = 0
    error: Dict[str, Any] = Field(default_factory=dict)
    time: Dict[str, Any] = Field(default_factory=dict)


class PatchPart(PartBase):
    """Records file diffs from a step."""
    type: Literal["patch"] = Field("patch")
    hash: str = ""
    files: List[str] = Field(default_factory=list)


Part = Union[
    TextPart, ReasoningPart, FilePart, ToolPart,
    StepStartPart, StepFinishPart, CompactionPart,
    SubtaskPart, AgentPart, RetryPart, PatchPart,
]

class Message(BaseModel):
    id: str
    session_id: str
    role: str
    time: Dict[str, Any]
    parent_id: Optional[str] = None
    model: Optional[Dict[str, str]] = None
    system: Optional[str] = None
    tools: Optional[Dict[str, bool]] = None
    parts: List[Dict[str, Any]] = []
    summary: Optional[Dict[str, Any]] = None

class WithParts(BaseModel):
    info: Dict[str, Any]
    parts: List[Dict[str, Any]]


class SessionManager:
    def __init__(self, storage: Optional[Storage] = None, events: Optional[EventBus] = None):
        # default to SQLite storage if not provided
        if storage is None:
            storage = SQLiteStorage()
            # ensure initialized
            import asyncio
            asyncio.get_event_loop().run_until_complete(storage.init())
        self.storage: Storage = storage
        self.events: EventBus = events or EventBus()

    async def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        session_id = next_id("session")
        await self.storage.create_session(session_id, metadata)
        ev = Event.create("session.created", {"session_id": session_id}, session_id=session_id, sequence=await self.storage.next_sequence(session_id))
        await self.events.publish(ev)
        return session_id

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        sess = await self.storage.get_session(session_id)
        if not sess:
            raise KeyError("session not found")
        return sess

    async def set_permission(self, session_id: str, rules: List[Dict[str, Any]]) -> None:
        sess = await self.storage.get_session(session_id)
        if not sess:
            raise KeyError("session not found")
        meta = sess.get('metadata', {}) or {}
        meta['permissions'] = rules
        # persist to metadata
        try:
            await self.storage.update_session(session_id, meta)
        except Exception:
            pass
        # Also persist to permissions table if storage supports it
        try:
            if hasattr(self.storage, 'save_permissions'):
                await self.storage.save_permissions(session_id, rules)
        except Exception:
            pass

    async def get_permission(self, session_id: str) -> List[Dict[str, Any]]:
        sess = await self.storage.get_session(session_id)
        if not sess:
            raise KeyError("session not found")
        meta = sess.get('metadata', {}) or {}
        rules = meta.get('permissions')
        if rules:
            return rules
        # Fall back to permissions table
        try:
            if hasattr(self.storage, 'load_permissions'):
                stored = await self.storage.load_permissions(session_id)
                if stored:
                    return stored
        except Exception:
            pass
        return []

    async def add_permission_rule(self, session_id: str, rule: Dict[str, Any]) -> None:
        """Append a single rule to the session's permission ruleset (e.g., for 'always' approval)."""
        current = await self.get_permission(session_id)
        current.append(rule)
        await self.set_permission(session_id, current)

    async def set_project_permission(self, project_id: str, rules: List[Dict[str, Any]]) -> None:
        """Save permission rules scoped to a project.

        Writes to .tcode/permissions.json (human-readable) and SQLite (backup).
        """
        # Write to .tcode/permissions.json
        try:
            from .instance import Instance
            inst = Instance.current()
            if not inst:
                # Try to find instance from cache by project_id
                from .instance import _instance_cache
                for cached in _instance_cache.values():
                    if cached.project_id == project_id:
                        inst = cached
                        break
            if inst:
                import json as _json, os as _os
                config_dir = inst.config_dir
                _os.makedirs(config_dir, exist_ok=True)
                path = _os.path.join(config_dir, "permissions.json")
                await asyncio.to_thread(
                    lambda: open(path, 'w', encoding='utf-8').write(
                        _json.dumps(rules, indent=2) + '\n'
                    )
                )
        except Exception:
            pass
        # Also persist to SQLite as backup
        try:
            if hasattr(self.storage, 'save_project_permissions'):
                await self.storage.save_project_permissions(project_id, rules)
        except Exception:
            pass

    async def get_project_permission(self, project_id: str) -> List[Dict[str, Any]]:
        """Load permission rules scoped to a project.

        Reads from .tcode/permissions.json first, falls back to SQLite.
        """
        # Try .tcode/permissions.json first (user may have edited it)
        try:
            from .instance import Instance
            inst = Instance.current()
            if not inst:
                from .instance import _instance_cache
                for cached in _instance_cache.values():
                    if cached.project_id == project_id:
                        inst = cached
                        break
            if inst:
                import json as _json, os as _os
                path = _os.path.join(inst.config_dir, "permissions.json")
                if _os.path.isfile(path):
                    text = await asyncio.to_thread(
                        lambda: open(path, 'r', encoding='utf-8').read()
                    )
                    rules = _json.loads(text)
                    if isinstance(rules, list):
                        return rules
        except Exception:
            pass
        # Fall back to SQLite
        try:
            if hasattr(self.storage, 'load_project_permissions'):
                stored = await self.storage.load_project_permissions(project_id)
                if stored:
                    return stored
        except Exception:
            pass
        return []

    async def add_project_permission_rule(self, project_id: str, rule: Dict[str, Any]) -> None:
        """Append a single rule to the project's permission ruleset."""
        current = await self.get_project_permission(project_id)
        current.append(rule)
        await self.set_project_permission(project_id, current)

    async def create_message(self, session_id: str, role: str, model: Optional[Dict[str, str]] = None,
                             system: Optional[str] = None, parent_id: Optional[str] = None) -> str:
        message_id = next_id("message")
        msg = {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "time": {"created": int(time.time())},
            "parent_id": parent_id,
            "model": model,
            "system": system,
            "parts": [],
        }
        await self.storage.save_message(msg)
        seq = await self.storage.next_sequence(session_id)
        ev = Event.create("message.updated", {"info": msg}, session_id=session_id, sequence=seq)
        await self.events.publish(ev)
        return message_id

    async def append_text_part(self, session_id: str, message_id: str, text: str, synthetic: bool = False) -> str:
        part_id = next_id("part")
        part = {
            "id": part_id,
            "session_id": session_id,
            "message_id": message_id,
            "type": "text",
            "text": text,
            "synthetic": synthetic,
            "time": {"start": int(time.time())},
        }
        await self.storage.append_part(part)
        seq = await self.storage.next_sequence(session_id)
        # delta event for low-latency
        delta = Event.create("message.part.delta", {"sessionID": session_id, "messageID": message_id, "partID": part_id, "field": "text", "delta": text}, session_id=session_id, sequence=seq)
        await self.events.publish(delta)
        # full part updated
        seq = await self.storage.next_sequence(session_id)
        ev = Event.create("message.part.updated", {"part": part}, session_id=session_id, sequence=seq)
        await self.events.publish(ev)
        return part_id

    async def insert_tool_part(self, session_id: str, message_id: str, call_id: str, tool: str, input: Dict[str, Any]) -> str:
        part_id = next_id("part")
        part = {
            "id": part_id,
            "session_id": session_id,
            "message_id": message_id,
            "type": "tool",
            "call_id": call_id,
            "tool": tool,
            "state": {"status": "pending", "input": input},
        }
        await self.storage.append_part(part)
        seq = await self.storage.next_sequence(session_id)
        ev = Event.create("message.part.updated", {"part": part}, session_id=session_id, sequence=seq)
        await self.events.publish(ev)
        return part_id

    async def update_part_state(self, session_id: str, message_id: str, part_id: str, new_state: Dict[str, Any]) -> None:
        await self.storage.update_part(part_id, {"state": new_state})
        part = await self.storage.get_part(part_id)
        seq = await self.storage.next_sequence(session_id)
        ev = Event.create("message.part.updated", {"part": part}, session_id=session_id, sequence=seq)
        await self.events.publish(ev)

    async def remove_part(self, session_id: str, message_id: str, part_id: str) -> None:
        await self.storage.delete_part(part_id)
        seq = await self.storage.next_sequence(session_id)
        ev = Event.create("message.part.removed", {"sessionID": session_id, "messageID": message_id, "partID": part_id}, session_id=session_id, sequence=seq)
        await self.events.publish(ev)

    async def insert_step_start_part(self, session_id: str, message_id: str, snapshot: Optional[str] = None) -> str:
        part_id = next_id("part")
        part = {
            "id": part_id, "session_id": session_id, "message_id": message_id,
            "type": "step-start", "snapshot": snapshot,
        }
        await self.storage.append_part(part)
        seq = await self.storage.next_sequence(session_id)
        ev = Event.create("message.part.updated", {"part": part}, session_id=session_id, sequence=seq)
        await self.events.publish(ev)
        return part_id

    async def insert_step_finish_part(self, session_id: str, message_id: str,
                                       reason: str = "", cost: float = 0.0,
                                       tokens: Optional[Dict[str, Any]] = None,
                                       snapshot: Optional[str] = None) -> str:
        part_id = next_id("part")
        part = {
            "id": part_id, "session_id": session_id, "message_id": message_id,
            "type": "step-finish", "reason": reason, "cost": cost,
            "tokens": tokens or {"input": 0, "output": 0, "reasoning": 0, "cache": {"read": 0, "write": 0}},
            "snapshot": snapshot,
        }
        await self.storage.append_part(part)
        seq = await self.storage.next_sequence(session_id)
        ev = Event.create("message.part.updated", {"part": part}, session_id=session_id, sequence=seq)
        await self.events.publish(ev)
        return part_id

    async def insert_compaction_part(self, session_id: str, message_id: str,
                                      text: str = "", auto: bool = True) -> str:
        part_id = next_id("part")
        part = {
            "id": part_id, "session_id": session_id, "message_id": message_id,
            "type": "compaction", "auto": auto, "text": text,
        }
        await self.storage.append_part(part)
        seq = await self.storage.next_sequence(session_id)
        ev = Event.create("message.part.updated", {"part": part}, session_id=session_id, sequence=seq)
        await self.events.publish(ev)
        return part_id

    async def insert_retry_part(self, session_id: str, message_id: str,
                                 attempt: int = 0, error: Optional[Dict[str, Any]] = None) -> str:
        part_id = next_id("part")
        part = {
            "id": part_id, "session_id": session_id, "message_id": message_id,
            "type": "retry", "attempt": attempt, "error": error or {},
            "time": {"created": int(time.time())},
        }
        await self.storage.append_part(part)
        seq = await self.storage.next_sequence(session_id)
        ev = Event.create("message.part.updated", {"part": part}, session_id=session_id, sequence=seq)
        await self.events.publish(ev)
        return part_id

    async def stream_messages(self, session_id: str) -> AsyncIterator[WithParts]:
        msgs = await self.storage.list_messages(session_id)
        # storage.list_messages returns newest first; reverse for chronological
        for m in reversed(msgs):
            # info as plain dict for compatibility with compose_messages
            info = m
            parts = m.get('parts', [])
            yield WithParts(info=info, parts=parts)

    async def get_message(self, session_id: str, message_id: str) -> WithParts:
        msg = await self.storage.get_message(message_id)
        if not msg:
            raise KeyError("message not found")
        # storage.get_message returns dict; return dict info for compatibility
        parts = msg.get('parts', [])
        return WithParts(info=msg, parts=parts)

    async def compose_messages(self, session_id: str, upto_message_id: Optional[str] = None,
                               system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """Compose provider messages from session history.

        Returns messages in OpenAI-compatible format:
          - {role: "system", content: str}
          - {role: "user", content: str | list[content_parts]}
          - {role: "assistant", content: str, tool_calls: [...]}
          - {role: "tool", tool_call_id: str, content: str}

        Follows opencode's toModelMessages() pattern (message-v2.ts:491-700):
          - User msgs: text parts (skip ignored), file parts, compaction parts as summary
          - Assistant msgs: text + reasoning + tool call parts
          - Tool results: separate role="tool" messages
          - Pending/running tools: emit error marker to prevent dangling tool_use blocks
        """
        result: List[Dict[str, Any]] = []

        # System prompt: explicit param > session metadata
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        else:
            sess = await self.storage.get_session(session_id)
            if sess:
                meta = sess.get("metadata") or {}
                sys_p = meta.get("system_prompt")
                if sys_p:
                    result.append({"role": "system", "content": sys_p})

        async for wp in self.stream_messages(session_id):
            info = wp.info
            role = info.get("role", "user")

            if role == "user":
                # Collect user content parts
                content_parts = []
                for part in wp.parts:
                    ptype = part.get("type")
                    if ptype == "text" and not part.get("ignored"):
                        content_parts.append(part.get("text", ""))
                    elif ptype == "file":
                        # Include file reference as text description for now
                        fname = part.get("filename") or part.get("url", "")
                        content_parts.append(f"[File: {fname}]")
                    elif ptype == "compaction":
                        # Compaction summary becomes context
                        summary = part.get("text") or part.get("summary") or "Previous conversation was summarized."
                        content_parts.append(summary)
                if content_parts:
                    result.append({"role": "user", "content": "\n".join(content_parts)})

            elif role == "assistant":
                # Collect assistant text and tool calls
                text_parts = []
                tool_calls = []
                tool_results_to_add = []

                for part in wp.parts:
                    ptype = part.get("type")
                    if ptype == "text":
                        text_parts.append(part.get("text", ""))
                    elif ptype == "reasoning":
                        # Reasoning can be included as text (provider-dependent)
                        pass
                    elif ptype == "tool":
                        state = part.get("state", {})
                        status = state.get("status", "pending")
                        call_id = part.get("call_id", "")
                        tool_name = part.get("tool", "")
                        tool_input = state.get("input", {})

                        # Add the tool call to the assistant message
                        import json as _json
                        tool_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": _json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input),
                            },
                        })

                        # Add corresponding tool result message
                        if status == "completed":
                            output = state.get("output", "")
                            # If output was compacted, use placeholder
                            if state.get("time", {}).get("compacted"):
                                output = "[Old tool result content cleared]"
                            tool_results_to_add.append({
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": output if isinstance(output, str) else _json.dumps(output),
                            })
                        elif status == "error":
                            error_msg = state.get("error", "Tool execution failed")
                            tool_results_to_add.append({
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": f"Error: {error_msg}",
                            })
                        elif status in ("pending", "running"):
                            # Prevent dangling tool_use blocks
                            tool_results_to_add.append({
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": "[Tool execution was interrupted]",
                            })

                # Build assistant message
                assistant_msg: Dict[str, Any] = {"role": "assistant"}
                content = "\n".join(text_parts) if text_parts else None
                if content:
                    assistant_msg["content"] = content
                else:
                    assistant_msg["content"] = None
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls

                # Only add if there's content or tool calls
                if content or tool_calls:
                    result.append(assistant_msg)
                    # Tool results must come right after the assistant message
                    result.extend(tool_results_to_add)

            # Stop if we've reached the target message
            if upto_message_id and info.get("id") == upto_message_id:
                break

        return result

    async def set_session_status(self, session_id: str, status: str) -> None:
        sess = await self.storage.get_session(session_id)
        if not sess:
            raise KeyError("session not found")
        sess["status"] = status
        seq = await self.storage.next_sequence(session_id)
        ev = Event.create("session.status.changed", {"sessionID": session_id, "status": status}, session_id=session_id, sequence=seq)
        await self.events.publish(ev)

    async def set_summary(self, session_id: str, summary: Dict[str, Any]) -> None:
        """Persist a session-level summary in metadata."""
        sess = await self.storage.get_session(session_id)
        if not sess:
            raise KeyError("session not found")
        meta = sess.get('metadata', {}) or {}
        meta['summary'] = summary
        try:
            await self.storage.update_session(session_id, meta)
        except Exception:
            pass
        # publish event
        seq = await self.storage.next_sequence(session_id)
        ev = Event.create("session.summary.updated", {"sessionID": session_id, "summary": summary}, session_id=session_id, sequence=seq)
        await self.events.publish(ev)
