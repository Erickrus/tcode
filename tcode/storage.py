from __future__ import annotations
from typing import Dict, Any, List, Optional
import asyncio
import atexit
import time

# Minimal in-memory storage for v1. Replace with SQLite in next steps.

class Storage:
    def __init__(self):
        # sessions: id -> {metadata, status, next_sequence}
        self._sessions: Dict[str, Dict[str, Any]] = {}
        # messages: id -> dict
        self._messages: Dict[str, Dict[str, Any]] = {}
        # parts: id -> dict
        self._parts: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._msg_seq = 0  # monotonic counter for insertion-order stability

    async def init(self, path: Optional[str] = None):
        # no-op for in-memory
        return

    async def update_session(self, session_id: str, metadata: Dict[str, Any]):
        """Update session metadata for in-memory storage"""
        async with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session not found")
            sess["metadata"] = metadata

    async def create_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None):
        async with self._lock:
            self._sessions[session_id] = {
                "metadata": metadata or {},
                "status": "idle",
                "next_sequence": 1,
            }

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    async def save_message(self, message: Dict[str, Any]):
        async with self._lock:
            # normalize message shape for in-memory store to be compatible with SQLiteStorage
            msg = dict(message)
            self._msg_seq += 1
            msg["_seq"] = self._msg_seq
            # ensure parts is a list of part dicts (may be empty)
            parts = msg.get("parts") or []
            normalized_parts: List[Dict[str, Any]] = []
            for p in parts:
                if isinstance(p, dict):
                    normalized_parts.append(p)
                    self._parts[p.get("id")] = p
                elif isinstance(p, str):
                    # part id reference: try to resolve
                    part_obj = self._parts.get(p)
                    if part_obj:
                        normalized_parts.append(part_obj)
            msg["parts"] = normalized_parts
            # ensure time_created for sorting compatibility
            msg_time = msg.get("time", {})
            msg["time_created"] = int(msg_time.get("created", int(time.time())))
            self._messages[msg["id"]] = msg

    async def append_part(self, part: Dict[str, Any]):
        async with self._lock:
            # store part
            self._parts[part["id"]] = part
            # Also add part dict to message ordering
            msg = self._messages.get(part["message_id"]) if part.get("message_id") else None
            if msg is not None:
                parts = msg.setdefault("parts", [])
                parts.append(part)

    async def update_part(self, part_id: str, patch: Dict[str, Any]):
        async with self._lock:
            part = self._parts.get(part_id)
            if not part:
                raise KeyError("part not found")
            part.update(patch)
            # also update in any message.parts lists
            msg_id = part.get("message_id")
            if msg_id:
                msg = self._messages.get(msg_id)
                if msg:
                    parts = msg.setdefault("parts", [])
                    for i, p in enumerate(parts):
                        if isinstance(p, dict) and p.get("id") == part_id:
                            parts[i] = part

    async def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        msg = self._messages.get(message_id)
        if not msg:
            return None
        # ensure returned object has parts as list of dicts
        parts = msg.get("parts", [])
        normalized = []
        for p in parts:
            if isinstance(p, dict):
                normalized.append(p)
            elif isinstance(p, str):
                part_obj = self._parts.get(p)
                if part_obj:
                    normalized.append(part_obj)
        out = dict(msg)
        out["parts"] = normalized
        return out

    async def list_messages(self, session_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        # Collect messages for this session, preserving insertion order via _seq
        msgs = [m for m in self._messages.values() if m.get("session_id") == session_id]
        # Sort by (time_created, _seq) descending to get newest first with stable ordering
        msgs.sort(key=lambda m: (m.get("time_created", 0), m.get("_seq", 0)), reverse=True)
        slice_ = msgs[offset: offset + limit]
        # return shallow copies
        return [dict(m) for m in slice_]

    async def transaction(self):
        class Tx:
            async def __aenter__(self_inner):
                await self._lock.acquire()
                return self
            async def __aexit__(self_inner, exc_type, exc, tb):
                self._lock.release()
        return Tx()

    # helper to increment sequence atomically
    async def next_sequence(self, session_id: str) -> int:
        async with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session not found")
            n = sess.get("next_sequence", 1)
            sess["next_sequence"] = n + 1
            return n

    # part helpers
    async def get_part(self, part_id: str) -> Optional[Dict[str, Any]]:
        return self._parts.get(part_id)

    async def delete_part(self, part_id: str) -> None:
        async with self._lock:
            part = self._parts.pop(part_id, None)
            if part:
                msg_id = part.get("message_id")
                if msg_id:
                    msg = self._messages.get(msg_id)
                    if msg:
                        parts = msg.get("parts", [])
                        msg["parts"] = [p for p in parts if not (isinstance(p, dict) and p.get("id") == part_id)]

# Ensure cleanup if needed
atexit.register(lambda: None)
