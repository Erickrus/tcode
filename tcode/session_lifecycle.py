"""Session lifecycle operations: fork, revert, archive.

Follows opencode session/index.ts and session/revert.ts patterns.
"""
from __future__ import annotations
import time
from typing import Optional, Dict, Any, List
from .session import SessionManager
from .event import EventBus, Event
from .util import next_id


async def fork_session(
    sessions: SessionManager,
    session_id: str,
    upto_message_id: Optional[str] = None,
) -> str:
    """Fork a session, creating a child with copies of messages up to an optional point.

    Returns the new session ID.
    """
    # Get original session
    orig = await sessions.get_session(session_id)
    orig_title = ""
    if orig:
        meta = orig.get("metadata", {}) or {}
        orig_title = orig.get("title", "") or meta.get("title", "")

    # Determine fork number
    fork_num = 1
    if orig_title:
        import re
        m = re.search(r"\(fork #(\d+)\)$", orig_title)
        if m:
            fork_num = int(m.group(1)) + 1

    fork_title = f"{orig_title} (fork #{fork_num})" if orig_title else f"Fork #{fork_num}"

    # Create new session
    new_sid = await sessions.create_session(
        metadata={"title": fork_title, "forked_from": session_id}
    )

    # Copy messages (chronological order)
    # Collect all messages first so we can handle upto_message_id correctly
    all_messages = []
    async for wp in sessions.stream_messages(session_id):
        all_messages.append(wp)

    # If upto_message_id specified, find its index and truncate
    if upto_message_id:
        upto_idx = None
        for i, wp in enumerate(all_messages):
            if wp.info.get("id") == upto_message_id:
                upto_idx = i
                break
        if upto_idx is not None:
            all_messages = all_messages[: upto_idx + 1]

    parent_map: Dict[str, str] = {}  # old_msg_id -> new_msg_id

    for wp in all_messages:
        old_msg_id = wp.info.get("id", "")
        role = wp.info.get("role", "user")
        model = wp.info.get("model")
        old_parent = wp.info.get("parent_id")

        # Map parent
        new_parent = parent_map.get(old_parent) if old_parent else None

        new_msg_id = await sessions.create_message(
            new_sid, role, model=model, parent_id=new_parent,
        )
        parent_map[old_msg_id] = new_msg_id

        # Copy parts
        for part in wp.parts:
            ptype = part.get("type", "")
            if ptype == "text":
                await sessions.append_text_part(
                    new_sid, new_msg_id, part.get("text", ""),
                    synthetic=part.get("synthetic", False),
                )
            elif ptype == "tool":
                part_id = await sessions.insert_tool_part(
                    new_sid, new_msg_id,
                    part.get("call_id", ""), part.get("tool", ""),
                    part.get("state", {}).get("input", {}),
                )
                state = part.get("state", {})
                if state.get("status") in ("completed", "error"):
                    await sessions.update_part_state(
                        new_sid, new_msg_id, part_id, state,
                    )
            elif ptype == "step-start":
                await sessions.insert_step_start_part(new_sid, new_msg_id)
            elif ptype == "step-finish":
                await sessions.insert_step_finish_part(
                    new_sid, new_msg_id,
                    reason=part.get("reason", ""),
                    cost=part.get("cost", 0.0),
                    tokens=part.get("tokens"),
                )

    # Publish event
    seq = await sessions.storage.next_sequence(new_sid)
    ev = Event.create(
        "session.forked",
        {"sessionID": new_sid, "forkedFrom": session_id},
        session_id=new_sid, sequence=seq,
    )
    await sessions.events.publish(ev)

    return new_sid


async def archive_session(
    sessions: SessionManager,
    session_id: str,
) -> None:
    """Soft-delete a session by setting time_archived."""
    sess = await sessions.get_session(session_id)
    if not sess:
        raise KeyError("session not found")
    meta = sess.get("metadata", {}) or {}
    meta["time_archived"] = int(time.time())
    try:
        await sessions.storage.update_session(session_id, meta)
    except Exception:
        pass
    seq = await sessions.storage.next_sequence(session_id)
    ev = Event.create(
        "session.archived",
        {"sessionID": session_id},
        session_id=session_id, sequence=seq,
    )
    await sessions.events.publish(ev)


async def unarchive_session(
    sessions: SessionManager,
    session_id: str,
) -> None:
    """Restore an archived session."""
    sess = await sessions.get_session(session_id)
    if not sess:
        raise KeyError("session not found")
    meta = sess.get("metadata", {}) or {}
    meta.pop("time_archived", None)
    try:
        await sessions.storage.update_session(session_id, meta)
    except Exception:
        pass


async def set_revert(
    sessions: SessionManager,
    session_id: str,
    message_id: str,
    snapshot: Optional[str] = None,
    diff: Optional[str] = None,
) -> None:
    """Mark a session revert point."""
    sess = await sessions.get_session(session_id)
    if not sess:
        raise KeyError("session not found")
    meta = sess.get("metadata", {}) or {}
    meta["revert"] = {
        "messageID": message_id,
        "snapshot": snapshot,
        "diff": diff,
        "time": int(time.time()),
    }
    await sessions.storage.update_session(session_id, meta)
    seq = await sessions.storage.next_sequence(session_id)
    ev = Event.create(
        "session.revert.set",
        {"sessionID": session_id, "messageID": message_id},
        session_id=session_id, sequence=seq,
    )
    await sessions.events.publish(ev)


async def clear_revert(
    sessions: SessionManager,
    session_id: str,
) -> None:
    """Clear a session revert point."""
    sess = await sessions.get_session(session_id)
    if not sess:
        raise KeyError("session not found")
    meta = sess.get("metadata", {}) or {}
    meta.pop("revert", None)
    await sessions.storage.update_session(session_id, meta)
