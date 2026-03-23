"""Flat-file storage backend for tcode.

Stores all persistent data as JSON files under a base directory (typically .tcode/):

    <base_dir>/
    ├── project.json
    ├── permissions.json
    └── sessions/
        └── <session_id>/
            ├── session.json
            ├── messages.json
            ├── todos.json
            └── permissions.json

Self-bootstrapping: init() creates the base dir. create_session() creates session subdirs.
Reads return sensible defaults (empty list, None) when files don't exist.
"""
from __future__ import annotations
import json
import os
import time as _time
import asyncio
import tempfile
import logging
from typing import Optional, Dict, Any, List

log = logging.getLogger(__name__)


_REPLACE_MAX_RETRIES = 5
_REPLACE_BASE_DELAY = 0.05  # seconds


def _replace_with_retry(src: str, dst: str) -> None:
    """os.replace with retry loop for transient PermissionError / OSError.

    On Windows (and occasionally other platforms), antivirus scanners, search
    indexers, or concurrent readers can briefly lock the target file, causing
    os.replace() to fail.  Retrying after a short, exponentially increasing
    delay resolves the issue in virtually all cases.
    """
    for attempt in range(_REPLACE_MAX_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_MAX_RETRIES - 1:
                raise
            _time.sleep(_REPLACE_BASE_DELAY * (2 ** attempt))


def _atomic_write(path: str, data: Any) -> None:
    """Write JSON atomically via temp file + os.replace()."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
            f.write("\n")
        _replace_with_retry(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: str, default=None):
    """Read JSON file, returning default if missing or corrupt."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


class FileStorage:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(os.getcwd(), ".tcode")
        self._lock = asyncio.Lock()
        # In-memory index: part_id -> (session_id, message_id)
        self._part_index: Dict[str, tuple] = {}

    def _session_dir(self, session_id: str) -> str:
        return os.path.join(self.base_dir, "sessions", session_id)

    def _project_path(self) -> str:
        return os.path.join(self.base_dir, "project.json")

    def _session_path(self, session_id: str) -> str:
        return os.path.join(self._session_dir(session_id), "session.json")

    def _messages_path(self, session_id: str) -> str:
        return os.path.join(self._session_dir(session_id), "messages.json")

    def _todos_path(self, session_id: str) -> str:
        return os.path.join(self._session_dir(session_id), "todos.json")

    def _session_permissions_path(self, session_id: str) -> str:
        return os.path.join(self._session_dir(session_id), "permissions.json")

    def _project_permissions_path(self) -> str:
        return os.path.join(self.base_dir, "permissions.json")

    # ---- Init / Close ----

    async def init(self):
        await asyncio.to_thread(os.makedirs, self.base_dir, exist_ok=True)

    async def close(self):
        pass  # no-op for file storage

    # ---- Project CRUD ----

    async def create_project(self, project_id: str, worktree: str, name: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None):
        def _write():
            ts = int(_time.time())
            project = _read_json(self._project_path(), {})
            if not isinstance(project, dict) or "projects" not in project:
                project = {"projects": {}}
            project["projects"][project_id] = {
                "id": project_id, "worktree": worktree, "name": name or "",
                "metadata": metadata or {}, "time_created": ts, "time_updated": ts,
            }
            _atomic_write(self._project_path(), project)
        await asyncio.to_thread(_write)

    async def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        def _read():
            data = _read_json(self._project_path(), {})
            projects = data.get("projects", {}) if isinstance(data, dict) else {}
            return projects.get(project_id)
        return await asyncio.to_thread(_read)

    async def list_projects(self) -> List[Dict[str, Any]]:
        def _read():
            data = _read_json(self._project_path(), {})
            projects = data.get("projects", {}) if isinstance(data, dict) else {}
            result = list(projects.values())
            result.sort(key=lambda p: p.get("time_updated", 0), reverse=True)
            return result
        return await asyncio.to_thread(_read)

    # ---- Session CRUD ----

    async def create_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None,
                              project_id: Optional[str] = None):
        def _write():
            sdir = self._session_dir(session_id)
            os.makedirs(sdir, exist_ok=True)
            ts = int(_time.time())
            session = {
                "id": session_id,
                "project_id": project_id,
                "metadata": metadata or {},
                "status": "idle",
                "title": "",
                "parent_id": None,
                "next_sequence": 1,
                "created_at": ts,
                "updated_at": ts,
            }
            _atomic_write(self._session_path(session_id), session)
            # Initialize empty messages and todos
            _atomic_write(self._messages_path(session_id), [])
            _atomic_write(self._todos_path(session_id), [])
        await asyncio.to_thread(_write)

    async def update_session(self, session_id: str, metadata: Dict[str, Any]):
        def _write():
            path = self._session_path(session_id)
            session = _read_json(path)
            if not session:
                return
            session["metadata"] = metadata or {}
            session["updated_at"] = int(_time.time())
            _atomic_write(path, session)
        await asyncio.to_thread(_write)

    async def update_session_field(self, session_id: str, field: str, value: Any):
        allowed = {"title", "status", "parent_id", "project_id"}
        if field not in allowed:
            raise ValueError(f"Cannot update field: {field}")

        def _write():
            path = self._session_path(session_id)
            session = _read_json(path)
            if not session:
                return
            session[field] = value
            session["updated_at"] = int(_time.time())
            _atomic_write(path, session)
        await asyncio.to_thread(_write)

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        def _read():
            session = _read_json(self._session_path(session_id))
            if not session:
                return None
            return {
                "id": session.get("id", session_id),
                "project_id": session.get("project_id"),
                "metadata": session.get("metadata", {}),
                "status": session.get("status", "idle"),
                "title": session.get("title", ""),
                "parent_id": session.get("parent_id"),
                "next_sequence": session.get("next_sequence", 1),
                "created_at": session.get("created_at"),
                "updated_at": session.get("updated_at"),
            }
        return await asyncio.to_thread(_read)

    async def list_sessions(self, project_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        def _read():
            sessions_dir = os.path.join(self.base_dir, "sessions")
            if not os.path.isdir(sessions_dir):
                return []
            result = []
            for entry in os.listdir(sessions_dir):
                spath = os.path.join(sessions_dir, entry, "session.json")
                session = _read_json(spath)
                if not session:
                    continue
                if project_id and session.get("project_id") != project_id:
                    continue
                result.append({
                    "id": session.get("id", entry),
                    "project_id": session.get("project_id"),
                    "metadata": session.get("metadata", {}),
                    "status": session.get("status", "idle"),
                    "title": session.get("title", ""),
                    "created_at": session.get("created_at"),
                    "updated_at": session.get("updated_at"),
                })
            result.sort(key=lambda s: s.get("updated_at", 0), reverse=True)
            return result[offset:offset + limit]
        return await asyncio.to_thread(_read)

    # ---- Message CRUD ----

    async def save_message(self, message: Dict[str, Any]):
        def _write():
            session_id = message["session_id"]
            path = self._messages_path(session_id)
            messages = _read_json(path, [])
            # Replace existing message or append new
            msg_data = {
                "id": message["id"],
                "session_id": message["session_id"],
                "role": message["role"],
                "parent_id": message.get("parent_id"),
                "model": message.get("model") or {},
                "metadata": message.get("metadata") or {},
                "time": message.get("time", {"created": int(_time.time())}),
                "parts": message.get("parts", []),
            }
            # Check if message already exists (replace)
            for i, m in enumerate(messages):
                if m.get("id") == message["id"]:
                    messages[i] = msg_data
                    _atomic_write(path, messages)
                    return
            messages.append(msg_data)
            _atomic_write(path, messages)
        await asyncio.to_thread(_write)

    async def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        def _read():
            # Try part index first for session_id hint
            hint = self._part_index.get(message_id)
            if hint:
                session_id = hint[0]
                messages = _read_json(self._messages_path(session_id), [])
                for m in messages:
                    if m.get("id") == message_id:
                        return {
                            "id": m["id"],
                            "session_id": m["session_id"],
                            "role": m["role"],
                            "time": m.get("time", {"created": 0}),
                            "parts": m.get("parts", []),
                        }
            # Scan all sessions
            sessions_dir = os.path.join(self.base_dir, "sessions")
            if not os.path.isdir(sessions_dir):
                return None
            for entry in os.listdir(sessions_dir):
                messages = _read_json(os.path.join(sessions_dir, entry, "messages.json"), [])
                for m in messages:
                    if m.get("id") == message_id:
                        return {
                            "id": m["id"],
                            "session_id": m.get("session_id", entry),
                            "role": m["role"],
                            "time": m.get("time", {"created": 0}),
                            "parts": m.get("parts", []),
                        }
            return None
        return await asyncio.to_thread(_read)

    async def list_messages(self, session_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        def _read():
            messages = _read_json(self._messages_path(session_id), [])
            # Add insertion index for stable sort (like SQLite ROWID)
            indexed = [(i, m) for i, m in enumerate(messages)]
            # Sort by (time_created DESC, index DESC) to match SQLite behavior
            indexed.sort(
                key=lambda pair: (pair[1].get("time", {}).get("created", 0), pair[0]),
                reverse=True,
            )
            sliced = indexed[offset:offset + limit]
            result = []
            for _, m in sliced:
                result.append({
                    "id": m["id"],
                    "session_id": m.get("session_id", session_id),
                    "role": m["role"],
                    "time": m.get("time", {"created": 0}),
                    "parts": m.get("parts", []),
                })
            return result
        return await asyncio.to_thread(_read)

    # ---- Part CRUD ----

    def _find_message_index(self, messages: list, message_id: str) -> int:
        for i, m in enumerate(messages):
            if m.get("id") == message_id:
                return i
        return -1

    def _find_part_in_messages(self, messages: list, part_id: str):
        """Find part by ID across all messages. Returns (msg_index, part_index) or (-1, -1)."""
        for mi, m in enumerate(messages):
            for pi, p in enumerate(m.get("parts", [])):
                if isinstance(p, dict) and p.get("id") == part_id:
                    return mi, pi
        return -1, -1

    async def append_part(self, part: Dict[str, Any]):
        def _write():
            session_id = part["session_id"]
            message_id = part["message_id"]
            path = self._messages_path(session_id)
            messages = _read_json(path, [])
            mi = self._find_message_index(messages, message_id)
            if mi == -1:
                return
            parts = messages[mi].setdefault("parts", [])
            parts.append(part)
            _atomic_write(path, messages)
            # Update index
            self._part_index[part["id"]] = (session_id, message_id)
        await asyncio.to_thread(_write)

    async def update_part(self, part_id: str, patch: Dict[str, Any]):
        def _write():
            # Try index first
            hint = self._part_index.get(part_id)
            if hint:
                session_id, message_id = hint
                path = self._messages_path(session_id)
                messages = _read_json(path, [])
                mi, pi = self._find_part_in_messages(messages, part_id)
                if mi >= 0:
                    messages[mi]["parts"][pi].update(patch)
                    _atomic_write(path, messages)
                    return
            # Scan all sessions
            sessions_dir = os.path.join(self.base_dir, "sessions")
            if not os.path.isdir(sessions_dir):
                raise KeyError("part not found")
            for entry in os.listdir(sessions_dir):
                path = os.path.join(sessions_dir, entry, "messages.json")
                messages = _read_json(path, [])
                mi, pi = self._find_part_in_messages(messages, part_id)
                if mi >= 0:
                    messages[mi]["parts"][pi].update(patch)
                    _atomic_write(path, messages)
                    # Update index
                    msg_id = messages[mi].get("id", "")
                    self._part_index[part_id] = (entry, msg_id)
                    return
            raise KeyError("part not found")
        await asyncio.to_thread(_write)

    async def get_part(self, part_id: str) -> Optional[Dict[str, Any]]:
        def _read():
            # Try index first
            hint = self._part_index.get(part_id)
            if hint:
                session_id, message_id = hint
                messages = _read_json(self._messages_path(session_id), [])
                mi, pi = self._find_part_in_messages(messages, part_id)
                if mi >= 0:
                    return messages[mi]["parts"][pi]
            # Scan all sessions
            sessions_dir = os.path.join(self.base_dir, "sessions")
            if not os.path.isdir(sessions_dir):
                return None
            for entry in os.listdir(sessions_dir):
                messages = _read_json(os.path.join(sessions_dir, entry, "messages.json"), [])
                mi, pi = self._find_part_in_messages(messages, part_id)
                if mi >= 0:
                    msg_id = messages[mi].get("id", "")
                    self._part_index[part_id] = (entry, msg_id)
                    return messages[mi]["parts"][pi]
            return None
        return await asyncio.to_thread(_read)

    async def delete_part(self, part_id: str) -> None:
        def _write():
            # Try index first
            hint = self._part_index.get(part_id)
            if hint:
                session_id, message_id = hint
                path = self._messages_path(session_id)
                messages = _read_json(path, [])
                mi, pi = self._find_part_in_messages(messages, part_id)
                if mi >= 0:
                    messages[mi]["parts"].pop(pi)
                    _atomic_write(path, messages)
                    self._part_index.pop(part_id, None)
                    return
            # Scan all sessions
            sessions_dir = os.path.join(self.base_dir, "sessions")
            if not os.path.isdir(sessions_dir):
                return
            for entry in os.listdir(sessions_dir):
                path = os.path.join(sessions_dir, entry, "messages.json")
                messages = _read_json(path, [])
                mi, pi = self._find_part_in_messages(messages, part_id)
                if mi >= 0:
                    messages[mi]["parts"].pop(pi)
                    _atomic_write(path, messages)
                    self._part_index.pop(part_id, None)
                    return
        await asyncio.to_thread(_write)

    # ---- Todo CRUD ----

    async def update_todos(self, session_id: str, todos: List[Dict[str, Any]]):
        def _write():
            ts = int(_time.time())
            result = []
            for i, todo in enumerate(todos):
                result.append({
                    "content": todo.get("content", ""),
                    "status": todo.get("status", "pending"),
                    "priority": todo.get("priority", "medium"),
                    "position": i,
                    "time_created": ts,
                    "time_updated": ts,
                })
            path = self._todos_path(session_id)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _atomic_write(path, result)
        await asyncio.to_thread(_write)

    async def get_todos(self, session_id: str) -> List[Dict[str, Any]]:
        def _read():
            todos = _read_json(self._todos_path(session_id), [])
            todos.sort(key=lambda t: t.get("position", 0))
            return todos
        return await asyncio.to_thread(_read)

    # ---- Permission persistence ----

    async def save_permissions(self, session_id: str, rules: list):
        def _write():
            path = self._session_permissions_path(session_id)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _atomic_write(path, rules)
        await asyncio.to_thread(_write)

    async def load_permissions(self, session_id: str) -> Optional[list]:
        def _read():
            data = _read_json(self._session_permissions_path(session_id))
            if data is None:
                return None
            if isinstance(data, list):
                return data
            return None
        return await asyncio.to_thread(_read)

    async def save_project_permissions(self, project_id: str, rules: list):
        def _write():
            _atomic_write(self._project_permissions_path(), rules)
        await asyncio.to_thread(_write)

    async def load_project_permissions(self, project_id: str) -> Optional[list]:
        def _read():
            data = _read_json(self._project_permissions_path())
            if data is None:
                return None
            if isinstance(data, list):
                return data
            return None
        return await asyncio.to_thread(_read)

    # ---- Sequence ----

    async def next_sequence(self, session_id: str) -> int:
        def _rw():
            path = self._session_path(session_id)
            session = _read_json(path)
            if not session:
                raise KeyError("session not found")
            n = session.get("next_sequence", 1)
            session["next_sequence"] = n + 1
            _atomic_write(path, session)
            return n
        return await asyncio.to_thread(_rw)

    # ---- Transaction ----

    async def transaction(self):
        class Tx:
            async def __aenter__(tx_self):
                await self._lock.acquire()
                return self

            async def __aexit__(tx_self, exc_type, exc, tb):
                self._lock.release()
        return Tx()
