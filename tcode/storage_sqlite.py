from __future__ import annotations
import sqlite3
import json
import os
import time as _time
import asyncio
import logging
from typing import Optional, Dict, Any, List

log = logging.getLogger(__name__)

# ---- Schema (v0 — initial) ----
# Applied via CREATE TABLE IF NOT EXISTS so it's safe for fresh DBs.
# Subsequent changes use the migration system below.

SCHEMA_V0 = '''
CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY,
  worktree TEXT NOT NULL,
  name TEXT,
  metadata TEXT,
  time_created INTEGER NOT NULL,
  time_updated INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  metadata TEXT,
  status TEXT,
  title TEXT DEFAULT '',
  parent_id TEXT,
  next_sequence INTEGER,
  created_at INTEGER,
  updated_at INTEGER,
  FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  role TEXT,
  parent_id TEXT,
  model TEXT,
  metadata TEXT,
  time_created INTEGER,
  time_completed INTEGER,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS parts (
  id TEXT PRIMARY KEY,
  message_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  type TEXT,
  data TEXT,
  time_created INTEGER,
  order_index INTEGER,
  FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS permissions (
  id TEXT PRIMARY KEY,
  session_id TEXT,
  project_id TEXT,
  rules TEXT,
  created_at INTEGER,
  updated_at INTEGER
);

CREATE TABLE IF NOT EXISTS todos (
  session_id TEXT NOT NULL,
  content TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  priority TEXT NOT NULL DEFAULT 'medium',
  position INTEGER NOT NULL,
  time_created INTEGER NOT NULL,
  time_updated INTEGER NOT NULL,
  PRIMARY KEY (session_id, position),
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS migrations (
  version INTEGER PRIMARY KEY,
  name TEXT,
  applied_at INTEGER NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_parts_message ON parts(message_id);
CREATE INDEX IF NOT EXISTS idx_parts_session ON parts(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_id);
CREATE INDEX IF NOT EXISTS idx_todos_session ON todos(session_id);
'''

# ---- Migration system ----
# Each migration is (version, name, sql). Applied in order; version tracked in `migrations` table.

MIGRATIONS: List[tuple] = [
    # v1: add session columns that may not exist in legacy DBs
    (1, "add_session_title_parent", """
        -- These are already in SCHEMA_V0 for new DBs; this handles upgrades of older DBs.
        -- SQLite ignores errors in executescript so we use ALTER TABLE ... if not exists pattern.
        -- We just try and ignore if column already exists.
    """),
]


class SQLiteStorage:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(os.getcwd(), "tcode.sqlite3")
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    def _ensure(self):
        if self._conn:
            return
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # PRAGMA optimizations (matching opencode db.ts)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA cache_size = -64000")  # 64MB
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

        # Apply base schema
        self._conn.executescript(SCHEMA_V0)
        self._conn.commit()

        # Run pending migrations
        self._run_migrations()

    def _run_migrations(self):
        """Apply pending migrations from MIGRATIONS list."""
        cur = self._conn.cursor()
        cur.execute("SELECT version FROM migrations ORDER BY version DESC LIMIT 1")
        row = cur.fetchone()
        current_version = row["version"] if row else 0

        for version, name, sql in MIGRATIONS:
            if version <= current_version:
                continue
            try:
                if sql.strip():
                    self._conn.executescript(sql)
                ts = int(_time.time())
                cur.execute(
                    "INSERT INTO migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (version, name, ts),
                )
                self._conn.commit()
                log.info("Applied migration v%d: %s", version, name)
            except Exception as e:
                log.warning("Migration v%d (%s) failed: %s", version, name, e)
                # Non-fatal — column may already exist etc.
                try:
                    cur.execute(
                        "INSERT OR IGNORE INTO migrations (version, name, applied_at) VALUES (?, ?, ?)",
                        (version, name, int(_time.time())),
                    )
                    self._conn.commit()
                except Exception:
                    pass

    def _get_migration_version(self) -> int:
        """Return current migration version."""
        cur = self._conn.cursor()
        cur.execute("SELECT version FROM migrations ORDER BY version DESC LIMIT 1")
        row = cur.fetchone()
        return row["version"] if row else 0

    async def init(self):
        await asyncio.to_thread(self._ensure)

    def _execute(self, sql: str, params=()):
        cur = self._conn.cursor()
        cur.execute(sql, params)
        self._conn.commit()
        return cur

    # ---- Project CRUD ----

    async def create_project(self, project_id: str, worktree: str, name: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None):
        await asyncio.to_thread(self._ensure)
        ts = int(_time.time())
        await asyncio.to_thread(
            self._execute,
            "INSERT OR REPLACE INTO projects (id, worktree, name, metadata, time_created, time_updated) VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, worktree, name or "", json.dumps(metadata or {}), ts, ts),
        )

    async def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        await asyncio.to_thread(self._ensure)
        cur = await asyncio.to_thread(self._execute, "SELECT * FROM projects WHERE id = ?", (project_id,))
        r = cur.fetchone()
        if not r:
            return None
        return {
            "id": r["id"], "worktree": r["worktree"], "name": r["name"],
            "metadata": json.loads(r["metadata"] or "{}"),
            "time_created": r["time_created"], "time_updated": r["time_updated"],
        }

    async def list_projects(self) -> List[Dict[str, Any]]:
        await asyncio.to_thread(self._ensure)
        cur = await asyncio.to_thread(self._execute, "SELECT * FROM projects ORDER BY time_updated DESC")
        rows = cur.fetchall()
        return [
            {"id": r["id"], "worktree": r["worktree"], "name": r["name"],
             "metadata": json.loads(r["metadata"] or "{}"),
             "time_created": r["time_created"], "time_updated": r["time_updated"]}
            for r in rows
        ]

    # ---- Session CRUD ----

    async def create_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None,
                              project_id: Optional[str] = None):
        await asyncio.to_thread(self._ensure)
        ts = int(_time.time())
        await asyncio.to_thread(
            self._execute,
            "INSERT OR REPLACE INTO sessions (id, project_id, metadata, status, title, next_sequence, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, project_id, json.dumps(metadata or {}), "idle", "", 1, ts, ts),
        )

    async def update_session(self, session_id: str, metadata: Dict[str, Any]):
        """Update session metadata."""
        await asyncio.to_thread(self._ensure)
        ts = int(_time.time())
        await asyncio.to_thread(
            self._execute,
            "UPDATE sessions SET metadata = ?, updated_at = ? WHERE id = ?",
            (json.dumps(metadata or {}), ts, session_id),
        )

    async def update_session_field(self, session_id: str, field: str, value: Any):
        """Update a single session column (title, status, etc.)."""
        allowed = {"title", "status", "parent_id", "project_id"}
        if field not in allowed:
            raise ValueError(f"Cannot update field: {field}")
        await asyncio.to_thread(self._ensure)
        ts = int(_time.time())
        await asyncio.to_thread(
            self._execute,
            f"UPDATE sessions SET {field} = ?, updated_at = ? WHERE id = ?",
            (value, ts, session_id),
        )

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        await asyncio.to_thread(self._ensure)
        cur = await asyncio.to_thread(self._execute, "SELECT * FROM sessions WHERE id = ?", (session_id,))
        r = cur.fetchone()
        if not r:
            return None
        return {
            "id": r["id"],
            "project_id": r["project_id"],
            "metadata": json.loads(r["metadata"] or "{}"),
            "status": r["status"],
            "title": r["title"] or "",
            "parent_id": r["parent_id"],
            "next_sequence": r["next_sequence"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }

    async def list_sessions(self, project_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List sessions, optionally filtered by project."""
        await asyncio.to_thread(self._ensure)
        if project_id:
            cur = await asyncio.to_thread(
                self._execute,
                "SELECT * FROM sessions WHERE project_id = ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (project_id, limit, offset),
            )
        else:
            cur = await asyncio.to_thread(
                self._execute,
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        rows = cur.fetchall()
        return [
            {"id": r["id"], "project_id": r["project_id"],
             "metadata": json.loads(r["metadata"] or "{}"),
             "status": r["status"], "title": r["title"] or "",
             "created_at": r["created_at"], "updated_at": r["updated_at"]}
            for r in rows
        ]

    # ---- Message CRUD ----

    async def save_message(self, message: Dict[str, Any]):
        await asyncio.to_thread(self._ensure)
        await asyncio.to_thread(
            self._execute,
            "INSERT OR REPLACE INTO messages (id, session_id, role, parent_id, model, metadata, time_created, time_completed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                message["id"], message["session_id"], message["role"],
                message.get("parent_id"),
                json.dumps(message.get("model") or {}),
                json.dumps(message.get("metadata") or {}),
                int(message["time"]["created"]),
                message["time"].get("completed"),
            ),
        )

    async def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        await asyncio.to_thread(self._ensure)
        cur = await asyncio.to_thread(self._execute, "SELECT * FROM messages WHERE id = ?", (message_id,))
        row = cur.fetchone()
        if not row:
            return None
        msg = {
            "id": row["id"], "session_id": row["session_id"], "role": row["role"],
            "time": {"created": row["time_created"], "completed": row["time_completed"]},
            "parts": [],
        }
        pcur = await asyncio.to_thread(
            self._execute,
            "SELECT data FROM parts WHERE message_id = ? ORDER BY time_created",
            (message_id,),
        )
        for pr in pcur.fetchall():
            try:
                msg["parts"].append(json.loads(pr["data"]))
            except Exception:
                continue
        return msg

    async def list_messages(self, session_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        await asyncio.to_thread(self._ensure)
        cur = await asyncio.to_thread(
            self._execute,
            "SELECT * FROM messages WHERE session_id = ? ORDER BY time_created DESC, ROWID DESC LIMIT ? OFFSET ?",
            (session_id, limit, offset),
        )
        rows = cur.fetchall()
        res = []
        for r in rows:
            pcur = await asyncio.to_thread(
                self._execute,
                "SELECT data FROM parts WHERE message_id = ? ORDER BY time_created",
                (r["id"],),
            )
            parts = []
            for pr in pcur.fetchall():
                try:
                    parts.append(json.loads(pr["data"]))
                except Exception:
                    continue
            res.append({
                "id": r["id"], "session_id": r["session_id"], "role": r["role"],
                "time": {"created": r["time_created"]}, "parts": parts,
            })
        return res

    # ---- Part CRUD ----

    async def append_part(self, part: Dict[str, Any]):
        await asyncio.to_thread(self._ensure)
        await asyncio.to_thread(
            self._execute,
            "INSERT INTO parts (id, message_id, session_id, type, data, time_created, order_index) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                part["id"], part["message_id"], part["session_id"],
                part["type"], json.dumps(part),
                int(part.get("time", {}).get("start", 0)), 0,
            ),
        )

    async def update_part(self, part_id: str, patch: Dict[str, Any]):
        await asyncio.to_thread(self._ensure)
        cur = await asyncio.to_thread(self._execute, "SELECT data FROM parts WHERE id = ?", (part_id,))
        row = cur.fetchone()
        if not row:
            raise KeyError("part not found")
        data = json.loads(row["data"])
        data.update(patch)
        await asyncio.to_thread(
            self._execute,
            "UPDATE parts SET data = ?, time_created = ? WHERE id = ?",
            (json.dumps(data), int(_time.time()), part_id),
        )

    async def get_part(self, part_id: str) -> Optional[Dict[str, Any]]:
        await asyncio.to_thread(self._ensure)
        cur = await asyncio.to_thread(self._execute, "SELECT data FROM parts WHERE id = ?", (part_id,))
        row = cur.fetchone()
        if not row:
            return None
        try:
            return json.loads(row["data"])
        except Exception:
            return None

    async def delete_part(self, part_id: str) -> None:
        await asyncio.to_thread(self._ensure)
        await asyncio.to_thread(self._execute, "DELETE FROM parts WHERE id = ?", (part_id,))

    # ---- Todo CRUD ----

    async def update_todos(self, session_id: str, todos: List[Dict[str, Any]]):
        """Replace all todos for a session (delete + re-insert, matching opencode pattern)."""
        await asyncio.to_thread(self._ensure)
        ts = int(_time.time())
        # Delete existing
        await asyncio.to_thread(self._execute, "DELETE FROM todos WHERE session_id = ?", (session_id,))
        # Insert new
        for i, todo in enumerate(todos):
            await asyncio.to_thread(
                self._execute,
                "INSERT INTO todos (session_id, content, status, priority, position, time_created, time_updated) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    todo.get("content", ""),
                    todo.get("status", "pending"),
                    todo.get("priority", "medium"),
                    i,
                    ts, ts,
                ),
            )

    async def get_todos(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all todos for a session, ordered by position."""
        await asyncio.to_thread(self._ensure)
        cur = await asyncio.to_thread(
            self._execute,
            "SELECT * FROM todos WHERE session_id = ? ORDER BY position",
            (session_id,),
        )
        rows = cur.fetchall()
        return [
            {
                "content": r["content"], "status": r["status"],
                "priority": r["priority"], "position": r["position"],
                "time_created": r["time_created"], "time_updated": r["time_updated"],
            }
            for r in rows
        ]

    # ---- Permission persistence ----

    async def save_permissions(self, session_id: str, rules: list):
        """Save permission rules for a session to the permissions table."""
        await asyncio.to_thread(self._ensure)
        ts = int(_time.time())
        await asyncio.to_thread(
            self._execute,
            "INSERT OR REPLACE INTO permissions (id, session_id, rules, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, session_id, json.dumps(rules), ts, ts),
        )

    async def load_permissions(self, session_id: str) -> Optional[list]:
        """Load permission rules for a session from the permissions table."""
        await asyncio.to_thread(self._ensure)
        cur = await asyncio.to_thread(
            self._execute,
            "SELECT rules FROM permissions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            return json.loads(row["rules"])
        except Exception:
            return None

    async def save_project_permissions(self, project_id: str, rules: list):
        """Save permission rules scoped to a project (persists across sessions)."""
        await asyncio.to_thread(self._ensure)
        ts = int(_time.time())
        row_id = f"project:{project_id}"
        await asyncio.to_thread(
            self._execute,
            "INSERT OR REPLACE INTO permissions (id, project_id, rules, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (row_id, project_id, json.dumps(rules), ts, ts),
        )

    async def load_project_permissions(self, project_id: str) -> Optional[list]:
        """Load permission rules scoped to a project."""
        await asyncio.to_thread(self._ensure)
        row_id = f"project:{project_id}"
        cur = await asyncio.to_thread(
            self._execute,
            "SELECT rules FROM permissions WHERE id = ?",
            (row_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            return json.loads(row["rules"])
        except Exception:
            return None

    # ---- Sequence ----

    async def next_sequence(self, session_id: str) -> int:
        await asyncio.to_thread(self._ensure)
        cur = await asyncio.to_thread(
            self._execute,
            "SELECT next_sequence FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            raise KeyError("session not found")
        n = row["next_sequence"]
        await asyncio.to_thread(
            self._execute,
            "UPDATE sessions SET next_sequence = ? WHERE id = ?",
            (n + 1, session_id),
        )
        return n

    # ---- Transaction ----

    async def transaction(self):
        class Tx:
            async def __aenter__(tx_self):
                await asyncio.to_thread(self._ensure)
                await asyncio.to_thread(self._execute, "BEGIN", ())
                return self

            async def __aexit__(tx_self, exc_type, exc, tb):
                if exc:
                    await asyncio.to_thread(self._execute, "ROLLBACK", ())
                else:
                    await asyncio.to_thread(self._execute, "COMMIT", ())
        return Tx()

    # ---- Diagnostics ----

    async def get_migration_version(self) -> int:
        await asyncio.to_thread(self._ensure)
        return self._get_migration_version()

    async def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
