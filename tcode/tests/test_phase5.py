"""Phase 5 tests: Storage and Infrastructure

Tests:
 - SQLite PRAGMAs (WAL, synchronous, foreign_keys, busy_timeout, cache_size)
 - Schema: projects, sessions (with project_id), messages, parts, permissions, todos, migrations tables
 - Indexes created
 - Migration system (version tracking)
 - Project CRUD (create, get, list)
 - Todo CRUD (update, get)
 - Session listing by project
 - Instance scoping (Instance class, context vars, path helpers)
 - Server lifespan (FastAPI lifespan pattern)
 - Session field update (title, status)
 - Storage close
"""
from __future__ import annotations
import asyncio
import os
import tempfile
import pytest

from tcode.storage_sqlite import SQLiteStorage
from tcode.session import SessionManager
from tcode.event import EventBus
from tcode.instance import Instance, dispose_all


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---- SQLite PRAGMAs ----

class TestSQLitePragmas:
    def _make_storage(self):
        td = tempfile.mkdtemp()
        db_path = os.path.join(td, "test.db")
        storage = SQLiteStorage(path=db_path)
        run(storage.init())
        return storage

    def test_wal_mode(self):
        s = self._make_storage()
        cur = s._conn.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        assert mode == "wal"

    def test_synchronous_normal(self):
        s = self._make_storage()
        cur = s._conn.execute("PRAGMA synchronous")
        val = cur.fetchone()[0]
        # 1 = NORMAL
        assert val == 1

    def test_foreign_keys_on(self):
        s = self._make_storage()
        cur = s._conn.execute("PRAGMA foreign_keys")
        val = cur.fetchone()[0]
        assert val == 1

    def test_busy_timeout(self):
        s = self._make_storage()
        cur = s._conn.execute("PRAGMA busy_timeout")
        val = cur.fetchone()[0]
        assert val == 5000

    def test_cache_size(self):
        s = self._make_storage()
        cur = s._conn.execute("PRAGMA cache_size")
        val = cur.fetchone()[0]
        assert val == -64000


# ---- Schema Tables ----

class TestSchemaTablesExist:
    def _make_storage(self):
        td = tempfile.mkdtemp()
        db_path = os.path.join(td, "test.db")
        storage = SQLiteStorage(path=db_path)
        run(storage.init())
        return storage

    def _table_names(self, s):
        cur = s._conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [r[0] for r in cur.fetchall()]

    def test_projects_table(self):
        s = self._make_storage()
        assert "projects" in self._table_names(s)

    def test_sessions_table(self):
        s = self._make_storage()
        assert "sessions" in self._table_names(s)

    def test_messages_table(self):
        s = self._make_storage()
        assert "messages" in self._table_names(s)

    def test_parts_table(self):
        s = self._make_storage()
        assert "parts" in self._table_names(s)

    def test_permissions_table(self):
        s = self._make_storage()
        assert "permissions" in self._table_names(s)

    def test_todos_table(self):
        s = self._make_storage()
        assert "todos" in self._table_names(s)

    def test_migrations_table(self):
        s = self._make_storage()
        assert "migrations" in self._table_names(s)

    def test_indexes_created(self):
        s = self._make_storage()
        cur = s._conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
        indexes = {r[0] for r in cur.fetchall()}
        assert "idx_messages_session" in indexes
        assert "idx_parts_message" in indexes
        assert "idx_parts_session" in indexes
        assert "idx_sessions_project" in indexes
        assert "idx_todos_session" in indexes


# ---- Migration System ----

class TestMigrationSystem:
    def test_migration_version_starts_at_1(self):
        td = tempfile.mkdtemp()
        db_path = os.path.join(td, "test.db")
        s = SQLiteStorage(path=db_path)
        run(s.init())
        version = run(s.get_migration_version())
        # Should be 1 since we have 1 migration defined
        assert version >= 1

    def test_migrations_idempotent(self):
        td = tempfile.mkdtemp()
        db_path = os.path.join(td, "test.db")
        s = SQLiteStorage(path=db_path)
        run(s.init())
        v1 = run(s.get_migration_version())
        # Re-init should not change version
        s._conn = None
        run(s.init())
        v2 = run(s.get_migration_version())
        assert v1 == v2


# ---- Project CRUD ----

class TestProjectCRUD:
    def _make(self):
        td = tempfile.mkdtemp()
        db_path = os.path.join(td, "test.db")
        s = SQLiteStorage(path=db_path)
        run(s.init())
        return s

    def test_create_and_get_project(self):
        s = self._make()
        run(s.create_project("proj1", "/home/user/myproject", name="My Project"))
        p = run(s.get_project("proj1"))
        assert p is not None
        assert p["id"] == "proj1"
        assert p["worktree"] == "/home/user/myproject"
        assert p["name"] == "My Project"

    def test_get_nonexistent_project(self):
        s = self._make()
        p = run(s.get_project("nonexistent"))
        assert p is None

    def test_list_projects(self):
        s = self._make()
        run(s.create_project("p1", "/a"))
        run(s.create_project("p2", "/b"))
        projects = run(s.list_projects())
        assert len(projects) == 2
        ids = {p["id"] for p in projects}
        assert "p1" in ids
        assert "p2" in ids

    def test_project_metadata(self):
        s = self._make()
        run(s.create_project("pm1", "/c", metadata={"key": "value"}))
        p = run(s.get_project("pm1"))
        assert p["metadata"]["key"] == "value"


# ---- Todo CRUD ----

class TestTodoCRUD:
    def _make(self):
        td = tempfile.mkdtemp()
        db_path = os.path.join(td, "test.db")
        s = SQLiteStorage(path=db_path)
        run(s.init())
        return s

    def test_update_and_get_todos(self):
        s = self._make()
        run(s.create_session("sess1"))
        todos = [
            {"content": "Fix bug", "status": "pending", "priority": "high"},
            {"content": "Write tests", "status": "in_progress", "priority": "medium"},
        ]
        run(s.update_todos("sess1", todos))
        result = run(s.get_todos("sess1"))
        assert len(result) == 2
        assert result[0]["content"] == "Fix bug"
        assert result[0]["priority"] == "high"
        assert result[0]["position"] == 0
        assert result[1]["content"] == "Write tests"
        assert result[1]["position"] == 1

    def test_update_replaces_existing_todos(self):
        s = self._make()
        run(s.create_session("sess2"))
        run(s.update_todos("sess2", [{"content": "Old task", "status": "pending", "priority": "low"}]))
        run(s.update_todos("sess2", [{"content": "New task", "status": "completed", "priority": "high"}]))
        result = run(s.get_todos("sess2"))
        assert len(result) == 1
        assert result[0]["content"] == "New task"
        assert result[0]["status"] == "completed"

    def test_empty_todos(self):
        s = self._make()
        run(s.create_session("sess3"))
        result = run(s.get_todos("sess3"))
        assert result == []

    def test_default_status_and_priority(self):
        s = self._make()
        run(s.create_session("sess4"))
        run(s.update_todos("sess4", [{"content": "Task"}]))
        result = run(s.get_todos("sess4"))
        assert result[0]["status"] == "pending"
        assert result[0]["priority"] == "medium"


# ---- Session with Project ----

class TestSessionWithProject:
    def _make(self):
        td = tempfile.mkdtemp()
        db_path = os.path.join(td, "test.db")
        s = SQLiteStorage(path=db_path)
        run(s.init())
        return s

    def test_session_with_project_id(self):
        s = self._make()
        run(s.create_project("proj1", "/test"))
        run(s.create_session("sess1", project_id="proj1"))
        sess = run(s.get_session("sess1"))
        assert sess["project_id"] == "proj1"

    def test_list_sessions_by_project(self):
        s = self._make()
        run(s.create_project("proj1", "/a"))
        run(s.create_project("proj2", "/b"))
        run(s.create_session("s1", project_id="proj1"))
        run(s.create_session("s2", project_id="proj1"))
        run(s.create_session("s3", project_id="proj2"))
        p1_sessions = run(s.list_sessions(project_id="proj1"))
        assert len(p1_sessions) == 2
        all_sessions = run(s.list_sessions())
        assert len(all_sessions) == 3

    def test_update_session_field(self):
        s = self._make()
        run(s.create_session("sf1"))
        run(s.update_session_field("sf1", "title", "My Session Title"))
        sess = run(s.get_session("sf1"))
        assert sess["title"] == "My Session Title"

    def test_update_session_field_status(self):
        s = self._make()
        run(s.create_session("sf2"))
        run(s.update_session_field("sf2", "status", "busy"))
        sess = run(s.get_session("sf2"))
        assert sess["status"] == "busy"

    def test_update_session_field_disallowed(self):
        s = self._make()
        run(s.create_session("sf3"))
        with pytest.raises(ValueError):
            run(s.update_session_field("sf3", "metadata", "injection"))


# ---- Instance Scoping ----

class TestInstance:
    def test_create_instance(self):
        inst = Instance(directory="/tmp/test_project")
        assert inst.directory == "/tmp/test_project"
        assert inst.project_id  # should be non-empty hash

    def test_project_id_stable(self):
        inst1 = Instance(directory="/tmp/stable")
        inst2 = Instance(directory="/tmp/stable")
        assert inst1.project_id == inst2.project_id

    def test_project_id_different_dirs(self):
        inst1 = Instance(directory="/tmp/dir1")
        inst2 = Instance(directory="/tmp/dir2")
        assert inst1.project_id != inst2.project_id

    def test_contains_path(self):
        inst = Instance(directory="/tmp/project", worktree="/tmp/project")
        assert inst.contains("/tmp/project/src/main.py")
        assert not inst.contains("/etc/passwd")

    def test_data_dir(self):
        inst = Instance(directory="/tmp/test_proj")
        assert "tcode" in inst.data_dir
        assert inst.project_id in inst.data_dir

    def test_config_dir(self):
        inst = Instance(directory="/tmp/test_proj")
        assert inst.config_dir == "/tmp/test_proj/.tcode"

    def test_state_bag(self):
        inst = Instance(directory="/tmp/state_test")
        inst.set_state("key1", "value1")
        assert inst.state("key1") == "value1"
        assert inst.state("nonexistent") is None
        assert inst.state("nonexistent", "default") == "default"

    def test_context_var(self):
        inst = Instance(directory="/tmp/ctx_test")

        async def _test():
            async with inst:
                current = Instance.current()
                assert current is inst
                assert Instance.require() is inst
            # After exit, should be None
            assert Instance.current() is None

        run(_test())

    def test_require_raises_outside_context(self):
        dispose_all()
        with pytest.raises(RuntimeError):
            Instance.require()

    def test_get_or_create_caching(self):
        dispose_all()
        inst1 = Instance.get_or_create("/tmp/cache_test")
        inst2 = Instance.get_or_create("/tmp/cache_test")
        assert inst1 is inst2

    def test_dispose_all(self):
        Instance.get_or_create("/tmp/dispose_test")
        dispose_all()
        # After dispose, get_or_create should create new
        inst = Instance.get_or_create("/tmp/dispose_test")
        assert inst is not None


# ---- Storage Close ----

class TestStorageClose:
    def test_close(self):
        td = tempfile.mkdtemp()
        db_path = os.path.join(td, "test.db")
        s = SQLiteStorage(path=db_path)
        run(s.init())
        assert s._conn is not None
        run(s.close())
        assert s._conn is None

    def test_reopen_after_close(self):
        td = tempfile.mkdtemp()
        db_path = os.path.join(td, "test.db")
        s = SQLiteStorage(path=db_path)
        run(s.init())
        run(s.create_session("reopen_s1"))
        run(s.close())
        # Re-init and verify data persists
        run(s.init())
        sess = run(s.get_session("reopen_s1"))
        assert sess is not None
        assert sess["id"] == "reopen_s1"


# ---- SessionManager with new storage ----

class TestSessionManagerWithNewStorage:
    def _make(self):
        td = tempfile.mkdtemp()
        db_path = os.path.join(td, "test.db")
        storage = SQLiteStorage(path=db_path)
        run(storage.init())
        events = EventBus()
        sm = SessionManager(storage=storage, events=events)
        return sm, storage

    def test_create_session_new_schema(self):
        sm, _ = self._make()
        sid = run(sm.create_session())
        sess = run(sm.get_session(sid))
        assert sess["id"] == sid

    def test_message_and_parts_new_schema(self):
        sm, _ = self._make()
        sid = run(sm.create_session())
        mid = run(sm.create_message(sid, "user"))
        pid = run(sm.append_text_part(sid, mid, "Hello!"))
        assert pid

    def test_compose_messages_new_schema(self):
        sm, _ = self._make()
        sid = run(sm.create_session())
        mid = run(sm.create_message(sid, "user"))
        run(sm.append_text_part(sid, mid, "Hello!"))
        msgs = run(sm.compose_messages(sid))
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello!"
