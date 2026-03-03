"""Phase 6 tests: Extended Features

Tests:
 - Snapshot class (non-git directory handling)
 - Session lifecycle (fork, archive, unarchive, revert)
 - Edit tool (replace, replace_all, not found, ambiguous)
 - TodoWrite / TodoRead tools
 - Task tool (basic structure)
 - New tool registration
"""
from __future__ import annotations
import asyncio
import os
import tempfile
import pytest

from tcode.storage_sqlite import SQLiteStorage
from tcode.session import SessionManager
from tcode.event import EventBus
from tcode.tools import ToolRegistry, ToolContext, ToolResult, ToolInfo
from tcode.builtin_tools import (
    register_builtin_tools, edit_execute, EditParams,
    todowrite_execute, todoread_execute,
    TodoWriteParams, TodoReadParams,
)
from tcode.snapshot import Snapshot, Patch, FileDiff
from tcode.session_lifecycle import fork_session, archive_session, unarchive_session, set_revert, clear_revert


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_sm():
    td = tempfile.mkdtemp()
    db_path = os.path.join(td, "test.db")
    storage = SQLiteStorage(path=db_path)
    run(storage.init())
    events = EventBus()
    sm = SessionManager(storage=storage, events=events)
    return sm, storage


# ---- Snapshot ----

class TestSnapshot:
    def test_snapshot_non_git_dir(self):
        td = tempfile.mkdtemp()
        snap = Snapshot(td)
        # track should return empty for non-git dir
        h = run(snap.track())
        assert h == ""

    def test_snapshot_patch_no_hash(self):
        td = tempfile.mkdtemp()
        snap = Snapshot(td)
        p = run(snap.patch(""))
        assert p.hash == ""
        assert p.files == []

    def test_snapshot_diff_non_git(self):
        td = tempfile.mkdtemp()
        snap = Snapshot(td)
        d = run(snap.diff("abc"))
        assert d == ""

    def test_snapshot_diff_full_non_git(self):
        td = tempfile.mkdtemp()
        snap = Snapshot(td)
        diffs = run(snap.diff_full("abc"))
        assert diffs == []

    def test_snapshot_restore_non_git(self):
        td = tempfile.mkdtemp()
        snap = Snapshot(td)
        assert run(snap.restore("abc")) is False

    def test_snapshot_revert_empty(self):
        td = tempfile.mkdtemp()
        snap = Snapshot(td)
        assert run(snap.revert([])) is False

    def test_patch_dataclass(self):
        p = Patch(hash="abc123", files=["a.py", "b.py"])
        assert p.hash == "abc123"
        assert len(p.files) == 2

    def test_file_diff_dataclass(self):
        fd = FileDiff(file="test.py", additions=10, deletions=3, status="modified")
        assert fd.file == "test.py"
        assert fd.additions == 10
        assert fd.status == "modified"


# ---- Session Lifecycle ----

class TestSessionFork:
    def test_fork_session(self):
        sm, _ = _make_sm()
        sid = run(sm.create_session(metadata={"title": "Original"}))
        mid = run(sm.create_message(sid, "user"))
        run(sm.append_text_part(sid, mid, "Hello"))
        mid2 = run(sm.create_message(sid, "assistant", parent_id=mid))
        run(sm.append_text_part(sid, mid2, "Hi there"))

        new_sid = run(fork_session(sm, sid))
        assert new_sid != sid

        # Verify forked session has messages
        msgs = run(sm.compose_messages(new_sid))
        assert len(msgs) >= 2

    def test_fork_session_upto(self):
        sm, _ = _make_sm()
        sid = run(sm.create_session())
        mid1 = run(sm.create_message(sid, "user"))
        run(sm.append_text_part(sid, mid1, "First"))
        mid2 = run(sm.create_message(sid, "user"))
        run(sm.append_text_part(sid, mid2, "Second"))

        # Fork only up to first message
        new_sid = run(fork_session(sm, sid, upto_message_id=mid1))
        msgs = run(sm.compose_messages(new_sid))
        # Should have only the first message
        assert len(msgs) == 1
        assert msgs[0]["content"] == "First"


class TestSessionArchive:
    def test_archive_and_unarchive(self):
        sm, _ = _make_sm()
        sid = run(sm.create_session())

        run(archive_session(sm, sid))
        sess = run(sm.get_session(sid))
        meta = sess.get("metadata", {})
        assert "time_archived" in meta

        run(unarchive_session(sm, sid))
        sess = run(sm.get_session(sid))
        meta = sess.get("metadata", {})
        assert "time_archived" not in meta

    def test_archive_nonexistent(self):
        sm, _ = _make_sm()
        with pytest.raises(KeyError):
            run(archive_session(sm, "nonexistent"))


class TestSessionRevert:
    def test_set_and_clear_revert(self):
        sm, _ = _make_sm()
        sid = run(sm.create_session())
        mid = run(sm.create_message(sid, "user"))

        run(set_revert(sm, sid, mid, snapshot="abc123"))
        sess = run(sm.get_session(sid))
        meta = sess.get("metadata", {})
        assert meta["revert"]["messageID"] == mid
        assert meta["revert"]["snapshot"] == "abc123"

        run(clear_revert(sm, sid))
        sess = run(sm.get_session(sid))
        meta = sess.get("metadata", {})
        assert "revert" not in meta


# ---- Edit Tool ----

class TestEditTool:
    def test_edit_simple_replace(self):
        td = tempfile.mkdtemp()
        fp = os.path.join(td, "test.py")
        with open(fp, "w") as f:
            f.write("hello world\ngoodbye world\n")

        ctx = ToolContext(session_id="s1", message_id="m1")
        result = run(edit_execute({"file_path": fp, "old_string": "hello", "new_string": "hi"}, ctx))
        assert "Edited" in result.output

        with open(fp) as f:
            content = f.read()
        assert "hi world" in content
        assert "hello" not in content

    def test_edit_replace_all(self):
        td = tempfile.mkdtemp()
        fp = os.path.join(td, "test.py")
        with open(fp, "w") as f:
            f.write("foo bar foo baz foo\n")

        ctx = ToolContext(session_id="s1", message_id="m1")
        result = run(edit_execute({
            "file_path": fp, "old_string": "foo", "new_string": "qux", "replace_all": True
        }, ctx))
        assert "Edited" in result.output

        with open(fp) as f:
            content = f.read()
        assert content == "qux bar qux baz qux\n"

    def test_edit_not_found(self):
        td = tempfile.mkdtemp()
        fp = os.path.join(td, "test.py")
        with open(fp, "w") as f:
            f.write("hello\n")

        ctx = ToolContext(session_id="s1", message_id="m1")
        result = run(edit_execute({"file_path": fp, "old_string": "xyz", "new_string": "abc"}, ctx))
        assert "not found" in result.output

    def test_edit_ambiguous(self):
        td = tempfile.mkdtemp()
        fp = os.path.join(td, "test.py")
        with open(fp, "w") as f:
            f.write("foo\nfoo\n")

        ctx = ToolContext(session_id="s1", message_id="m1")
        result = run(edit_execute({"file_path": fp, "old_string": "foo", "new_string": "bar"}, ctx))
        assert "found 2 times" in result.output

    def test_edit_file_not_exists(self):
        ctx = ToolContext(session_id="s1", message_id="m1")
        result = run(edit_execute({"file_path": "/nonexistent/file.py", "old_string": "x", "new_string": "y"}, ctx))
        assert "not found" in result.output

    def test_edit_directory(self):
        td = tempfile.mkdtemp()
        ctx = ToolContext(session_id="s1", message_id="m1")
        result = run(edit_execute({"file_path": td, "old_string": "x", "new_string": "y"}, ctx))
        assert "directory" in result.output


# ---- Todo Tools ----

class TestTodoTools:
    def test_todowrite_and_read(self):
        sm, storage = _make_sm()
        sid = run(sm.create_session())
        ctx = ToolContext(session_id=sid, message_id="m1", extra={"sessions": sm})

        todos = [
            {"content": "Fix bug", "status": "pending", "priority": "high"},
            {"content": "Write tests", "status": "in_progress", "priority": "medium"},
        ]
        result = run(todowrite_execute({"todos": todos}, ctx))
        assert "Fix bug" in result.output

        # Read back
        result2 = run(todoread_execute({}, ctx))
        assert "Fix bug" in result2.output
        assert "Write tests" in result2.output

    def test_todowrite_replaces(self):
        sm, _ = _make_sm()
        sid = run(sm.create_session())
        ctx = ToolContext(session_id=sid, message_id="m1", extra={"sessions": sm})

        run(todowrite_execute({"todos": [{"content": "Old"}]}, ctx))
        run(todowrite_execute({"todos": [{"content": "New"}]}, ctx))

        result = run(todoread_execute({}, ctx))
        assert "New" in result.output
        assert "Old" not in result.output

    def test_todoread_empty(self):
        sm, _ = _make_sm()
        sid = run(sm.create_session())
        ctx = ToolContext(session_id=sid, message_id="m1", extra={"sessions": sm})
        result = run(todoread_execute({}, ctx))
        assert result.output == "[]"

    def test_todowrite_no_session(self):
        ctx = ToolContext(session_id="", message_id="m1")
        result = run(todowrite_execute({"todos": []}, ctx))
        assert "Error" in result.output


# ---- Tool Registration ----

class TestToolRegistration:
    def test_phase6_tools_registered(self):
        registry = ToolRegistry()
        register_builtin_tools(registry)
        tool_ids = registry.list()
        assert "builtin_edit" in tool_ids
        assert "builtin_todowrite" in tool_ids
        assert "builtin_todoread" in tool_ids
        assert "builtin_task" in tool_ids

    def test_total_tool_count(self):
        registry = ToolRegistry()
        register_builtin_tools(registry)
        # 10 original + 4 Phase 6 = 14
        assert len(registry.list()) == 14


# ---- Task Tool ----

class TestTaskTool:
    def test_task_no_runner(self):
        """Task tool should fail gracefully without agent_runner."""
        from tcode.builtin_tools import task_execute
        ctx = ToolContext(session_id="s1", message_id="m1", extra={})
        result = run(task_execute({
            "description": "test", "prompt": "hello", "subagent_type": "explore"
        }, ctx))
        assert "Error" in result.output
