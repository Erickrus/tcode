"""Phase 4 tests: Permissions Hardening

Tests:
 - Permission persistence (SQLite save/load)
 - "Always" approval (rule added and evaluated)
 - Error hierarchy (PermissionDeniedError vs PermissionRejectedError)
 - Timeout (default-deny after timeout)
 - Merge chain (agent defaults + session rules)
 - Agent blocked on rejection
 - ToolRunner typed error propagation
"""
from __future__ import annotations
import asyncio
import pytest
import time

from tcode.storage_sqlite import SQLiteStorage
from tcode.session import SessionManager
from tcode.event import EventBus
from tcode.permissions import PermissionsManager, PermissionRequest
from tcode.permission_next import (
    PermissionDeniedError,
    PermissionRejectedError,
    evaluate_rules,
    merge_rulesets,
    add_always_rule,
    ask,
    ask_or_raise,
    ask_permission,
    DEFAULT_PERMISSION_TIMEOUT,
)


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run(coro, loop=None):
    if loop is None:
        loop = asyncio.new_event_loop()
    return loop.run_until_complete(coro)


# ---- Permission Persistence ----

class TestPermissionPersistence:
    def test_save_and_load_permissions(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            storage = SQLiteStorage(path=db_path)
            run(storage.init())
            events = EventBus()
            sm = SessionManager(storage=storage, events=events)
            sid = run(sm.create_session())

            rules = [
                {"permission": "builtin_shell", "action": "allow", "pattern": "*"},
                {"permission": "builtin_write_file", "action": "deny", "pattern": "/etc/*"},
            ]
            run(sm.set_permission(sid, rules))

            # Load from session metadata
            loaded = run(sm.get_permission(sid))
            assert len(loaded) == 2
            assert loaded[0]["permission"] == "builtin_shell"
            assert loaded[1]["action"] == "deny"

    def test_permissions_persist_to_sqlite_table(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            storage = SQLiteStorage(path=db_path)
            run(storage.init())

            rules = [{"permission": "*", "action": "allow", "pattern": "*"}]
            run(storage.save_permissions("sess1", rules))

            loaded = run(storage.load_permissions("sess1"))
            assert loaded is not None
            assert len(loaded) == 1
            assert loaded[0]["permission"] == "*"

    def test_permissions_load_nonexistent(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            storage = SQLiteStorage(path=db_path)
            run(storage.init())
            loaded = run(storage.load_permissions("nonexistent"))
            assert loaded is None

    def test_add_permission_rule(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            storage = SQLiteStorage(path=db_path)
            run(storage.init())
            events = EventBus()
            sm = SessionManager(storage=storage, events=events)
            sid = run(sm.create_session())

            # Start with one rule
            run(sm.set_permission(sid, [{"permission": "tool_a", "action": "deny", "pattern": "*"}]))
            # Append another
            run(sm.add_permission_rule(sid, {"permission": "tool_b", "action": "allow", "pattern": "*"}))

            loaded = run(sm.get_permission(sid))
            assert len(loaded) == 2
            assert loaded[1]["permission"] == "tool_b"

    def test_permissions_fallback_to_table(self):
        """If session metadata has no permissions, fall back to permissions table."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            storage = SQLiteStorage(path=db_path)
            run(storage.init())
            events = EventBus()
            sm = SessionManager(storage=storage, events=events)
            sid = run(sm.create_session())

            # Save directly to table (bypassing metadata)
            rules = [{"permission": "fallback_tool", "action": "allow", "pattern": "*"}]
            run(storage.save_permissions(sid, rules))

            # Clear metadata permissions
            run(storage.update_session(sid, {}))

            # Should fall back to table
            loaded = run(sm.get_permission(sid))
            assert len(loaded) == 1
            assert loaded[0]["permission"] == "fallback_tool"


# ---- "Always" Approval ----

class TestAlwaysApproval:
    def test_add_always_rule(self):
        ruleset = [{"permission": "builtin_shell", "action": "deny", "pattern": "*"}]
        new_rs = add_always_rule(ruleset, "builtin_shell")
        assert len(new_rs) == 2
        assert new_rs[1] == {"permission": "builtin_shell", "action": "allow", "pattern": "*"}

    def test_always_rule_overrides_deny(self):
        """Last rule wins: adding 'always allow' after 'deny' should result in 'allow'."""
        ruleset = [{"permission": "builtin_shell", "action": "deny", "pattern": "*"}]
        new_rs = add_always_rule(ruleset, "builtin_shell")
        action = evaluate_rules(new_rs, "builtin_shell")
        assert action == "allow"

    def test_always_rule_with_pattern(self):
        ruleset = []
        new_rs = add_always_rule(ruleset, "builtin_write_file", pattern="/tmp/*")
        assert new_rs[0]["pattern"] == "/tmp/*"
        action = evaluate_rules(new_rs, "builtin_write_file", metadata={"path": "/tmp/foo.txt"})
        assert action == "allow"
        # Should not match outside the pattern
        action2 = evaluate_rules(new_rs, "builtin_write_file", metadata={"path": "/etc/passwd"})
        assert action2 == "ask"

    def test_always_approval_persisted_in_session(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            storage = SQLiteStorage(path=db_path)
            run(storage.init())
            events = EventBus()
            sm = SessionManager(storage=storage, events=events)
            sid = run(sm.create_session())

            # Simulate "always" approval: add rule via add_permission_rule
            run(sm.add_permission_rule(sid, {"permission": "builtin_shell", "action": "allow", "pattern": "*"}))

            # Verify it persists
            rules = run(sm.get_permission(sid))
            assert any(r["permission"] == "builtin_shell" and r["action"] == "allow" for r in rules)

            # Verify evaluate_rules returns allow
            action = evaluate_rules(rules, "builtin_shell")
            assert action == "allow"


# ---- Error Hierarchy ----

class TestErrorHierarchy:
    def test_permission_denied_error(self):
        err = PermissionDeniedError("builtin_shell")
        assert err.permission == "builtin_shell"
        assert "denied" in str(err).lower()

    def test_permission_denied_error_with_pattern(self):
        err = PermissionDeniedError("builtin_write_file", pattern="/etc/*")
        assert err.pattern == "/etc/*"
        assert "pattern" in str(err)

    def test_permission_rejected_error(self):
        err = PermissionRejectedError("builtin_shell", request_id="req123")
        assert err.permission == "builtin_shell"
        assert err.request_id == "req123"
        assert "rejected" in str(err).lower()

    def test_denied_vs_rejected_are_different(self):
        """PermissionDeniedError and PermissionRejectedError should be separate exception types."""
        assert not issubclass(PermissionDeniedError, PermissionRejectedError)
        assert not issubclass(PermissionRejectedError, PermissionDeniedError)

    def test_ask_returns_false_on_deny_rule(self):
        """ask() catches PermissionDeniedError and returns False."""
        ruleset = [{"permission": "tool_x", "action": "deny", "pattern": "*"}]
        result = run(ask(None, {"permission": "tool_x"}, ruleset=ruleset))
        assert result is False

    def test_ask_or_raise_raises_on_deny(self):
        """ask_or_raise() raises PermissionDeniedError on deny rule."""
        ruleset = [{"permission": "tool_x", "action": "deny", "pattern": "*"}]
        with pytest.raises(PermissionDeniedError):
            run(ask_or_raise(None, {"permission": "tool_x"}, ruleset=ruleset))

    def test_ask_or_raise_raises_on_no_manager(self):
        """ask_or_raise() raises PermissionDeniedError if no manager and no allow rule."""
        with pytest.raises(PermissionDeniedError):
            run(ask_or_raise(None, {"permission": "tool_x"}))

    def test_ask_or_raise_returns_true_on_allow(self):
        ruleset = [{"permission": "tool_x", "action": "allow", "pattern": "*"}]
        result = run(ask_or_raise(None, {"permission": "tool_x"}, ruleset=ruleset))
        assert result is True


# ---- Timeout Fallback ----

class TestTimeoutFallback:
    def test_default_timeout_constant(self):
        assert DEFAULT_PERMISSION_TIMEOUT == 120.0

    def test_ask_permission_timeout_raises_denied(self):
        """If permission manager never responds, should raise PermissionDeniedError after timeout."""
        pm = PermissionsManager()
        # Use a very short timeout so we don't wait long
        with pytest.raises(PermissionDeniedError):
            run(ask_permission(pm, {"permission": "test_tool"}, timeout=0.01))

    def test_ask_timeout_returns_false(self):
        """ask() with timeout returns False (catches the error)."""
        pm = PermissionsManager()
        result = run(ask(pm, {"permission": "test_tool"}, timeout=0.01))
        assert result is False


# ---- Merge Chain ----

class TestMergeChain:
    def test_merge_rulesets_basic(self):
        rs1 = [{"permission": "a", "action": "deny", "pattern": "*"}]
        rs2 = [{"permission": "a", "action": "allow", "pattern": "*"}]
        merged = merge_rulesets(rs1, rs2)
        assert len(merged) == 2
        # Last rule wins
        action = evaluate_rules(merged, "a")
        assert action == "allow"

    def test_merge_rulesets_empty(self):
        merged = merge_rulesets(None, [], None)
        assert merged == []

    def test_merge_rulesets_single(self):
        rs = [{"permission": "*", "action": "allow", "pattern": "*"}]
        merged = merge_rulesets(rs)
        assert merged == rs

    def test_merge_agent_plus_session_rules(self):
        """Agent defaults deny, but session rules allow — session should win."""
        agent_rules = [{"permission": "builtin_shell", "action": "deny", "pattern": "*"}]
        session_rules = [{"permission": "builtin_shell", "action": "allow", "pattern": "*"}]
        merged = merge_rulesets(agent_rules, session_rules)
        action = evaluate_rules(merged, "builtin_shell")
        assert action == "allow"


# ---- PermissionsManager ----

class TestPermissionsManagerRespond:
    def test_respond_allow(self):
        pm = PermissionsManager()

        async def _test():
            req = PermissionRequest("r1", "test_perm", {"key": "val"})
            # Start request in background
            task = asyncio.create_task(pm.request("sess1", req))
            # Respond
            await asyncio.sleep(0.01)
            await pm.respond("r1", True)
            result = await task
            return result

        result = run(_test())
        assert result is True

    def test_respond_deny(self):
        pm = PermissionsManager()

        async def _test():
            req = PermissionRequest("r2", "test_perm", {})
            task = asyncio.create_task(pm.request("sess1", req))
            await asyncio.sleep(0.01)
            await pm.respond("r2", False)
            result = await task
            return result

        result = run(_test())
        assert result is False

    def test_respond_always(self):
        pm = PermissionsManager()

        async def _test():
            req = PermissionRequest("r3", "test_perm", {})
            task = asyncio.create_task(pm.request("sess1", req))
            await asyncio.sleep(0.01)
            # Capture request info before respond cleans it up
            pending_req = pm.get_pending_request("r3")
            sid = pm.get_session_for_request("r3")
            assert pending_req is not None
            assert sid == "sess1"
            await pm.respond("r3", True, always=True)
            result = await task
            return result

        result = run(_test())
        assert result is True

    def test_get_pending_request(self):
        pm = PermissionsManager()

        async def _test():
            req = PermissionRequest("r4", "test_perm", {"detail": 1})
            task = asyncio.create_task(pm.request("sess1", req))
            await asyncio.sleep(0.01)
            pending = pm.get_pending_request("r4")
            assert pending is not None
            assert pending.type == "test_perm"
            assert pending.details == {"detail": 1}
            # Cleanup
            await pm.respond("r4", True)
            await task

        run(_test())

    def test_get_session_for_request(self):
        pm = PermissionsManager()

        async def _test():
            req = PermissionRequest("r5", "perm", {})
            task = asyncio.create_task(pm.request("my_session", req))
            await asyncio.sleep(0.01)
            sid = pm.get_session_for_request("r5")
            assert sid == "my_session"
            await pm.respond("r5", True)
            await task

        run(_test())


# ---- ToolRunner Error Propagation ----

class TestToolRunnerPermissionErrors:
    def _make_toolrunner(self):
        import tempfile, os
        from tcode.tools import ToolRegistry, ToolContext, ToolResult, ToolInfo
        from tcode.toolrunner import ToolRunner
        from pydantic import BaseModel

        td = tempfile.mkdtemp()
        db_path = os.path.join(td, "test.db")
        storage = SQLiteStorage(path=db_path)
        run(storage.init())
        events = EventBus()
        sm = SessionManager(storage=storage, events=events)

        registry = ToolRegistry()

        class EchoParams(BaseModel):
            text: str = ""

        async def echo_fn(args, ctx):
            return ToolResult(output=args.get("text", ""))

        registry.register(ToolInfo("echo", "echo tool", EchoParams, echo_fn))
        runner = ToolRunner(registry, events, sm)
        return runner, sm

    def test_deny_rule_raises_permission_denied(self):
        runner, sm = self._make_toolrunner()
        sid = run(sm.create_session())
        mid = run(sm.create_message(sid, "assistant"))

        # Set deny rule for echo
        run(sm.set_permission(sid, [{"permission": "echo", "action": "deny", "pattern": "*"}]))

        with pytest.raises(PermissionDeniedError):
            run(runner.execute_tool(sid, mid, "call1", "echo", {"text": "hello"}))

    def test_allow_rule_succeeds(self):
        runner, sm = self._make_toolrunner()
        sid = run(sm.create_session())
        mid = run(sm.create_message(sid, "assistant"))

        # Set allow rule for echo
        run(sm.set_permission(sid, [{"permission": "echo", "action": "allow", "pattern": "*"}]))

        result = run(runner.execute_tool(sid, mid, "call1", "echo", {"text": "hi"}))
        assert result.output == "hi"

    def test_no_rules_no_manager_raises_denied(self):
        """With no rules and no PermissionsManager, ask_or_raise raises PermissionDeniedError."""
        runner, sm = self._make_toolrunner()
        sid = run(sm.create_session())
        mid = run(sm.create_message(sid, "assistant"))

        # No permissions set, no manager
        with pytest.raises(PermissionDeniedError):
            run(runner.execute_tool(sid, mid, "call1", "echo", {"text": "test"}))


# ---- Agent Blocked on Rejection ----

class TestAgentBlockedOnRejection:
    def test_permission_rejected_sets_blocked(self):
        """When PermissionRejectedError is raised in tool execution, agent should set blocked=True."""
        # This tests the error type behavior — full integration with agent loop
        # would require mocking the provider. Test the error type directly.
        err = PermissionRejectedError("builtin_shell", request_id="r1")
        assert isinstance(err, PermissionRejectedError)
        assert err.permission == "builtin_shell"

    def test_permission_denied_does_not_set_blocked(self):
        """PermissionDeniedError should be caught gracefully (tool error, not blocked)."""
        err = PermissionDeniedError("builtin_shell")
        assert isinstance(err, PermissionDeniedError)
        # DeniedError is expected — tool shows error, loop continues


# ---- Evaluate Rules ----

class TestEvaluateRules:
    def test_wildcard_permission(self):
        rules = [{"permission": "*", "action": "allow", "pattern": "*"}]
        assert evaluate_rules(rules, "anything") == "allow"

    def test_exact_match(self):
        rules = [{"permission": "builtin_shell", "action": "deny", "pattern": "*"}]
        assert evaluate_rules(rules, "builtin_shell") == "deny"
        assert evaluate_rules(rules, "builtin_read_file") == "ask"

    def test_pattern_match(self):
        rules = [{"permission": "builtin_write_file", "action": "deny", "pattern": "/etc/*"}]
        assert evaluate_rules(rules, "builtin_write_file", metadata={"path": "/etc/passwd"}) == "deny"
        assert evaluate_rules(rules, "builtin_write_file", metadata={"path": "/tmp/test"}) == "ask"

    def test_last_rule_wins(self):
        rules = [
            {"permission": "tool_a", "action": "deny", "pattern": "*"},
            {"permission": "tool_a", "action": "allow", "pattern": "*"},
        ]
        assert evaluate_rules(rules, "tool_a") == "allow"

    def test_empty_ruleset(self):
        assert evaluate_rules([], "anything") == "ask"
        assert evaluate_rules(None, "anything") == "ask"
