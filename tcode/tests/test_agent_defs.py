"""Tests for tcode agent definitions and registry."""
from __future__ import annotations
import pytest
from tcode.agent_defs import (
    AgentInfo, AgentRegistry, disabled_tools,
    DEFAULT_PERMISSIONS, PROMPT_BUILD, PROMPT_PLAN,
)
from tcode.config import AgentConfig, ModelRef
from tcode.permission_next import evaluate_rules


# ---- Built-in agents ----

def test_builtin_agents_registered():
    registry = AgentRegistry()
    assert registry.get("build") is not None
    assert registry.get("plan") is not None
    assert registry.get("explore") is not None
    assert registry.get("compaction") is not None
    assert registry.get("title") is not None
    assert registry.get("summary") is not None


def test_build_agent_is_primary():
    registry = AgentRegistry()
    build = registry.get("build")
    assert build.mode == "primary"
    assert build.hidden is False
    assert build.prompt == PROMPT_BUILD


def test_explore_agent_is_subagent():
    registry = AgentRegistry()
    explore = registry.get("explore")
    assert explore.mode == "subagent"


def test_hidden_agents():
    registry = AgentRegistry()
    compaction = registry.get("compaction")
    title = registry.get("title")
    summary = registry.get("summary")
    assert compaction.hidden is True
    assert title.hidden is True
    assert summary.hidden is True


def test_list_visible_excludes_hidden():
    registry = AgentRegistry()
    visible = registry.list_visible()
    names = [a.name for a in visible]
    assert "build" in names
    assert "plan" in names
    assert "explore" in names
    assert "compaction" not in names
    assert "title" not in names
    assert "summary" not in names


def test_list_all_includes_hidden():
    registry = AgentRegistry()
    all_agents = registry.list()
    names = [a.name for a in all_agents]
    assert "compaction" in names
    assert "title" in names


def test_default_agent():
    registry = AgentRegistry()
    assert registry.default_agent() == "build"


# ---- Permission evaluation per agent ----

def test_build_agent_allows_question():
    registry = AgentRegistry()
    build = registry.get("build")
    action = evaluate_rules(build.permission, "question")
    assert action == "allow"


def test_plan_agent_denies_write():
    registry = AgentRegistry()
    plan = registry.get("plan")
    action = evaluate_rules(plan.permission, "write_file")
    assert action == "deny"


def test_plan_agent_allows_question():
    registry = AgentRegistry()
    plan = registry.get("plan")
    action = evaluate_rules(plan.permission, "question")
    assert action == "allow"


def test_compaction_agent_denies_all_tools():
    registry = AgentRegistry()
    compaction = registry.get("compaction")
    action = evaluate_rules(compaction.permission, "read_file")
    assert action == "deny"
    action = evaluate_rules(compaction.permission, "shell")
    assert action == "deny"


def test_explore_agent_denies_write():
    registry = AgentRegistry()
    explore = registry.get("explore")
    action = evaluate_rules(explore.permission, "write_file")
    assert action == "deny"


def test_explore_agent_allows_read():
    registry = AgentRegistry()
    explore = registry.get("explore")
    action = evaluate_rules(explore.permission, "read_file")
    assert action == "allow"


# ---- Config-defined agents ----

def test_load_config_agent_new():
    registry = AgentRegistry()
    registry.load_from_config({
        "custom": AgentConfig(
            prompt="Custom prompt",
            steps=20,
            mode="subagent",
            description="A custom agent",
        )
    })
    custom = registry.get("custom")
    assert custom is not None
    assert custom.prompt == "Custom prompt"
    assert custom.steps == 20
    assert custom.mode == "subagent"


def test_load_config_agent_override_builtin():
    registry = AgentRegistry()
    original_prompt = registry.get("build").prompt
    registry.load_from_config({
        "build": AgentConfig(
            prompt="Overridden prompt",
            temperature=0.7,
        )
    })
    build = registry.get("build")
    assert build.prompt == "Overridden prompt"
    assert build.temperature == 0.7
    # Other fields preserved
    assert build.mode == "primary"


# ---- disabled_tools ----

def test_disabled_tools_from_deny_rules():
    ruleset = [
        {"permission": "write_file", "action": "deny", "pattern": "*"},
        {"permission": "shell", "action": "deny", "pattern": "*"},
        {"permission": "read_file", "action": "allow", "pattern": "*"},
    ]
    denied = disabled_tools(ruleset, ["write_file", "shell", "read_file", "grep"])
    assert "write_file" in denied
    assert "shell" in denied
    assert "read_file" not in denied


def test_disabled_tools_compaction_agent():
    registry = AgentRegistry()
    compaction = registry.get("compaction")
    tool_ids = ["read_file", "write_file", "grep", "shell", "echo"]
    denied = disabled_tools(compaction.permission, tool_ids)
    # All tools should be denied for compaction agent
    assert set(denied) == set(tool_ids)


def test_disabled_tools_empty_ruleset():
    denied = disabled_tools(None, ["read_file", "write_file"])
    assert denied == []
