"""Tests for tcode command system."""
from __future__ import annotations
import pytest
import asyncio
from tcode.command import (
    CommandInfo, CommandRegistry, extract_hints, render_template,
)
from tcode.config import CommandConfig
from tcode.event import EventBus


# ---- Template variable extraction ----

def test_extract_hints_numbered():
    hints = extract_hints("Deploy $1 to $2 environment")
    assert "1" in hints
    assert "2" in hints


def test_extract_hints_arguments():
    hints = extract_hints("Run command with $ARGUMENTS")
    assert "ARGUMENTS" in hints


def test_extract_hints_mixed():
    hints = extract_hints("$1 does $2 with $ARGUMENTS")
    assert sorted(hints) == ["1", "2", "ARGUMENTS"]


def test_extract_hints_none():
    hints = extract_hints("No variables here")
    assert hints == []


# ---- Template rendering ----

def test_render_numbered_args():
    result = render_template("Deploy $1 to $2", ["myapp", "production"])
    assert result == "Deploy myapp to production"


def test_render_arguments():
    result = render_template("Run: $ARGUMENTS", ["one", "two", "three"])
    assert result == "Run: one two three"


def test_render_mixed():
    result = render_template("$1 with $ARGUMENTS", ["first", "second", "third"])
    assert result == "first with first second third"


def test_render_no_args():
    result = render_template("Hello world", [])
    assert result == "Hello world"


# ---- Built-in commands ----

def test_builtin_commands_registered():
    registry = CommandRegistry()
    assert registry.get("init") is not None
    assert registry.get("review") is not None


def test_builtin_init_command():
    registry = CommandRegistry()
    init = registry.get("init")
    assert init.source == "command"
    assert "AGENTS.md" in init.template


def test_builtin_review_command():
    registry = CommandRegistry()
    review = registry.get("review")
    assert review.subtask is True


# ---- Config commands ----

def test_load_config_commands():
    registry = CommandRegistry()
    registry.load_from_config({
        "deploy": CommandConfig(
            description="Deploy the app",
            template="Deploy $1 to $2",
        ),
        "test": CommandConfig(
            description="Run tests",
            template="pytest $ARGUMENTS",
            agent="explore",
        ),
    })
    deploy = registry.get("deploy")
    assert deploy is not None
    assert deploy.description == "Deploy the app"
    assert "1" in deploy.hints
    assert "2" in deploy.hints

    test = registry.get("test")
    assert test.agent == "explore"
    assert "ARGUMENTS" in test.hints


# ---- MCP prompts as commands ----

def test_load_mcp_prompts():
    registry = CommandRegistry()
    prompts = [
        {
            "name": "summarize",
            "description": "Summarize the content",
            "arguments": [
                {"name": "text", "required": True},
            ],
        },
        {
            "name": "translate",
            "description": "Translate content",
            "arguments": [
                {"name": "text", "required": True},
                {"name": "lang", "required": True},
            ],
        },
    ]
    registry.load_from_mcp_prompts("myserver", prompts)

    summarize = registry.get("mcp.myserver.summarize")
    assert summarize is not None
    assert summarize.source == "mcp"
    assert "1" in summarize.hints

    translate = registry.get("mcp.myserver.translate")
    assert translate is not None
    assert "2" in translate.hints


# ---- Skills as commands ----

def test_load_skills_as_commands():
    registry = CommandRegistry()
    skills = [
        {"name": "lint", "description": "Run linter", "content": "Run the linter on $ARGUMENTS"},
        {"name": "format", "description": "Format code", "content": "Format all files"},
    ]
    registry.load_from_skills(skills)

    lint = registry.get("lint")
    assert lint is not None
    assert lint.source == "skill"
    assert lint.template == "Run the linter on $ARGUMENTS"


def test_skills_dont_override_existing_commands():
    registry = CommandRegistry()
    # init is a built-in
    skills = [{"name": "init", "description": "custom init", "content": "custom"}]
    registry.load_from_skills(skills)
    init = registry.get("init")
    assert init.source == "command"  # kept built-in


# ---- Command listing ----

def test_list_commands():
    registry = CommandRegistry()
    commands = registry.list()
    names = [c.name for c in commands]
    assert "init" in names
    assert "review" in names


# ---- Command execution ----

@pytest.mark.asyncio
async def test_execute_command():
    events = EventBus()
    registry = CommandRegistry(events=events)
    registry.register(CommandInfo(
        name="greet",
        template="Hello $1!",
        source="command",
    ))

    # Track events
    executed_events = []
    events.subscribe("command.executed", lambda ev: asyncio.ensure_future(_append(executed_events, ev)))

    result = await registry.execute("greet", ["world"])
    assert result == "Hello world!"


@pytest.mark.asyncio
async def test_execute_nonexistent_command():
    registry = CommandRegistry()
    result = await registry.execute("nonexistent")
    assert result is None


async def _append(lst, ev):
    lst.append(ev)
