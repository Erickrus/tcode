"""Command system for tcode.

Commands are invokable actions from: config definitions, MCP prompts, and skills.
Each command has a name, template, and optional agent/model override.

Reference: opencode command/index.ts
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from .event import EventBus, Event


@dataclass
class CommandInfo:
    """Command definition."""
    name: str
    description: str = ""
    agent: Optional[str] = None
    model: Optional[str] = None
    source: str = "command"  # "command" | "mcp" | "skill"
    template: str = ""
    subtask: bool = False
    hints: List[str] = field(default_factory=list)


# ---- Template variable extraction ----

_VAR_RE = re.compile(r'\$(\d+|\bARGUMENTS\b)')


def extract_hints(template: str) -> List[str]:
    """Extract template variable names ($1, $2, $ARGUMENTS) from a template."""
    matches = _VAR_RE.findall(template)
    return sorted(set(matches))


def render_template(template: str, arguments: List[str]) -> str:
    """Render a command template with positional arguments.

    $1, $2, etc. are replaced with corresponding arguments.
    $ARGUMENTS is replaced with all arguments joined by space.
    """
    result = template
    for i, arg in enumerate(arguments, 1):
        result = result.replace(f'${i}', arg)
    result = result.replace('$ARGUMENTS', ' '.join(arguments))
    return result


# ---- Built-in commands ----

INIT_TEMPLATE = (
    "Create or update an AGENTS.md file in the project root that documents "
    "how to work with this codebase. Include: project structure, key commands, "
    "coding conventions, and any special instructions for AI agents."
)

REVIEW_TEMPLATE = (
    "Review the current changes in the working directory. Check for: "
    "correctness, potential bugs, style issues, and missing tests. "
    "If there are no uncommitted changes, review the most recent commit."
)

_BUILTIN_COMMANDS: Dict[str, CommandInfo] = {
    "init": CommandInfo(
        name="init",
        description="Create or update AGENTS.md",
        source="command",
        template=INIT_TEMPLATE,
    ),
    "review": CommandInfo(
        name="review",
        description="Review current changes",
        source="command",
        template=REVIEW_TEMPLATE,
        subtask=True,
    ),
}


# ---- Command Registry ----

class CommandRegistry:
    """Unified command registry across built-in, config, MCP, and skill sources."""

    def __init__(self, events: Optional[EventBus] = None):
        self._commands: Dict[str, CommandInfo] = {}
        self._events = events
        self._load_builtins()

    def _load_builtins(self):
        for name, cmd in _BUILTIN_COMMANDS.items():
            self._commands[name] = cmd

    def load_from_config(self, config_commands: Dict[str, Any]):
        """Load commands from config.command dict."""
        for name, cfg in config_commands.items():
            hints = extract_hints(cfg.template) if cfg.template else []
            self._commands[name] = CommandInfo(
                name=name,
                description=cfg.description or "",
                agent=cfg.agent,
                model=cfg.model,
                source="command",
                template=cfg.template or "",
                subtask=cfg.subtask,
                hints=hints,
            )

    def load_from_mcp_prompts(self, mcp_name: str, prompts: List[Dict[str, Any]]):
        """Convert MCP prompts into commands.

        Each prompt becomes a command with name 'mcp.<server>.<prompt_name>'.
        """
        for prompt in prompts:
            pname = prompt.get('name', '')
            if not pname:
                continue
            cmd_name = f"mcp.{mcp_name}.{pname}"
            if cmd_name in self._commands:
                continue

            description = prompt.get('description', '')
            args = prompt.get('arguments', [])
            # Build template from prompt arguments
            template_parts = [description] if description else []
            hints = []
            for i, arg in enumerate(args, 1):
                arg_name = arg.get('name', f'arg{i}')
                template_parts.append(f'{arg_name}: ${i}')
                hints.append(str(i))

            self._commands[cmd_name] = CommandInfo(
                name=cmd_name,
                description=description,
                source="mcp",
                template='\n'.join(template_parts),
                hints=hints,
            )

    def load_from_skills(self, skills: List[Dict[str, Any]]):
        """Convert loaded skills into commands.

        Each skill with content becomes an invokable command.
        """
        for skill in skills:
            name = skill.get('name', '')
            if not name or name in self._commands:
                continue
            self._commands[name] = CommandInfo(
                name=name,
                description=skill.get('description', ''),
                source="skill",
                template=skill.get('content', ''),
            )

    def register(self, cmd: CommandInfo):
        """Register a single command."""
        self._commands[cmd.name] = cmd

    def get(self, name: str) -> Optional[CommandInfo]:
        return self._commands.get(name)

    def list(self) -> List[CommandInfo]:
        """List all commands sorted by name."""
        return sorted(self._commands.values(), key=lambda c: c.name)

    async def execute(self, name: str, arguments: List[str] = None,
                      session_id: Optional[str] = None,
                      message_id: Optional[str] = None) -> Optional[str]:
        """Render a command template with arguments and publish executed event.

        Returns the rendered template text, or None if command not found.
        """
        cmd = self.get(name)
        if not cmd:
            return None

        arguments = arguments or []
        rendered = render_template(cmd.template, arguments)

        if self._events:
            await self._events.publish(Event.create(
                "command.executed",
                {
                    "name": name,
                    "arguments": arguments,
                    "session_id": session_id,
                    "message_id": message_id,
                },
                session_id=session_id,
            ))

        return rendered
