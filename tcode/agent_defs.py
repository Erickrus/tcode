"""Agent definitions and registry for tcode.

Defines built-in agents (build, plan, explore, compaction, title, summary)
and supports config-defined agents. Each agent has a name, prompt, model,
permissions, tool set, and step limit.

Reference: opencode agent/agent.ts
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .permission_next import merge_rulesets, evaluate_rules


@dataclass
class AgentInfo:
    """Agent definition."""
    name: str
    description: str = ""
    mode: str = "primary"  # "primary" | "subagent"
    hidden: bool = False
    model: Optional[Dict[str, str]] = None  # {"provider_id": ..., "model_id": ...}
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    permission: Optional[List[Dict[str, Any]]] = None  # ruleset
    steps: int = 50
    options: Dict[str, Any] = field(default_factory=dict)


# ---- Default permission rulesets ----

DEFAULT_PERMISSIONS: List[Dict[str, Any]] = [
    {"permission": "*", "action": "allow", "pattern": "*"},
    {"permission": "doom_loop", "action": "ask", "pattern": "*"},
    {"permission": "external_directory", "action": "ask", "pattern": "*"},
    {"permission": "question", "action": "deny", "pattern": "*"},
    {"permission": "plan_enter", "action": "deny", "pattern": "*"},
    {"permission": "plan_exit", "action": "deny", "pattern": "*"},
]

# ---- Agent-specific prompts ----
# Copied from packages/opencode/src/agent/prompt/*.txt

# build and plan agents have no custom prompt in opencode — they use the session system prompt.
PROMPT_BUILD = None
PROMPT_PLAN = None

PROMPT_EXPLORE = """\
You are a file search specialist. You excel at thoroughly navigating and exploring codebases.

Your strengths:
- Rapidly finding files using glob patterns
- Searching code and text with powerful regex patterns
- Reading and analyzing file contents

Guidelines:
- Use Glob for broad file pattern matching
- Use Grep for searching file contents with regex
- Use Read when you know the specific file path you need to read
- Use Bash for file operations like copying, moving, or listing directory contents
- Adapt your search approach based on the thoroughness level specified by the caller
- Return file paths as absolute paths in your final response
- For clear communication, avoid using emojis
- Do not create any files, or run bash commands that modify the user's system state in any way

Complete the user's search request efficiently and report your findings clearly.
"""

PROMPT_COMPACTION = """\
You are a helpful AI assistant tasked with summarizing conversations.

When asked to summarize, provide a detailed but concise summary of the conversation.
Focus on information that would be helpful for continuing the conversation, including:
- What was done
- What is currently being worked on
- Which files are being modified
- What needs to be done next
- Key user requests, constraints, or preferences that should persist
- Important technical decisions and why they were made

Your summary should be comprehensive enough to provide context but concise enough to be quickly understood.

Do not respond to any questions in the conversation, only output the summary.
"""

PROMPT_TITLE = """\
You are a title generator. You output ONLY a thread title. Nothing else.

<task>
Generate a brief title that would help the user find this conversation later.

Follow all rules in <rules>
Use the <examples> so you know what a good title looks like.
Your output must be:
- A single line
- ≤50 characters
- No explanations
</task>

<rules>
- you MUST use the same language as the user message you are summarizing
- Title must be grammatically correct and read naturally - no word salad
- Never include tool names in the title (e.g. "read tool", "bash tool", "edit tool")
- Focus on the main topic or question the user needs to retrieve
- Vary your phrasing - avoid repetitive patterns like always starting with "Analyzing"
- When a file is mentioned, focus on WHAT the user wants to do WITH the file, not just that they shared it
- Keep exact: technical terms, numbers, filenames, HTTP codes
- Remove: the, this, my, a, an
- Never assume tech stack
- Never use tools
- NEVER respond to questions, just generate a title for the conversation
- The title should NEVER include "summarizing" or "generating" when generating a title
- DO NOT SAY YOU CANNOT GENERATE A TITLE OR COMPLAIN ABOUT THE INPUT
- Always output something meaningful, even if the input is minimal.
- If the user message is short or conversational (e.g. "hello", "lol", "what's up", "hey"):
  → create a title that reflects the user's tone or intent (such as Greeting, Quick check-in, Light chat, Intro message, etc.)
</rules>

<examples>
"debug 500 errors in production" → Debugging production 500 errors
"refactor user service" → Refactoring user service
"why is app.js failing" → app.js failure investigation
"implement rate limiting" → Rate limiting implementation
"how do I connect postgres to my API" → Postgres API connection
"best practices for React hooks" → React hooks best practices
"@src/auth.ts can you add refresh token support" → Auth refresh token support
"@utils/parser.ts this is broken" → Parser bug fix
"look at @config.json" → Config review
"@App.tsx add dark mode toggle" → Dark mode toggle in App
</examples>
"""

PROMPT_SUMMARY = """\
Summarize what was done in this conversation. Write like a pull request description.

Rules:
- 2-3 sentences max
- Describe the changes made, not the process
- Do not mention running tests, builds, or other validation steps
- Do not explain what the user asked for
- Write in first person (I added..., I fixed...)
- Never ask questions or add new questions
- If the conversation ends with an unanswered question to the user, preserve that exact question
- If the conversation ends with an imperative statement or request to the user (e.g. "Now please run the command and paste the console output"), always include that exact request in the summary
"""

# ---- Built-in agents ----

def _build_agent() -> AgentInfo:
    return AgentInfo(
        name="build",
        description="Default agent with full tool access",
        mode="primary",
        prompt=PROMPT_BUILD,
        permission=merge_rulesets(
            DEFAULT_PERMISSIONS,
            [
                {"permission": "question", "action": "allow", "pattern": "*"},
                {"permission": "plan_enter", "action": "allow", "pattern": "*"},
            ],
        ),
    )


def _plan_agent() -> AgentInfo:
    return AgentInfo(
        name="plan",
        description="Read-only planning agent",
        mode="primary",
        prompt=PROMPT_PLAN,
        permission=merge_rulesets(
            DEFAULT_PERMISSIONS,
            [
                {"permission": "question", "action": "allow", "pattern": "*"},
                {"permission": "plan_exit", "action": "allow", "pattern": "*"},
                # Deny all edit tools except plan files
                {"permission": "write_file", "action": "deny", "pattern": "*"},
                {"permission": "shell", "action": "deny", "pattern": "*"},
                {"permission": "edit", "action": "deny", "pattern": "*"},
                {"permission": "apply_patch", "action": "deny", "pattern": "*"},
            ],
        ),
    )


def _explore_agent() -> AgentInfo:
    return AgentInfo(
        name="explore",
        description="Fast codebase exploration agent",
        mode="subagent",
        prompt=PROMPT_EXPLORE,
        permission=merge_rulesets(
            DEFAULT_PERMISSIONS,
            [
                # Only allow read-only tools
                {"permission": "grep", "action": "allow", "pattern": "*"},
                {"permission": "read_file", "action": "allow", "pattern": "*"},
                {"permission": "list_files", "action": "allow", "pattern": "*"},
                {"permission": "shell", "action": "allow", "pattern": "*"},
                {"permission": "http_fetch", "action": "allow", "pattern": "*"},
                # Deny write tools
                {"permission": "write_file", "action": "deny", "pattern": "*"},
                {"permission": "edit", "action": "deny", "pattern": "*"},
                {"permission": "apply_patch", "action": "deny", "pattern": "*"},
                {"permission": "todowrite", "action": "deny", "pattern": "*"},
                {"permission": "todoread", "action": "deny", "pattern": "*"},
            ],
        ),
    )


def _compaction_agent() -> AgentInfo:
    return AgentInfo(
        name="compaction",
        description="Session compaction agent",
        mode="primary",
        hidden=True,
        prompt=PROMPT_COMPACTION,
        temperature=0.3,
        permission=[
            # Deny ALL tools — text-only
            {"permission": "*", "action": "deny", "pattern": "*"},
        ],
    )


def _title_agent() -> AgentInfo:
    return AgentInfo(
        name="title",
        description="Session title generator",
        mode="primary",
        hidden=True,
        prompt=PROMPT_TITLE,
        temperature=0.5,
        permission=[
            {"permission": "*", "action": "deny", "pattern": "*"},
        ],
    )


def _summary_agent() -> AgentInfo:
    return AgentInfo(
        name="summary",
        description="Session summary generator",
        mode="primary",
        hidden=True,
        prompt=PROMPT_SUMMARY,
        permission=[
            {"permission": "*", "action": "deny", "pattern": "*"},
        ],
    )


# ---- Agent Registry ----

_BUILTIN_AGENTS = {
    "build": _build_agent,
    "plan": _plan_agent,
    "explore": _explore_agent,
    "compaction": _compaction_agent,
    "title": _title_agent,
    "summary": _summary_agent,
}


class AgentRegistry:
    """Registry of all available agents (built-in + config-defined)."""

    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
        self._load_builtins()

    def _load_builtins(self):
        for name, factory in _BUILTIN_AGENTS.items():
            self._agents[name] = factory()

    def load_from_config(self, config_agents: Dict[str, Any]):
        """Load agent definitions from config.agent dict.

        Config agents can override built-in agents or define new ones.
        Config fields are merged on top of built-in defaults if agent exists.
        """
        for name, cfg in config_agents.items():
            if name in self._agents:
                # Merge on top of existing
                agent = self._agents[name]
                if cfg.model is not None:
                    agent.model = {"provider_id": cfg.model.provider_id, "model_id": cfg.model.model_id}
                if cfg.prompt is not None:
                    agent.prompt = cfg.prompt
                if cfg.temperature is not None:
                    agent.temperature = cfg.temperature
                if cfg.top_p is not None:
                    agent.top_p = cfg.top_p
                if cfg.steps is not None:
                    agent.steps = cfg.steps
                if cfg.mode is not None:
                    agent.mode = cfg.mode
                if cfg.hidden is not None:
                    agent.hidden = cfg.hidden
                if cfg.description is not None:
                    agent.description = cfg.description
                if cfg.options:
                    agent.options.update(cfg.options)
                if cfg.permission is not None:
                    # Merge config permissions on top of agent defaults
                    agent.permission = merge_rulesets(agent.permission, cfg.permission)
            else:
                # New agent from config
                model_dict = None
                if cfg.model:
                    model_dict = {"provider_id": cfg.model.provider_id, "model_id": cfg.model.model_id}
                self._agents[name] = AgentInfo(
                    name=name,
                    description=cfg.description or "",
                    mode=cfg.mode or "primary",
                    hidden=cfg.hidden or False,
                    model=model_dict,
                    prompt=cfg.prompt,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    permission=merge_rulesets(DEFAULT_PERMISSIONS, cfg.permission) if cfg.permission else DEFAULT_PERMISSIONS,
                    steps=cfg.steps or 50,
                    options=cfg.options or {},
                )

    def get(self, name: str) -> Optional[AgentInfo]:
        return self._agents.get(name)

    def list(self) -> List[AgentInfo]:
        """List all agents, non-hidden first."""
        visible = [a for a in self._agents.values() if not a.hidden]
        hidden = [a for a in self._agents.values() if a.hidden]
        return visible + hidden

    def list_visible(self) -> List[AgentInfo]:
        """List only visible (non-hidden) agents."""
        return [a for a in self._agents.values() if not a.hidden]

    def default_agent(self) -> str:
        """Return the name of the default agent."""
        return "build"


def disabled_tools(ruleset: Optional[List[Dict[str, Any]]], tool_ids: List[str]) -> List[str]:
    """Return list of tool IDs that are denied by the ruleset for all patterns.

    These tools should be excluded from the LLM tool set entirely.
    """
    if not ruleset:
        return []
    denied = []
    for tool_id in tool_ids:
        action = evaluate_rules(ruleset, tool_id)
        if action == 'deny':
            denied.append(tool_id)
    return denied
