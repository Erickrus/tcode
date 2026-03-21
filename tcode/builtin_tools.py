from __future__ import annotations
from pydantic import BaseModel
from .tools import ToolInfo, ToolRegistry, ToolResult, ToolContext
from typing import Dict, Any, Optional, List
import os
import glob
import re
from .attachments import AttachmentStore
import httpx

# Implement built-in tools for MVP


def _verbose_log(ctx: ToolContext, tool_name: str, args: Dict[str, Any]):
    """Print tool invocation details when verbose mode is enabled."""
    if not (ctx.extra and ctx.extra.get("verbose")):
        return
    import sys, json as _json
    summary = {}
    for k, v in args.items():
        s = str(v)
        summary[k] = s if len(s) <= 120 else s[:117] + "..."
    print(f"\033[2m[verbose] tool={tool_name} args={_json.dumps(summary, ensure_ascii=False)}\033[0m", file=sys.stderr)

class EchoParams(BaseModel):
    text: str

async def echo_execute(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    _verbose_log(ctx, "echo", args)
    return ToolResult(title="echo", metadata={}, output=args.get("text"), attachments=None)

class ReadFileParams(BaseModel):
    path: str
    max_bytes: int = 1000000

async def read_file_execute(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    _verbose_log(ctx, "read_file", args)
    path = args.get("path")
    max_bytes = args.get("max_bytes", 1000000)
    # safe canonicalization: prevent accessing outside cwd
    base = os.getcwd()
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(base):
        return ToolResult(title="read_file", metadata={}, output=f"Error: access denied {path}")
    if not os.path.exists(abs_path):
        return ToolResult(title="read_file", metadata={}, output=f"Error: file not found: {path}")
    size = os.path.getsize(abs_path)
    truncated = False
    data = ""
    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        data = data[:max_bytes]
        truncated = True
    return ToolResult(title="read_file", metadata={"truncated": truncated, "size": size}, output=data)

class WriteFileParams(BaseModel):
    path: str
    content: str
    overwrite: bool = False

async def write_file_execute(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    _verbose_log(ctx, "write_file", args)
    import asyncio as _aio
    path = args.get("path")
    content = args.get("content", "")
    overwrite = args.get("overwrite", False)
    base = os.getcwd()
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(base):
        return ToolResult(title="write_file", metadata={}, output=f"Error: access denied {path}")
    if os.path.exists(abs_path) and not overwrite:
        return ToolResult(title="write_file", metadata={}, output=f"Error: file exists and overwrite=False")
    def _write():
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
    try:
        await _aio.to_thread(_write)
    except Exception as e:
        return ToolResult(title="write_file", metadata={}, output=f"Error: {e}")
    return ToolResult(title="write_file", metadata={"path": abs_path}, output="OK")

class ListFilesParams(BaseModel):
    pattern: str
    root: Optional[str] = None

async def list_files_execute(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    _verbose_log(ctx, "list_files", args)
    pattern = args.get("pattern")
    root = args.get("root") or os.getcwd()
    abs_root = os.path.abspath(root)
    if not abs_root.startswith(os.getcwd()):
        return ToolResult(title="list_files", metadata={}, output=f"Error: access denied {root}")
    found = glob.glob(os.path.join(abs_root, pattern))
    return ToolResult(title="list_files", metadata={}, output=str(found))

class GrepParams(BaseModel):
    pattern: str
    path: Optional[str] = None
    max_results: int = 100

async def grep_execute(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    _verbose_log(ctx, "grep", args)
    pattern = args.get("pattern")
    path = args.get("path") or os.getcwd()
    max_results = args.get("max_results", 100)
    regex = re.compile(pattern)
    matches = []
    for root, dirs, files in os.walk(path):
        for fn in files:
            full = os.path.join(root, fn)
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, start=1):
                        if regex.search(line):
                            matches.append({"file": full, "line": i, "snippet": line.strip()})
                            if len(matches) >= max_results:
                                return ToolResult(title="grep", metadata={}, output=str(matches))
            except Exception:
                continue
    return ToolResult(title="grep", metadata={}, output=str(matches))

class FileAttachParams(BaseModel):
    content_b64: str
    filename: str
    mime: str

async def file_attach_execute(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    _verbose_log(ctx, "file_attach", args)
    import base64
    content_b64 = args.get("content_b64")
    filename = args.get("filename")
    mime = args.get("mime")
    data = base64.b64decode(content_b64)
    store = AttachmentStore()
    url = store.store(data, filename, mime)
    return ToolResult(title="file_attach", metadata={}, output=url)


# http_fetch tool
class HttpFetchParams(BaseModel):
    url: str
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    timeout: int = 10
    max_bytes: int = 100000

async def http_fetch_execute(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    _verbose_log(ctx, "http_fetch", args)
    # permission gating: request network access
    perm_allowed = False
    try:
        perm_allowed = await ctx.ask('network_access', {'url': args.get('url')})
    except Exception:
        perm_allowed = False
    if not perm_allowed:
        return ToolResult(title='http_fetch', metadata={}, output='Error: network access not permitted')

    url = args.get('url')
    method = (args.get('method') or 'GET').upper()
    headers = args.get('headers') or {}
    timeout = args.get('timeout', 10)
    max_bytes = args.get('max_bytes', 100000)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, url, headers=headers)
            resp.raise_for_status()
            content = resp.content
            truncated = False
            if len(content) > max_bytes:
                truncated = True
                content = content[:max_bytes]
            # attempt to decode as text
            try:
                text = content.decode('utf-8', errors='replace')
            except Exception:
                text = str(content)
            meta = {'status_code': resp.status_code, 'headers': dict(resp.headers), 'truncated': truncated}
            return ToolResult(title='http_fetch', metadata=meta, output=text)
    except Exception as e:
        return ToolResult(title='http_fetch', metadata={}, output=f'Error: {e}')


# registration helper
import shutil
import subprocess
from typing import Tuple

async def shell_execute(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    _verbose_log(ctx, "shell", args)
    cmd = args.get('cmd')
    timeout = args.get('timeout', 30)
    # permission gating: require ctx.extra.get('allow_shell') or ask permission
    allowed = False
    try:
        allowed = await ctx.ask('shell_execute', {'cmd': cmd})
    except Exception:
        allowed = False
    if not allowed:
        return ToolResult(title='shell', metadata={}, output='Error: shell execution not permitted')
    # Use shutil.which to validate executable presence
    if isinstance(cmd, str):
        parts = cmd.split()
    else:
        parts = list(cmd)
    if not parts:
        return ToolResult(title='shell', metadata={}, output='Error: empty command')
    exe = shutil.which(parts[0])
    if not exe:
        return ToolResult(title='shell', metadata={}, output=f'Error: executable not found: {parts[0]}')
    try:
        proc = subprocess.run(parts, capture_output=True, text=True, timeout=timeout, shell=True)
        return ToolResult(title='shell', metadata={'code': proc.returncode}, output=proc.stdout + '\n' + proc.stderr)
    except subprocess.TimeoutExpired:
        return ToolResult(title='shell', metadata={'timeout': True}, output='Error: timeout')
    except Exception as e:
        return ToolResult(title='shell', metadata={}, output=f'Error: {e}')

class ShellParams(BaseModel):
    cmd: str
    timeout: int = 30

# ---- Edit tool ----

class EditParams(BaseModel):
    file_path: str
    old_string: str
    new_string: str
    replace_all: bool = False

async def edit_execute(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Targeted string replacement in a file (matching opencode edit tool)."""
    _verbose_log(ctx, "edit", args)
    file_path = args.get("file_path", "")
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")
    replace_all = args.get("replace_all", False)

    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        return ToolResult(title="edit", output=f"Error: file not found: {file_path}")
    if os.path.isdir(abs_path):
        return ToolResult(title="edit", output=f"Error: path is a directory: {file_path}")

    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    if old_string not in content:
        return ToolResult(title="edit", output=f"Error: old_string not found in {file_path}")

    if not replace_all:
        count = content.count(old_string)
        if count > 1:
            return ToolResult(
                title="edit",
                output=f"Error: old_string found {count} times in {file_path}. Use replace_all=true or provide more context to make it unique.",
            )

    if replace_all:
        new_content = content.replace(old_string, new_string)
    else:
        new_content = content.replace(old_string, new_string, 1)
    
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    return ToolResult(
        title="edit",
        metadata={"file_path": abs_path, "replace_all": replace_all},
        output=f"Edited {file_path}",
    )


# ---- TodoWrite / TodoRead tools ----

class TodoWriteParams(BaseModel):
    todos: List[Dict[str, Any]]

async def todowrite_execute(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Write/replace all todos for the current session."""
    _verbose_log(ctx, "todowrite", args)
    import json as _json
    todos = args.get("todos", [])
    sessions = ctx.extra.get("sessions") if ctx.extra else None
    if not sessions or not ctx.session_id:
        return ToolResult(title="todowrite", output="Error: no session context")
    try:
        await sessions.storage.update_todos(ctx.session_id, todos)
        return ToolResult(
            title="todowrite",
            metadata={"todos": todos},
            output=_json.dumps(todos),
        )
    except Exception as e:
        return ToolResult(title="todowrite", output=f"Error: {e}")


class TodoReadParams(BaseModel):
    pass

async def todoread_execute(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Read all todos for the current session."""
    _verbose_log(ctx, "todoread", args)
    import json as _json
    sessions = ctx.extra.get("sessions") if ctx.extra else None
    if not sessions or not ctx.session_id:
        return ToolResult(title="todoread", output="Error: no session context")
    try:
        todos = await sessions.storage.get_todos(ctx.session_id)
        return ToolResult(
            title="todoread",
            metadata={"todos": todos},
            output=_json.dumps(todos),
        )
    except Exception as e:
        return ToolResult(title="todoread", output=f"Error: {e}")


# ---- Task tool (subagent spawning) ----

class TaskParams(BaseModel):
    description: str
    prompt: str
    subagent_type: str = "explore"

async def task_execute(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Spawn a subagent in a child session. Requires agent_runner in ctx.extra."""
    _verbose_log(ctx, "task", args)
    description = args.get("description", "")
    prompt = args.get("prompt", "")
    subagent_type = args.get("subagent_type", "explore")

    sessions = ctx.extra.get("sessions") if ctx.extra else None
    agent_runner = ctx.extra.get("agent_runner") if ctx.extra else None

    if not sessions or not agent_runner:
        return ToolResult(title="task", output="Error: agent_runner not available in context")

    # Create child session
    child_sid = await sessions.create_session(
        metadata={"parent_session": ctx.session_id, "task_description": description}
    )

    # Create user message with prompt
    child_mid = await sessions.create_message(child_sid, "user")
    await sessions.append_text_part(child_sid, child_mid, prompt)

    # Run subagent
    try:
        result = await agent_runner.run(
            agent_runner.providers._adapters.get("litellm") and "litellm" or "openai",
            "",  # model determined by agent config
            child_sid, child_mid,
            agent_name=subagent_type,
            max_steps=20,
        )
        final_text = result.get("final_text", "")
        return ToolResult(
            title="task",
            metadata={"sessionId": child_sid, "steps": result.get("steps", 0),
                       "cost": result.get("cost", 0)},
            output=final_text or "(task completed with no text output)",
        )
    except Exception as e:
        return ToolResult(title="task", output=f"Error running subagent: {e}")


# register all tools
def register_builtin_tools(registry: ToolRegistry):
    registry.register(ToolInfo(id="builtin_echo", description="Echo text", parameters=EchoParams, execute=echo_execute, permission="low"))
    registry.register(ToolInfo(id="builtin_read_file", description="Read file", parameters=ReadFileParams, execute=read_file_execute, permission="medium"))
    registry.register(ToolInfo(id="builtin_write_file", description="Write file", parameters=WriteFileParams, execute=write_file_execute, permission="high"))
    registry.register(ToolInfo(id="builtin_list_files", description="List files", parameters=ListFilesParams, execute=list_files_execute, permission="medium"))
    registry.register(ToolInfo(id="builtin_grep", description="Grep", parameters=GrepParams, execute=grep_execute, permission="medium"))
    registry.register(ToolInfo(id="builtin_file_attach", description="Store attachment", parameters=FileAttachParams, execute=file_attach_execute, permission="medium"))
    registry.register(ToolInfo(id="builtin_shell", description="Execute shell command (restricted)", parameters=ShellParams, execute=shell_execute, permission="high"))
    registry.register(ToolInfo(id="builtin_http_fetch", description="HTTP fetch (GET)", parameters=HttpFetchParams, execute=http_fetch_execute, permission="medium"))
    registry.register(ToolInfo(id="builtin_structured_output", description="Produce structured JSON output following a schema", parameters=StructuredParams, execute=structured_execute, permission="low"))
    registry.register(ToolInfo(id="builtin_plan_exit", description="Signal that planning is complete and request user approval", parameters=None, execute=plan_exit_execute, permission="high"))

    # Phase 6 tools
    registry.register(ToolInfo(id="builtin_edit", description="Targeted string replacement in a file", parameters=EditParams, execute=edit_execute, permission="high"))
    registry.register(ToolInfo(id="builtin_todowrite", description="Write/replace all todos for the session", parameters=TodoWriteParams, execute=todowrite_execute, permission="low"))
    registry.register(ToolInfo(id="builtin_todoread", description="Read all todos for the session", parameters=TodoReadParams, execute=todoread_execute, permission="low"))
    registry.register(ToolInfo(id="builtin_task", description="Spawn a subagent in a child session", parameters=TaskParams, execute=task_execute, permission="high"))

# module-level plan_exit implementation so tests can import it
async def plan_exit_execute(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    _verbose_log(ctx, "plan_exit", args)
    # only allowed in plan mode; mark session plan as ready
    session_id = ctx.session_id
    try:
        sessions = None
        try:
            sessions = ctx.extra.get('sessions') if ctx.extra else None
        except Exception:
            sessions = None
        if sessions:
            sess = await sessions.get_session(session_id)
            meta = sess.get('metadata', {}) or {}
            meta['plan_active'] = False
            meta['plan_ready'] = True
            await sessions.set_summary(session_id, {'plan': meta.get('plan', {})})
    except Exception:
        pass
    return ToolResult(title='plan_exit', metadata={}, output='Plan exit acknowledged')


# Structured output definitions (module-level) - parameters and execution function
class StructuredParams(BaseModel):
    schema: Dict[str, Any]


# expose structured_execute for unit tests
async def structured_execute(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    _verbose_log(ctx, "structured_output", args)
    # Ask provider to produce JSON matching the schema, then validate basic shape.
    import json
    schema = args.get('schema')
    provider = None
    try:
        provider = ctx.extra.get('provider') if ctx.extra else None
    except Exception:
        provider = None
    prompt = f"Please output a JSON object that conforms to the following schema, and respond ONLY with the JSON object (no extra text):\n{schema}\n"
    text = str(schema)
    if provider and hasattr(provider, 'chat'):
        resp = await provider.chat([{"role": "user", "content": prompt}], None, {})
        if isinstance(resp, dict):
            choices = resp.get('choices')
            if choices and isinstance(choices, list):
                c0 = choices[0]
                text = c0.get('text') or (c0.get('message') and c0.get('message').get('content')) or ''
            else:
                text = resp.get('text') or ''
        else:
            text = str(resp)
    # try to parse JSON
    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        # return error with raw text
        return ToolResult(title='structured_output', metadata={'schema': schema, 'error': 'invalid_json'}, output=text)
    # basic shape check: ensure parsed is dict
    if not isinstance(parsed, dict):
        return ToolResult(title='structured_output', metadata={'schema': schema, 'error': 'not_object'}, output=str(parsed))
    # perform full JSON Schema validation if jsonschema available
    try:
        import jsonschema
        try:
            jsonschema.validate(instance=parsed, schema=schema)
        except jsonschema.ValidationError as ve:
            return ToolResult(title='structured_output', metadata={'schema': schema, 'error': 'validation_error', 'message': str(ve)}, output=parsed)
        except Exception as e:
            return ToolResult(title='structured_output', metadata={'schema': schema, 'error': 'validation_failed', 'message': str(e)}, output=parsed)
    except Exception:
        # jsonschema not available, skip strict validation but return parsed
        pass
    return ToolResult(title='structured_output', metadata={'schema': schema}, output=parsed)

# ---- Skill tool (dynamic, agent-callable) ----

class SkillParams(BaseModel):
    name: str


def make_skill_tool(discovered_skills: Dict[str, Dict[str, Any]]) -> ToolInfo:
    """Create a skill tool with dynamic description listing available skills."""
    if not discovered_skills:
        desc = "Load a specialized skill. No skills are currently available."
    else:
        lines = [
            "Load a specialized skill that provides domain-specific instructions and workflows.",
            "",
            "When you recognize that a task matches one of the available skills listed below,",
            "use this tool to load the full skill instructions.",
            "",
            "<available_skills>",
        ]
        for name, info in discovered_skills.items():
            lines.extend([
                "  <skill>",
                f"    <name>{name}</name>",
                f"    <description>{info.get('description', '')}</description>",
                f"    <location>{info.get('location', '')}</location>",
                "  </skill>",
            ])
        lines.append("</available_skills>")
        desc = "\n".join(lines)

    async def skill_execute(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        skill_name = args.get("name", "")
        skill = discovered_skills.get(skill_name)
        if not skill:
            available = ", ".join(discovered_skills.keys()) or "none"
            return ToolResult(
                title="skill",
                output=f'Skill "{skill_name}" not found. Available: {available}',
            )

        # List files in skill directory (sampled, max 10)
        skill_dir = os.path.dirname(skill["location"])
        skill_files: List[str] = []
        for root, dirs, files in os.walk(skill_dir):
            for f in files:
                if f.upper() == "SKILL.MD":
                    continue
                skill_files.append(os.path.join(root, f))
                if len(skill_files) >= 10:
                    break
            if len(skill_files) >= 10:
                break

        files_xml = "\n".join(f"<file>{f}</file>" for f in skill_files)
        output = "\n".join([
            f'<skill_content name="{skill_name}">',
            f"# Skill: {skill_name}",
            "",
            skill["content"].strip(),
            "",
            f"Base directory for this skill: {skill_dir}",
            "Relative paths in this skill are relative to this base directory.",
            "",
            "<skill_files>",
            files_xml,
            "</skill_files>",
            "</skill_content>",
        ])
        return ToolResult(
            title=f"Loaded skill: {skill_name}",
            metadata={"name": skill_name, "dir": skill_dir},
            output=output,
        )

    return ToolInfo(
        id="builtin_skill",
        description=desc,
        parameters=SkillParams,
        execute=skill_execute,
        permission="low",
    )


# ---- MCP list tool ----

class McpListParams(BaseModel):
    pass


def make_mcp_list_tool(mcp_manager) -> ToolInfo:
    """Create a tool that lists MCP server status and their tools."""

    async def mcp_list_execute(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not mcp_manager or not mcp_manager.status:
            return ToolResult(title="mcp_list", output="No MCP servers configured.")
        lines = []
        for name, status in mcp_manager.status.items():
            tools: List[str] = []
            if name in mcp_manager.clients:
                try:
                    tools = await mcp_manager.list_tools(name)
                except Exception:
                    pass
            tool_str = ", ".join(tools) if tools else "none"
            lines.append(f"Server: {name} | Status: {status} | Tools: {tool_str}")
        return ToolResult(title="mcp_list", output="\n".join(lines))

    return ToolInfo(
        id="builtin_mcp_list",
        description="List connected MCP servers and their available tools",
        parameters=McpListParams,
        execute=mcp_list_execute,
        permission="low",
    )


# Also make structured_execute importable at module level
__all__ = ['structured_execute', 'make_skill_tool', 'make_mcp_list_tool']
