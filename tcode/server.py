from __future__ import annotations
import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from .event import EventBus, Event
from .storage_file import FileStorage
from .session import SessionManager
from .tools import ToolRegistry
from .permissions import PermissionsManager
from .builtin_tools import register_builtin_tools
from .providers.factory import ProviderFactory
from .mcp import MCPManager
from .attachments import AttachmentStore
from .agent import AgentRunner
from .agent_defs import AgentRegistry
from .command import CommandRegistry
from .config import get_config, set_project_dir, get_provider_config
from .instance import Instance

# ---- Module-level state (populated during lifespan) ----

config = None
events: Optional[EventBus] = None
storage: Optional[FileStorage] = None
sessions: Optional[SessionManager] = None
tool_registry: Optional[ToolRegistry] = None
permissions: Optional[PermissionsManager] = None
provider_factory: Optional[ProviderFactory] = None
agent_registry: Optional[AgentRegistry] = None
command_registry: Optional[CommandRegistry] = None
mcp_manager: Optional[MCPManager] = None
attachments: Optional[AttachmentStore] = None
agent_runner: Optional[AgentRunner] = None
_abort_events: Dict[str, asyncio.Event] = {}
_instance: Optional[Instance] = None


# ---- Lifespan ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler — initializes all singletons on startup, cleans up on shutdown."""
    global config, events, storage, sessions, tool_registry, permissions
    global provider_factory, agent_registry, command_registry
    global mcp_manager, attachments, agent_runner, _instance

    # Set project directory
    project_dir = os.environ.get("TCODE_PROJECT_DIR", os.getcwd())
    set_project_dir(project_dir)

    # Instance scoping
    _instance = Instance.get_or_create(project_dir)

    # Load configuration
    config = get_config()

    # Core infrastructure
    events = EventBus()
    storage = FileStorage(base_dir=os.path.join(project_dir, ".tcode"))
    await storage.init()

    sessions = SessionManager(storage=storage, events=events)

    # Tools and providers
    tool_registry = ToolRegistry()
    permissions = PermissionsManager(events)
    register_builtin_tools(tool_registry)

    # Load skills
    from .skills.loader import load_skills, discover_and_load_skills

    discovered_skills = {}
    try:
        discovered_skills = discover_and_load_skills(
            project_dir=project_dir,
            extra_dirs=config.skill.paths or None,
            remote_urls=config.skill.urls or None,
            registry=tool_registry,
        )
    except Exception:
        pass
    try:
        legacy_skills_dir = os.path.join(project_dir, "tcode", "skills")
        load_skills([legacy_skills_dir], tool_registry)
    except Exception:
        pass

    # Register provider adapters
    provider_factory = ProviderFactory()
    from .providers.openai_adapter import OpenAIAdapter
    from .providers.litellm_adapter import LitellmAdapter
    from .providers.azure_openai_adapter import AzureOpenAIAdapter
    from .providers.anthropic_adapter import AnthropicAdapter
    from .providers.gemini_adapter import GeminiAdapter
    from .providers.ollama_adapter import OllamaAdapter

    def _make_adapter_factory(adapter_cls, provider_id):
        def factory(**kwargs):
            pcfg = get_provider_config(provider_id)
            opts = dict(pcfg.options)
            if pcfg.api_key:
                opts.setdefault("api_key", pcfg.api_key)
            if pcfg.base_url:
                opts.setdefault("base_url", pcfg.base_url)
            opts.update(kwargs)
            return adapter_cls(**opts)
        return factory

    provider_factory.register_adapter("openai", _make_adapter_factory(OpenAIAdapter, "openai"))
    provider_factory.register_adapter("azure_openai", _make_adapter_factory(AzureOpenAIAdapter, "azure_openai"))
    provider_factory.register_adapter("anthropic", _make_adapter_factory(AnthropicAdapter, "anthropic"))
    provider_factory.register_adapter("gemini", _make_adapter_factory(GeminiAdapter, "gemini"))
    provider_factory.register_adapter("ollama", _make_adapter_factory(OllamaAdapter, "ollama"))
    provider_factory.register_adapter("litellm", _make_adapter_factory(LitellmAdapter, "litellm"))

    # Agent and command registries
    agent_registry = AgentRegistry()
    agent_registry.load_from_config(config.agent)

    command_registry = CommandRegistry(events=events)
    command_registry.load_from_config(config.command)
    if discovered_skills:
        command_registry.load_from_skills(list(discovered_skills.values()))

    mcp_manager = MCPManager()
    attachments = AttachmentStore()
    agent_runner = AgentRunner(
        provider_factory, tool_registry, mcp_manager, sessions, events, attachments,
        agent_registry=agent_registry,
    )
    agent_runner.toolrunner.permissions = permissions

    # Planner coordinator
    try:
        from .todos import PlannerCoordinator
        from .plan import PlanManager
        PlannerCoordinator(
            events,
            lambda: PlanManager(sessions, provider_factory, events, tool_registry, mcp_manager, attachments),
        )
    except Exception:
        pass

    # Auto-connect MCP servers from config
    for name, mcp_cfg in config.mcp.items():
        if not mcp_cfg.enabled:
            continue
        try:
            cfg_dict: Dict[str, Any] = {"type": mcp_cfg.type}
            if mcp_cfg.url:
                cfg_dict["url"] = mcp_cfg.url
            if mcp_cfg.command:
                cfg_dict["command"] = mcp_cfg.command
            if mcp_cfg.headers:
                cfg_dict["headers"] = mcp_cfg.headers
            if mcp_cfg.timeout:
                cfg_dict["timeout"] = mcp_cfg.timeout
            await mcp_manager.add(name, cfg_dict)
            try:
                if hasattr(mcp_manager, "list_prompts"):
                    prompts = await mcp_manager.list_prompts(name)
                    if prompts:
                        command_registry.load_from_mcp_prompts(name, prompts)
            except Exception:
                pass
        except Exception:
            pass

    # Start CLI permission prompt if TTY
    _start_permission_cli_if_applicable()

    yield

    # Shutdown
    if storage:
        await storage.close()
    from .instance import dispose_all
    dispose_all()


app = FastAPI(lifespan=lifespan)


# ---- CLI permission prompt ----

import threading
import queue as _queue
import sys

_cli_queue: Optional[_queue.Queue] = None
_cli_unsub = None
_main_loop = None


def _start_permission_cli_if_applicable():
    global _cli_queue, _cli_unsub, _main_loop
    try:
        enabled = os.environ.get("TCODE_CLI_PROMPT", "1") == "1" and sys.stdin and sys.stdin.isatty()
        if not enabled:
            return
    except Exception:
        return

    _cli_queue = _queue.Queue()
    try:
        _main_loop = asyncio.get_running_loop()
    except RuntimeError:
        _main_loop = asyncio.get_event_loop()

    async def _ev_cb(ev):
        try:
            _cli_queue.put(ev.to_dict())
        except Exception:
            pass

    _cli_unsub = events.subscribe_all(lambda ev: asyncio.create_task(_ev_cb(ev)))

    def _thread_fn():
        while True:
            try:
                item = _cli_queue.get()
            except Exception:
                break
            if item is None:
                break
            try:
                etype = item.get("type")
                payload = item.get("payload", {})
                if etype == "permission.requested":
                    pid = payload.get("id")
                    ptype = payload.get("type")
                    details = payload.get("details")
                    print("\nPermission request:")
                    print(f"  id: {pid}\n  type: {ptype}\n  details: {details}")
                    ans = input("Allow? (y/N/a=always): ").strip().lower()
                    allow = ans in ("y", "yes", "a", "always")
                    always = ans in ("a", "always")
                    try:
                        if _main_loop:
                            future = asyncio.run_coroutine_threadsafe(
                                permissions.respond(pid, allow, always=always), _main_loop
                            )
                            result = future.result(timeout=10)
                            always_rule = result.get("always_rule") if isinstance(result, dict) else None
                            resp_session_id = result.get("session_id") if isinstance(result, dict) else None
                            if always_rule and resp_session_id:
                                async def _save_always(sid, rule):
                                    await sessions.add_permission_rule(sid, rule)
                                    if _instance:
                                        await sessions.add_project_permission_rule(
                                            _instance.project_id, rule
                                        )
                                asyncio.run_coroutine_threadsafe(
                                    _save_always(resp_session_id, always_rule), _main_loop
                                )
                    except Exception:
                        pass
            except Exception:
                continue

    t = threading.Thread(target=_thread_fn, daemon=True)
    t.start()


# ---- SSE events ----

@app.get("/events")
async def events_stream(request: Request):
    queue: asyncio.Queue = asyncio.Queue()

    unsub = events.subscribe_all(lambda ev: asyncio.create_task(queue.put(ev.to_dict())))

    async def streamer():
        try:
            await queue.put({"type": "server.connected", "payload": {}})
            while True:
                try:
                    item = await queue.get()
                    if item is None:
                        break
                    yield f"data: {json.dumps(item)}\n\n"
                except asyncio.CancelledError:
                    break
                await asyncio.sleep(0)
        finally:
            unsub()

    return StreamingResponse(streamer(), media_type="text/event-stream")


# ---- Attachments ----

@app.get("/attachments/{id}")
async def get_attachment(id: str):
    try:
        path = attachments.get_path(f"attachment://{id}")
        return StreamingResponse(open(path, "rb"), media_type="application/octet-stream")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="attachment not found")


# ---- Session endpoints ----

@app.post("/session")
async def create_session():
    session_id = await sessions.create_session()
    return JSONResponse({"session_id": session_id})


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    try:
        sess = await sessions.get_session(session_id)
        return JSONResponse(sess)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")


@app.post("/session/{session_id}/permissions")
async def set_session_permissions(session_id: str, payload: Dict[str, Any]):
    rules = payload.get("rules") or []
    try:
        await sessions.set_permission(session_id, rules)
        return JSONResponse({"status": "ok"})
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")


@app.get("/session/{session_id}/permissions")
async def get_session_permissions(session_id: str):
    try:
        rules = await sessions.get_permission(session_id)
        return JSONResponse({"rules": rules})
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")


@app.post("/session/{session_id}/message")
async def create_message(session_id: str, payload: Dict[str, Any]):
    role = payload.get("role", "user")
    model = payload.get("model")
    message_id = await sessions.create_message(session_id, role=role, model=model)
    text = payload.get("text")
    if text:
        await sessions.append_text_part(session_id, message_id, text)
    return JSONResponse({"message_id": message_id})


@app.get("/session/{session_id}/message/{message_id}")
async def get_message_endpoint(session_id: str, message_id: str):
    wp = await sessions.get_message(session_id, message_id)
    return JSONResponse(wp.model_dump())


@app.post("/session/{session_id}/compact")
async def compact_session(session_id: str, payload: Dict[str, Any] = {}):
    keep_last_n = payload.get("keep_last_n", 2)
    provider = payload.get("provider", "litellm")
    model = payload.get("model", "gpt-5-mini")
    try:
        from .session_compaction import SessionCompaction
        compactor = SessionCompaction(sessions, provider_factory, events)
        res = await compactor.compact(session_id, keep_last_n=keep_last_n, provider=provider, model=model)
        return JSONResponse({"status": "ok", "result": res})
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/summarize")
async def summarize_session(session_id: str):
    try:
        from .session_summary import SessionSummary
        s = SessionSummary(sessions, events)
        res = await s.summarize(session_id)
        return JSONResponse({"status": "ok", "summary": res})
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/plan")
async def create_plan(session_id: str, payload: Dict[str, Any]):
    title = payload.get("title")
    instructions = payload.get("instructions")
    provider = payload.get("provider", "litellm")
    model = payload.get("model", "gpt-5-mini")
    schema = payload.get("schema")
    try:
        from .plan import PlanManager
        pm = PlanManager(sessions, provider_factory, events)
        if schema:
            sess = await sessions.get_session(session_id)
            meta = sess.get("metadata", {}) or {}
            meta["schema"] = schema
            await storage.update_session(session_id, meta)
        res = await pm.create_plan(session_id, title, instructions, provider=provider, model=model)
        return JSONResponse({"status": "ok", "plan": res})
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/todo/{part_id}/verify")
async def verify_todo_endpoint(session_id: str, part_id: str):
    try:
        from .todos import verify_todo_handler
        res = await verify_todo_handler(session_id, part_id, sessions, agent_runner.toolrunner)
        return JSONResponse({"status": "ok", "result": res})
    except KeyError:
        raise HTTPException(status_code=404, detail="part not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Todo endpoints ----

@app.get("/session/{session_id}/todos")
async def list_todos(session_id: str):
    try:
        todos = await storage.get_todos(session_id)
        return JSONResponse({"todos": todos})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/todos")
async def update_todos(session_id: str, payload: Dict[str, Any]):
    """Replace all todos for a session."""
    todos = payload.get("todos", [])
    try:
        await storage.update_todos(session_id, todos)
        # Publish event
        seq = await storage.next_sequence(session_id)
        ev = Event.create("session.todo.updated", {"sessionID": session_id, "todos": todos},
                          session_id=session_id, sequence=seq)
        await events.publish(ev)
        return JSONResponse({"status": "ok"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Agent run ----

@app.post("/agent/run")
async def run_agent(payload: Dict[str, Any], background: BackgroundTasks):
    session_id = payload.get("session_id")
    message_id = payload.get("message_id")
    default_model = config.model
    provider = payload.get("provider", default_model.provider_id)
    model = payload.get("model", default_model.model_id)
    agent_name = payload.get("agent")
    system_prompt = payload.get("system_prompt")

    if not session_id or not message_id:
        raise HTTPException(status_code=400, detail="session_id and message_id required")

    if system_prompt:
        sess = await storage.get_session(session_id)
        if sess:
            meta = sess.get("metadata", {})
            meta["system_prompt"] = system_prompt
            await storage.update_session(session_id, meta)

    abort_event = asyncio.Event()
    _abort_events[session_id] = abort_event

    async def _run():
        try:
            await agent_runner.run(
                provider, model, session_id, message_id,
                system_prompt=system_prompt,
                agent_name=agent_name,
                abort_event=abort_event,
            )
        finally:
            _abort_events.pop(session_id, None)

    asyncio.create_task(_run())
    return JSONResponse({"status": "started"})


@app.post("/session/{session_id}/abort")
async def abort_session(session_id: str):
    """Abort a running agent loop for a session."""
    abort_event = _abort_events.get(session_id)
    if abort_event:
        abort_event.set()
        return JSONResponse({"status": "aborted"})
    return JSONResponse({"status": "no_active_run"})


# ---- Tool call ----

@app.post("/tools/{tool_id}/call")
async def call_tool(tool_id: str, payload: Dict[str, Any]):
    session_id = payload.get("session_id")
    message_id = payload.get("message_id")
    args = payload.get("args", {})
    call_id = payload.get("call_id") or "call-" + tool_id
    result = await agent_runner.toolrunner.execute_tool(session_id, message_id, call_id, tool_id, args)
    try:
        if payload.get("verify_todo"):
            from .todos import verify_todo_handler
            await verify_todo_handler(session_id, payload.get("part_id"), sessions, agent_runner.toolrunner)
    except Exception:
        pass
    return JSONResponse(result.model_dump())


# ---- Permission endpoints ----

@app.get("/permissions/pending")
async def list_pending_permissions():
    pending = []
    try:
        for pid, pr in permissions._requests.items():
            pending.append({"id": pid, "type": pr.type, "details": pr.details})
    except Exception:
        pass
    return JSONResponse({"pending": pending})


@app.post("/permissions/respond")
async def respond_permission(payload: Dict[str, Any]):
    pid = payload.get("id")
    allow = bool(payload.get("allow"))
    always = bool(payload.get("always", False))
    result = await permissions.respond(pid, allow, always=always)
    always_rule = result.get("always_rule")
    resp_session_id = result.get("session_id")
    if always_rule and resp_session_id:
        try:
            # Save to session for immediate effect
            await sessions.add_permission_rule(resp_session_id, always_rule)
            # Save to project-level for cross-session persistence
            if _instance:
                await sessions.add_project_permission_rule(
                    _instance.project_id, always_rule
                )
        except Exception:
            pass
    return JSONResponse({"status": "ok", "id": pid, "allow": allow, "always": always})


# ---- Command endpoints ----

@app.get("/commands")
async def list_commands():
    commands = command_registry.list()
    return JSONResponse({
        "commands": [
            {"name": c.name, "description": c.description, "source": c.source,
             "hints": c.hints, "agent": c.agent, "subtask": c.subtask}
            for c in commands
        ]
    })


@app.post("/commands/{name}/execute")
async def execute_command(name: str, payload: Dict[str, Any] = {}):
    arguments = payload.get("arguments", [])
    session_id = payload.get("session_id")
    message_id = payload.get("message_id")
    rendered = await command_registry.execute(name, arguments, session_id, message_id)
    if rendered is None:
        raise HTTPException(status_code=404, detail=f"command not found: {name}")
    return JSONResponse({"rendered": rendered})


# ---- Agent endpoints ----

@app.get("/agents")
async def list_agents():
    agents = agent_registry.list_visible()
    return JSONResponse({
        "agents": [
            {"name": a.name, "description": a.description, "mode": a.mode}
            for a in agents
        ]
    })


# ---- Instance endpoint ----

# ---- Session lifecycle endpoints ----

@app.post("/session/{session_id}/fork")
async def fork_session_endpoint(session_id: str, payload: Dict[str, Any] = {}):
    """Fork a session, optionally up to a specific message."""
    upto = payload.get("upto_message_id")
    try:
        from .session_lifecycle import fork_session
        new_sid = await fork_session(sessions, session_id, upto_message_id=upto)
        return JSONResponse({"status": "ok", "session_id": new_sid})
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/archive")
async def archive_session_endpoint(session_id: str):
    try:
        from .session_lifecycle import archive_session
        await archive_session(sessions, session_id)
        return JSONResponse({"status": "ok"})
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")


@app.post("/session/{session_id}/unarchive")
async def unarchive_session_endpoint(session_id: str):
    try:
        from .session_lifecycle import unarchive_session
        await unarchive_session(sessions, session_id)
        return JSONResponse({"status": "ok"})
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")


@app.post("/session/{session_id}/revert")
async def revert_session_endpoint(session_id: str, payload: Dict[str, Any]):
    message_id = payload.get("message_id")
    if not message_id:
        raise HTTPException(status_code=400, detail="message_id required")
    try:
        from .session_lifecycle import set_revert
        await set_revert(sessions, session_id, message_id,
                          snapshot=payload.get("snapshot"), diff=payload.get("diff"))
        return JSONResponse({"status": "ok"})
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")


# ---- Instance endpoint ----

@app.get("/instance")
async def get_instance_info():
    return JSONResponse({
        "directory": _instance.directory if _instance else None,
        "worktree": _instance.worktree if _instance else None,
        "project_id": _instance.project_id if _instance else None,
    })


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
