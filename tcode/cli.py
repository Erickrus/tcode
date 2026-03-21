#!/usr/bin/env python3
"""tcode CLI — interactive chat with the tcode agent.

Usage:
    python -m tcode.cli                    # interactive REPL
    python -m tcode.cli "fix the bug"      # single prompt
    python -m tcode.cli -a explore "find all API endpoints"
    python -m tcode.cli --model claude-sonnet-4-20250514

Environment variables:
    TCODE_PROJECT_DIR     Project directory (default: cwd)
    TCODE_MODEL_PROVIDER  Provider (default: litellm)
    TCODE_MODEL_ID        Model ID
    TCODE_PROVIDER_*_API_KEY  Provider API keys
"""
from __future__ import annotations
import argparse
import asyncio
import os
import sys
import signal
import json
import time
from typing import Optional, Dict, Any

from .config import get_config, set_project_dir, get_provider_config
from .event import EventBus, Event
from .storage_file import FileStorage
from .session import SessionManager
from .tools import ToolRegistry
from .permissions import PermissionsManager
from .builtin_tools import register_builtin_tools, make_skill_tool, make_mcp_list_tool
from .providers.factory import ProviderFactory
from .mcp import MCPManager
from .attachments import AttachmentStore
from .agent import AgentRunner
from .agent_defs import AgentRegistry
from .command import CommandRegistry
from .instance import Instance
from .skills.loader import discover_and_load_skills, load_skills


class TcodeCLI:
    """Interactive CLI for tcode agent."""

    def __init__(self, project_dir: str):
        self.project_dir = os.path.abspath(project_dir)
        self.session_id: Optional[str] = None
        self.abort_event: Optional[asyncio.Event] = None
        self.tui_mode: bool = False  # suppress print output when True

        # Components (initialized in setup)
        self.config = None
        self.events: Optional[EventBus] = None
        self.storage: Optional[FileStorage] = None
        self.sessions: Optional[SessionManager] = None
        self.tool_registry: Optional[ToolRegistry] = None
        self.permissions: Optional[PermissionsManager] = None
        self.provider_factory: Optional[ProviderFactory] = None
        self.agent_registry: Optional[AgentRegistry] = None
        self.command_registry: Optional[CommandRegistry] = None
        self.mcp_manager: Optional[MCPManager] = None
        self.attachments: Optional[AttachmentStore] = None
        self.agent_runner: Optional[AgentRunner] = None
        self.instance: Optional[Instance] = None

    async def setup(self):
        """Initialize all tcode components."""
        set_project_dir(self.project_dir)
        self.config = get_config()
        self.instance = Instance.get_or_create(self.project_dir)

        # Core
        self.events = EventBus()
        self.storage = FileStorage(
            base_dir=os.path.join(self.project_dir, ".tcode")
        )
        await self.storage.init()
        self.sessions = SessionManager(storage=self.storage, events=self.events)

        # Tools
        self.tool_registry = ToolRegistry()
        self.permissions = PermissionsManager(self.events)
        register_builtin_tools(self.tool_registry)

        # Skills
        self._discovered_skills: Dict[str, Any] = {}
        try:
            self._discovered_skills = discover_and_load_skills(
                project_dir=self.project_dir,
                extra_dirs=self.config.skill.paths or None,
                remote_urls=self.config.skill.urls or None,
                registry=self.tool_registry,
            )
        except Exception:
            pass
        try:
            load_skills(
                [os.path.join(self.project_dir, "tcode", "skills")],
                self.tool_registry,
            )
        except Exception:
            pass

        # Register skill tool (agent-callable, with discovered skills)
        skill_tool = make_skill_tool(self._discovered_skills)
        self.tool_registry.register(skill_tool)

        # Providers
        self.provider_factory = ProviderFactory()
        self._register_providers()

        # Agents and commands
        self.agent_registry = AgentRegistry()
        self.agent_registry.load_from_config(self.config.agent)

        self.command_registry = CommandRegistry(events=self.events)
        self.command_registry.load_from_config(self.config.command)
        if self._discovered_skills:
            self.command_registry.load_from_skills(list(self._discovered_skills.values()))

        # MCP
        self.mcp_manager = MCPManager(events=self.events, tool_registry=self.tool_registry)
        await self._connect_mcp()

        # Register MCP list tool (agent-callable)
        mcp_list_tool = make_mcp_list_tool(self.mcp_manager)
        self.tool_registry.register(mcp_list_tool)

        # Attachments
        self.attachments = AttachmentStore()

        # Agent runner
        self.agent_runner = AgentRunner(
            self.provider_factory,
            self.tool_registry,
            self.mcp_manager,
            self.sessions,
            self.events,
            self.attachments,
            agent_registry=self.agent_registry,
        )
        self.agent_runner.toolrunner.permissions = self.permissions

        # Subscribe to events for live output
        self._subscribe_events()

    def _register_providers(self):
        """Register all provider adapters with config."""
        from .providers.openai_adapter import OpenAIAdapter
        from .providers.litellm_adapter import LitellmAdapter
        from .providers.azure_openai_adapter import AzureOpenAIAdapter
        from .providers.anthropic_adapter import AnthropicAdapter
        from .providers.gemini_adapter import GeminiAdapter
        from .providers.ollama_adapter import OllamaAdapter

        adapters = {
            "openai": OpenAIAdapter,
            "azure_openai": AzureOpenAIAdapter,
            "anthropic": AnthropicAdapter,
            "gemini": GeminiAdapter,
            "ollama": OllamaAdapter,
            "litellm": LitellmAdapter,
        }
        for name, cls in adapters.items():
            self.provider_factory.register_adapter(
                name, self._make_adapter_factory(cls, name)
            )

    def _make_adapter_factory(self, adapter_cls, provider_id):
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

    async def _connect_mcp(self):
        """Auto-connect MCP servers from config."""
        for name, mcp_cfg in self.config.mcp.items():
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
                # Pass through args for stdio transports
                if mcp_cfg.args:
                    cfg_dict["args"] = mcp_cfg.args
                await self.mcp_manager.add(name, cfg_dict)
                # Register MCP tools into the tool registry
                if self.mcp_manager.status.get(name) == "connected":
                    try:
                        client = self.mcp_manager.clients.get(name)
                        if client:
                            raw_tools = await client.list_tools()
                            for t in raw_tools:
                                toolinfo = self.mcp_manager.convert_mcp_tool(name, t)
                                self.tool_registry.register(toolinfo)
                    except Exception:
                        pass
                # Load MCP prompts as commands
                try:
                    if hasattr(self.mcp_manager, "list_prompts"):
                        prompts = await self.mcp_manager.list_prompts(name)
                        if prompts:
                            self.command_registry.load_from_mcp_prompts(name, prompts)
                except Exception:
                    pass
            except Exception:
                pass

    def _subscribe_events(self):
        """Subscribe to events for terminal output."""
        self._unsub_permission = self.events.subscribe("permission.requested", self._handle_permission)

    async def _handle_permission(self, ev: Event):
        """Handle permission requests interactively."""
        payload = ev.payload
        pid = payload.get("id")
        ptype = payload.get("type")
        details = payload.get("details", {})

        # Format details concisely
        detail_str = ""
        if isinstance(details, dict):
            for k, v in details.items():
                val = str(v)
                if len(val) > 80:
                    val = val[:77] + "..."
                detail_str += f"\n    {k}: {val}"

        print(f"\n\033[33m[Permission]\033[0m {ptype}{detail_str}")

        # Use asyncio-compatible input
        loop = asyncio.get_running_loop()
        try:
            ans = await loop.run_in_executor(
                None, lambda: input("  Allow? (y/N/a=always): ").strip().lower()
            )
        except (EOFError, KeyboardInterrupt):
            ans = "n"

        allow = ans in ("y", "yes", "a", "always")
        always = ans in ("a", "always")

        result = await self.permissions.respond(pid, allow, always=always)

        always_rule = result.get("always_rule")
        resp_session_id = result.get("session_id") or ev.session_id
        if always_rule and resp_session_id:
            try:
                # Save to session for immediate effect
                await self.sessions.add_permission_rule(resp_session_id, always_rule)
                # Save to project-level for cross-session persistence
                if self.instance:
                    await self.sessions.add_project_permission_rule(
                        self.instance.project_id, always_rule
                    )
            except Exception:
                pass

    async def new_session(self) -> str:
        """Create a new session and return its ID."""
        self.session_id = await self.sessions.create_session()
        return self.session_id

    async def send(
        self,
        text: str,
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a message and run the agent. Returns the run result."""
        if not self.session_id:
            await self.new_session()

        # Create user message
        message_id = await self.sessions.create_message(self.session_id, "user")
        await self.sessions.append_text_part(self.session_id, message_id, text)

        # Resolve provider/model
        default_model = self.config.model
        pid = provider_id or default_model.provider_id
        mid = model_id or default_model.model_id

        # Abort event
        self.abort_event = asyncio.Event()

        if not self.tui_mode:
            print(f"\033[2m[{pid}/{mid}] thinking...\033[0m")

        # Run agent
        result = await self.agent_runner.run(
            provider_id=pid,
            model=mid,
            session_id=self.session_id,
            message_id=message_id,
            system_prompt=system_prompt,
            agent_name=agent_name,
            abort_event=self.abort_event,
        )

        # Print result
        final_text = result.get("final_text", "")
        if final_text and not self.tui_mode:
            print(f"\n{final_text}")

        # Print stats
        steps = result.get("steps", 0)
        cost = result.get("cost", 0.0)
        tokens = result.get("tokens", {})
        input_t = tokens.get("input", 0)
        output_t = tokens.get("output", 0)
        blocked = result.get("blocked", False)

        stats_parts = []
        if steps > 0:
            stats_parts.append(f"steps={steps}")
        if input_t or output_t:
            stats_parts.append(f"tokens={input_t}+{output_t}")
        if cost > 0:
            stats_parts.append(f"cost=${cost:.4f}")
        if blocked:
            stats_parts.append("BLOCKED")

        if stats_parts and not self.tui_mode:
            print(f"\033[2m[{' | '.join(stats_parts)}]\033[0m")

        return result

    async def run_command(self, name: str, arguments: list = None) -> Optional[str]:
        """Execute a command by name."""
        rendered = await self.command_registry.execute(
            name, arguments or [], self.session_id, None
        )
        if rendered is None:
            print(f"Command not found: {name}")
            return None
        return rendered

    async def repl(
        self,
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """Interactive REPL loop."""
        await self.new_session()

        default_model = self.config.model
        pid = provider_id or default_model.provider_id
        mid = model_id or default_model.model_id

        print(f"tcode REPL — {pid}/{mid}")
        print(f"Project: {self.project_dir}")
        print(f"Session: {self.session_id}")
        print(f"Tools: {len(self.tool_registry.list())} registered")
        agents = self.agent_registry.list_visible()
        if agents:
            print(f"Agents: {', '.join(a.name for a in agents)}")
        commands = self.command_registry.list()
        if commands:
            print(f"Commands: {', '.join(c.name for c in commands)}")
        print(f"Type /help for commands, /quit to exit.\n")

        loop = asyncio.get_running_loop()
        while True:
            try:
                user_input = await loop.run_in_executor(
                    None, lambda: input("\033[1m> \033[0m")
                )
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            text = user_input.strip()
            if not text:
                continue

            # Handle special commands
            if text == "/quit" or text == "/exit":
                print("Bye.")
                break
            elif text == "/help":
                self._print_help()
                continue
            elif text == "/new":
                await self.new_session()
                print(f"New session: {self.session_id}")
                continue
            elif text == "/tools":
                tools = self.tool_registry.list()
                for t in tools:
                    print(f"  {t}")
                continue
            elif text == "/agents":
                for a in self.agent_registry.list_visible():
                    marker = " *" if a.name == (agent_name or "build") else ""
                    print(f"  {a.name}: {a.description}{marker}")
                continue
            elif text == "/skills":
                if self._discovered_skills:
                    for name, info in self._discovered_skills.items():
                        desc = info.get('description', '')
                        loc = info.get('location', '')
                        line = f"  {name}"
                        if desc:
                            line += f": {desc}"
                        if loc:
                            line += f"  ({loc})"
                        print(line)
                else:
                    print("  No skills discovered.")
                    print("  Place SKILL.md files in .tcode/skills/ or ~/.config/tcode/skills/")
                continue
            elif text == "/commands":
                for c in self.command_registry.list():
                    print(f"  /{c.name}: {c.description}")
                continue
            elif text == "/mcp":
                if self.mcp_manager and self.mcp_manager.status:
                    for name, st in self.mcp_manager.status.items():
                        tools = []
                        if name in self.mcp_manager.clients:
                            tools = [
                                t for t in self.tool_registry.list()
                                if t.startswith(f"mcp_{name}_")
                            ]
                        tool_count = f"  ({len(tools)} tools)" if tools else ""
                        print(f"  {name}: {st}{tool_count}")
                else:
                    print("  No MCP servers configured.")
                    print("  Add servers in tcode.json under \"mcp\".")
                continue
            elif text == "/cost":
                # Show session cost from step-finish parts
                total_cost = 0.0
                async for wp in self.sessions.stream_messages(self.session_id):
                    for p in wp.parts:
                        if p.get("type") == "step-finish":
                            total_cost += p.get("cost", 0.0)
                print(f"Session cost: ${total_cost:.4f}")
                continue
            elif text == "/compact":
                try:
                    from .session_compaction import SessionCompaction
                    compactor = SessionCompaction(
                        self.sessions, self.provider_factory, self.events
                    )
                    await compactor.compact(self.session_id, provider=pid, model=mid)
                    print("Session compacted.")
                except Exception as e:
                    print(f"Compaction failed: {e}")
                continue
            elif text == "/memory" or text == "/memory show":
                from .memory import read_memory
                content = read_memory(self.storage.base_dir)
                if content:
                    print(content)
                else:
                    print("No project memory. Use memory_write tool or 'remember X' to create entries.")
                continue
            elif text == "/memory compact":
                try:
                    from .memory import consolidate_memory
                    new_content = await consolidate_memory(
                        self.storage.base_dir, self.provider_factory,
                        provider_id=pid, model=mid,
                    )
                    print("Memory consolidated.")
                    from .memory import parse_entries
                    entries = parse_entries(new_content)
                    print(f"  {len(entries)} entries remaining.")
                except Exception as e:
                    print(f"Memory consolidation failed: {e}")
                continue
            elif text.startswith("/agent "):
                new_agent = text[7:].strip()
                agent_info = self.agent_registry.get(new_agent)
                if agent_info:
                    agent_name = new_agent
                    print(f"Switched to agent: {agent_name}")
                else:
                    print(f"Unknown agent: {new_agent}")
                continue
            elif text.startswith("/model "):
                parts = text[7:].strip().split("/", 1)
                if len(parts) == 2:
                    pid, mid = parts
                else:
                    mid = parts[0]
                # Save to tcode.json for persistence
                from .config import save_model_to_project
                save_model_to_project(self.project_dir, pid, mid)
                print(f"Switched to model: {pid}/{mid} (saved to tcode.json)")
                continue
            elif text.startswith("/"):
                # Try as command
                parts = text[1:].split(None, 1)
                cmd_name = parts[0]
                cmd_args = parts[1].split() if len(parts) > 1 else []
                rendered = await self.run_command(cmd_name, cmd_args)
                if rendered:
                    # Feed rendered command as user prompt
                    await self.send(
                        rendered,
                        provider_id=pid,
                        model_id=mid,
                        agent_name=agent_name,
                        system_prompt=system_prompt,
                    )
                continue

            # Regular message
            try:
                await self.send(
                    text,
                    provider_id=pid,
                    model_id=mid,
                    agent_name=agent_name,
                    system_prompt=system_prompt,
                )
            except KeyboardInterrupt:
                if self.abort_event:
                    self.abort_event.set()
                print("\n\033[33m[Aborted]\033[0m")
            except Exception as e:
                print(f"\033[31mError: {e}\033[0m")

    def _print_help(self):
        print("""
tcode REPL commands:
  /help         Show this help
  /quit         Exit the REPL
  /new          Start a new session
  /tools        List available tools
  /agents       List available agents
  /commands     List available commands
  /skills       List discovered skills
  /mcp          List MCP servers and status
  /agent <name> Switch to a different agent
  /model [provider/]model  Switch model
  /cost         Show session cost
  /compact      Compact session history
  /memory       Show project memory
  /memory compact  Consolidate memory entries
  /<command>    Run a registered command

Ctrl+C during generation aborts the current run.
""")

    async def teardown(self):
        """Clean up resources."""
        # Close MCP transports (subprocess pipes) before the event loop closes
        if self.mcp_manager:
            for name in list(self.mcp_manager.transports):
                try:
                    await self.mcp_manager.remove(name)
                except Exception:
                    pass
        if self.storage:
            await self.storage.close()
        from .instance import dispose_all
        dispose_all()

banner = """
████████╗     ██████╗ ██████╗ ██████╗ ███████╗
╚══██╔══╝    ██╔════╝██╔═══██╗██╔══██╗██╔════╝
   ██║       ██║     ██║   ██║██║  ██║█████╗  
   ██║       ██║     ██║   ██║██║  ██║██╔══╝  
   ██║       ╚██████╗╚██████╔╝██████╔╝███████╗
   ╚═╝        ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
"""

async def async_main():
    parser = argparse.ArgumentParser(
        description="tcode — interactive AI coding agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt", nargs="*", help="Prompt to send (omit for interactive REPL)"
    )
    parser.add_argument(
        "-d", "--dir", default=os.getcwd(),
        help="Project directory (default: cwd)",
    )
    parser.add_argument(
        "-p", "--provider", default=None,
        help="Provider ID (e.g., litellm, openai, anthropic)",
    )
    parser.add_argument(
        "-m", "--model", default=None,
        help="Model ID (e.g., claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "-a", "--agent", default=None,
        help="Agent name (e.g., build, explore, plan)",
    )
    parser.add_argument(
        "-s", "--system", default=None,
        help="System prompt override",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose mode: log tool invocations and other debug info",
    )
    parser.add_argument(
        "--repl", action="store_true",
        help="Use the old line-based REPL instead of the TUI",
    )
    parser.add_argument(
        "--tui", action="store_true",
        help="Use the full-screen TUI (default for interactive mode)",
    )

    args = parser.parse_args()

    if args.prompt:
        # Single-shot mode — no TUI
        print(banner)
        cli = TcodeCLI(project_dir=args.dir)

        def signal_handler(sig, frame):
            if cli.abort_event:
                cli.abort_event.set()
            else:
                print("\nBye.")
                sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        try:
            await cli.setup()
            if args.verbose:
                cli.agent_runner.toolrunner.verbose = True
            prompt_text = " ".join(args.prompt)
            await cli.new_session()
            await cli.send(
                prompt_text,
                provider_id=args.provider,
                model_id=args.model,
                agent_name=args.agent,
                system_prompt=args.system,
            )
        finally:
            await cli.teardown()

    elif args.repl:
        # Old REPL mode
        print(banner)
        cli = TcodeCLI(project_dir=args.dir)

        def signal_handler(sig, frame):
            if cli.abort_event:
                cli.abort_event.set()
            else:
                print("\nBye.")
                sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        try:
            await cli.setup()
            if args.verbose:
                cli.agent_runner.toolrunner.verbose = True
            await cli.repl(
                provider_id=args.provider,
                model_id=args.model,
                agent_name=args.agent,
                system_prompt=args.system,
            )
        finally:
            await cli.teardown()

    else:
        # TUI mode (default)
        from tcode_app import TcodeApp
        tui_app = TcodeApp(args)
        try:
            await tui_app._run_async()
        except KeyboardInterrupt:
            pass
        finally:
            tui_app._shutdown()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
