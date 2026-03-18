"""TcodeBridge — async bridge between tcode backend and runtui paint cycle."""
from __future__ import annotations

import asyncio
import traceback
from typing import Any, TYPE_CHECKING

from ..event import Event

if TYPE_CHECKING:
    from tcode_app import TcodeApp
    from ..cli import TcodeCLI


class TcodeBridge:
    """Connects tcode async events to TUI state + repaint."""

    def __init__(self, app: TcodeApp, cli: TcodeCLI) -> None:
        self.app = app
        self.cli = cli
        self.state = app.state
        self._unsubs: list[Any] = []
        self._current_task: asyncio.Task | None = None
        self._subscribe_events()

    def _subscribe_events(self) -> None:
        bus = self.cli.events

        self._unsubs.append(
            bus.subscribe("permission.requested", self._on_permission_requested)
        )
        self._unsubs.append(
            bus.subscribe("permission.responded", self._on_permission_responded)
        )
        self._unsubs.append(
            bus.subscribe("session.status.changed", self._on_session_status)
        )
        self._unsubs.append(
            bus.subscribe("tool.started", self._on_tool_started)
        )
        self._unsubs.append(
            bus.subscribe("tool.completed", self._on_tool_completed)
        )
        self._unsubs.append(
            bus.subscribe("message.part.updated", self._on_part_updated)
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_permission_requested(self, ev: Event) -> None:
        payload = ev.payload
        self.state.pending_permission = {
            "id": payload.get("id"),
            "type": payload.get("type"),
            "details": payload.get("details", {}),
            "session_id": payload.get("session_id"),
        }
        self.state.status = "waiting_permission"
        self.app.schedule_repaint()

    async def _on_permission_responded(self, ev: Event) -> None:
        self.state.pending_permission = None
        if self.state.status == "waiting_permission":
            self.state.status = "running"
        self.app.schedule_repaint()

    async def _on_session_status(self, ev: Event) -> None:
        status = ev.payload.get("status", "")
        if status == "retry":
            self.state.status = "running"
        self.app.schedule_repaint()

    async def _on_tool_started(self, ev: Event) -> None:
        payload = ev.payload
        self.state.messages.append({
            "type": "tool",
            "tool": payload.get("tool", ""),
            "call_id": payload.get("callID", ""),
            "status": "running",
            "input": payload.get("input", {}),
            "output": None,
        })
        self.app.schedule_repaint()

    async def _on_tool_completed(self, ev: Event) -> None:
        call_id = ev.payload.get("callID", "")
        result = ev.payload.get("result", {})
        for msg in reversed(self.state.messages):
            if msg.get("type") == "tool" and msg.get("call_id") == call_id:
                msg["status"] = "error" if result.get("error") else "done"
                msg["output"] = result
                break
        self.app.schedule_repaint()

    async def _on_part_updated(self, ev: Event) -> None:
        self.app.schedule_repaint()

    # ------------------------------------------------------------------
    # Spinner management
    # ------------------------------------------------------------------

    def _start_spinner(self) -> None:
        session_screen = self.app._session_screen
        if session_screen and hasattr(session_screen, 'start_spinner'):
            session_screen.start_spinner()

    def _stop_spinner(self) -> None:
        session_screen = self.app._session_screen
        if session_screen and hasattr(session_screen, 'stop_spinner'):
            session_screen.stop_spinner()

    # ------------------------------------------------------------------
    # State reset — the single place that cleans up after any run
    # ------------------------------------------------------------------

    def _reset_to_idle(self) -> None:
        """Unconditionally reset state to idle. Called after every run path."""
        self.state.status = "idle"
        self.state.streaming_text = ""
        self.state.pending_permission = None
        self.state.cancel_pending = False
        self._stop_spinner()
        self.app.schedule_repaint()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    async def start_session(self) -> str:
        session_id = await self.cli.new_session()
        self.state.session_id = session_id
        self.state.messages.clear()
        self.state.cost = 0.0
        self.state.tokens = {}
        self.state.streaming_text = ""
        return session_id

    def submit_prompt(self, text: str) -> None:
        """Submit a prompt. If running, queue it."""
        text = text.strip()
        if not text:
            return

        # Add to history
        if not self.state.prompt_history or self.state.prompt_history[-1] != text:
            self.state.prompt_history.append(text)

        if self.state.status in ("running", "waiting_permission"):
            self.state.queued_prompts.append(text)
            self.app.schedule_repaint()
            return

        self._launch_prompt(text)

    def _launch_prompt(self, text: str) -> None:
        """Create a tracked task for the prompt."""
        task = asyncio.create_task(self._run_prompt(text))
        self._current_task = task
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Safety net: if the task ended with an exception, force reset."""
        self._current_task = None
        exc = task.exception() if not task.cancelled() else None
        if exc:
            self.state.messages.append({
                "type": "error",
                "text": f"Internal error: {exc}",
            })
            self._reset_to_idle()

    async def _run_prompt(self, text: str) -> None:
        """Run a single prompt through the agent."""
        self.state.status = "running"
        self.state.streaming_text = ""
        self._start_spinner()
        self.app.schedule_repaint()

        # Add user message to state
        self.state.messages.append({
            "type": "user",
            "text": text,
        })
        self.app.schedule_repaint()

        try:
            # Handle slash commands
            if text.startswith("/"):
                await self._handle_slash_command(text)
                return

            # Handle shell commands (! prefix from shell mode)
            if text.startswith("!"):
                await self._handle_shell_command(text[1:])
                return

            if not self.state.session_id:
                await self.start_session()

            result = await self.cli.send(
                text,
                provider_id=self.state.provider_id,
                model_id=self.state.model_id,
                agent_name=self.state.agent_name,
            )

            # Update state from result
            final_text = result.get("final_text", "")
            if final_text:
                self.state.messages.append({
                    "type": "assistant",
                    "text": final_text,
                })

            self.state.cost += result.get("cost", 0.0)
            r_tokens = result.get("tokens", {})
            for k in ("input", "output", "reasoning"):
                self.state.tokens[k] = self.state.tokens.get(k, 0) + r_tokens.get(k, 0)
            cache = r_tokens.get("cache", {})
            if cache:
                tc = self.state.tokens.setdefault("cache", {})
                for k in ("read", "write"):
                    tc[k] = tc.get(k, 0) + cache.get(k, 0)

            if result.get("blocked"):
                self.state.messages.append({
                    "type": "error",
                    "text": "Agent run was blocked.",
                })

        except asyncio.CancelledError:
            self.state.messages.append({
                "type": "system",
                "text": "Aborted.",
            })
        except Exception as e:
            self.state.messages.append({
                "type": "error",
                "text": str(e),
            })
        finally:
            self._reset_to_idle()

            # Process queued prompts
            if self.state.queued_prompts:
                next_prompt = self.state.queued_prompts.pop(0)
                self._launch_prompt(next_prompt)

    async def _handle_slash_command(self, text: str) -> None:
        """Handle slash commands within the TUI.

        NOTE: This is called inside _run_prompt's try/finally, so any
        exception here will be caught and state will be reset properly.
        """
        parts = text[1:].split(None, 1)
        cmd = parts[0].lower() if parts else ""
        cmd_args = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit"):
            self.app.quit()
            return

        elif cmd == "new":
            await self.start_session()
            self.state.messages.append({
                "type": "system",
                "text": f"New session: {self.state.session_id}",
            })

        elif cmd == "model":
            model_parts = cmd_args.strip().split("/", 1)
            if len(model_parts) == 2:
                self.state.provider_id, self.state.model_id = model_parts
            elif model_parts[0]:
                self.state.model_id = model_parts[0]
            self.state.messages.append({
                "type": "system",
                "text": f"Model: {self.state.provider_id}/{self.state.model_id}",
            })

        elif cmd == "agent":
            name = cmd_args.strip()
            if name and self.cli.agent_registry.get(name):
                self.state.agent_name = name
                self.state.messages.append({
                    "type": "system",
                    "text": f"Agent: {self.state.agent_name}",
                })
            else:
                self.state.messages.append({
                    "type": "error",
                    "text": f"Unknown agent: {name}",
                })

        elif cmd == "cost":
            self.state.messages.append({
                "type": "system",
                "text": f"Session cost: ${self.state.cost:.4f}",
            })

        elif cmd == "help":
            self.state.messages.append({
                "type": "system",
                "text": (
                    "Commands:\n"
                    "  /new          Start a new session\n"
                    "  /tools        List available tools\n"
                    "  /agents       List available agents\n"
                    "  /skills       List discovered skills\n"
                    "  /commands     List registered commands\n"
                    "  /mcp          List MCP servers and status\n"
                    "  /model [p/]m  Switch model\n"
                    "  /agent <name> Switch agent\n"
                    "  /cost         Show session cost\n"
                    "  /compact      Compact session history\n"
                    "  /help         Show this help\n"
                    "  /quit         Exit\n"
                    "\n"
                    "Ctrl+C aborts current run. PageUp/PageDown to scroll.\n"
                    "Select text with mouse to copy. Type ! to toggle shell mode."
                ),
            })

        elif cmd == "compact":
            from ..session_compaction import SessionCompaction
            compactor = SessionCompaction(
                self.cli.sessions, self.cli.provider_factory, self.cli.events
            )
            await compactor.compact(
                self.state.session_id,
                provider=self.state.provider_id,
                model=self.state.model_id,
            )
            self.state.messages.append({
                "type": "system",
                "text": "Session compacted.",
            })

        elif cmd == "tools":
            tools = self.cli.tool_registry.list()
            text = "\n".join(f"  {t}" for t in tools) if tools else "  No tools registered."
            self.state.messages.append({"type": "system", "text": text})

        elif cmd == "agents":
            agents = self.cli.agent_registry.list_visible()
            lines = []
            for a in agents:
                marker = " *" if a.name == self.state.agent_name else ""
                lines.append(f"  {a.name}: {a.description}{marker}")
            text = "\n".join(lines) if lines else "  No agents."
            self.state.messages.append({"type": "system", "text": text})

        elif cmd == "skills":
            skills = getattr(self.cli, '_discovered_skills', {})
            if skills:
                lines = []
                for name, info in skills.items():
                    desc = info.get('description', '')
                    line = f"  {name}"
                    if desc:
                        line += f": {desc}"
                    lines.append(line)
                text = "\n".join(lines)
            else:
                text = "  No skills discovered."
            self.state.messages.append({"type": "system", "text": text})

        elif cmd == "commands":
            commands = self.cli.command_registry.list()
            lines = [f"  /{c.name}: {c.description}" for c in commands]
            text = "\n".join(lines) if lines else "  No commands."
            self.state.messages.append({"type": "system", "text": text})

        elif cmd == "mcp":
            if self.cli.mcp_manager and self.cli.mcp_manager.status:
                lines = []
                for name, st in self.cli.mcp_manager.status.items():
                    tools = [
                        t for t in self.cli.tool_registry.list()
                        if t.startswith(f"mcp_{name}_")
                    ]
                    tool_count = f"  ({len(tools)} tools)" if tools else ""
                    lines.append(f"  {name}: {st}{tool_count}")
                text = "\n".join(lines)
            else:
                text = "  No MCP servers configured."
            self.state.messages.append({"type": "system", "text": text})

        else:
            # Try as registered command
            rendered = await self.cli.run_command(cmd, cmd_args.split() if cmd_args else [])
            if rendered:
                # Feed rendered prompt back — will be picked up by queued prompts
                self.state.queued_prompts.append(rendered)
            else:
                self.state.messages.append({
                    "type": "error",
                    "text": f"Unknown command: /{cmd}",
                })

    async def _handle_shell_command(self, cmd: str) -> None:
        """Run a shell command and display output in the message list."""
        cmd = cmd.strip()
        if not cmd:
            return

        self.state.messages.append({
            "type": "tool",
            "tool": "shell",
            "call_id": "",
            "status": "running",
            "input": {"command": cmd},
            "output": None,
        })
        self.app.schedule_repaint()

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            output_text = ""
            if stdout:
                output_text += stdout.decode(errors="replace")
            if stderr:
                if output_text:
                    output_text += "\n"
                output_text += stderr.decode(errors="replace")

            # Update tool message status
            for msg in reversed(self.state.messages):
                if msg.get("type") == "tool" and msg.get("tool") == "shell" and msg.get("status") == "running":
                    msg["status"] = "error" if proc.returncode != 0 else "done"
                    msg["output"] = {"text": output_text}
                    break

            if output_text.strip():
                self.state.messages.append({
                    "type": "system",
                    "text": output_text.rstrip(),
                })

            if proc.returncode != 0:
                self.state.messages.append({
                    "type": "error",
                    "text": f"Exit code: {proc.returncode}",
                })

        except Exception as e:
            for msg in reversed(self.state.messages):
                if msg.get("type") == "tool" and msg.get("tool") == "shell" and msg.get("status") == "running":
                    msg["status"] = "error"
                    msg["output"] = {"error": str(e)}
                    break
            self.state.messages.append({
                "type": "error",
                "text": f"Shell error: {e}",
            })

        self.app.schedule_repaint()

    async def respond_permission(self, allow: bool, always: bool = False) -> None:
        """Respond to a pending permission request."""
        perm = self.state.pending_permission
        if not perm:
            return

        request_id = perm["id"]
        try:
            result = await self.cli.permissions.respond(request_id, allow, always=always)

            # Save always rule
            always_rule = result.get("always_rule")
            session_id = result.get("session_id") or self.state.session_id
            if always_rule and session_id:
                try:
                    await self.cli.sessions.add_permission_rule(session_id, always_rule)
                    if self.cli.instance:
                        await self.cli.sessions.add_project_permission_rule(
                            self.cli.instance.project_id, always_rule
                        )
                except Exception:
                    pass
        except Exception as e:
            self.state.messages.append({
                "type": "error",
                "text": f"Permission response failed: {e}",
            })
        finally:
            self.state.pending_permission = None
            if self.state.status == "waiting_permission":
                self.state.status = "running"
            self.app.schedule_repaint()

    def abort(self) -> None:
        """Abort current agent run. Always resets to idle."""
        # Signal the agent to stop
        if self.cli.abort_event:
            self.cli.abort_event.set()

        # Cancel the running task if any
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

        # Reject any pending permission (unblock the future)
        perm = self.state.pending_permission
        if perm:
            perm_id = perm.get("id")
            if perm_id:
                fut = self.cli.permissions._responses.get(perm_id)
                if fut and not fut.done():
                    fut.set_result(False)
            # Clean up permission state
            self.cli.permissions._requests.pop(perm.get("id", ""), None)
            self.cli.permissions._responses.pop(perm.get("id", ""), None)
            self.cli.permissions._session_for_request.pop(perm.get("id", ""), None)

        # Clear queued prompts on abort
        self.state.queued_prompts.clear()

        self.state.messages.append({
            "type": "system",
            "text": "Aborted.",
        })
        self._reset_to_idle()

    def force_reset(self) -> None:
        """Emergency reset — called when user can't interact at all."""
        if self.cli.abort_event:
            self.cli.abort_event.set()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        self.state.queued_prompts.clear()
        self._reset_to_idle()
