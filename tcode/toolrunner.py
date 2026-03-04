from __future__ import annotations
from typing import Dict, Any, Optional
from .tools import ToolRegistry, ToolContext, ToolResult
from .event import Event
from .event import EventBus
from .permission_next import PermissionDeniedError, PermissionRejectedError
import asyncio
import time

class ToolRunner:
    def __init__(self, registry: ToolRegistry, events: EventBus, sessions: object, permissions: object = None):
        """sessions should be a SessionManager instance."""
        self.registry = registry
        self.events = events
        self.sessions = sessions
        self.permissions = permissions

    async def execute_tool(self, session_id: str, message_id: str, call_id: str, tool_id: str, args: Dict[str, Any], timeout: Optional[int] = None, part_id: Optional[str] = None) -> ToolResult:
        tool = self.registry.get(tool_id)
        if not tool:
            raise KeyError("tool not found")

        # Ensure there's a part to update; create if not supplied
        if part_id is None and session_id and message_id:
            try:
                part_id = await self.sessions.insert_tool_part(session_id, message_id, call_id, tool_id, args)
            except Exception:
                part_id = None

        # Merge permission rulesets: project rules → session rules (last wins)
        extra = {"permissions": getattr(self, 'permissions', None), "sessions": self.sessions, "verbose": getattr(self, 'verbose', False)}
        try:
            from .permission_next import merge_rulesets
            from .instance import Instance
            # Session-level rules (includes "always" approved patterns)
            session_rules = await self.sessions.get_permission(session_id)
            # Project-level rules (persist across sessions)
            project_rules = []
            inst = Instance.current()
            if inst:
                project_rules = await self.sessions.get_project_permission(inst.project_id)
            # Project first, session last → session overrides project
            merged = merge_rulesets(project_rules, session_rules)
            if merged:
                extra['rules'] = merged
        except Exception:
            pass
        ctx = ToolContext(session_id=session_id, message_id=message_id, call_id=call_id, extra=extra)

        # publish tool.started
        seq = await self.sessions.storage.next_sequence(session_id) if session_id else None
        ev = Event.create("tool.started", {"callID": call_id, "tool": tool_id, "sessionID": session_id, "messageID": message_id, "input": args}, session_id=session_id, sequence=seq)
        await self.events.publish(ev)

        # update session part to running state if available
        if part_id and session_id:
            try:
                await self.sessions.update_part_state(session_id, message_id, part_id, {"status": "running", "input": args, "time": {"start": int(time.time())}})
            except Exception:
                pass

        # Consult session-level permissions via ctx.ask (permission engine).
        try:
            perm_details = args if isinstance(args, dict) else {}
            allowed = await ctx.ask(tool_id, perm_details)
            if not allowed:
                if part_id and session_id:
                    await self.sessions.update_part_state(session_id, message_id, part_id, {"status": "error", "error": "Permission denied"})
                raise PermissionDeniedError(tool_id)
        except PermissionDeniedError:
            if part_id and session_id:
                try:
                    await self.sessions.update_part_state(session_id, message_id, part_id, {"status": "error", "error": "Permission denied by rule"})
                except Exception:
                    pass
            raise
        except PermissionRejectedError:
            if part_id and session_id:
                try:
                    await self.sessions.update_part_state(session_id, message_id, part_id, {"status": "error", "error": "Permission rejected by user"})
                except Exception:
                    pass
            raise
        except (PermissionError, Exception) as e:
            # Other permission-related errors default to deny
            if part_id and session_id:
                try:
                    await self.sessions.update_part_state(session_id, message_id, part_id, {"status": "error", "error": str(e)})
                except Exception:
                    pass
            raise PermissionDeniedError(tool_id)

        # Enforce plan-mode write restrictions: if this is a write_file tool and session is plan_active, only allow writing to plan file
        if tool_id in ("builtin_write_file", "builtin_shell") and session_id:
            try:
                sess = await self.sessions.get_session(session_id)
                meta = sess.get('metadata', {}) or {}
                if meta.get('plan_active'):
                    plan_file = meta.get('plan_file')
                    target = args.get('path') if isinstance(args, dict) else None
                    if target and plan_file:
                        # normalize paths to prevent bypass via relative paths
                        import os
                        try:
                            abs_plan = os.path.abspath(plan_file)
                            abs_target = os.path.abspath(target)
                            # allow only if the target exactly equals the plan file (strict enforcement)
                            allowed = (abs_plan == abs_target)
                            if not allowed:
                                await self.sessions.update_part_state(session_id, message_id, part_id, {"status": "error", "error": "Write disallowed in plan mode"})
                                raise PermissionError("Write disallowed in plan mode")
                        except Exception:
                            await self.sessions.update_part_state(session_id, message_id, part_id, {"status": "error", "error": "Write disallowed in plan mode"})
                            raise PermissionError("Write disallowed in plan mode")
            except Exception:
                # if permission denied, raise to caller
                raise

        # Execute tool
        coro = tool.execute(args, ctx)
        try:
            if timeout:
                result = await asyncio.wait_for(coro, timeout)
            else:
                result = await coro
        except Exception as e:
            err = {"error": str(e)}
            seq = await self.sessions.storage.next_sequence(session_id) if session_id else None
            ev2 = Event.create("tool.failed", {"callID": call_id, "tool": tool_id, "sessionID": session_id, "messageID": message_id, "error": err}, session_id=session_id, sequence=seq)
            await self.events.publish(ev2)
            # update part to error
            if part_id and session_id:
                try:
                    await self.sessions.update_part_state(session_id, message_id, part_id, {"status": "error", "error": str(e), "time": {"end": int(time.time())}})
                except Exception:
                    pass
            raise

        # Convert result to dict if pydantic
        try:
            res_obj = result.model_dump() if hasattr(result, 'model_dump') else (result if isinstance(result, dict) else {"output": str(result)})
        except Exception:
            res_obj = {"output": str(result)}

        seq = await self.sessions.storage.next_sequence(session_id) if session_id else None
        ev3 = Event.create("tool.completed", {"callID": call_id, "tool": tool_id, "sessionID": session_id, "messageID": message_id, "result": res_obj}, session_id=session_id, sequence=seq)
        await self.events.publish(ev3)

        # update part to completed in session
        if part_id and session_id:
            try:
                completed_state = {
                    "status": "completed",
                    "input": args,
                    "output": res_obj.get('output') if isinstance(res_obj, dict) else str(res_obj),
                    "title": res_obj.get('title') if isinstance(res_obj, dict) else None,
                    "metadata": res_obj.get('metadata') if isinstance(res_obj, dict) else {},
                    "time": {"end": int(time.time())},
                }
                await self.sessions.update_part_state(session_id, message_id, part_id, completed_state)
            except Exception:
                pass

        # Evaluate todo automatically after completion
        try:
            from .todos import evaluate_todo
            part = await self.sessions.storage.get_part(part_id)
            eval_res = await evaluate_todo(session_id, message_id, part, self.sessions, self)
            # publish verification outcome as event
            seq2 = await self.sessions.storage.next_sequence(session_id)
            if eval_res.get('status') == 'verified':
                self.events.publish_fire_and_forget(Event.create('session.todo.verified', {'sessionID': session_id, 'partID': part_id, 'details': eval_res.get('details')}, session_id=session_id, sequence=seq2))
            elif eval_res.get('status') == 'failed':
                self.events.publish_fire_and_forget(Event.create('session.todo.verification.failed', {'sessionID': session_id, 'partID': part_id, 'details': eval_res.get('details')}, session_id=session_id, sequence=seq2))
        except Exception:
            pass

        return result

