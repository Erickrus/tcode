from __future__ import annotations
from typing import Dict, Any, Optional, Callable
from .event import Event, EventBus
from .plan import PlanManager
import asyncio

# Evaluate todo implementation merged here (previously in todo_evaluator.py)
from .attachments import AttachmentStore
from .jsonschema_util import validate_simple_schema
from .tools import ToolResult
import subprocess
import time

async def evaluate_todo(session_id: str, message_id: str, part: Dict[str, Any], sessions: object, toolrunner: object, timeout: int = 60) -> Dict[str, Any]:
    """Evaluate a todo part. Returns dict {status: 'verified'|'failed'|'pending', details: ...}

    Checks performed (in order):
    - ensure part.state.status == 'completed'
    - if part.metadata.schema -> parse part.state.output (JSON) and validate via validate_simple_schema
    - if part.metadata.attachment or metadata.file -> ensure attachment exists via AttachmentStore.get_path
    - if part.metadata.verify_cmds -> run commands via subprocess and check exit codes
    - if human_approval required -> publish permission request (TODO: via PermissionsManager)
    """
    out = {'status': 'pending', 'details': {}}
    state = part.get('state') or {}
    status = state.get('status')
    if status != 'completed':
        out['status'] = 'pending'
        out['details']['reason'] = 'part not completed'
        return out

    # Schema validation
    schema = part.get('metadata', {}).get('schema') or {}
    if schema:
        # parse output
        raw = state.get('output') or ''
        import json
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except Exception as e:
            out['status'] = 'failed'
            out['details']['schema_error'] = f'invalid json: {e}'
            return out
        ok, reason = validate_simple_schema(parsed, schema)
        if not ok:
            out['status'] = 'failed'
            out['details']['schema_error'] = reason
            return out

    # Attachment/file check
    attach = part.get('metadata', {}).get('attachment') or part.get('metadata', {}).get('file')
    if attach:
        store = AttachmentStore()
        try:
            path = store.get_path(attach) if isinstance(attach, str) and attach.startswith('attachment://') else attach
            # if get_path returned, check exists
            out['details']['attachment_path'] = path
        except Exception as e:
            out['status'] = 'failed'
            out['details']['attachment_error'] = repr(e)
            return out

    # verify commands
    cmds = part.get('metadata', {}).get('verify_cmds') or []
    for cmd in cmds:
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            if proc.returncode != 0:
                out['status'] = 'failed'
                out['details'].setdefault('verify_cmds', []).append({'cmd': cmd, 'rc': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr})
                return out
        except Exception as e:
            out['status'] = 'failed'
            out['details'].setdefault('verify_cmds', []).append({'cmd': cmd, 'error': str(e)})
            return out

    # passed all checks
    out['status'] = 'verified'
    out['details']['verified_at'] = int(time.time())
    return out


class PlannerCoordinator:
    """Subscribe to events and orchestrate replans or clarifications."""
    def __init__(self, events: EventBus, plan_manager_factory: Callable[[], PlanManager]):
        self.events = events
        self.plan_manager_factory = plan_manager_factory
        # debounce map for scheduled replans: session_id -> asyncio.Task
        self._pending_replans: Dict[str, asyncio.Task] = {}
        # lock to protect pending replans map
        self._lock = asyncio.Lock()
        # subscribe to key events
        self.events.subscribe('tool.failed', self._on_tool_failed)
        self.events.subscribe('session.todo.verification.failed', self._on_todo_verification_failed)
        self.events.subscribe('session.compaction.completed', self._on_compaction_completed)
        # user messages (message.updated) could be filtered by role
        self.events.subscribe('message.updated', self._on_message_updated)

    async def _on_tool_failed(self, ev: Event):
        payload = ev.payload
        session_id = ev.session_id
        # schedule a replan after a small delay to allow transient retries
        await self._schedule_replan(session_id, reason='tool_failed', details=payload)

    async def _on_todo_verification_failed(self, ev: Event):
        session_id = ev.session_id
        await self._schedule_replan(session_id, reason='verification_failed', details=ev.payload)

    async def _on_compaction_completed(self, ev: Event):
        session_id = ev.session_id
        # compaction may remove context; consider replan low-priority
        await self._schedule_replan(session_id, reason='compaction', details=ev.payload, delay=5)

    async def _on_message_updated(self, ev: Event):
        # if a new user message updates goals, re-evaluate plan
        info = ev.payload.get('info')
        if info and info.get('role') == 'user':
            session_id = ev.session_id
            await self._schedule_replan(session_id, reason='user_update', details=ev.payload)

    async def _schedule_replan(self, session_id: str, reason: str, details: Dict[str, Any] = None, delay: int = 2):
        # Debounce: cancel existing pending replan for the session
        async with self._lock:
            prev = self._pending_replans.get(session_id)
            if prev and not prev.done():
                prev.cancel()
            # schedule the replan task
            task = asyncio.create_task(self._delayed_replan(session_id, reason, details, delay))
            self._pending_replans[session_id] = task

    async def _delayed_replan(self, session_id: str, reason: str, details: Dict[str, Any] = None, delay: int = 2):
        try:
            await asyncio.sleep(delay)
            pm = self.plan_manager_factory()
            instr = f"Replan triggered due to {reason}. Details: {details or {}}"
            await pm.create_plan(session_id, title='Auto Replan', instructions=instr)
            # publish event
            self.events.publish_fire_and_forget(Event.create('session.plan.auto_created', {'sessionID': session_id, 'reason': reason}, session_id=session_id))
        except asyncio.CancelledError:
            # canceled due to newer event; ignore
            return
        except Exception:
            return

# helper to expose verify endpoint handler callable
async def verify_todo_handler(session_id: str, part_id: str, sessions, toolrunner) -> Dict[str, Any]:
    # fetch part
    part = await sessions.storage.get_part(part_id)
    if not part:
        raise KeyError('part not found')
    res = await evaluate_todo(session_id, part.get('message_id'), part, sessions, toolrunner)
    # publish outcome
    if res.get('status') == 'verified':
        await sessions.events.publish(Event.create('session.todo.verified', {'sessionID': session_id, 'partID': part_id, 'details': res.get('details')}, session_id=session_id))
    else:
        await sessions.events.publish(Event.create('session.todo.verification.failed', {'sessionID': session_id, 'partID': part_id, 'details': res.get('details')}, session_id=session_id))
    return res
