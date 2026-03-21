from __future__ import annotations
from typing import Dict, Any, List, Optional
import asyncio


class PermissionRequest:
    def __init__(self, id: str, type: str, details: Dict[str, Any]):
        self.id = id
        self.type = type
        self.details = details


class PermissionsManager:
    def __init__(self, events: Any = None):
        self._requests: Dict[str, PermissionRequest] = {}
        self._responses: Dict[str, asyncio.Future] = {}
        self._session_for_request: Dict[str, str] = {}  # request_id -> session_id
        # optional EventBus to emit permission events
        self.events = events

    async def request(self, session_id: str, permission: PermissionRequest) -> bool:
        fut = asyncio.get_event_loop().create_future()
        self._requests[permission.id] = permission
        self._responses[permission.id] = fut
        self._session_for_request[permission.id] = session_id
        # publish event to EventBus if available so external agents/clients can react
        try:
            if self.events:
                from .event import Event
                self.events.publish_fire_and_forget(Event.create(
                    'permission.requested',
                    {'id': permission.id, 'session_id': session_id,
                     'type': permission.type, 'details': permission.details},
                    session_id=session_id,
                ))
        except Exception:
            pass
        return await fut

    async def respond(self, request_id: str, allow: bool, always: bool = False) -> Dict[str, Any]:
        """Respond to a permission request.

        Args:
            request_id: The permission request ID
            allow: Whether to allow this request
            always: If True and allow is True, add a permanent "always allow" rule
                    for this permission type in the session

        Returns a dict with 'ok', 'always_rule' (if applicable), and 'session_id'.
        """
        fut = self._responses.get(request_id)
        req = self._requests.get(request_id)

        if fut and not fut.done():
            fut.set_result(bool(allow))

        # Handle "always" approval — capture before cleanup
        always_rule = None
        if always and allow and req:
            always_rule = {
                "permission": req.type,
                "action": "allow",
                "pattern": "*",
            }

        # Capture session_id before cleanup
        session_id = self._session_for_request.get(request_id)

        # publish responded event
        try:
            if self.events:
                from .event import Event
                self.events.publish_fire_and_forget(Event.create(
                    'permission.responded',
                    {'id': request_id, 'allow': bool(allow), 'always': always,
                     'always_rule': always_rule},
                    session_id=session_id,
                ))
        except Exception:
            pass

        # Cleanup
        self._requests.pop(request_id, None)
        self._responses.pop(request_id, None)
        self._session_for_request.pop(request_id, None)

        return {"ok": True, "always_rule": always_rule, "session_id": session_id}

    def get_pending_request(self, request_id: str) -> Optional[PermissionRequest]:
        """Get a pending permission request by ID."""
        return self._requests.get(request_id)

    def get_session_for_request(self, request_id: str) -> Optional[str]:
        """Get the session ID for a pending request."""
        return self._session_for_request.get(request_id)
