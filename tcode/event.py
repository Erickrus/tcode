from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Callable, Awaitable, Dict, List
import time
import asyncio
import uuid

@dataclass
class Event:
    id: str
    type: str
    timestamp: float
    session_id: Optional[str]
    payload: dict
    sequence: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "payload": self.payload,
            "sequence": self.sequence,
        }

    @staticmethod
    def create(type: str, payload: dict, session_id: Optional[str] = None, sequence: Optional[int] = None) -> "Event":
        return Event(id=str(uuid.uuid4()), type=type, timestamp=time.time(), session_id=session_id, payload=payload, sequence=sequence)


class EventBus:
    def __init__(self):
        self._subscriptions: Dict[str, List[Callable[[Event], Awaitable[None]]]] = {}
        self._wildcard: List[Callable[[Event], Awaitable[None]]] = []
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: str, callback: Callable[[Event], Awaitable[None]]):
        """Subscribe to a specific event type. Returns unsubscribe function."""
        if event_type == "*":
            self._wildcard.append(callback)
            def unsub():
                try:
                    self._wildcard.remove(callback)
                except ValueError:
                    pass
            return unsub

        lst = self._subscriptions.get(event_type)
        if lst is None:
            lst = []
            self._subscriptions[event_type] = lst
        lst.append(callback)

        def unsubscribe():
            try:
                lst.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    def subscribe_all(self, callback: Callable[[Event], Awaitable[None]]):
        return self.subscribe("*", callback)

    def once(self, event_type: str, callback: Callable[[Event], Awaitable[None]]):
        async def wrapper(event: Event):
            try:
                await callback(event)
            finally:
                unsubscribe()

        unsubscribe = self.subscribe(event_type, wrapper)

    async def publish(self, event: Event):
        """Publish event to subscribers. Awaits subscriber coroutines."""
        # Collect subscribers
        subs = list(self._wildcard)
        subs += list(self._subscriptions.get(event.type, []))
        if not subs:
            return
        # Execute subscribers concurrently but don't swallow exceptions
        async def run_sub(scb):
            try:
                await scb(event)
            except Exception:
                # Subscriber errors should not break publisher
                # In production we would log this
                pass

        await asyncio.gather(*(run_sub(s) for s in subs))

    def publish_fire_and_forget(self, event: Event):
        """Schedule event delivery without awaiting subscribers."""
        asyncio.create_task(self.publish(event))
