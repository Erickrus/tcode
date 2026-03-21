from __future__ import annotations
from typing import Dict, Any, List
from .session import SessionManager
from .event import Event
from .event import EventBus
from .util import next_id
import time

class SessionSummary:
    def __init__(self, sessions: SessionManager, events: EventBus):
        self.sessions = sessions
        self.events = events

    async def summarize(self, session_id: str) -> Dict[str, Any]:
        # gather messages
        msgs = await self.sessions.storage.list_messages(session_id)
        msgs = list(reversed(msgs))
        # simple summary: count messages and list recent files (naive)
        total_msgs = len(msgs)
        recent_text = []
        for m in msgs[-5:]:
            for p in m.get('parts', []):
                if p.get('type') == 'text':
                    recent_text.append(p.get('text'))
        summary_text = ' '.join([t for t in recent_text if t])[:1000]
        summary = {'total_messages': total_msgs, 'recent_snippet': summary_text, 'timestamp': int(time.time())}
        await self.sessions.set_summary(session_id, summary)
        # emit event
        seq = await self.sessions.storage.next_sequence(session_id)
        self.events.publish_fire_and_forget(Event.create('session.summary.completed', {'sessionID': session_id}, session_id=session_id, sequence=seq))
        return summary
