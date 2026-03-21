from __future__ import annotations
from typing import Dict, Any, Optional
from .agent import AgentRunner
from .session import SessionManager
from .providers.factory import ProviderFactory
from .event import EventBus
from .attachments import AttachmentStore

# A simple wrapper that runs a scoped agent for exploration or design.
# For MVP we will run the AgentRunner.run with the provided messages and return the final assistant part.

class SubagentRunner:
    def __init__(self, providers: ProviderFactory, tools: object, mcp: object, sessions: SessionManager, events: EventBus, attachments: AttachmentStore):
        self.providers = providers
        self.tools = tools
        self.mcp = mcp
        self.sessions = sessions
        self.events = events
        self.attachments = attachments
        # ensure a real ToolRegistry and MCPManager are passed to AgentRunner
        self.agent_runner = AgentRunner(providers, tools or __import__('tcode').tcode.tools.ToolRegistry(), mcp or __import__('tcode').tcode.mcp.MCPManager(), sessions, events, attachments or AttachmentStore())

    async def run_explore(self, provider_id: str, model: str, session_id: str, instructions: str) -> Dict[str, Any]:
        # create a temporary message to run the agent against
        message_id = await self.sessions.create_message(session_id, 'user')
        await self.sessions.append_text_part(session_id, message_id, instructions)
        # create assistant message to collect
        await self.sessions.create_message(session_id, 'assistant')
        # run agent runner on this message
        res = await self.agent_runner.run(provider_id, model, session_id, message_id, stream=False)
        return res
