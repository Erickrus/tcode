from __future__ import annotations
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Callable, Awaitable

import warnings

warnings.filterwarnings(
    "ignore",
    message=r'Field name "schema" in .* shadows an attribute in parent "BaseModel"',
    category=UserWarning
)

from pydantic import ConfigDict

class ToolContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    message_id: str
    agent: Optional[str] = None
    abort_token: Optional[Any] = None
    call_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    messages: Optional[List[Dict[str, Any]]] = None

    # permission helper will be injected at runtime (PermissionsManager)
    async def ask(self, permission_type: str, details: Dict[str, Any]) -> bool:
        """Ask for permission. Evaluates rules first, falls back to PermissionsManager.

        Raises PermissionDeniedError or PermissionRejectedError on denial/rejection.
        Returns True if allowed, False if denied (backward-compatible mode).
        """
        mgr = None
        try:
            if self.extra and isinstance(self.extra, dict):
                mgr = self.extra.get('permissions')
        except Exception:
            mgr = None
        # support passing a ruleset in extra to auto-evaluate
        ruleset = None
        try:
            if self.extra and isinstance(self.extra, dict):
                ruleset = self.extra.get('rules')
        except Exception:
            ruleset = None
        # Use permission_next ask_or_raise to propagate typed errors
        from .permission_next import ask_or_raise
        return await ask_or_raise(mgr, {'permission': permission_type, 'metadata': details}, ruleset=ruleset, session_id=self.session_id)

class ToolResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    output: Optional[Any] = None
    attachments: Optional[List[Dict[str, Any]]] = None

class ToolInfo:
    def __init__(self, id: str, description: str, parameters: BaseModel, execute: Callable[[Dict[str, Any], ToolContext], Awaitable[ToolResult]], permission: str = "low", streaming: bool = False):
        self.id = id
        self.description = description
        self.parameters = parameters
        self._execute = execute
        self.permission = permission
        self.streaming = streaming

    async def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        # validate args via pydantic if parameters is model
        if isinstance(self.parameters, type) and issubclass(self.parameters, BaseModel):
            params = self.parameters(**args)
            args = params.model_dump()
        return await self._execute(args, ctx)

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolInfo] = {}

    def register(self, tool: ToolInfo):
        self._tools[tool.id] = tool

    def get(self, id: str) -> Optional[ToolInfo]:
        return self._tools.get(id)

    def list(self) -> List[str]:
        return list(self._tools.keys())
