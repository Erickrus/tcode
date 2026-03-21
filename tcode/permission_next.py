from __future__ import annotations
from typing import List, Dict, Any, Optional
from .permissions import PermissionRequest, PermissionsManager
import asyncio
import time
from .util import next_id

# ---- Permission error hierarchy ----

class PermissionDeniedError(Exception):
    """A rule explicitly denies this permission."""
    def __init__(self, permission: str = "", pattern: str = ""):
        self.permission = permission
        self.pattern = pattern
        super().__init__(f"Permission denied: {permission}" + (f" (pattern: {pattern})" if pattern else ""))


class PermissionRejectedError(Exception):
    """User interactively rejected this permission request."""
    def __init__(self, permission: str = "", request_id: str = ""):
        self.permission = permission
        self.request_id = request_id
        super().__init__(f"Permission rejected by user: {permission}")


# Default timeout for interactive permission requests (seconds)
DEFAULT_PERMISSION_TIMEOUT = 120.0

# A minimal PermissionNext rule engine for tcode MVP.
# Rules are simple dicts: {"permission": "tool.id" or "*", "action": "allow"|"deny", "pattern": "*"}

def merge_rulesets(*rulesets: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge multiple rulesets. Later rulesets override earlier ones via later-rule precedence during evaluation."""
    merged: List[Dict[str, Any]] = []
    for rs in rulesets:
        if not rs:
            continue
        for r in rs:
            merged.append(r)
    return merged

def evaluate_rules(ruleset: List[Dict[str, Any]], permission: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Evaluate rules for a permission. Returns 'allow', 'deny', or 'ask'.
    Last matching rule wins (we iterate reversed). Permission can match by exact or wildcard '*'.
    Supports optional 'pattern' key in rules to match against metadata (e.g., file paths).
    """
    if not ruleset:
        return 'ask'
    import fnmatch
    for r in reversed(ruleset):
        try:
            perm = r.get('permission')
            action = r.get('action')
            pattern = r.get('pattern')
            # permission match (exact or wildcard)
            if perm != '*' and perm != permission:
                continue
            # if pattern present, try to match against metadata path-like fields
            if pattern and pattern != '*':
                if not metadata:
                    continue
                # look for likely fields to match
                target = metadata.get('path') or metadata.get('target') or metadata.get('filename') or metadata.get('file') or ''
                if not target:
                    continue
                # use fnmatch for wildcard support
                if not fnmatch.fnmatch(target, pattern):
                    continue
            if action in ('allow', 'deny'):
                return action
        except Exception:
            continue
    return 'ask'

async def ask_permission(pm: PermissionsManager, req: Dict[str, Any], ruleset: Optional[List[Dict[str, Any]]] = None, session_id: Optional[str] = None, timeout: Optional[float] = None) -> bool:
    """High-level ask that consults ruleset and falls back to interactive PermissionsManager if needed.

    req is expected shape: {permission: str, metadata?: dict, patterns?: list, always?: list}

    Raises:
        PermissionDeniedError: if a rule explicitly denies
        PermissionRejectedError: if user interactively rejects
    """
    permission = req.get('permission') if isinstance(req, dict) else str(req)
    if not permission:
        return False

    # evaluate ruleset if present
    if ruleset:
        metadata = req.get('metadata') if isinstance(req, dict) else None
        action = evaluate_rules(ruleset, permission, metadata=metadata)
        if action == 'allow':
            return True
        if action == 'deny':
            raise PermissionDeniedError(permission)
    # otherwise, fall back to using PermissionsManager to ask the user
    pid = next_id('perm')
    pr = PermissionRequest(pid, permission, req.get('metadata') or {})
    # ask via PermissionsManager with timeout
    if timeout is None:
        timeout = DEFAULT_PERMISSION_TIMEOUT
    coro = pm.request(session_id or '', pr)
    try:
        allowed = await asyncio.wait_for(coro, timeout)
    except asyncio.TimeoutError:
        # timeout denies by default
        raise PermissionDeniedError(permission)
    if not allowed:
        raise PermissionRejectedError(permission, request_id=pid)
    return True


# convenience wrapper that accepts either a PermissionsManager instance or None
async def ask(pm: Optional[PermissionsManager], req: Dict[str, Any], ruleset: Optional[List[Dict[str, Any]]] = None, session_id: Optional[str] = None, timeout: Optional[float] = None) -> bool:
    """Evaluate permissions. Returns True if allowed, False if denied.

    For backward compatibility, catches PermissionDeniedError/PermissionRejectedError and returns False.
    Use ask_or_raise() for the typed-error version.
    """
    # If we have rules, evaluate them first even without a PermissionsManager
    if ruleset:
        permission = req.get('permission') if isinstance(req, dict) else str(req)
        metadata = req.get('metadata') if isinstance(req, dict) else None
        action = evaluate_rules(ruleset, permission, metadata=metadata)
        if action == 'allow':
            return True
        if action == 'deny':
            return False
    # Fall back to PermissionsManager for "ask" actions
    if pm is None:
        return False
    try:
        return await ask_permission(pm, req, ruleset=ruleset, session_id=session_id, timeout=timeout)
    except (PermissionDeniedError, PermissionRejectedError):
        return False


async def ask_or_raise(pm: Optional[PermissionsManager], req: Dict[str, Any],
                       ruleset: Optional[List[Dict[str, Any]]] = None,
                       session_id: Optional[str] = None,
                       timeout: Optional[float] = None) -> bool:
    """Like ask() but raises PermissionDeniedError/PermissionRejectedError instead of returning False."""
    if ruleset:
        permission = req.get('permission') if isinstance(req, dict) else str(req)
        metadata = req.get('metadata') if isinstance(req, dict) else None
        action = evaluate_rules(ruleset, permission, metadata=metadata)
        if action == 'allow':
            return True
        if action == 'deny':
            raise PermissionDeniedError(permission)
    if pm is None:
        raise PermissionDeniedError(req.get('permission', '') if isinstance(req, dict) else str(req))
    return await ask_permission(pm, req, ruleset=ruleset, session_id=session_id, timeout=timeout)


# ---- "Always" approval ----

def add_always_rule(ruleset: List[Dict[str, Any]], permission: str, pattern: str = "*") -> List[Dict[str, Any]]:
    """Add an 'always allow' rule for a specific permission+pattern.

    Returns a new ruleset with the rule appended (last rule wins).
    """
    new_rule = {"permission": permission, "action": "allow", "pattern": pattern}
    return ruleset + [new_rule]
