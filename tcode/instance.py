"""Project/instance scoping for tcode.

Provides per-project isolation using contextvars, matching opencode's
Instance.provide() / Instance.state() pattern.

Usage:
    instance = Instance(directory="/path/to/project")
    async with instance:
        # All operations scoped to this project
        print(Instance.current().directory)
"""
from __future__ import annotations
import os
import hashlib
import contextvars
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field

# Context variable holding the current Instance
_current_instance: contextvars.ContextVar[Optional[Instance]] = contextvars.ContextVar(
    "tcode_instance", default=None
)

# Global cache of instances by directory
_instance_cache: Dict[str, Instance] = {}


@dataclass
class Instance:
    """Represents a project instance scoped to a directory."""

    directory: str
    worktree: str = ""
    project_id: str = ""
    _state: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.directory = os.path.abspath(self.directory)
        if not self.worktree:
            self.worktree = self._find_worktree(self.directory)
        if not self.project_id:
            self.project_id = self._make_project_id(self.worktree or self.directory)

    @staticmethod
    def _find_worktree(directory: str) -> str:
        """Walk up to find a VCS root (git repository)."""
        d = directory
        while True:
            if os.path.isdir(os.path.join(d, ".git")):
                return d
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        return directory

    @staticmethod
    def _make_project_id(worktree: str) -> str:
        """Create a stable project ID from the worktree path."""
        return hashlib.sha256(worktree.encode()).hexdigest()[:16]

    # ---- Context management ----

    async def __aenter__(self):
        _current_instance.set(self)
        _instance_cache[self.directory] = self
        return self

    async def __aexit__(self, exc_type, exc, tb):
        _current_instance.set(None)

    @staticmethod
    def current() -> Optional[Instance]:
        """Get the current instance from context."""
        return _current_instance.get()

    @staticmethod
    def require() -> Instance:
        """Get the current instance or raise."""
        inst = _current_instance.get()
        if inst is None:
            raise RuntimeError("No active Instance — use 'async with Instance(dir)' first")
        return inst

    @staticmethod
    def get_or_create(directory: str) -> Instance:
        """Get a cached instance or create a new one.  Also sets it as the current instance."""
        directory = os.path.abspath(directory)
        if directory in _instance_cache:
            inst = _instance_cache[directory]
        else:
            inst = Instance(directory=directory)
            _instance_cache[directory] = inst
        _current_instance.set(inst)
        return inst

    # ---- State bag (per-instance key-value store) ----

    def state(self, key: str, default: Any = None) -> Any:
        """Get per-instance state value."""
        return self._state.get(key, default)

    def set_state(self, key: str, value: Any):
        """Set per-instance state value."""
        self._state[key] = value

    # ---- Path helpers ----

    def contains(self, path: str) -> bool:
        """Check if a path is within this instance's worktree."""
        try:
            abs_path = os.path.abspath(path)
            return abs_path.startswith(self.worktree + os.sep) or abs_path == self.worktree
        except Exception:
            return False

    @property
    def data_dir(self) -> str:
        """Per-project data directory (~/.local/share/tcode/<project_id>/)."""
        base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
        return os.path.join(base, "tcode", self.project_id)

    @property
    def config_dir(self) -> str:
        """Per-project config directory."""
        return os.path.join(self.directory, ".tcode")


def dispose_all():
    """Clean up all cached instances."""
    _instance_cache.clear()
    _current_instance.set(None)
