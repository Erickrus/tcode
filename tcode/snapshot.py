"""Snapshot/patch tracking for tcode.

Records git state before/after agent steps so changes can be diffed and reverted.
Follows opencode snapshot/index.ts pattern.
"""
from __future__ import annotations
import os
import subprocess
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class FileDiff:
    """Diff for a single file between two snapshots."""
    file: str
    before: str = ""
    after: str = ""
    additions: int = 0
    deletions: int = 0
    status: str = "modified"  # "added" | "deleted" | "modified"


@dataclass
class Patch:
    """Result of comparing two snapshots."""
    hash: str
    files: List[str] = field(default_factory=list)


class Snapshot:
    """Git-based snapshot tracking for a project directory."""

    def __init__(self, worktree: str):
        self.worktree = worktree
        self._git_dir: Optional[str] = None

    def _run_git(self, *args: str, check: bool = True) -> str:
        """Run a git command in the worktree."""
        cmd = ["git", "-C", self.worktree] + list(args)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=30, check=check,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return ""

    def _is_git_repo(self) -> bool:
        """Check if worktree is a git repository."""
        return os.path.isdir(os.path.join(self.worktree, ".git"))

    async def track(self) -> str:
        """Capture current tree hash. Returns hash string."""
        if not self._is_git_repo():
            return ""
        # Use git write-tree to capture index state, or rev-parse HEAD for committed state
        def _track():
            # Add all to index (staging area) to capture working state
            self._run_git("add", "-A", "--intent-to-add", check=False)
            # Get tree hash from current working tree state
            tree_hash = self._run_git("stash", "create", check=False)
            if not tree_hash:
                # No changes — use HEAD
                tree_hash = self._run_git("rev-parse", "HEAD", check=False)
            return tree_hash or ""
        return await asyncio.to_thread(_track)

    async def patch(self, from_hash: str) -> Patch:
        """Get changed files between from_hash and current state."""
        if not from_hash or not self._is_git_repo():
            return Patch(hash="", files=[])

        def _patch():
            # Get current state
            current = self._run_git("rev-parse", "HEAD", check=False)
            # Diff between from_hash and working tree
            diff_output = self._run_git(
                "diff", "--name-only", from_hash, check=False,
            )
            files = [f for f in diff_output.split("\n") if f.strip()]
            # Also check untracked files
            untracked = self._run_git(
                "ls-files", "--others", "--exclude-standard", check=False,
            )
            for f in untracked.split("\n"):
                if f.strip() and f.strip() not in files:
                    files.append(f.strip())
            return Patch(hash=current or from_hash, files=files)

        return await asyncio.to_thread(_patch)

    async def diff(self, from_hash: str) -> str:
        """Get unified diff between from_hash and current state."""
        if not from_hash or not self._is_git_repo():
            return ""

        def _diff():
            return self._run_git("diff", from_hash, check=False)

        return await asyncio.to_thread(_diff)

    async def diff_full(self, from_hash: str, to_hash: Optional[str] = None) -> List[FileDiff]:
        """Get detailed per-file diffs between two hashes."""
        if not from_hash or not self._is_git_repo():
            return []

        def _diff_full():
            args = ["diff", "--numstat", from_hash]
            if to_hash:
                args.append(to_hash)
            stat_output = self._run_git(*args, check=False)
            diffs = []
            for line in stat_output.split("\n"):
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    adds = int(parts[0]) if parts[0] != "-" else 0
                    dels = int(parts[1]) if parts[1] != "-" else 0
                    fname = parts[2]
                    status = "modified"
                    if adds > 0 and dels == 0:
                        status = "added"
                    elif adds == 0 and dels > 0:
                        status = "deleted"
                    diffs.append(FileDiff(
                        file=fname, additions=adds, deletions=dels, status=status,
                    ))
            return diffs

        return await asyncio.to_thread(_diff_full)

    async def restore(self, snapshot_hash: str) -> bool:
        """Restore working tree to a previous snapshot."""
        if not snapshot_hash or not self._is_git_repo():
            return False

        def _restore():
            result = self._run_git("checkout", snapshot_hash, "--", ".", check=False)
            return bool(result is not None)

        return await asyncio.to_thread(_restore)

    async def revert(self, patches: List[Patch]) -> bool:
        """Revert specific patches by restoring files to their pre-patch state."""
        if not patches or not self._is_git_repo():
            return False

        def _revert():
            for p in patches:
                if p.hash and p.files:
                    for f in p.files:
                        self._run_git("checkout", p.hash, "--", f, check=False)
            return True

        return await asyncio.to_thread(_revert)
