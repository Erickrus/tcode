"""Project memory system — file-based, human-editable memory for tcode.

Stores timestamped entries in .tcode/MEMORY.md. Only a compact index
(titles + timestamps) is injected into the system prompt; the agent
uses tools to read full details on demand.
"""
from __future__ import annotations

import os
import re
import tempfile
import time as _time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Entry pattern: ## [2026-03-21 14:30] Title text
_ENTRY_RE = re.compile(r"^## \[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] (.+)$", re.MULTILINE)

MEMORY_FILENAME = "MEMORY.md"


def memory_path(base_dir: str) -> str:
    """Return the path to MEMORY.md inside the .tcode directory."""
    return os.path.join(base_dir, MEMORY_FILENAME)


def read_memory(base_dir: str) -> str:
    """Read the memory file. Returns empty string if missing."""
    path = memory_path(base_dir)
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


_REPLACE_MAX_RETRIES = 5
_REPLACE_BASE_DELAY = 0.05  # seconds


def _replace_with_retry(src: str, dst: str) -> None:
    """os.replace with retry loop for transient PermissionError / OSError.

    On Windows (and occasionally other platforms), antivirus scanners, search
    indexers, or concurrent readers can briefly lock the target file, causing
    os.replace() to fail.  Retrying after a short, exponentially increasing
    delay resolves the issue in virtually all cases.
    """
    for attempt in range(_REPLACE_MAX_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_MAX_RETRIES - 1:
                raise
            _time.sleep(_REPLACE_BASE_DELAY * (2 ** attempt))


def write_memory(base_dir: str, content: str) -> None:
    """Atomic write: write to temp file then replace."""
    path = memory_path(base_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        _replace_with_retry(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def parse_entries(content: str) -> List[Dict[str, str]]:
    """Split markdown content into a list of entries.

    Each entry: {timestamp, title, body}.
    """
    if not content.strip():
        return []

    entries: List[Dict[str, str]] = []
    matches = list(_ENTRY_RE.finditer(content))

    for i, m in enumerate(matches):
        timestamp = m.group(1)
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        entries.append({"timestamp": timestamp, "title": title, "body": body})

    return entries


def format_entries(entries: List[Dict[str, str]]) -> str:
    """Rebuild markdown from entry list."""
    if not entries:
        return "# Project Memory\n"

    parts = ["# Project Memory\n"]
    for entry in entries:
        parts.append(f"## [{entry['timestamp']}] {entry['title']}")
        if entry.get("body"):
            parts.append(entry["body"])
        parts.append("")  # blank line between entries
    return "\n".join(parts)


def add_entry(base_dir: str, title: str, body: str) -> Dict[str, str]:
    """Append a new entry with the current timestamp."""
    content = read_memory(base_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    entry = {"timestamp": timestamp, "title": title, "body": body}

    entries = parse_entries(content)
    entries.append(entry)
    write_memory(base_dir, format_entries(entries))
    return entry


def delete_entry(base_dir: str, title_substring: str) -> int:
    """Remove entries matching title substring (case-insensitive). Returns count removed."""
    content = read_memory(base_dir)
    entries = parse_entries(content)
    query = title_substring.lower()
    remaining = [e for e in entries if query not in e["title"].lower()]
    removed = len(entries) - len(remaining)
    if removed > 0:
        write_memory(base_dir, format_entries(remaining))
    return removed


def search_entries(base_dir: str, query: str) -> List[Dict[str, str]]:
    """Return entries whose title or body contains query (case-insensitive)."""
    content = read_memory(base_dir)
    entries = parse_entries(content)
    q = query.lower()
    return [
        e for e in entries
        if q in e["title"].lower() or q in e.get("body", "").lower()
    ]


def memory_for_prompt(base_dir: str) -> str:
    """Return a compact index of memory entries for the system prompt.

    Only titles and timestamps — NOT full content. Wrapped in XML tags.
    Returns empty string if no memory file or no entries.
    """
    content = read_memory(base_dir)
    entries = parse_entries(content)
    if not entries:
        return ""

    lines = ["<project_memory_index>"]
    for entry in entries:
        lines.append(f"- [{entry['timestamp']}] {entry['title']}")
    lines.append("Use memory_read or memory_search tools to access full details.")
    lines.append("</project_memory_index>")
    return "\n".join(lines)


def instructions_for_prompt(instructions: List[str]) -> str:
    """Format config instructions list into XML-tagged block for system prompt."""
    if not instructions:
        return ""
    lines = ["<project_instructions>"]
    for inst in instructions:
        lines.append(f"- {inst}")
    lines.append("</project_instructions>")
    return "\n".join(lines)


async def consolidate_memory(
    base_dir: str,
    provider_factory: Any,
    provider_id: str = "litellm",
    model: str = "",
) -> str:
    """Use a separate LLM call to consolidate/prune memory entries.

    Runs outside the main conversation context. Returns the new content.
    """
    content = read_memory(base_dir)
    entries = parse_entries(content)
    if len(entries) < 2:
        return content  # nothing to consolidate

    prompt = (
        "You are a memory consolidation assistant. Below are timestamped memory entries "
        "for a software project. Merge related entries, remove duplicates, and prune "
        "outdated information. Keep the same markdown format:\n"
        "## [YYYY-MM-DD HH:MM] Title\n"
        "- bullet points\n\n"
        "Preserve all important facts. Use the most recent timestamp when merging.\n\n"
        "Current memory:\n\n"
        f"{content}\n\n"
        "Output ONLY the consolidated markdown (starting with '# Project Memory')."
    )

    try:
        adapter = provider_factory.get(provider_id)
        messages = [{"role": "user", "content": prompt}]
        response = await adapter.chat(messages, model, {})

        # Extract text from response
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                c0 = choices[0]
                new_content = (
                    c0.get("text")
                    or (c0.get("message") and c0["message"].get("content"))
                    or ""
                )
            else:
                new_content = response.get("text", "")
        else:
            new_content = str(response)

        new_content = new_content.strip()
        if not new_content or "# Project Memory" not in new_content:
            return content  # safety: don't overwrite with garbage

        # Validate we can parse it
        new_entries = parse_entries(new_content)
        if not new_entries:
            return content

        write_memory(base_dir, new_content)
        return new_content

    except Exception:
        return content  # on error, leave memory unchanged
