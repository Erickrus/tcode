"""MessageBlock — render helper for a single message."""
from __future__ import annotations

import os
from difflib import unified_diff
from typing import Any

from runtui import Color, Attrs
from runtui.rendering.painter import Painter

_BG = Color.from_rgb(22, 22, 30)

# Diff colors — muted to fit the dark theme
_DIFF_DEL_BG = Color.from_rgb(60, 20, 20)     # dark red background
_DIFF_DEL_FG = Color.from_rgb(240, 140, 140)   # light red text
_DIFF_ADD_BG = Color.from_rgb(20, 50, 20)      # dark green background
_DIFF_ADD_FG = Color.from_rgb(140, 230, 140)   # light green text
_DIFF_HUNK_FG = Color.from_rgb(130, 160, 210)  # blue for @@ lines
_DIFF_LINE_NO_FG = Color.from_rgb(100, 100, 120)  # dim for line numbers
_DIFF_CONTEXT_FG = Color.from_rgb(160, 165, 180)  # normal text for context
_DIFF_FILE_FG = Color.from_rgb(180, 180, 200)   # file path


def render_message(painter: Painter, msg: dict[str, Any], y: int, width: int,
                   theme_color=None) -> int:
    """Render a single message block at position y. Returns height used."""
    msg_type = msg.get("type", "")

    if msg_type == "user":
        return _render_user(painter, msg, y, width, theme_color)
    elif msg_type == "assistant":
        return _render_assistant(painter, msg, y, width, theme_color)
    elif msg_type == "tool":
        return _render_tool(painter, msg, y, width, theme_color)
    elif msg_type == "error":
        return _render_error(painter, msg, y, width, theme_color)
    elif msg_type == "system":
        return _render_system(painter, msg, y, width, theme_color)
    return 0


def _wrap_text(text: str, width: int) -> list[str]:
    """Simple word-wrap that respects newlines."""
    if width < 1:
        width = 1
    lines = []
    for raw_line in text.split("\n"):
        if not raw_line:
            lines.append("")
            continue
        while len(raw_line) > width:
            break_at = raw_line.rfind(" ", 0, width)
            if break_at <= 0:
                break_at = width
            lines.append(raw_line[:break_at])
            raw_line = raw_line[break_at:].lstrip()
        lines.append(raw_line)
    return lines


def _render_user(painter: Painter, msg: dict, y: int, width: int, theme_color) -> int:
    text = msg.get("text", "")
    border_color = Color.from_rgb(100, 140, 255)
    fg = Color.from_rgb(240, 240, 245)
    bg = _BG
    if theme_color:
        border_color = theme_color("message.user.border", border_color)
        fg = theme_color("message.user.fg", fg)
        bg = theme_color("container.bg", bg)

    lines = _wrap_text(text, width - 3)
    for i, line in enumerate(lines):
        painter.put_char(0, y + i, "\u2502", fg=border_color, bg=bg)
        painter.put_str(2, y + i, line, fg=fg, bg=bg, max_width=width - 3)
    return max(len(lines), 1)


def _render_assistant(painter: Painter, msg: dict, y: int, width: int, theme_color) -> int:
    text = msg.get("text", "")
    fg = Color.from_rgb(200, 200, 210)
    bg = _BG
    if theme_color:
        fg = theme_color("message.assistant.fg", fg)
        bg = theme_color("container.bg", bg)

    lines = _wrap_text(text, width - 1)
    for i, line in enumerate(lines):
        painter.put_str(0, y + i, line, fg=fg, bg=bg, max_width=width - 1)
    return max(len(lines), 1)


# ------------------------------------------------------------------
# Diff rendering for edit tools
# ------------------------------------------------------------------

def _shorten_path(file_path: str) -> str:
    """Shorten a file path relative to cwd."""
    try:
        cwd = os.getcwd()
        if file_path.startswith(cwd):
            return file_path[len(cwd):].lstrip("/")
    except Exception:
        pass
    return file_path


def _build_diff_lines(inp: dict) -> list[dict]:
    """Build diff display lines from edit tool input.

    Returns list of {"type": "hunk"|"del"|"add"|"ctx", "text": str}.
    """
    old_string = inp.get("old_string", "")
    new_string = inp.get("new_string", "")

    if not old_string and not new_string:
        return []

    old_lines = old_string.split("\n") if old_string else []
    new_lines = new_string.split("\n") if new_string else []

    diff = list(unified_diff(old_lines, new_lines, lineterm="", n=2))
    if not diff:
        return []

    result: list[dict] = []
    for line in diff:
        if line.startswith("@@"):
            result.append({"type": "hunk", "text": line})
        elif line.startswith("---") or line.startswith("+++"):
            continue  # skip unified_diff file headers
        elif line.startswith("-"):
            result.append({"type": "del", "text": line[1:]})
        elif line.startswith("+"):
            result.append({"type": "add", "text": line[1:]})
        else:
            text = line[1:] if line.startswith(" ") else line
            result.append({"type": "ctx", "text": text})

    return result


def _render_diff(painter: Painter, diff_lines: list[dict], y: int, width: int, bg: Color) -> int:
    """Render diff lines. Returns height used."""
    if not diff_lines:
        return 0

    code_width = max(1, width - 2)
    row = 0

    for dl in diff_lines:
        dtype = dl["type"]
        text = dl["text"]

        if dtype == "hunk":
            painter.put_str(0, y + row, "  " + text, fg=_DIFF_HUNK_FG, bg=bg, max_width=width)
            row += 1
        elif dtype == "del":
            painter.fill_rect(0, y + row, width, 1, bg=_DIFF_DEL_BG)
            painter.put_str(0, y + row, "- ", fg=_DIFF_DEL_FG, bg=_DIFF_DEL_BG)
            painter.put_str(2, y + row, text, fg=_DIFF_DEL_FG, bg=_DIFF_DEL_BG, max_width=code_width)
            row += 1
        elif dtype == "add":
            painter.fill_rect(0, y + row, width, 1, bg=_DIFF_ADD_BG)
            painter.put_str(0, y + row, "+ ", fg=_DIFF_ADD_FG, bg=_DIFF_ADD_BG)
            painter.put_str(2, y + row, text, fg=_DIFF_ADD_FG, bg=_DIFF_ADD_BG, max_width=code_width)
            row += 1
        else:  # ctx
            painter.put_str(0, y + row, "  ", fg=_DIFF_LINE_NO_FG, bg=bg)
            painter.put_str(2, y + row, text, fg=_DIFF_CONTEXT_FG, bg=bg, max_width=code_width)
            row += 1

    return row


def _render_tool(painter: Painter, msg: dict, y: int, width: int, theme_color) -> int:
    tool = msg.get("tool", "?")
    status = msg.get("status", "running")
    inp = msg.get("input", {})
    bg = _BG

    if status == "running":
        icon = "\u2699"
        status_text = "running..."
        color = Color.from_rgb(240, 200, 60)
    elif status == "done":
        icon = "\u2713"
        status_text = "done"
        color = Color.from_rgb(80, 200, 120)
    else:
        icon = "\u2717"
        status_text = "error"
        color = Color.from_rgb(240, 70, 70)

    if theme_color:
        bg = theme_color("container.bg", bg)
        if status == "running":
            color = theme_color("tool.running", color)
        elif status == "done":
            color = theme_color("tool.done", color)
        else:
            color = theme_color("tool.error", color)

    # Friendly display name
    display_name = tool.replace("builtin_", "")

    # For edit tools, show file path in header
    is_edit = tool in ("builtin_edit", "Edit")
    if is_edit and inp.get("file_path"):
        short_path = _shorten_path(inp["file_path"])
        header = f"{icon} {display_name} {short_path} [{status_text}]"
    else:
        header = f"{icon} {display_name} [{status_text}]"

    painter.put_str(0, y, header, fg=color, bg=bg, max_width=width)
    height = 1

    # For edit tools, render inline diff below the header
    if is_edit and inp.get("old_string") is not None:
        diff_lines = _build_diff_lines(inp)
        if diff_lines:
            diff_height = _render_diff(painter, diff_lines, y + height, width, bg)
            height += diff_height

    return height


def _render_error(painter: Painter, msg: dict, y: int, width: int, theme_color) -> int:
    text = msg.get("text", "Error")
    fg = Color.from_rgb(240, 70, 70)
    bg = _BG
    if theme_color:
        fg = theme_color("tool.error", fg)
        bg = theme_color("container.bg", bg)

    lines = _wrap_text(text, width - 3)
    for i, line in enumerate(lines):
        painter.put_char(0, y + i, "\u2502", fg=fg, bg=bg)
        painter.put_str(2, y + i, line, fg=fg, bg=bg, max_width=width - 3)
    return max(len(lines), 1)


def _render_system(painter: Painter, msg: dict, y: int, width: int, theme_color) -> int:
    text = msg.get("text", "")
    fg = Color.from_rgb(160, 165, 180)
    bg = _BG
    if theme_color:
        bg = theme_color("container.bg", bg)

    lines = _wrap_text(text, width - 1)
    for i, line in enumerate(lines):
        painter.put_str(0, y + i, line, fg=fg, bg=bg, max_width=width - 1)
    return max(len(lines), 1)
