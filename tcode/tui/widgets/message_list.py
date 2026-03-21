"""MessageList — scrollable message container with auto-scroll."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from runtui import Color, Attrs
from runtui.core.event import KeyEvent, MouseEvent
from runtui.core.keys import Keys, Modifiers, MouseAction, MouseButton
from runtui.rendering.painter import Painter
from runtui.widgets.base import Widget

from .message_block import render_message

if TYPE_CHECKING:
    from ..state import TuiState


class MessageList(Widget):
    """Scrollable list of message blocks."""

    def __init__(self, state: TuiState) -> None:
        super().__init__()
        self.state = state
        self._scroll_offset = 0
        self._total_height = 0
        self._auto_scroll = True
        self._last_msg_count = 0
        self.can_focus = False

        self.on(KeyEvent, self._handle_key)
        self.on(MouseEvent, self._handle_mouse)

    def paint(self, painter: Painter) -> None:
        sr = self._screen_rect
        lx = sr.x - painter._offset.x
        ly = sr.y - painter._offset.y
        w = sr.width - 1  # leave 1 col for scrollbar
        h = sr.height

        bg = self.theme_color("container.bg", Color.from_rgb(22, 22, 30))
        painter.fill_rect(lx, ly, sr.width, h, bg=bg)

        messages = self.state.messages
        content_width = max(1, w - 2)  # 1 padding on each side

        # Calculate total content height
        row_heights: list[int] = []
        for msg in messages:
            # Calculate height without drawing
            msg_h = self._calc_message_height(msg, content_width)
            row_heights.append(msg_h + 1)  # +1 for spacing between messages

        # Add permission block height if pending
        perm = self.state.pending_permission
        perm_height = 0
        if perm:
            perm_height = self._calc_permission_height(perm, content_width)

        self._total_height = sum(row_heights) + perm_height

        # Auto-scroll if new messages arrived
        if len(messages) > self._last_msg_count:
            self._auto_scroll = True
            self._last_msg_count = len(messages)

        # Clamp scroll offset to valid range (handles permission disappearing, resize, etc.)
        max_scroll = max(0, self._total_height - h)
        if self._scroll_offset > max_scroll:
            self._scroll_offset = max_scroll

        if self._auto_scroll:
            self._scroll_offset = max_scroll

        # Render visible messages
        content_painter = painter.sub_painter(lx + 1, ly, content_width, h)
        y = -self._scroll_offset

        for i, msg in enumerate(messages):
            msg_h = row_heights[i] - 1  # without spacing
            # Only render if visible
            if y + msg_h > 0 and y < h:
                render_message(content_painter, msg, y, content_width,
                               theme_color=self.theme_color)
            y += msg_h + 1  # +1 spacing

        # Render permission prompt if pending
        if perm:
            if y + perm_height > 0 and y < h:
                self._render_permission(content_painter, perm, y, content_width)

        # Render scrollbar
        if self._total_height > h:
            self._render_scrollbar(painter, lx + sr.width - 1, ly, h)

    def _calc_message_height(self, msg: dict[str, Any], width: int) -> int:
        """Calculate the rendered height of a message without drawing."""
        msg_type = msg.get("type", "")
        if msg_type == "tool":
            h = 1  # header line
            # Edit tools have inline diff
            tool = msg.get("tool", "")
            inp = msg.get("input", {})
            if tool in ("builtin_edit", "Edit") and inp.get("old_string") is not None:
                from .message_block import _build_diff_lines
                diff_lines = _build_diff_lines(inp)
                h += len(diff_lines)
            return h

        text = msg.get("text", "")
        if not text:
            return 1

        # Count wrapped lines
        lines = 0
        for raw_line in text.split("\n"):
            if not raw_line:
                lines += 1
                continue
            effective_width = width - 3 if msg_type in ("user", "error") else width - 1
            effective_width = max(1, effective_width)
            line_count = (len(raw_line) + effective_width - 1) // effective_width
            lines += max(line_count, 1)
        return max(lines, 1)

    @staticmethod
    def _format_permission_details(perm: dict) -> list[tuple[str, str]]:
        """Extract human-readable detail lines from a permission request.

        Returns list of (label, value) tuples to display.
        Filters out noisy fields (e.g. full file content) and formats
        the remaining fields for readability.
        """
        ptype = perm.get("type", "")
        details = perm.get("details", {})
        if not isinstance(details, dict):
            return []

        lines: list[tuple[str, str]] = []

        # Tool-specific formatting
        if "write_file" in ptype:
            if details.get("path"):
                lines.append(("File", str(details["path"])))
            content = details.get("content", "")
            if content:
                size = len(content)
                preview = content[:60].replace("\n", "\\n")
                if len(content) > 60:
                    preview += "..."
                lines.append(("Size", f"{size} chars"))
                lines.append(("Preview", preview))
            if details.get("overwrite"):
                lines.append(("Overwrite", "yes"))

        elif "read_file" in ptype:
            if details.get("path"):
                lines.append(("File", str(details["path"])))

        elif "shell" in ptype:
            if details.get("cmd"):
                lines.append(("Command", str(details["cmd"])))
            if details.get("timeout"):
                lines.append(("Timeout", f"{details['timeout']}s"))

        elif "http_fetch" in ptype:
            if details.get("url"):
                lines.append(("URL", str(details["url"])))
            method = details.get("method", "GET")
            if method and method != "GET":
                lines.append(("Method", method))

        elif "edit" in ptype:
            if details.get("file_path"):
                lines.append(("File", str(details["file_path"])))
            old = details.get("old_string", "")
            new = details.get("new_string", "")
            if old:
                preview = old[:50].replace("\n", "\\n")
                if len(old) > 50:
                    preview += "..."
                lines.append(("Replace", preview))
            if new:
                preview = new[:50].replace("\n", "\\n")
                if len(new) > 50:
                    preview += "..."
                lines.append(("With", preview))

        elif "network" in ptype:
            if details.get("url"):
                lines.append(("URL", str(details["url"])))

        else:
            # Generic: show relevant fields, skip large values
            skip_keys = {"content", "content_b64", "data"}
            for k, v in list(details.items())[:4]:
                if k in skip_keys:
                    continue
                val = str(v)
                if len(val) > 80:
                    val = val[:77] + "..."
                lines.append((k, val))

        return lines

    def _calc_permission_height(self, perm: dict, width: int) -> int:
        """Calculate the height needed for the permission block."""
        detail_lines = self._format_permission_details(perm)
        # 1 blank + 1 header + detail lines + 1 separator + 1 hint line
        return 3 + len(detail_lines) + 1

    def _render_permission(self, painter: Painter, perm: dict, y: int, width: int) -> None:
        fg = self.theme_color("permission.fg", Color.from_rgb(240, 200, 60))
        border_color = self.theme_color("permission.border", Color.from_rgb(240, 200, 60))
        dim_fg = self.theme_color("permission.dim", Color.from_rgb(180, 160, 80))
        bg = self.theme_color("container.bg", Color.from_rgb(22, 22, 30))

        ptype = perm.get("type", "unknown")

        # Human-readable tool names
        tool_labels = {
            "builtin_write_file": "Write File",
            "builtin_read_file": "Read File",
            "builtin_shell": "Shell Command",
            "builtin_http_fetch": "HTTP Request",
            "builtin_edit": "Edit File",
            "builtin_file_attach": "File Attachment",
            "builtin_grep": "Search Files",
            "builtin_list_files": "List Files",
            "shell_execute": "Shell Command",
            "network_access": "Network Access",
        }
        label = tool_labels.get(ptype, ptype)

        row = y

        # Blank line before
        painter.put_char(0, row, "┌", fg=border_color, bg=bg)
        painter.put_str(1, row, "─" * (width - 2), fg=border_color, bg=bg)
        row += 1

        # Header
        painter.put_char(0, row, "│", fg=border_color, bg=bg)
        painter.put_str(2, row, f"Allow {label}?", fg=fg, bg=bg, attrs=Attrs.BOLD,
                        max_width=width - 3)
        row += 1

        # Detail lines
        detail_lines = self._format_permission_details(perm)
        for lbl, val in detail_lines:
            painter.put_char(0, row, "│", fg=border_color, bg=bg)
            label_str = f"  {lbl}: "
            painter.put_str(2, row, label_str, fg=dim_fg, bg=bg, max_width=width - 3)
            val_start = 2 + len(label_str)
            remaining = width - val_start - 1
            if remaining > 0:
                display_val = val if len(val) <= remaining else val[:remaining - 3] + "..."
                painter.put_str(val_start, row, display_val, fg=fg, bg=bg, max_width=remaining)
            row += 1

        # Separator
        painter.put_char(0, row, "│", fg=border_color, bg=bg)
        row += 1

        # Action hints
        painter.put_char(0, row, "└", fg=border_color, bg=bg)
        hint = " [y] Allow  [n] Deny  [a] Always allow"
        painter.put_str(2, row, hint, fg=fg, bg=bg, attrs=Attrs.BOLD, max_width=width - 3)

    def _render_scrollbar(self, painter: Painter, x: int, y: int, height: int) -> None:
        track_color = self.theme_color("scrollbar.track", Color.from_rgb(35, 35, 45))
        thumb_color = self.theme_color("scrollbar.thumb", Color.from_rgb(100, 100, 120))

        # Draw track
        for row in range(height):
            painter.put_char(x - painter._offset.x, y + row - painter._offset.y,
                             "░", fg=track_color)

        # Draw thumb
        if self._total_height > 0:
            thumb_h = max(1, height * height // self._total_height)
            thumb_pos = self._scroll_offset * (height - thumb_h) // max(1, self._total_height - height)
            thumb_pos = max(0, min(thumb_pos, height - thumb_h))
            for row in range(thumb_h):
                painter.put_char(x - painter._offset.x,
                                 y + thumb_pos + row - painter._offset.y,
                                 "█", fg=thumb_color)

    def _handle_key(self, event: KeyEvent) -> None:
        if event.key == Keys.PAGE_UP:
            self._scroll_offset = max(0, self._scroll_offset - self._screen_rect.height)
            self._auto_scroll = False
            self.invalidate()
            event.mark_handled()
        elif event.key == Keys.PAGE_DOWN:
            max_scroll = max(0, self._total_height - self._screen_rect.height)
            self._scroll_offset = min(max_scroll, self._scroll_offset + self._screen_rect.height)
            if self._scroll_offset >= max_scroll:
                self._auto_scroll = True
            self.invalidate()
            event.mark_handled()

    def _handle_mouse(self, event: MouseEvent) -> None:
        if event.button == MouseButton.SCROLL_UP:
            self._scroll_offset = max(0, self._scroll_offset - 3)
            self._auto_scroll = False
            self.invalidate()
            event.mark_handled()
        elif event.button == MouseButton.SCROLL_DOWN:
            max_scroll = max(0, self._total_height - self._screen_rect.height)
            self._scroll_offset = min(max_scroll, self._scroll_offset + 3)
            if self._scroll_offset >= max_scroll:
                self._auto_scroll = True
            self.invalidate()
            event.mark_handled()
