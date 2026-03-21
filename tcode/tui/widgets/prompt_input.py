"""PromptInput — multiline text input with `> ` prefix, history, and mode toggle."""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from runtui import (
    Container, Color, Attrs,
)
from runtui.core.event import KeyEvent
from runtui.core.keys import Keys, Modifiers
from runtui.rendering.painter import Painter

if TYPE_CHECKING:
    from ..state import TuiState

# Mode colors
_CHAT_FG = Color.from_rgb(100, 140, 255)
_SHELL_FG = Color.from_rgb(240, 200, 60)
_INPUT_BG = Color.from_rgb(30, 30, 40)
_TEXT_FG = Color.from_rgb(240, 240, 245)
_PLACEHOLDER_FG = Color.from_rgb(100, 100, 120)

_MODE_CONFIG = {
    "chat":  {"prefix": "> ", "fg": _CHAT_FG,  "placeholder": "Type a message... (Shift+Enter for new line)"},
    "shell": {"prefix": "$ ", "fg": _SHELL_FG, "placeholder": "Shell command... (Shift+Enter for new line)"},
}

MAX_INPUT_HEIGHT = 10


class PromptInput(Container):
    """Multiline prompt with mode toggle and text wrapping.

    Press `!` on an empty line to toggle between chat and shell mode.
    Press Enter to add a newline. Press Ctrl+Enter to submit.
    Long lines wrap automatically (up to MAX_INPUT_HEIGHT).
    """

    def __init__(
        self,
        on_submit: Callable[[str], None] | None = None,
        state: TuiState | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id, height=1)
        self._on_submit = on_submit
        self._state = state
        self._history_index = -1
        self._mode = "chat"

        # Text state
        self._text = ""
        self._cursor_pos = 0
        self._view_scroll = 0
        self.can_focus = True
        self._focused = False

        self.on(KeyEvent, self._handle_key)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        self._text = value
        self._cursor_pos = len(value)
        self._recalc_height()

    def focus(self) -> None:
        self._focused = True
        self.invalidate()

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self.invalidate()

    # ------------------------------------------------------------------
    # Wrapping helpers
    # ------------------------------------------------------------------

    def _content_width(self) -> int:
        """Available width for text (total width minus 2-char prefix)."""
        if self._screen_rect:
            return max(1, self._screen_rect.width - 2)
        return 40

    def _wrap_lines(self, width: int) -> list[str]:
        """Wrap text into display lines, respecting newlines and width."""
        if not self._text:
            return [""]

        display_lines: list[str] = []
        for line in self._text.split("\n"):
            if not line:
                display_lines.append("")
            else:
                while len(line) > width:
                    display_lines.append(line[:width])
                    line = line[width:]
                display_lines.append(line)
        return display_lines

    def _cursor_to_display_pos(self, width: int) -> tuple[int, int]:
        """Convert flat cursor position to (row, col) in wrapped display."""
        text_before_cursor = self._text[:self._cursor_pos]
        display_lines = self._wrap_lines(width)

        # Count through display lines to find cursor position
        char_count = 0
        for row, line in enumerate(display_lines):
            if char_count + len(line) >= len(text_before_cursor):
                col = len(text_before_cursor) - char_count
                return (row, col)
            char_count += len(line)
            # Account for newline character (if this line ended with \n in original)
            if char_count < len(self._text) and self._text[char_count] == "\n":
                if char_count >= len(text_before_cursor):
                    return (row, len(line))
                char_count += 1

        # Cursor at end
        return (len(display_lines) - 1, len(display_lines[-1]) if display_lines else 0)

    def _recalc_height(self) -> None:
        """Adjust container height based on wrapped line count."""
        cw = self._content_width()
        display_lines = self._wrap_lines(cw)
        new_height = min(max(1, len(display_lines)), MAX_INPUT_HEIGHT)
        if new_height != self.height:
            self.height = new_height
            # Ask parent to re-layout
            parent = self.parent
            if parent and hasattr(parent, "arrange") and parent._screen_rect:
                parent.arrange(parent._screen_rect)
            self.invalidate()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paint(self, painter: Painter) -> None:
        sr = self._screen_rect
        if not sr:
            return
        lx = sr.x - painter._offset.x
        ly = sr.y - painter._offset.y
        w = sr.width
        h = sr.height

        cfg = _MODE_CONFIG[self._mode]
        prefix = cfg["prefix"]
        prefix_fg = cfg["fg"]
        cw = max(1, w - 2)
        display_lines = self._wrap_lines(cw)

        # Scroll so cursor line is visible
        cursor_row, cursor_col = self._cursor_to_display_pos(cw)
        if cursor_row < self._view_scroll:
            self._view_scroll = cursor_row
        elif cursor_row >= self._view_scroll + h:
            self._view_scroll = cursor_row - h + 1
        self._view_scroll = max(0, min(self._view_scroll, max(0, len(display_lines) - h)))

        # Fill background
        painter.fill_rect(lx, ly, w, h, bg=_INPUT_BG)

        # Paint prefix on first visible line
        painter.put_str(lx, ly, prefix, fg=prefix_fg, bg=_INPUT_BG)
        # Fill prefix column on remaining lines
        for row in range(1, h):
            painter.put_str(lx, ly + row, "  ", fg=prefix_fg, bg=_INPUT_BG)

        # Paint text lines
        for row in range(h):
            line_idx = self._view_scroll + row
            if line_idx >= len(display_lines):
                break
            line = display_lines[line_idx]
            painter.put_str(lx + 2, ly + row, line, fg=_TEXT_FG, bg=_INPUT_BG,
                            max_width=cw)

        # Placeholder when empty
        if not self._text:
            placeholder = cfg.get("placeholder", "")
            painter.put_str(lx + 2, ly, placeholder, fg=_PLACEHOLDER_FG, bg=_INPUT_BG,
                            max_width=cw)

        # Paint cursor
        if self._focused:
            screen_row = cursor_row - self._view_scroll
            if 0 <= screen_row < h:
                ch = self._text[self._cursor_pos] if self._cursor_pos < len(self._text) else " "
                painter.put_char(lx + 2 + cursor_col, ly + screen_row, ch,
                                 fg=_INPUT_BG, bg=_TEXT_FG)

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _handle_key(self, event: KeyEvent) -> None:
        if not self._focused:
            return

        # Shift+Enter: insert newline
        if event.key == Keys.ENTER and Modifiers.SHIFT in event.modifiers:
            self._insert_text("\n")
            event.mark_handled()
            return

        # Enter: submit
        if event.key == Keys.ENTER:
            self._submit()
            event.mark_handled()
            return

        # Mode toggle: if text is exactly "!" on empty, toggle mode
        if self._text == "!":
            self._text = ""
            self._cursor_pos = 0
            if self._mode == "chat":
                self._set_mode("shell")
            else:
                self._set_mode("chat")
            self._recalc_height()
            event.mark_handled()
            return

        # Navigation
        if event.key == Keys.LEFT:
            self._cursor_pos = max(0, self._cursor_pos - 1)
            self.invalidate()
            event.mark_handled()
        elif event.key == Keys.RIGHT:
            self._cursor_pos = min(len(self._text), self._cursor_pos + 1)
            self.invalidate()
            event.mark_handled()
        elif event.key == Keys.HOME:
            # Move to start of current line
            line_start = self._text.rfind("\n", 0, self._cursor_pos)
            self._cursor_pos = line_start + 1 if line_start >= 0 else 0
            self.invalidate()
            event.mark_handled()
        elif event.key == Keys.END:
            # Move to end of current line
            line_end = self._text.find("\n", self._cursor_pos)
            self._cursor_pos = line_end if line_end >= 0 else len(self._text)
            self.invalidate()
            event.mark_handled()
        elif event.key == Keys.UP:
            if not self._text:
                self._navigate_history(-1)
            event.mark_handled()
        elif event.key == Keys.DOWN:
            if self._history_index >= 0:
                self._navigate_history(1)
            event.mark_handled()

        # Editing
        elif event.key == Keys.BACKSPACE:
            if self._cursor_pos > 0:
                self._text = self._text[:self._cursor_pos - 1] + self._text[self._cursor_pos:]
                self._cursor_pos -= 1
                self._recalc_height()
                self.invalidate()
            event.mark_handled()
        elif event.key == Keys.DELETE:
            if self._cursor_pos < len(self._text):
                self._text = self._text[:self._cursor_pos] + self._text[self._cursor_pos + 1:]
                self._recalc_height()
                self.invalidate()
            event.mark_handled()

        # Character input
        elif event.key == Keys.CHAR and event.char and Modifiers.CTRL not in event.modifiers:
            self._insert_text(event.char)
            event.mark_handled()

    def _insert_text(self, text: str) -> None:
        self._text = self._text[:self._cursor_pos] + text + self._text[self._cursor_pos:]
        self._cursor_pos += len(text)
        self._recalc_height()
        self.invalidate()

    def _submit(self) -> None:
        text = self._text.strip()
        if not text:
            return

        # In shell mode, prefix with ! unless it's a slash command
        if self._mode == "shell" and not text.startswith("/"):
            text = "!" + text

        self._text = ""
        self._cursor_pos = 0
        self._history_index = -1
        self._view_scroll = 0
        self._recalc_height()
        if self._on_submit:
            self._on_submit(text)

    def _navigate_history(self, direction: int) -> None:
        if not self._state or not self._state.prompt_history:
            return
        history = self._state.prompt_history
        if direction < 0:
            if self._history_index < 0:
                self._history_index = len(history) - 1
            else:
                self._history_index = max(0, self._history_index - 1)
        else:
            self._history_index += 1
            if self._history_index >= len(history):
                self._history_index = -1
                self._text = ""
                self._cursor_pos = 0
                self._recalc_height()
                return

        self._text = history[self._history_index]
        self._cursor_pos = len(self._text)
        self._recalc_height()
        self.invalidate()
