"""HeaderBar — 1-row widget showing session title, tokens, and cost."""
from __future__ import annotations

from typing import TYPE_CHECKING

from runtui import Color, Attrs
from runtui.rendering.painter import Painter
from runtui.widgets.base import Widget

if TYPE_CHECKING:
    from ..state import TuiState


class HeaderBar(Widget):
    """Single-row header: session title on left, tokens + cost on right."""

    def __init__(self, state: TuiState) -> None:
        super().__init__(height=1)
        self.state = state
        self.dock = "top"

    def paint(self, painter: Painter) -> None:
        sr = self._screen_rect
        lx = sr.x - painter._offset.x
        ly = sr.y - painter._offset.y
        w = sr.width

        bg = self.theme_color("header.bg", Color.from_rgb(25, 25, 35))
        fg = self.theme_color("header.fg", Color.from_rgb(200, 200, 210))
        accent = self.theme_color("streaming.cursor", Color.from_rgb(100, 140, 255))

        # Fill background
        painter.fill_rect(lx, ly, w, 1, bg=bg)

        # Left: session title or "tcode"
        title = "tcode"
        if self.state.session_id:
            title = f"tcode — {self.state.session_id[:8]}"
        painter.put_str(lx + 1, ly, title, fg=accent, bg=bg, attrs=Attrs.BOLD)

        # Status indicator
        status = self.state.status
        if status == "running":
            status_text = " ● running"
            status_fg = Color.from_rgb(240, 200, 60)
        elif status == "waiting_permission":
            status_text = " ⚠ permission"
            status_fg = Color.from_rgb(240, 200, 60)
        else:
            status_text = ""
            status_fg = fg

        if status_text:
            title_len = len(title) + 1
            painter.put_str(lx + 1 + title_len, ly, status_text, fg=status_fg, bg=bg)

        # Right: tokens + cost
        tokens = self.state.tokens
        input_t = tokens.get("input", 0)
        output_t = tokens.get("output", 0)
        cost = self.state.cost

        right_parts = []
        if input_t or output_t:
            total_k = (input_t + output_t) / 1000
            right_parts.append(f"{total_k:.1f}k tok")
        if cost > 0:
            right_parts.append(f"${cost:.4f}")

        right_text = "  ".join(right_parts)
        if right_text:
            rx = w - len(right_text) - 1
            if rx > 0:
                painter.put_str(lx + rx, ly, right_text, fg=fg, bg=bg)
