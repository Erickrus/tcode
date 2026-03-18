"""FooterBar — 1-row widget showing directory, MCP status, model, agent."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from runtui import Color, Attrs
from runtui.rendering.painter import Painter
from runtui.widgets.base import Widget

if TYPE_CHECKING:
    from ..state import TuiState


class FooterBar(Widget):
    """Single-row footer: directory on left, MCP + model + agent on right."""

    def __init__(self, state: TuiState, project_dir: str = "") -> None:
        super().__init__(height=1)
        self.state = state
        self.project_dir = project_dir
        self.dock = "bottom"
        self._spinner_frame = 0
        self._spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def advance_spinner(self) -> None:
        """Called by a timer to advance the spinner frame."""
        if self.state.status == "running":
            self._spinner_frame = (self._spinner_frame + 1) % len(self._spinner_chars)

    def paint(self, painter: Painter) -> None:
        sr = self._screen_rect
        lx = sr.x - painter._offset.x
        ly = sr.y - painter._offset.y
        w = sr.width

        bg = self.theme_color("footer.bg", Color.from_rgb(25, 25, 35))
        fg = self.theme_color("footer.fg", Color.from_rgb(100, 100, 120))
        accent = Color.from_rgb(100, 140, 255)

        # Fill background
        painter.fill_rect(lx, ly, w, 1, bg=bg)

        # Left: directory (truncated)
        dir_text = self.project_dir
        home = os.path.expanduser("~")
        if dir_text.startswith(home):
            dir_text = "~" + dir_text[len(home):]

        # Spinner when running
        if self.state.status == "running":
            spinner = self._spinner_chars[self._spinner_frame] + " "
            painter.put_str(lx + 1, ly, spinner, fg=accent, bg=bg)
            offset = 3
        else:
            offset = 1

        max_dir_len = w // 2
        if len(dir_text) > max_dir_len:
            dir_text = "..." + dir_text[-(max_dir_len - 3):]
        painter.put_str(lx + offset, ly, dir_text, fg=fg, bg=bg)

        # Right: MCP status | model | agent | queue
        right_parts = []

        # MCP
        mcp = self.state.mcp_status
        connected = sum(1 for s in mcp.values() if s == "connected")
        if connected:
            right_parts.append(f"MCP:{connected}")

        # Model
        model_str = self.state.model_id
        if model_str:
            if len(model_str) > 20:
                model_str = model_str[:17] + "..."
            right_parts.append(model_str)

        # Agent
        if self.state.agent_name and self.state.agent_name != "build":
            right_parts.append(f"[{self.state.agent_name}]")

        # Queue
        qlen = len(self.state.queued_prompts)
        if qlen > 0:
            right_parts.append(f"queue:{qlen}")

        right_text = "  ".join(right_parts)
        rx = w - len(right_text) - 1
        if rx > 0:
            painter.put_str(lx + rx, ly, right_text, fg=fg, bg=bg)
