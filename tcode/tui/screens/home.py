"""HomeScreen — logo + info + prompt input."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from runtui import Container, Label, Color, Attrs, VBoxLayout, Size, Rect
from runtui.rendering.painter import Painter

from ..widgets.prompt_input import PromptInput

if TYPE_CHECKING:
    from tcode_app import TcodeApp


BANNER = """\
████████╗     ██████╗ ██████╗ ██████╗ ███████╗
╚══██╔══╝    ██╔════╝██╔═══██╗██╔══██╗██╔════╝
  ██║       ██║     ██║   ██║██║  ██║█████╗
  ██║       ██║     ██║   ██║██║  ██║██╔══╝
   ██║       ╚██████╗╚██████╔╝██████╔╝███████╗
   ╚═╝        ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝\
"""


class HomeScreen(Container):
    """Home screen with ASCII logo, info labels, and prompt input."""

    def __init__(self, app: TcodeApp) -> None:
        super().__init__()
        self.app = app
        self._layout_manager = VBoxLayout()
        self._build()

    def _build(self) -> None:
        state = self.app.state

        # Top spacer (flex)
        spacer_top = Container(id="spacer_top")
        spacer_top.flex = 1.0
        self.add_child(spacer_top)

        # Banner lines — render as individual centered labels
        banner_lines = BANNER.split("\n")
        for line in banner_lines:
            lbl = Label(
                text=line,
                align="center",
                fg=Color.from_rgb(100, 140, 255),
                height=1,
            )
            lbl.flex = 0.0
            self.add_child(lbl)

        # Blank line
        blank = Label(text="", height=1)
        blank.flex = 0.0
        self.add_child(blank)

        # Info labels
        project_dir = self.app.args.dir
        home = os.path.expanduser("~")
        display_dir = project_dir
        if display_dir.startswith(home):
            display_dir = "~" + display_dir[len(home):]

        info_lines = [
            f"  Project: {display_dir}",
            f"  Model:   {state.provider_id}/{state.model_id}",
        ]

        mcp = state.mcp_status
        connected = sum(1 for s in mcp.values() if s == "connected")
        if connected:
            info_lines.append(f"  MCP:     {connected} server{'s' if connected != 1 else ''} connected")

        for text in info_lines:
            lbl = Label(
                text=text,
                align="center",
                fg=Color.from_rgb(100, 100, 120),
                height=1,
            )
            lbl.flex = 0.0
            self.add_child(lbl)

        # Blank line
        blank2 = Label(text="", height=1)
        blank2.flex = 0.0
        self.add_child(blank2)

        # Bottom spacer (flex)
        spacer_bottom = Container(id="spacer_bottom")
        spacer_bottom.flex = 1.0
        self.add_child(spacer_bottom)

        # Prompt input at bottom
        self._prompt = PromptInput(
            on_submit=self._on_submit,
            state=state,
            id="home_prompt",
        )
        self._prompt.flex = 0.0
        self._prompt.height = 1
        self.add_child(self._prompt)

    def _on_submit(self, text: str) -> None:
        import asyncio

        # Start session and switch to session screen
        async def _start():
            await self.app.bridge.start_session()
            self.app.switch_to_session()
            self.app.bridge.submit_prompt(text)

        asyncio.create_task(_start())
