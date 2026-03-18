"""SessionScreen — header + message list + prompt + footer."""
from __future__ import annotations

from typing import TYPE_CHECKING

from runtui import Container, DockLayout, VBoxLayout

from ..widgets.header_bar import HeaderBar
from ..widgets.footer_bar import FooterBar
from ..widgets.message_list import MessageList
from ..widgets.prompt_input import PromptInput

if TYPE_CHECKING:
    from tcode_app import TcodeApp


class SessionScreen(Container):
    """Active session screen with header, messages, prompt, and footer."""

    def __init__(self, app: TcodeApp) -> None:
        super().__init__()
        self.app = app
        self._layout_manager = DockLayout()
        self._spinner_timer = None
        self._build()

    def _build(self) -> None:
        state = self.app.state

        # Header bar (dock top)
        self._header = HeaderBar(state)
        self._header.dock = "top"
        self._header.height = 1
        self.add_child(self._header)

        # Footer bar (dock bottom)
        self._footer = FooterBar(state, project_dir=self.app.args.dir)
        self._footer.dock = "bottom"
        self._footer.height = 1
        self.add_child(self._footer)

        # Prompt input (dock bottom, above footer)
        self._prompt = PromptInput(
            on_submit=self._on_submit,
            state=state,
            id="session_prompt",
        )
        self._prompt.dock = "bottom"
        self._prompt.height = 1
        self.add_child(self._prompt)

        # Message list (fill remaining)
        self._message_list = MessageList(state)
        self._message_list.dock = "fill"
        self.add_child(self._message_list)

    def start_spinner(self) -> None:
        """Start the footer spinner timer."""
        if self._spinner_timer is None:
            self._spinner_timer = self.app.set_interval(0.2, self._tick_spinner)

    def stop_spinner(self) -> None:
        """Stop the footer spinner timer."""
        if self._spinner_timer is not None:
            self._spinner_timer.cancel()
            self._spinner_timer = None

    def _tick_spinner(self) -> None:
        self._footer.advance_spinner()
        self.app.schedule_repaint()

    def _on_submit(self, text: str) -> None:
        if self.app.bridge:
            self.app.bridge.submit_prompt(text)
