"""TcodeApp — fullscreen runtui application for tcode."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from runtui import (
    App, Container, Color, Rect, Size, Keys, Modifiers,
    DockLayout, ThemeDefinition,
)
from runtui.core.event import KeyEvent, ResizeEvent, Event
from runtui.rendering.painter import Painter
from runtui.widgets.base import Widget, _find_focused

from tcode.tui.state import TuiState
from tcode.tui.theme import tcode_theme


class TcodeApp(App):
    """Full-screen TUI for tcode, built on runtui.

    Skips window manager / taskbar / desktop — uses a single root Container
    with DockLayout to hold tcode screens.
    """

    def __init__(self, args: Any) -> None:
        super().__init__(theme="tcode")
        self.args = args
        self.state = TuiState()
        self.bridge: Any = None  # set after setup

        # Screens (set in _run_async)
        self._home_screen: Widget | None = None
        self._session_screen: Widget | None = None

        # CLI backend
        self._cli: Any = None

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    async def _run_async(self) -> None:
        """Initialize backend, build UI, setup tcode, run event loop."""
        from runtui.backend.detect import create_backend
        from runtui.rendering.screen import Screen
        from runtui.core.event_loop import EventLoop
        from runtui.mouse.cursor import MouseCursor
        from runtui.mouse.tracker import MouseTracker

        self._backend = create_backend()
        self._backend.init()

        # Disable mouse tracking entirely so users can select + copy text
        # natively in their terminal. We use PageUp/PageDown for scrolling.
        self._backend.write("\x1b[?1003l")  # disable any-event tracking
        self._backend.write("\x1b[?1006l")  # disable SGR mouse mode
        self._backend.write("\x1b[?1000l")  # disable normal button tracking
        self._backend.flush()

        cols, rows = self._backend.get_size()
        self._screen = Screen(self._backend)

        # We don't use window_manager / taskbar / desktop
        self._window_manager = None
        self._taskbar = None
        self._desktop = None

        # Build root
        self._build_root(cols, rows)

        # Register tcode theme and set it
        self._theme_engine.register(tcode_theme)
        self._theme_engine.set_theme("tcode")

        self._event_loop = EventLoop(self._backend)
        self.running = True

        # Setup tcode backend
        from tcode.cli import TcodeCLI
        self._cli = TcodeCLI(project_dir=self.args.dir, db_path=self.args.db)
        self._cli.tui_mode = True
        try:
            await self._cli.setup()

            if self.args.verbose:
                self._cli.agent_runner.toolrunner.verbose = True

            # Unsubscribe CLI's interactive permission handler (uses input())
            # TUI handles permissions via the bridge instead
            if hasattr(self._cli, '_unsub_permission') and self._cli._unsub_permission:
                self._cli._unsub_permission()

            # Populate state from config
            default_model = self._cli.config.model
            self.state.provider_id = self.args.provider or default_model.provider_id
            self.state.model_id = self.args.model or default_model.model_id
            self.state.agent_name = self.args.agent or "build"
            if self._cli.mcp_manager and self._cli.mcp_manager.status:
                self.state.mcp_status = dict(self._cli.mcp_manager.status)

            # Build bridge
            from tcode.tui.bridge import TcodeBridge
            self.bridge = TcodeBridge(self, self._cli)

            # Build screens
            from tcode.tui.screens.home import HomeScreen
            from tcode.tui.screens.session import SessionScreen

            self._home_screen = HomeScreen(self)
            self._session_screen = SessionScreen(self)

            # Start with home screen
            self.switch_to_home()

            # Force initial paint
            self._needs_repaint = True
            if self._screen:
                self._screen.force_full_redraw()

            # Run event loop (yields via asyncio.sleep each frame)
            await self._event_loop.run(self)
        finally:
            await self._cli.teardown()

    def _build_root(self, cols: int, rows: int) -> None:
        """Create a single fullscreen Container as root."""
        self.root = Container(width=cols, height=rows)
        self.root._screen_rect = Rect(0, 0, cols, rows)
        self.root._theme_engine = self._theme_engine
        self.root._layout_manager = DockLayout()

    # ------------------------------------------------------------------
    # Screen switching
    # ------------------------------------------------------------------

    def switch_to_home(self) -> None:
        if not self.root:
            return
        self.root.clear_children()
        if self._home_screen:
            self._home_screen.dock = "fill"
            self.root.add_child(self._home_screen)
            self.root.arrange(self.root._screen_rect)
            # Focus the prompt input
            prompt = self._home_screen.find_by_id("home_prompt")
            if prompt:
                prompt.focus()
        self.state.screen = "home"
        self.schedule_repaint()

    def switch_to_session(self) -> None:
        if not self.root:
            return
        self.root.clear_children()
        if self._session_screen:
            self._session_screen.dock = "fill"
            self.root.add_child(self._session_screen)
            self.root.arrange(self.root._screen_rect)
            # Focus the prompt input
            prompt = self._session_screen.find_by_id("session_prompt")
            if prompt:
                prompt.focus()
        self.state.screen = "session"
        self.schedule_repaint()

    # ------------------------------------------------------------------
    # Repaint + timers
    # ------------------------------------------------------------------

    def schedule_repaint(self) -> None:
        self._needs_repaint = True

    def set_interval(self, interval: float, callback) -> Any:
        """Schedule a repeating callback. Returns a handle with .cancel()."""
        if self._event_loop:
            return self._event_loop.set_interval(interval, callback)
        return None

    def call_later(self, delay: float, callback) -> Any:
        """Schedule a one-shot callback after delay."""
        if self._event_loop:
            return self._event_loop.call_later(delay, callback)
        return None

    def invalidate_all(self) -> None:
        """Force full repaint."""
        self._needs_repaint = True
        if self._screen:
            self._screen.force_full_redraw()

    # ------------------------------------------------------------------
    # Event handling overrides
    # ------------------------------------------------------------------

    def _handle_mouse(self, event) -> None:
        """Mouse tracking is disabled for native terminal text selection."""
        pass

    def _handle_resize(self, event: ResizeEvent) -> None:
        if self._screen:
            self._screen.resize(event.width, event.height)
        if self.root:
            self.root.width = event.width
            self.root.height = event.height
            self.root._screen_rect = Rect(0, 0, event.width, event.height)
            self.root.arrange(self.root._screen_rect)
        self.invalidate_all()

    def _handle_key(self, event: KeyEvent) -> None:
        # Ctrl+C: abort regardless of current state
        if event.key == Keys.CHAR and event.char == "c" and Modifiers.CTRL in event.modifiers:
            if self.bridge and self.state.status != "idle":
                self.bridge.abort()
                self.schedule_repaint()
            event.mark_handled()
            return

        # Escape: abort if not idle (double-tap for running, single for other stuck states)
        if event.key == Keys.ESCAPE:
            if self.bridge and self.state.status != "idle":
                now = time.monotonic()
                if self.state.cancel_pending and (now - self.state.cancel_pending_time) < 1.0:
                    self.bridge.abort()
                    self.state.cancel_pending = False
                else:
                    self.state.cancel_pending = True
                    self.state.cancel_pending_time = now
                    self.call_later(1.0, self._clear_cancel_pending)
                self.schedule_repaint()
            event.mark_handled()
            return

        # Ctrl+Q: always quit, no matter what
        if event.key == Keys.CHAR and event.char == "q" and Modifiers.CTRL in event.modifiers:
            if self.bridge and self.state.status != "idle":
                self.bridge.force_reset()
            self.quit()
            event.mark_handled()
            return

        # Permission shortcuts when waiting for permission
        if self.state.pending_permission and self.state.screen == "session":
            if event.key == Keys.CHAR and event.char in ("y", "n", "a"):
                allow = event.char in ("y", "a")
                always = event.char == "a"
                if self.bridge:
                    asyncio.create_task(self.bridge.respond_permission(allow, always))
                event.mark_handled()
                return

        # PageUp/PageDown: route to message list
        if event.key in (Keys.PAGE_UP, Keys.PAGE_DOWN) and self.state.screen == "session":
            if self._session_screen:
                msg_list = self._session_screen.find_by_id("message_list")
                if not msg_list:
                    msg_list = getattr(self._session_screen, '_message_list', None)
                if msg_list:
                    self._dispatch_to(event, msg_list)
                    self._needs_repaint = True
                    return

        # Delegate to focused widget
        focused = self._find_focused()
        if focused:
            self._dispatch_to(event, focused)
            self._needs_repaint = True

    def _find_focused(self) -> Widget | None:
        if not self.root:
            return None
        return _find_focused(self.root)

    # ------------------------------------------------------------------
    # Paint overrides — no desktop/taskbar/window_manager
    # ------------------------------------------------------------------

    def _paint(self) -> None:
        if not self._screen or not self.root:
            return
        if not self._needs_repaint:
            return

        buf = self._screen.back
        bg = self._theme_engine.get_color("desktop.bg", Color.from_rgb(22, 22, 30))
        fg = self._theme_engine.get_color("desktop.fg", Color.from_rgb(100, 100, 120))
        buf.clear(fg, bg)

        painter = Painter(buf, Rect(0, 0, self._screen.width, self._screen.height))

        # Layout if needed
        self.root.layout_if_needed()

        # Paint root and all children
        self.root.paint(painter)
        self._paint_widget_tree(painter, self.root)

        self._needs_repaint = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clear_cancel_pending(self) -> None:
        self.state.cancel_pending = False
        self.schedule_repaint()

