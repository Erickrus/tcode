"""tcode theme extending runtui's dark theme."""
from __future__ import annotations

from runtui import Color, ThemeDefinition

_BG = Color.from_rgb(22, 22, 30)
_FG = Color.from_rgb(200, 200, 210)
_SURFACE = Color.from_rgb(30, 30, 40)
_BORDER = Color.from_rgb(55, 55, 70)
_ACCENT = Color.from_rgb(100, 140, 255)
_ACCENT_DIM = Color.from_rgb(60, 90, 180)
_GREEN = Color.from_rgb(80, 200, 120)
_RED = Color.from_rgb(240, 70, 70)
_YELLOW = Color.from_rgb(240, 200, 60)
_DIM = Color.from_rgb(100, 100, 120)
_WHITE = Color.from_rgb(240, 240, 245)
_HEADER_BG = Color.from_rgb(25, 25, 35)
_FOOTER_BG = Color.from_rgb(25, 25, 35)

tcode_theme = ThemeDefinition(
    name="tcode",
    colors={
        # Desktop / root
        "desktop.bg": _BG,
        "desktop.fg": _DIM,

        # Container defaults
        "container.bg": _BG,
        "container.fg": _FG,
        "container.border": _BORDER,
        "container.title": _FG,

        # Label
        "label.fg": _FG,
        "label.bg": _BG,

        # Input
        "input.fg": _FG,
        "input.bg": _SURFACE,
        "input.focused.fg": _WHITE,
        "input.focused.bg": Color.from_rgb(35, 35, 50),
        "input.cursor": _WHITE,
        "input.placeholder": _DIM,

        # Header / footer bars
        "header.bg": _HEADER_BG,
        "header.fg": _FG,
        "footer.bg": _FOOTER_BG,
        "footer.fg": _DIM,

        # Messages
        "message.user.border": _ACCENT,
        "message.user.fg": _WHITE,
        "message.assistant.fg": _FG,

        # Tool status
        "tool.running": _YELLOW,
        "tool.done": _GREEN,
        "tool.error": _RED,

        # Permission
        "permission.border": _YELLOW,
        "permission.fg": _YELLOW,

        # Streaming cursor
        "streaming.cursor": _ACCENT,

        # Scrollbar
        "scrollbar.track": Color.from_rgb(35, 35, 45),
        "scrollbar.thumb": _DIM,

        # Dialog fallbacks
        "dialog.bg": _SURFACE,
        "dialog.fg": _FG,
        "dialog.border": _ACCENT,
        "dialog.title": _WHITE,
    },
    glyphs={
        "scrollbar.thumb": "█",
        "scrollbar.track": "░",
        "scrollbar.up": "▲",
        "scrollbar.down": "▼",
    },
)
