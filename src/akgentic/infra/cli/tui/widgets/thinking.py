"""Animated spinner shown while waiting for agent response."""

from __future__ import annotations

from rich.console import RenderableType
from rich.text import Text
from textual.widgets import Static


class ThinkingIndicator(Static):
    """Animated spinner shown while waiting for agent response."""

    DEFAULT_CSS = """
    ThinkingIndicator {
        margin: 0 0 1 0;
        padding: 0 1;
    }
    """

    _FRAMES = list("\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f")
    _INTERVAL = 0.1

    def __init__(self) -> None:
        super().__init__()
        self._frame_idx: int = 0
        self._timer_handle: object | None = None

    def on_mount(self) -> None:
        """Start the animation timer when mounted."""
        self._timer_handle = self.set_interval(self._INTERVAL, self._advance_frame)

    def _advance_frame(self) -> None:
        """Advance to the next spinner frame."""
        self._frame_idx = (self._frame_idx + 1) % len(self._FRAMES)
        self.refresh()

    def render(self) -> RenderableType:
        """Render spinner frame with thinking text."""
        frame = self._FRAMES[self._frame_idx]
        return Text(f"{frame} Agent is thinking...", style="dim italic")
