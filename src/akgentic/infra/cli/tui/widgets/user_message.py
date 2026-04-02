"""User message widget for conversation display."""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.text import Text
from textual.widgets import Static


class UserMessage(Static):
    """The user's own message in the conversation."""

    DEFAULT_CSS = """
    UserMessage {
        margin: 0 0 1 6;
        padding: 0 1;
        background: $boost;
    }
    """

    def __init__(self, content: str, timestamp: str | None = None) -> None:
        self._content = content
        self._timestamp = timestamp
        super().__init__()

    def on_mount(self) -> None:
        """Apply right border accent for visual distinction."""
        self.styles.border_right = ("tall", "white")

    def render(self) -> RenderableType:
        """Render user message with [You] prefix and optional timestamp."""
        header = Text("[You]", style="bold white")
        if self._timestamp:
            header.append(f"  {self._timestamp}", style="dim")
        body = Text(self._content)
        return Group(header, body)
