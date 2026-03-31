"""Agent message widget for conversation display."""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.text import Text
from textual.widgets import Static


class AgentMessage(Static):
    """A single agent message in the conversation."""

    DEFAULT_CSS = """
    AgentMessage {
        margin: 0 0 1 0;
        padding: 0 1;
    }
    """

    def __init__(self, sender: str, content: str, color: str) -> None:
        self._sender = sender
        self._content = content
        self._color = color
        super().__init__()

    def render(self) -> RenderableType:
        """Render sender name with color and markdown body."""
        sender_text = Text(f"[@{self._sender}]", style=f"bold {self._color}")
        body = Markdown(self._content)
        return Group(sender_text, body)
