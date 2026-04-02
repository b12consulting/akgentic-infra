"""User message widget for conversation display."""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.text import Text
from textual.widgets import Static


class UserMessage(Static):
    """The user's own message in the conversation."""

    DEFAULT_CSS = """
    UserMessage {
        margin: 0 0 1 0;
        padding: 0 1;
    }
    """

    def __init__(self, content: str) -> None:
        self._content = content
        super().__init__()

    def render(self) -> RenderableType:
        """Render user message with [You] prefix."""
        sender = Text("[You]", style="bold white")
        body = Text(self._content)
        return Group(sender, body)
