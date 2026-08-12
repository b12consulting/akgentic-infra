"""Warning message widget for conversation display."""

from __future__ import annotations

from rich.console import RenderableType
from rich.text import Text
from textual.widgets import Static


class WarningWidget(Static):
    """Yellow warning message display."""

    DEFAULT_CSS = """
    WarningWidget {
        margin: 0 0 1 0;
        padding: 0 1;
    }
    """

    def __init__(self, content: str) -> None:
        self._content = content
        super().__init__()

    def render(self) -> RenderableType:
        """Render yellow warning message with [warning] prefix."""
        return Text(f"[warning] {self._content}", style="bold yellow")
