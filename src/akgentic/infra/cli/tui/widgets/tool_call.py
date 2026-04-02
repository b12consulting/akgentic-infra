"""Tool call widget for conversation display."""

from __future__ import annotations

import json

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


class ToolCallWidget(Static):
    """A tool invocation -- collapsed by default, expandable on click."""

    DEFAULT_CSS = """
    ToolCallWidget {
        margin: 0 0 1 2;
        padding: 0 1;
        color: $text-muted;
    }
    """

    collapsed: reactive[bool] = reactive(True)

    def __init__(self, tool_name: str, tool_input: str, tool_output: str | None) -> None:
        self._tool_name = tool_name
        self._tool_input = tool_input
        self._tool_output = tool_output
        super().__init__(Text(f"\u25b8 Tool: {tool_name}", style="dim"))

    def _build_collapsed(self) -> RenderableType:
        """Build the collapsed one-line summary."""
        return Text(f"\u25b8 Tool: {self._tool_name}", style="dim")

    def _build_expanded(self) -> RenderableType:
        """Build the expanded panel with JSON input/output."""
        parts: list[Text | Syntax] = [Text("Input:")]
        try:
            parsed = json.loads(self._tool_input)
            formatted = json.dumps(parsed, indent=2)
            parts.append(Syntax(formatted, "json", theme="monokai"))
        except (json.JSONDecodeError, TypeError):
            parts.append(Text(self._tool_input))
        if self._tool_output is not None:
            parts.append(Text("\nOutput:"))
            parts.append(Text(self._tool_output))
        return Panel(
            Group(*parts),
            title=f"Tool: {self._tool_name}",
            border_style="dim",
        )

    def watch_collapsed(self, value: bool) -> None:
        """Re-render when collapsed state changes."""
        self.update(self._build_collapsed() if value else self._build_expanded())

    def on_click(self) -> None:
        """Toggle collapsed state."""
        self.collapsed = not self.collapsed
