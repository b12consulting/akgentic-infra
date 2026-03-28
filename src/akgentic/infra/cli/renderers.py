"""Rich terminal rendering engine for chat events."""

from __future__ import annotations

import json

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


class RichRenderer:
    """Renders chat events with rich formatting, markdown, and color-coded agents."""

    _PALETTE: list[str] = ["cyan", "green", "magenta", "yellow", "blue", "red"]

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console(highlight=False)
        self._agent_colors: dict[str, str] = {}
        self._color_idx: int = 0

    def _get_agent_color(self, agent_name: str) -> str:
        """Return a consistent color for a given agent name (round-robin assignment)."""
        if agent_name not in self._agent_colors:
            self._agent_colors[agent_name] = self._PALETTE[self._color_idx % len(self._PALETTE)]
            self._color_idx += 1
        return self._agent_colors[agent_name]

    def render_agent_message(self, sender: str, content: str) -> None:
        """Render an agent message with color-coded sender and markdown content."""
        color = self._get_agent_color(sender)
        escaped = sender.replace("[", "\\[")
        self._console.print(f"[bold {color}]\\[{escaped}][/bold {color}]")
        self._console.print(Markdown(content))

    def render_error(self, content: str) -> None:
        """Render an error message with red styling."""
        self._console.print(f"[bold red]\\[error][/bold red] {content}")

    def render_tool_call(
        self,
        tool_name: str,
        tool_input: str,
        tool_output: str | None = None,
    ) -> None:
        """Render a tool invocation as a panel with optional syntax highlighting."""
        # Try to pretty-print JSON input
        try:
            parsed = json.loads(tool_input)
            input_text = json.dumps(parsed, indent=2)
        except (json.JSONDecodeError, TypeError):
            input_text = tool_input

        content_lines = f"Input:\n{input_text}"

        if tool_output is not None:
            content_lines += f"\n\nOutput:\n{tool_output}"

        panel = Panel(
            content_lines,
            title=f"Tool: {tool_name}",
            border_style="dim",
        )
        self._console.print(panel)

    def render_human_input_request(self, prompt_text: str) -> None:
        """Render a highlighted human input prompt."""
        panel = Panel(
            prompt_text,
            title="Human Input Required",
            border_style="bold yellow",
        )
        self._console.print(panel)

    def render_history_separator(self) -> None:
        """Render a styled history separator."""
        self._console.rule("history", style="dim")

    def render_system_message(self, text: str) -> None:
        """Render system/status messages in dim style."""
        self._console.print(f"[dim]{text}[/dim]")
