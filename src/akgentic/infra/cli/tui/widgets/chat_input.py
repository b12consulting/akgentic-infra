"""Multi-line chat input with Enter-to-send, history, and slash completion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import events
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import TextArea

if TYPE_CHECKING:
    from akgentic.infra.cli.commands import CommandRegistry


class ChatInput(TextArea):
    """Multi-line chat input with Enter-to-send, history navigation, and slash completion."""

    DEFAULT_CSS = """
    ChatInput {
        dock: bottom;
        height: auto;
        max-height: 10;
        min-height: 3;
        border-top: solid $accent;
    }
    """

    input_mode: reactive[str] = reactive("chat")

    class Submitted(Message):
        """Fired when user presses Enter (without Shift)."""

        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def __init__(
        self,
        command_registry: CommandRegistry | None = None,
    ) -> None:
        super().__init__(language=None, soft_wrap=True, show_line_numbers=False)
        self._history: list[str] = []
        self._history_idx: int = -1
        self._browsing_history: bool = False
        self._reply_target: str = ""
        self._command_registry = command_registry
        self._palette: CommandPalette | None = None
        self._suppress_palette: bool = False

    def watch_input_mode(self, mode: str) -> None:
        """Update border title based on input mode."""
        labels = {
            "chat": "> ",
            "reply": f"Reply to @{self._reply_target}: ",
            "disconnected": "\\[disconnected] > ",
            "reconnecting": "\\[reconnecting...] > ",
        }
        self.border_title = labels.get(mode, "> ")

    def _show_palette(self) -> None:
        """Mount the CommandPalette if a registry is available."""
        if self._command_registry is None or self._palette is not None:
            return
        palette = CommandPalette(self._command_registry.commands)
        self._palette = palette
        self.mount(palette)

    def _dismiss_palette(self) -> None:
        """Remove the CommandPalette from the DOM."""
        if self._palette is not None:
            self._palette.remove()
            self._palette = None

    def _update_palette_filter(self) -> None:
        """Update palette filter based on current input text."""
        if self._palette is None:
            return
        text = self.text
        if text.startswith("/"):
            self._palette.filter_text = text[1:]
        else:
            self._dismiss_palette()

    def _select_palette_command(self) -> str | None:
        """Select the highlighted palette command and dismiss palette."""
        if self._palette is None:
            return None
        cmd = self._palette.selected_command
        self._dismiss_palette()
        if cmd is not None:
            self._suppress_palette = True
            self.text = f"/{cmd} "
            self.move_cursor_to_end()
        return cmd

    def _submit_text(self) -> None:
        """Submit current text, add to history, and clear input."""
        text = self.text.strip()
        if text:
            self._history.append(text)
            self._history_idx = -1
            self._browsing_history = False
            self.post_message(self.Submitted(text))
            self.clear()

    def _handle_palette_key(self, event: events.Key) -> bool:
        """Handle key events when the palette is visible. Returns True if handled."""
        if self._palette is None:
            return False
        if event.key == "escape":
            event.prevent_default()
            self._dismiss_palette()
            return True
        if event.key == "tab":
            event.prevent_default()
            self._select_palette_command()
            return True
        if event.key == "up":
            event.prevent_default()
            self._palette.move_up()
            return True
        if event.key == "down":
            event.prevent_default()
            self._palette.move_down()
            return True
        if event.key == "enter":
            event.prevent_default()
            cmd = self._select_palette_command()
            if cmd is not None:
                self._submit_text()
            return True
        return False

    def _handle_history_key(self, event: events.Key) -> bool:
        """Handle up/down arrow keys for history navigation. Returns True if handled."""
        if event.key == "up" and (not self.text or self._browsing_history):
            event.prevent_default()
            if self._history:
                if self._history_idx < 0:
                    self._history_idx = len(self._history) - 1
                elif self._history_idx > 0:
                    self._history_idx -= 1
                self._browsing_history = True
                self.text = self._history[self._history_idx]
                self.move_cursor_to_end()
            return True
        if event.key == "down" and self._browsing_history:
            event.prevent_default()
            if self._history_idx < len(self._history) - 1:
                self._history_idx += 1
                self.text = self._history[self._history_idx]
                self.move_cursor_to_end()
            else:
                self._history_idx = -1
                self._browsing_history = False
                self.text = ""
            return True
        return False

    async def _on_key(self, event: events.Key) -> None:
        if self._handle_palette_key(event):
            return
        if event.key == "enter":
            event.prevent_default()
            self._submit_text()
            self._dismiss_palette()
            return
        if self._handle_history_key(event):
            return
        # For non-navigation keys, reset history browsing
        if event.key not in {"up", "down", "enter", "shift+enter", "escape", "tab"}:
            self._browsing_history = False

    def _on_text_area_changed(self, _event: TextArea.Changed) -> None:
        """React to text changes for palette show/dismiss/filter."""
        if self._suppress_palette:
            self._suppress_palette = False
            return
        text = self.text
        if text.startswith("/") and self._command_registry is not None:
            if self._palette is None:
                self._show_palette()
            self._update_palette_filter()
        else:
            self._dismiss_palette()

    def move_cursor_to_end(self) -> None:
        """Move cursor to the end of the text."""
        lines = self.text.split("\n")
        last_line = len(lines) - 1
        last_col = len(lines[-1])
        self.move_cursor((last_line, last_col))
        # Shift+Enter is "shift+enter" key name -- let it pass through for newline


# Import at bottom to avoid circular imports
from akgentic.infra.cli.tui.widgets.command_palette import CommandPalette  # noqa: E402
