"""Multi-line chat input with Enter-to-send."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import events
from textual.message import Message
from textual.widgets import TextArea

if TYPE_CHECKING:
    from akgentic.infra.cli.tui.messages import ConnectionStateChanged


class ChatInput(TextArea):
    """Multi-line chat input with Enter-to-send."""

    DEFAULT_CSS = """
    ChatInput {
        dock: bottom;
        height: auto;
        max-height: 10;
        min-height: 3;
        border-top: solid $accent;
    }
    """

    class Submitted(Message):
        """Fired when user presses Enter (without Shift)."""

        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def __init__(self) -> None:
        super().__init__(language=None, soft_wrap=True, show_line_numbers=False)
        self._history: list[str] = []
        self._history_idx: int = -1

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            text = self.text.strip()
            if text:
                self._history.append(text)
                self._history_idx = -1
                self.post_message(self.Submitted(text))
                self.clear()
        # Shift+Enter is "shift+enter" key name -- let it pass through for newline

    def on_connection_state_changed(
        self,
        event: ConnectionStateChanged,
    ) -> None:
        """Update input mode based on connection state."""
        state = event.state.value
        if state == "disconnected":
            self.border_title = "\\[disconnected] > "
        elif state == "reconnecting":
            self.border_title = "\\[reconnecting...] > "
        elif state == "connected":
            self.border_title = "> "
