"""Multi-line chat input with Enter-to-send."""

from __future__ import annotations

from textual import events
from textual.message import Message
from textual.widgets import TextArea


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

