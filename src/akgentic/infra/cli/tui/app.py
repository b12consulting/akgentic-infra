"""Top-level Textual application for ak-infra chat."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from akgentic.infra.cli.tui.widgets.chat_input import ChatInput
from akgentic.infra.cli.tui.widgets.hint_bar import HintBar
from akgentic.infra.cli.tui.widgets.status_header import StatusHeader

_CSS_PATH = Path(__file__).parent / "styles" / "chat.tcss"


class ChatApp(App):  # type: ignore[type-arg]
    """Top-level Textual application for ak-infra chat."""

    TITLE = "Akgentic Chat"
    CSS_PATH = _CSS_PATH

    def __init__(
        self,
        team_name: str = "(no team)",
        team_id: str = "",
        team_status: str = "running",
    ) -> None:
        super().__init__()
        self._team_name = team_name
        self._team_id = team_id
        self._team_status = team_status

    def compose(self) -> ComposeResult:
        """Compose the four-zone layout."""
        yield StatusHeader(
            team_name=self._team_name,
            team_id=self._team_id,
            team_status=self._team_status,
        )
        yield VerticalScroll(
            Static(
                "Welcome to Akgentic Chat. Send a message to begin.",
                id="welcome",
            ),
            id="conversation",
        )
        yield ChatInput()
        yield HintBar()
