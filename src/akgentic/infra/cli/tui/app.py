"""Top-level Textual application for ak-infra chat."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from akgentic.infra.cli.connection import ConnectionState
from akgentic.infra.cli.tui.colors import AgentColorRegistry
from akgentic.infra.cli.tui.messages import ConnectionStateChanged
from akgentic.infra.cli.tui.widgets.chat_input import ChatInput
from akgentic.infra.cli.tui.widgets.hint_bar import HintBar
from akgentic.infra.cli.tui.widgets.status_header import StatusHeader

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from akgentic.infra.cli.connection import ConnectionManager
    from akgentic.infra.cli.event_router import EventRouter

_CSS_PATH = Path(__file__).parent / "styles" / "chat.tcss"


class ChatApp(App[None]):
    """Top-level Textual application for ak-infra chat."""

    TITLE = "Akgentic Chat"
    CSS_PATH = _CSS_PATH

    def __init__(
        self,
        team_name: str = "(no team)",
        team_id: str = "",
        team_status: str = "running",
        connection_manager: ConnectionManager | None = None,
        event_router: EventRouter | None = None,
    ) -> None:
        super().__init__()
        self._team_name = team_name
        self._team_id = team_id
        self._team_status = team_status
        self._connection_manager = connection_manager
        self._event_router = event_router
        self._color_registry = AgentColorRegistry()

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

    def on_mount(self) -> None:
        """Wire connection manager callback and start streaming."""
        if self._connection_manager is not None:
            self._connection_manager._on_state_change = self._on_conn_state_change
        self.stream_events()

    def _on_conn_state_change(self, state: ConnectionState) -> None:
        """Callback for ConnectionManager -- post as Textual message."""
        self.post_message(ConnectionStateChanged(state))

    def on_connection_state_changed(
        self,
        event: ConnectionStateChanged,
    ) -> None:
        """Forward connection state changes to child widgets."""
        state_str = event.state.value
        try:
            self.query_one(StatusHeader).update_connection(state_str)
        except Exception:  # noqa: BLE001
            _log.debug("StatusHeader not available for connection state update")
        try:
            chat_input = self.query_one(ChatInput)
            if state_str == "disconnected":
                chat_input.border_title = "\\[disconnected] > "
            elif state_str == "reconnecting":
                chat_input.border_title = "\\[reconnecting...] > "
            elif state_str == "connected":
                chat_input.border_title = "> "
        except Exception:  # noqa: BLE001
            _log.debug("ChatInput not available for connection state update")

    @work(exclusive=True)
    async def stream_events(self) -> None:
        """Background worker: stream WebSocket events and mount widgets."""
        if self._connection_manager is None or self._event_router is None:
            return
        from akgentic.infra.cli.ws_client import WsConnectionError

        conversation = self.query_one("#conversation", VerticalScroll)
        while True:
            try:
                event_data = await self._connection_manager.receive_event()
                widget = self._event_router.to_widget(event_data, self._color_registry)
                if widget is not None:
                    self._remove_thinking_indicator(conversation)
                    await conversation.mount(widget)
                    widget.scroll_visible(animate=False)
            except WsConnectionError:
                self._remove_thinking_indicator(conversation)
                break

        # Mount disconnection message after worker exits
        from akgentic.infra.cli.tui.widgets.system_message import SystemMessage

        msg = SystemMessage(content="Connection lost. Use /reconnect to try again.")
        await conversation.mount(msg)
        msg.scroll_visible(animate=False)

    def _remove_thinking_indicator(self, conversation: VerticalScroll) -> None:
        """Remove ThinkingIndicator if present."""
        from akgentic.infra.cli.tui.widgets.thinking import ThinkingIndicator

        indicators = conversation.query(ThinkingIndicator)
        for indicator in indicators:
            indicator.remove()

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle user message submission."""
        text = event.text
        if text.startswith("/"):
            # Slash commands -- do NOT mount ThinkingIndicator
            # Slash command dispatch is Story 12.6 -- for now, skip
            return
        conversation = self.query_one("#conversation", VerticalScroll)

        # Remove welcome placeholder on first message
        welcome_nodes = conversation.query("#welcome")
        for node in welcome_nodes:
            node.remove()

        # Mount user message widget
        from akgentic.infra.cli.tui.widgets.user_message import UserMessage

        user_msg = UserMessage(content=text)
        await conversation.mount(user_msg)
        user_msg.scroll_visible(animate=False)

        # Mount ThinkingIndicator
        from akgentic.infra.cli.tui.widgets.thinking import ThinkingIndicator

        indicator = ThinkingIndicator()
        await conversation.mount(indicator)
        indicator.scroll_visible(animate=False)
        # TODO (Story 12.6): Actually send message via API
