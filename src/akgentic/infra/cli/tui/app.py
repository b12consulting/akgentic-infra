"""Top-level Textual application for ak-infra chat."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from akgentic.infra.cli.client import ApiError
from akgentic.infra.cli.connection import ConnectionState
from akgentic.infra.cli.tui.colors import AgentColorRegistry
from akgentic.infra.cli.tui.command_adapter import TuiCommandAdapter
from akgentic.infra.cli.tui.messages import ConnectionStateChanged
from akgentic.infra.cli.tui.widgets.chat_input import ChatInput
from akgentic.infra.cli.tui.widgets.hint_bar import HintBar
from akgentic.infra.cli.tui.widgets.status_header import StatusHeader

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from akgentic.infra.cli.client import ApiClient
    from akgentic.infra.cli.commands import CommandRegistry
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
        command_registry: CommandRegistry | None = None,
        client: ApiClient | None = None,
    ) -> None:
        super().__init__()
        self._team_name = team_name
        self._team_id = team_id
        self._team_status = team_status
        self._connection_manager = connection_manager
        self._event_router = event_router
        self._command_registry = command_registry
        self._client = client
        self._color_registry = AgentColorRegistry()
        self._tui_adapter = TuiCommandAdapter(command_registry) if command_registry else None

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
        yield ChatInput(command_registry=self._command_registry)
        yield HintBar()

    def on_mount(self) -> None:
        """Wire connection manager callback and start streaming."""
        if not self._team_id:
            self._select_team()
            return
        if self._connection_manager is not None:
            self._connection_manager._on_state_change = self._on_conn_state_change
        self.stream_events()

    @work(exclusive=True)
    async def _select_team(self) -> None:
        """Push TeamSelectScreen and wait for result."""
        from akgentic.infra.cli.tui.screens.team_select import TeamSelectScreen

        team_id = await self.push_screen_wait(TeamSelectScreen(client=self._client))
        if team_id is None:
            self.exit()
            return
        # Fetch team info and update header
        if self._client is not None:
            team_info = self._client.get_team(team_id)
            self._team_id = team_id
            self._team_name = team_info.name
            self._team_status = team_info.status
            self.query_one(StatusHeader).update_team(team_info.name, team_id, team_info.status)
        else:
            self._team_id = team_id
        if self._connection_manager is not None:
            self._connection_manager._on_state_change = self._on_conn_state_change
            # Switch to the selected team (updates ConnectionManager's team_id)
            await self._connection_manager.switch_team(team_id)
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
                chat_input.input_mode = "disconnected"
            elif state_str == "reconnecting":
                chat_input.input_mode = "reconnecting"
            elif state_str == "connected":
                chat_input.input_mode = "chat"
        except Exception:  # noqa: BLE001
            _log.debug("ChatInput not available for connection state update")

    @work(exclusive=True)
    async def stream_events(self) -> None:
        """Background worker: stream WebSocket events and mount widgets."""
        if self._connection_manager is None or self._event_router is None:
            return
        from akgentic.infra.cli.ws_client import WsConnectionError

        # Ensure WebSocket is connected before entering the receive loop.
        # ConnectionManager.receive_event() raises immediately if not connected,
        # unlike the old REPL which called connect() explicitly before looping.
        if self._connection_manager.state == ConnectionState.DISCONNECTED:
            try:
                await self._connection_manager.connect()
            except WsConnectionError:
                pass  # Fall through — receive loop will show "Connection lost"

        conversation = self.query_one("#conversation", VerticalScroll)
        while True:
            try:
                event_data = await self._connection_manager.receive_event()
                _log.debug("WS event received: %s", event_data)
                widget = self._event_router.to_widget(event_data, self._color_registry)
                _log.debug("to_widget result: %s", type(widget).__name__ if widget else None)
                if widget is not None:
                    self._remove_thinking_indicator(conversation)
                    await conversation.mount(widget)
                    widget.scroll_visible(animate=False)
            except WsConnectionError as exc:
                _log.debug("WS connection error in stream loop: %s", exc)
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
            if self._tui_adapter is not None:
                await self._tui_adapter.dispatch(text, self)
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

        # Send message via API in background thread
        self._send_message(text)

    @work(thread=True)
    def _send_message(self, text: str) -> None:
        """Send a user message via the REST API in a background thread."""
        if self._client is not None:
            try:
                self._client.send_message(self._team_id, text)
            except ApiError as exc:
                self.call_from_thread(self._mount_send_error, str(exc.detail))

    def _mount_send_error(self, detail: str) -> None:
        """Schedule ErrorWidget mount for a failed message send (called from thread)."""
        self.run_worker(self._do_mount_send_error(detail), exclusive=False)

    async def _do_mount_send_error(self, detail: str) -> None:
        """Mount an ErrorWidget for a failed message send."""
        from akgentic.infra.cli.tui.widgets.error import ErrorWidget

        conversation = self.query_one("#conversation", VerticalScroll)
        widget = ErrorWidget(content=f"Failed to send message: {detail}")
        await conversation.mount(widget)
        widget.scroll_visible(animate=False)
