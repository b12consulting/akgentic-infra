"""Pilot tests for WebSocket streaming worker, ThinkingIndicator, and connection state."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from akgentic.infra.cli.connection import ConnectionState
from akgentic.infra.cli.tui.app import ChatApp
from akgentic.infra.cli.tui.messages import ConnectionStateChanged
from akgentic.infra.cli.tui.widgets.agent_message import AgentMessage
from akgentic.infra.cli.tui.widgets.chat_input import ChatInput
from akgentic.infra.cli.tui.widgets.status_header import StatusHeader
from akgentic.infra.cli.tui.widgets.system_message import SystemMessage
from akgentic.infra.cli.tui.widgets.thinking import ThinkingIndicator
from akgentic.infra.cli.ws_client import WsConnectionError


def _make_app(
    connection_manager: object | None = None,
    event_router: object | None = None,
) -> ChatApp:
    """Create a ChatApp with optional mock dependencies."""
    return ChatApp(
        team_name="test",
        team_id="123",
        team_status="running",
        connection_manager=connection_manager,  # type: ignore[arg-type]
        event_router=event_router,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# ThinkingIndicator tests (Task 12)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_indicator_renders() -> None:
    """ThinkingIndicator renders spinner text."""
    app = _make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        conv = pilot.app.query_one("#conversation")
        indicator = ThinkingIndicator()
        await conv.mount(indicator)
        rendered = str(indicator.render())
        assert "Agent is thinking" in rendered


@pytest.mark.asyncio
async def test_thinking_indicator_mounted_on_send() -> None:
    """ThinkingIndicator is mounted when user sends a message."""
    app = _make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.click(ChatInput)
        await pilot.press("h", "i")
        await pilot.press("enter")
        await pilot.pause()
        indicators = pilot.app.query(ThinkingIndicator)
        assert len(indicators) == 1


@pytest.mark.asyncio
async def test_no_thinking_indicator_on_slash_command() -> None:
    """ThinkingIndicator is NOT mounted for slash commands."""
    app = _make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.click(ChatInput)
        await pilot.press("/", "h", "e", "l", "p")
        await pilot.press("enter")
        await pilot.pause()
        indicators = pilot.app.query(ThinkingIndicator)
        assert len(indicators) == 0


@pytest.mark.asyncio
async def test_thinking_indicator_removed_on_response() -> None:
    """ThinkingIndicator is removed when first response widget is mounted."""
    import asyncio

    # Use an event to delay the first receive_event until ThinkingIndicator is mounted
    gate = asyncio.Event()

    async def _receive_event_delayed() -> dict[str, Any]:
        # Wait until the test signals that the user message was sent
        await gate.wait()
        gate.clear()
        return {
            "event": {
                "__model__": "SentMessage",
                "sender": "bot",
                "message": {"content": "hi"},
            }
        }

    call_count = 0

    async def _receive_side_effect() -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return await _receive_event_delayed()
        raise WsConnectionError("done", retryable=False)

    mock_cm = MagicMock()
    mock_cm._on_state_change = None
    mock_cm.receive_event = AsyncMock(side_effect=_receive_side_effect)

    mock_router = MagicMock()
    agent_msg = AgentMessage(sender="bot", content="hi", color="cyan")
    mock_router.to_widget = MagicMock(return_value=agent_msg)

    app = _make_app(connection_manager=mock_cm, event_router=mock_router)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        # Send a message to mount ThinkingIndicator
        await pilot.click(ChatInput)
        await pilot.press("h", "i")
        await pilot.press("enter")
        await pilot.pause()

        # Verify ThinkingIndicator is present
        indicators = pilot.app.query(ThinkingIndicator)
        assert len(indicators) == 1

        # Release the gate so the worker can process the event
        gate.set()
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        # ThinkingIndicator should be gone (removed by stream_events)
        indicators = pilot.app.query(ThinkingIndicator)
        assert len(indicators) == 0


# ---------------------------------------------------------------------------
# stream_events worker tests (Task 11)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_events_mounts_widgets() -> None:
    """stream_events worker mounts AgentMessage widgets."""
    mock_cm = MagicMock()
    mock_cm._on_state_change = None
    mock_cm.receive_event = AsyncMock(
        side_effect=[
            {
                "event": {
                    "__model__": "SentMessage",
                    "sender": "bot",
                    "message": {"content": "hello"},
                }
            },
            WsConnectionError("done", retryable=False),
        ]
    )

    agent_msg = AgentMessage(sender="bot", content="hello", color="cyan")
    mock_router = MagicMock()
    mock_router.to_widget = MagicMock(return_value=agent_msg)

    app = _make_app(connection_manager=mock_cm, event_router=mock_router)
    async with app.run_test(size=(80, 24)) as pilot:
        # Wait for worker to start and process
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        # Verify AgentMessage was mounted
        messages = pilot.app.query(AgentMessage)
        assert len(messages) >= 1


@pytest.mark.asyncio
async def test_stream_events_exits_on_ws_error() -> None:
    """Worker exits gracefully on WsConnectionError, mounts SystemMessage."""
    mock_cm = MagicMock()
    mock_cm._on_state_change = None
    mock_cm.receive_event = AsyncMock(
        side_effect=WsConnectionError("disconnected", retryable=False)
    )

    mock_router = MagicMock()
    mock_router.to_widget = MagicMock(return_value=None)

    app = _make_app(connection_manager=mock_cm, event_router=mock_router)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        # SystemMessage "Connection lost" should be mounted
        sys_msgs = pilot.app.query(SystemMessage)
        found = any("Connection lost" in m._content for m in sys_msgs)
        assert found, "Expected 'Connection lost' SystemMessage after WsConnectionError"


@pytest.mark.asyncio
async def test_stream_events_skips_none_widgets() -> None:
    """Worker skips events where to_widget returns None."""
    mock_cm = MagicMock()
    mock_cm._on_state_change = None
    mock_cm.receive_event = AsyncMock(
        side_effect=[
            {"event": {"__model__": "UnknownEvent"}},
            WsConnectionError("done", retryable=False),
        ]
    )

    mock_router = MagicMock()
    mock_router.to_widget = MagicMock(return_value=None)

    app = _make_app(connection_manager=mock_cm, event_router=mock_router)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        # No AgentMessage should be mounted
        messages = pilot.app.query(AgentMessage)
        assert len(messages) == 0


@pytest.mark.asyncio
async def test_no_streaming_without_deps() -> None:
    """stream_events returns immediately without connection_manager/event_router."""
    app = _make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        # No crash, app runs fine without deps
        assert pilot.app.query_one("#conversation") is not None


# ---------------------------------------------------------------------------
# Connection state propagation tests (Task 13)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_header_updates_on_disconnected() -> None:
    """StatusHeader updates connection indicator on DISCONNECTED."""
    app = _make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        header = pilot.app.query_one(StatusHeader)
        pilot.app.post_message(ConnectionStateChanged(ConnectionState.DISCONNECTED))
        await pilot.pause()
        assert header._connection_state == "disconnected"


@pytest.mark.asyncio
async def test_status_header_updates_on_reconnecting() -> None:
    """StatusHeader updates connection indicator on RECONNECTING."""
    app = _make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        header = pilot.app.query_one(StatusHeader)
        pilot.app.post_message(ConnectionStateChanged(ConnectionState.RECONNECTING))
        await pilot.pause()
        assert header._connection_state == "reconnecting"


@pytest.mark.asyncio
async def test_status_header_updates_on_connected() -> None:
    """StatusHeader restores connected state."""
    app = _make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        header = pilot.app.query_one(StatusHeader)
        # Set to disconnected first
        pilot.app.post_message(ConnectionStateChanged(ConnectionState.DISCONNECTED))
        await pilot.pause()
        assert header._connection_state == "disconnected"
        # Restore to connected
        pilot.app.post_message(ConnectionStateChanged(ConnectionState.CONNECTED))
        await pilot.pause()
        await pilot.pause()
        assert header._connection_state == "connected"


@pytest.mark.asyncio
async def test_chat_input_updates_on_disconnected() -> None:
    """ChatInput updates border_title on DISCONNECTED."""
    app = _make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        chat_input = pilot.app.query_one(ChatInput)
        pilot.app.post_message(ConnectionStateChanged(ConnectionState.DISCONNECTED))
        await pilot.pause()
        assert chat_input.border_title == "\\[disconnected] > "


@pytest.mark.asyncio
async def test_chat_input_updates_on_reconnecting() -> None:
    """ChatInput updates border_title on RECONNECTING."""
    app = _make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        chat_input = pilot.app.query_one(ChatInput)
        pilot.app.post_message(ConnectionStateChanged(ConnectionState.RECONNECTING))
        await pilot.pause()
        assert chat_input.border_title == "\\[reconnecting...] > "


@pytest.mark.asyncio
async def test_chat_input_updates_on_connected() -> None:
    """ChatInput restores default border_title on CONNECTED."""
    app = _make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        chat_input = pilot.app.query_one(ChatInput)
        # Set to disconnected first
        pilot.app.post_message(ConnectionStateChanged(ConnectionState.DISCONNECTED))
        await pilot.pause()
        assert chat_input.border_title == "\\[disconnected] > "
        # Restore to connected
        pilot.app.post_message(ConnectionStateChanged(ConnectionState.CONNECTED))
        await pilot.pause()
        await pilot.pause()
        assert chat_input.border_title == "> "


@pytest.mark.asyncio
async def test_welcome_removed_on_first_message() -> None:
    """Welcome placeholder is removed when first message is sent."""
    app = _make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        # Welcome should exist
        welcome = pilot.app.query("#welcome")
        assert len(welcome) == 1
        # Send a message
        await pilot.click(ChatInput)
        await pilot.press("h", "i")
        await pilot.press("enter")
        await pilot.pause()
        # Welcome should be removed
        welcome = pilot.app.query("#welcome")
        assert len(welcome) == 0
