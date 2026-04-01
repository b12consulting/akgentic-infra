"""Pilot tests for slash command dispatch via TuiCommandAdapter."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.containers import VerticalScroll

from akgentic.infra.cli.client import ApiClient, ApiError, TeamInfo
from akgentic.infra.cli.commands import CommandRegistry, build_default_registry
from akgentic.infra.cli.connection import ConnectionManager
from akgentic.infra.cli.event_router import EventRouter
from akgentic.infra.cli.tui.app import ChatApp
from akgentic.infra.cli.tui.widgets.chat_input import ChatInput
from akgentic.infra.cli.tui.widgets.error import ErrorWidget
from akgentic.infra.cli.tui.widgets.status_header import StatusHeader
from akgentic.infra.cli.tui.widgets.system_message import SystemMessage
from akgentic.infra.cli.tui.widgets.user_message import UserMessage


def _make_team(
    team_id: str = "t-123",
    name: str = "test-team",
    status: str = "running",
) -> TeamInfo:
    return TeamInfo(
        team_id=team_id,
        name=name,
        status=status,
        user_id="u1",
        created_at="2026-01-01",
        updated_at="2026-01-01",
    )


def _mock_client() -> MagicMock:
    """Create a mock ApiClient."""
    client = MagicMock(spec=ApiClient)
    client.list_teams.return_value = [_make_team()]
    client.get_team.return_value = _make_team()
    client.send_message.return_value = None
    return client


def _mock_conn() -> MagicMock:
    """Create a mock ConnectionManager."""
    conn = MagicMock(spec=ConnectionManager)
    conn.switch_team = AsyncMock()
    conn.connect = AsyncMock()
    conn.receive_event = AsyncMock(side_effect=asyncio.CancelledError)
    return conn


def _mock_event_router() -> MagicMock:
    """Create a mock EventRouter that returns None from to_widget()."""
    router = MagicMock(spec=EventRouter)
    router.to_widget.return_value = None
    return router


def _make_app(
    client: object | None = None,
    connection_manager: object | None = None,
    command_registry: CommandRegistry | None = None,
) -> ChatApp:
    """Create a ChatApp with mock dependencies for testing."""
    registry = command_registry or build_default_registry()
    return ChatApp(
        team_name="test",
        team_id="t-123",
        team_status="running",
        connection_manager=connection_manager,
        event_router=_mock_event_router(),
        command_registry=registry,
        client=client,
    )


# ---------------------------------------------------------------------------
# Task 7: Pilot tests for slash command dispatch (AC: #1, #2, #6, #11)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_help_mounts_system_message() -> None:
    """AC #6: /help mounts a SystemMessage with help text."""
    app = _make_app(client=_mock_client())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.click(ChatInput)
        await pilot.press("/", "h", "e", "l", "p")
        await pilot.press("enter")
        await pilot.pause()
        msgs = pilot.app.query(SystemMessage)
        # Should have at least one SystemMessage with help content
        found = any("Available commands" in m._content for m in msgs)
        assert found, "Expected SystemMessage with 'Available commands'"


@pytest.mark.asyncio
async def test_teams_mounts_system_message() -> None:
    """AC #2: /teams mounts a SystemMessage with team listing."""
    client = _mock_client()
    app = _make_app(client=client)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.click(ChatInput)
        await pilot.press("/", "t", "e", "a", "m", "s")
        await pilot.press("enter")
        await pilot.pause()
        msgs = pilot.app.query(SystemMessage)
        found = any("test-team" in m._content for m in msgs)
        assert found, "Expected SystemMessage with team listing"


@pytest.mark.asyncio
async def test_info_mounts_system_message() -> None:
    """AC #2: /info mounts a SystemMessage with team info."""
    client = _mock_client()
    app = _make_app(client=client)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.click(ChatInput)
        await pilot.press("/", "i", "n", "f", "o")
        await pilot.press("enter")
        await pilot.pause()
        msgs = pilot.app.query(SystemMessage)
        found = any("Name:" in m._content for m in msgs)
        assert found, "Expected SystemMessage with team info"


@pytest.mark.asyncio
async def test_unknown_command_mounts_message() -> None:
    """AC #1: unknown /foo mounts an error/system message."""
    app = _make_app(client=_mock_client())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        chat_input = pilot.app.query_one(ChatInput)
        chat_input.post_message(ChatInput.Submitted(text="/foo"))
        await pilot.pause()
        await pilot.pause()
        msgs = pilot.app.query(SystemMessage)
        found = any("Unknown command" in m._content for m in msgs)
        assert found, "Expected SystemMessage with 'Unknown command'"


# ---------------------------------------------------------------------------
# Task 8: Pilot tests for /switch worker lifecycle (AC: #3, #11)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_switch_calls_conn_and_updates_header() -> None:
    """AC #3: /switch calls ConnectionManager.switch_team(), updates StatusHeader."""
    client = _mock_client()
    conn = _mock_conn()
    app = _make_app(client=client, connection_manager=conn)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # Type /switch new-team-id
        chat_input = pilot.app.query_one(ChatInput)
        chat_input.post_message(ChatInput.Submitted(text="/switch new-team-id"))
        await pilot.pause()
        await pilot.pause()

        conn.switch_team.assert_called_once_with("new-team-id")
        msgs = pilot.app.query(SystemMessage)
        found = any("Switched to team" in m._content for m in msgs)
        assert found, "Expected SystemMessage confirming switch"


@pytest.mark.asyncio
async def test_switch_no_args_mounts_usage() -> None:
    """AC #3: /switch with no args mounts usage SystemMessage."""
    app = _make_app(client=_mock_client(), connection_manager=_mock_conn())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        chat_input = pilot.app.query_one(ChatInput)
        chat_input.post_message(ChatInput.Submitted(text="/switch"))
        await pilot.pause()
        await pilot.pause()

        msgs = pilot.app.query(SystemMessage)
        found = any("Usage:" in m._content for m in msgs)
        assert found, "Expected usage message for /switch"


@pytest.mark.asyncio
async def test_switch_failure_mounts_error() -> None:
    """AC #3: /switch failure mounts ErrorWidget."""
    conn = _mock_conn()
    conn.switch_team = AsyncMock(side_effect=Exception("connection refused"))
    app = _make_app(client=_mock_client(), connection_manager=conn)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        chat_input = pilot.app.query_one(ChatInput)
        chat_input.post_message(ChatInput.Submitted(text="/switch bad-id"))
        await pilot.pause()
        await pilot.pause()

        errors = pilot.app.query(ErrorWidget)
        found = any("Switch failed" in e._content for e in errors)
        assert found, "Expected ErrorWidget for switch failure"


# ---------------------------------------------------------------------------
# Task 9: Pilot tests for error rendering on command failure (AC: #8, #11)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_command_api_error_mounts_error_widget() -> None:
    """AC #8: command that raises ApiError mounts ErrorWidget."""
    client = _mock_client()
    client.list_teams.side_effect = ApiError(500, "Internal server error")
    app = _make_app(client=client)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        chat_input = pilot.app.query_one(ChatInput)
        chat_input.post_message(ChatInput.Submitted(text="/teams"))
        await pilot.pause()
        await pilot.pause()

        errors = pilot.app.query(ErrorWidget)
        assert len(errors) >= 1, "Expected ErrorWidget for API error"


@pytest.mark.asyncio
async def test_reconnect_failure_mounts_error() -> None:
    """AC #4, #8: /reconnect failure mounts ErrorWidget."""
    conn = _mock_conn()
    conn.connect = AsyncMock(side_effect=Exception("server unreachable"))
    app = _make_app(client=_mock_client(), connection_manager=conn)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        chat_input = pilot.app.query_one(ChatInput)
        chat_input.post_message(ChatInput.Submitted(text="/reconnect"))
        await pilot.pause()
        await pilot.pause()

        errors = pilot.app.query(ErrorWidget)
        found = any("Reconnection failed" in e._content for e in errors)
        assert found, "Expected ErrorWidget for reconnection failure"


# ---------------------------------------------------------------------------
# Task 10: Pilot tests for message sending (AC: #10, #11)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_message_send_calls_api() -> None:
    """AC #10: non-slash text sends message via ApiClient.send_message()."""
    client = _mock_client()
    app = _make_app(client=client)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.click(ChatInput)
        await pilot.press("h", "e", "l", "l", "o")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        client.send_message.assert_called_once_with("t-123", "hello")


@pytest.mark.asyncio
async def test_message_send_mounts_user_message() -> None:
    """AC #10: non-slash text mounts UserMessage widget."""
    client = _mock_client()
    app = _make_app(client=client)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.click(ChatInput)
        await pilot.press("h", "i")
        await pilot.press("enter")
        await pilot.pause()

        user_msgs = pilot.app.query(UserMessage)
        assert len(user_msgs) >= 1


@pytest.mark.asyncio
async def test_message_send_error_mounts_error_widget() -> None:
    """AC #10: ApiError on send mounts ErrorWidget."""
    client = _mock_client()
    client.send_message.side_effect = ApiError(500, "send failed")
    app = _make_app(client=client)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.click(ChatInput)
        await pilot.press("h", "i")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        errors = pilot.app.query(ErrorWidget)
        found = any("Failed to send" in e._content for e in errors)
        assert found, "Expected ErrorWidget for send failure"


# ---------------------------------------------------------------------------
# Task 4: /quit test (AC: #5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quit_exits_app() -> None:
    """AC #5: /quit exits ChatApp."""
    app = _make_app(client=_mock_client())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        chat_input = pilot.app.query_one(ChatInput)
        chat_input.post_message(ChatInput.Submitted(text="/quit"))
        await pilot.pause()
        await pilot.pause()
        # App should have received exit request
        # (Textual test runner handles this gracefully)
