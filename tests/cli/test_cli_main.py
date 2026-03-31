"""Tests for main.py top-level error boundary."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

from akgentic.infra.cli.client import ApiError
from akgentic.infra.cli.ws_client import WsConnectionError


@pytest.fixture()
def _mock_state() -> MagicMock:
    """Patch _state with a mock client and server config."""
    with patch("akgentic.infra.cli.main._state") as state:
        state.client = MagicMock()
        state.server = "http://localhost:8000"
        state.api_key = None
        state.fmt = MagicMock()
        yield state


@pytest.fixture()
def _mock_ws() -> AsyncMock:
    """Patch WsClient constructor."""
    ws = AsyncMock()
    ws.__aenter__ = AsyncMock(return_value=ws)
    ws.__aexit__ = AsyncMock(return_value=None)
    with patch("akgentic.infra.cli.main.WsClient", return_value=ws):
        yield ws


class TestChatErrorBoundary:
    """Test the top-level error boundary in chat()."""

    def test_api_error_renders_error_and_exits(
        self, _mock_state: MagicMock, _mock_ws: AsyncMock
    ) -> None:
        """ApiError during session.run() renders error and exits cleanly."""
        error = ApiError(500, "internal server error")
        with (
            patch("akgentic.infra.cli.main.RichRenderer") as MockRenderer,
            patch(
                "akgentic.infra.cli.main.ChatSession"
            ) as MockSession,
            patch("akgentic.infra.cli.main.asyncio") as mock_asyncio,
        ):
            renderer_instance = MockRenderer.return_value
            mock_asyncio.run.side_effect = error

            from akgentic.infra.cli.main import chat

            with pytest.raises(typer.Exit):
                chat(team_id="t1")

            renderer_instance.render_error.assert_called_once()
            call_args = renderer_instance.render_error.call_args[0][0]
            assert "Server error" in call_args
            _mock_state.client.close.assert_called_once()

    def test_ws_connection_error_renders_error_and_exits(
        self, _mock_state: MagicMock, _mock_ws: AsyncMock
    ) -> None:
        """WsConnectionError during session.run() renders error and exits cleanly."""
        error = WsConnectionError("refused")
        with (
            patch("akgentic.infra.cli.main.RichRenderer") as MockRenderer,
            patch(
                "akgentic.infra.cli.main.ChatSession"
            ) as MockSession,
            patch("akgentic.infra.cli.main.asyncio") as mock_asyncio,
        ):
            renderer_instance = MockRenderer.return_value
            mock_asyncio.run.side_effect = error

            from akgentic.infra.cli.main import chat

            with pytest.raises(typer.Exit):
                chat(team_id="t1")

            renderer_instance.render_error.assert_called_once()
            call_args = renderer_instance.render_error.call_args[0][0]
            assert "Connection failed" in call_args
            _mock_state.client.close.assert_called_once()

    def test_generic_exception_renders_error_and_exits(
        self, _mock_state: MagicMock, _mock_ws: AsyncMock
    ) -> None:
        """Generic Exception during session.run() renders error and exits cleanly."""
        error = RuntimeError("something broke")
        with (
            patch("akgentic.infra.cli.main.RichRenderer") as MockRenderer,
            patch(
                "akgentic.infra.cli.main.ChatSession"
            ) as MockSession,
            patch("akgentic.infra.cli.main.asyncio") as mock_asyncio,
        ):
            renderer_instance = MockRenderer.return_value
            mock_asyncio.run.side_effect = error

            from akgentic.infra.cli.main import chat

            with pytest.raises(typer.Exit):
                chat(team_id="t1")

            renderer_instance.render_error.assert_called_once()
            call_args = renderer_instance.render_error.call_args[0][0]
            assert "Unexpected error" in call_args
            _mock_state.client.close.assert_called_once()

    def test_keyboard_interrupt_exits_cleanly(
        self, _mock_state: MagicMock, _mock_ws: AsyncMock
    ) -> None:
        """KeyboardInterrupt during session.run() exits without error message."""
        with (
            patch("akgentic.infra.cli.main.RichRenderer") as MockRenderer,
            patch(
                "akgentic.infra.cli.main.ChatSession"
            ) as MockSession,
            patch("akgentic.infra.cli.main.asyncio") as mock_asyncio,
        ):
            renderer_instance = MockRenderer.return_value
            mock_asyncio.run.side_effect = KeyboardInterrupt

            from akgentic.infra.cli.main import chat

            # KeyboardInterrupt should be caught silently
            chat(team_id="t1")

            renderer_instance.render_error.assert_not_called()
            _mock_state.client.close.assert_called_once()

    def test_client_close_called_on_success(
        self, _mock_state: MagicMock, _mock_ws: AsyncMock
    ) -> None:
        """client.close() is called even on successful exit."""
        with (
            patch("akgentic.infra.cli.main.RichRenderer"),
            patch(
                "akgentic.infra.cli.main.ChatSession"
            ),
            patch("akgentic.infra.cli.main.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = None

            from akgentic.infra.cli.main import chat

            chat(team_id="t1")
            _mock_state.client.close.assert_called_once()
