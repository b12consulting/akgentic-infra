"""Tests for ChatSession REPL core."""

from __future__ import annotations

import asyncio
import io
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import websockets.exceptions
from rich.console import Console

from akgentic.infra.cli.client import EventInfo
from akgentic.infra.cli.formatters import OutputFormat
from akgentic.infra.cli.renderers import RichRenderer
from akgentic.infra.cli.repl import ChatSession, _print_event, _render_event_impl


def _captured_renderer() -> tuple[RichRenderer, io.StringIO]:
    """Build a RichRenderer that captures output to a StringIO buffer."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120, no_color=True)
    return RichRenderer(console=console), buf


def _mock_client(**overrides: Any) -> MagicMock:
    """Build a mock ApiClient."""
    mock = MagicMock()
    mock.get_events.return_value = []
    mock.send_message.return_value = None
    for k, v in overrides.items():
        setattr(mock, k, v)
    return mock


def _mock_ws() -> AsyncMock:
    """Build a mock WsClient."""
    ws = AsyncMock()
    ws.__aenter__ = AsyncMock(return_value=ws)
    ws.__aexit__ = AsyncMock(return_value=None)
    ws.receive_event = AsyncMock(side_effect=asyncio.CancelledError)
    return ws


def _make_session(
    client: MagicMock | None = None,
    ws: AsyncMock | None = None,
    renderer: RichRenderer | None = None,
) -> ChatSession:
    """Create a ChatSession with mocked dependencies and optional captured renderer."""
    if client is None:
        client = _mock_client()
    if ws is None:
        ws = _mock_ws()
    return ChatSession(client, ws, "t1", OutputFormat.table, renderer=renderer)


class TestReplayHistory:
    def test_replay_displays_sent_messages(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client(
            get_events=MagicMock(
                return_value=[
                    EventInfo(
                        team_id="t1",
                        sequence=1,
                        event={
                            "__model__": "SentMessage",
                            "sender": "bot",
                            "content": "hello",
                        },
                        timestamp="2026-01-01T00:00:00",
                    ),
                    EventInfo(
                        team_id="t1",
                        sequence=2,
                        event={
                            "__model__": "StateChangedMessage",
                            "state": "running",
                        },
                        timestamp="2026-01-01T00:00:00",
                    ),
                ]
            )
        )
        session = _make_session(client=client, renderer=renderer)
        session._replay_history()
        out = buf.getvalue()
        assert "bot" in out
        assert "hello" in out
        assert "history" in out  # separator
        assert "StateChanged" not in out

    def test_replay_no_events(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)
        session._replay_history()
        out = buf.getvalue()
        assert "history" not in out

    def test_replay_handles_error(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        client.get_events.side_effect = SystemExit(1)
        session = _make_session(client=client, renderer=renderer)
        session._replay_history()  # Should not raise


class TestQuitHandling:
    async def test_quit_command(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)

        with patch("akgentic.infra.cli.repl._read_input", side_effect=["/quit"]):
            await session.run()

        out = buf.getvalue()
        assert "Connected to team t1" in out
        assert "Session closed." in out

    async def test_ctrl_c_exits(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)

        with patch(
            "akgentic.infra.cli.repl._read_input",
            side_effect=KeyboardInterrupt,
        ):
            await session.run()

        out = buf.getvalue()
        assert "Session closed." in out

    async def test_eof_exits(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)

        with patch("akgentic.infra.cli.repl._read_input", side_effect=EOFError):
            await session.run()

        out = buf.getvalue()
        assert "Session closed." in out


class TestMessageSending:
    async def test_sends_message(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)

        with patch(
            "akgentic.infra.cli.repl._read_input",
            side_effect=["hello world", "/quit"],
        ):
            await session.run()

        client.send_message.assert_called_once_with("t1", "hello world")

    async def test_empty_input_skipped(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)

        with patch(
            "akgentic.infra.cli.repl._read_input",
            side_effect=["", "  ", "/quit"],
        ):
            await session.run()

        client.send_message.assert_not_called()


class TestReceiveLoop:
    async def test_renders_sent_message(self) -> None:
        renderer, buf = _captured_renderer()
        events = [
            {"event": {"__model__": "SentMessage", "sender": "agent1", "content": "hi there"}},
        ]
        client = _mock_client()
        ws = _mock_ws()
        call_count = 0

        async def recv_events() -> dict[str, Any]:
            nonlocal call_count
            if call_count < len(events):
                evt = events[call_count]
                call_count += 1
                return evt
            await asyncio.sleep(10)
            return {}

        ws.receive_event = AsyncMock(side_effect=recv_events)
        session = _make_session(client=client, ws=ws, renderer=renderer)

        with patch(
            "akgentic.infra.cli.repl._read_input",
            side_effect=["/quit"],
        ):
            await session.run()

        out = buf.getvalue()
        assert "agent1" in out
        assert "hi there" in out

    async def test_skips_state_changed(self) -> None:
        renderer, buf = _captured_renderer()
        events = [
            {"event": {"__model__": "StateChangedMessage", "state": "running"}},
        ]
        client = _mock_client()
        ws = _mock_ws()
        call_count = 0

        async def recv_events() -> dict[str, Any]:
            nonlocal call_count
            if call_count < len(events):
                evt = events[call_count]
                call_count += 1
                return evt
            await asyncio.sleep(10)
            return {}

        ws.receive_event = AsyncMock(side_effect=recv_events)
        session = _make_session(client=client, ws=ws, renderer=renderer)

        with patch(
            "akgentic.infra.cli.repl._read_input",
            side_effect=["/quit"],
        ):
            await session.run()

        out = buf.getvalue()
        assert "StateChanged" not in out


class TestReceiveLoopCloseCode:
    async def test_close_code_4004_prints_error(self) -> None:
        renderer, buf = _captured_renderer()
        close_frame = MagicMock()
        close_frame.code = 4004
        close_frame.reason = "Team not found"
        exc = websockets.exceptions.ConnectionClosedError(rcvd=close_frame, sent=None)
        client = _mock_client()
        ws = _mock_ws()
        ws.receive_event = AsyncMock(side_effect=exc)
        session = _make_session(client=client, ws=ws, renderer=renderer)

        with patch("akgentic.infra.cli.repl._read_input", side_effect=["/quit"]):
            await session.run()

        out = buf.getvalue()
        assert "team not found" in out

    async def test_close_code_1000_no_error(self) -> None:
        renderer, buf = _captured_renderer()
        close_frame = MagicMock()
        close_frame.code = 1000
        close_frame.reason = "OK"
        exc = websockets.exceptions.ConnectionClosedOK(rcvd=close_frame, sent=None)
        client = _mock_client()
        ws = _mock_ws()
        ws.receive_event = AsyncMock(side_effect=exc)
        session = _make_session(client=client, ws=ws, renderer=renderer)

        with patch("akgentic.infra.cli.repl._read_input", side_effect=["/quit"]):
            await session.run()

        out = buf.getvalue()
        # Should NOT contain connection error — only system messages (Connected, Session closed)
        assert "Connection closed" not in out


class TestRenderEvent:
    def test_sent_message(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event(
            {"event": {"__model__": "SentMessage", "sender": "bot", "content": "reply"}}
        )
        assert result is True
        out = buf.getvalue()
        assert "bot" in out
        assert "reply" in out

    def test_error_message(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event(
            {"event": {"__model__": "ErrorMessage", "content": "something broke"}}
        )
        assert result is True
        out = buf.getvalue()
        assert "error" in out
        assert "something broke" in out

    def test_skip_start_message(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event({"event": {"__model__": "StartMessage"}})
        assert result is False

    def test_skip_received_message(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event({"event": {"__model__": "ReceivedMessage"}})
        assert result is False

    def test_skip_processed_message(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event({"event": {"__model__": "ProcessedMessage"}})
        assert result is False

    def test_sent_message_no_content(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event(
            {"event": {"__model__": "SentMessage", "sender": "bot", "content": ""}}
        )
        assert result is False

    def test_empty_event(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event({})
        assert result is False

    def test_event_message_tool_call(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event(
            {
                "event": {
                    "__model__": "EventMessage",
                    "event": {
                        "tool_name": "search",
                        "args": {"query": "test"},
                    },
                }
            }
        )
        assert result is True
        out = buf.getvalue()
        assert "search" in out

    def test_event_message_human_input(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event(
            {
                "event": {
                    "__model__": "EventMessage",
                    "event": {
                        "__model__": "HumanInputRequest",
                        "prompt": "Enter your name",
                    },
                }
            }
        )
        assert result is True
        out = buf.getvalue()
        assert "Human Input Required" in out
        assert "Enter your name" in out

    def test_event_message_unknown_nested(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event(
            {
                "event": {
                    "__model__": "EventMessage",
                    "event": {"__model__": "SomeOtherEvent", "data": "xyz"},
                }
            }
        )
        assert result is False


class TestPrintEventBackwardCompat:
    """Test the module-level _print_event backward compatibility wrapper."""

    def test_sent_message(self) -> None:
        result = _print_event(
            {"event": {"__model__": "SentMessage", "sender": "bot", "content": "reply"}}
        )
        assert result is True

    def test_skip_unknown(self) -> None:
        result = _print_event({"event": {"__model__": "StartMessage"}})
        assert result is False

    def test_empty_event(self) -> None:
        result = _print_event({})
        assert result is False
