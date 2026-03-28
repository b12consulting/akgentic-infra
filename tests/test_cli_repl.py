"""Tests for ChatSession REPL core."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets.exceptions

from akgentic.infra.cli.formatters import OutputFormat
from akgentic.infra.cli.repl import ChatSession, _print_event


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


class TestReplayHistory:
    def test_replay_displays_sent_messages(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client(
            get_events=MagicMock(
                return_value=[
                    {
                        "event": {
                            "__model__": "SentMessage",
                            "sender": "bot",
                            "content": "hello",
                        }
                    },
                    {
                        "event": {
                            "__model__": "StateChangedMessage",
                            "state": "running",
                        }
                    },
                ]
            )
        )
        ws = _mock_ws()
        session = ChatSession(client, ws, "t1", OutputFormat.table)
        session._replay_history()
        out = capsys.readouterr().out
        assert "[bot] hello" in out
        assert "--- history ---" in out
        assert "StateChanged" not in out

    def test_replay_no_events(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        ws = _mock_ws()
        session = ChatSession(client, ws, "t1", OutputFormat.table)
        session._replay_history()
        out = capsys.readouterr().out
        assert "--- history ---" not in out

    def test_replay_handles_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.get_events.side_effect = SystemExit(1)
        ws = _mock_ws()
        session = ChatSession(client, ws, "t1", OutputFormat.table)
        session._replay_history()  # Should not raise


class TestQuitHandling:
    async def test_quit_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        ws = _mock_ws()
        session = ChatSession(client, ws, "t1", OutputFormat.table)

        with patch("akgentic.infra.cli.repl._read_input", side_effect=["/quit"]):
            await session.run()

        out = capsys.readouterr().out
        assert "Connected to team t1" in out
        assert "Session closed." in out

    async def test_ctrl_c_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        ws = _mock_ws()
        session = ChatSession(client, ws, "t1", OutputFormat.table)

        with patch(
            "akgentic.infra.cli.repl._read_input",
            side_effect=KeyboardInterrupt,
        ):
            await session.run()

        out = capsys.readouterr().out
        assert "Session closed." in out

    async def test_eof_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        ws = _mock_ws()
        session = ChatSession(client, ws, "t1", OutputFormat.table)

        with patch("akgentic.infra.cli.repl._read_input", side_effect=EOFError):
            await session.run()

        out = capsys.readouterr().out
        assert "Session closed." in out


class TestMessageSending:
    async def test_sends_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        ws = _mock_ws()
        session = ChatSession(client, ws, "t1", OutputFormat.table)

        with patch(
            "akgentic.infra.cli.repl._read_input",
            side_effect=["hello world", "/quit"],
        ):
            await session.run()

        client.send_message.assert_called_once_with("t1", "hello world")

    async def test_empty_input_skipped(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        ws = _mock_ws()
        session = ChatSession(client, ws, "t1", OutputFormat.table)

        with patch(
            "akgentic.infra.cli.repl._read_input",
            side_effect=["", "  ", "/quit"],
        ):
            await session.run()

        client.send_message.assert_not_called()


class TestReceiveLoop:
    async def test_prints_sent_message(self, capsys: pytest.CaptureFixture[str]) -> None:
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
        session = ChatSession(client, ws, "t1", OutputFormat.table)

        with patch(
            "akgentic.infra.cli.repl._read_input",
            side_effect=["/quit"],
        ):
            await session.run()

        out = capsys.readouterr().out
        assert "[agent1] hi there" in out

    async def test_skips_state_changed(self, capsys: pytest.CaptureFixture[str]) -> None:
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
        session = ChatSession(client, ws, "t1", OutputFormat.table)

        with patch(
            "akgentic.infra.cli.repl._read_input",
            side_effect=["/quit"],
        ):
            await session.run()

        out = capsys.readouterr().out
        assert "StateChanged" not in out


class TestReceiveLoopCloseCode:
    async def test_close_code_4004_prints_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        close_frame = MagicMock()
        close_frame.code = 4004
        close_frame.reason = "Team not found"
        exc = websockets.exceptions.ConnectionClosedError(rcvd=close_frame, sent=None)
        client = _mock_client()
        ws = _mock_ws()
        ws.receive_event = AsyncMock(side_effect=exc)
        session = ChatSession(client, ws, "t1", OutputFormat.table)

        with patch("akgentic.infra.cli.repl._read_input", side_effect=["/quit"]):
            await session.run()

        err = capsys.readouterr().err
        assert "team not found" in err

    async def test_close_code_1000_no_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        close_frame = MagicMock()
        close_frame.code = 1000
        close_frame.reason = "OK"
        exc = websockets.exceptions.ConnectionClosedOK(rcvd=close_frame, sent=None)
        client = _mock_client()
        ws = _mock_ws()
        ws.receive_event = AsyncMock(side_effect=exc)
        session = ChatSession(client, ws, "t1", OutputFormat.table)

        with patch("akgentic.infra.cli.repl._read_input", side_effect=["/quit"]):
            await session.run()

        err = capsys.readouterr().err
        assert err == ""


class TestPrintEvent:
    def test_sent_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = _print_event(
            {"event": {"__model__": "SentMessage", "sender": "bot", "content": "reply"}}
        )
        assert result is True
        assert "[bot] reply" in capsys.readouterr().out

    def test_error_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = _print_event(
            {"event": {"__model__": "ErrorMessage", "content": "something broke"}}
        )
        assert result is True
        assert "[error] something broke" in capsys.readouterr().out

    def test_skip_start_message(self) -> None:
        result = _print_event({"event": {"__model__": "StartMessage"}})
        assert result is False

    def test_skip_received_message(self) -> None:
        result = _print_event({"event": {"__model__": "ReceivedMessage"}})
        assert result is False

    def test_skip_processed_message(self) -> None:
        result = _print_event({"event": {"__model__": "ProcessedMessage"}})
        assert result is False

    def test_sent_message_no_content(self) -> None:
        result = _print_event(
            {"event": {"__model__": "SentMessage", "sender": "bot", "content": ""}}
        )
        assert result is False

    def test_empty_event(self) -> None:
        result = _print_event({})
        assert result is False
