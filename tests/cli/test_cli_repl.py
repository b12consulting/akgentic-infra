"""Tests for ChatSession REPL core."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import websockets.exceptions

from akgentic.infra.cli.client import EventInfo
from akgentic.infra.cli.formatters import OutputFormat
from akgentic.infra.cli.renderers import RichRenderer
from akgentic.infra.cli.commands import build_default_registry
from akgentic.infra.cli.repl import ChatSession, _SlashCompleter, _print_event, _render_event_impl

from tests.fixtures.events import (
    make_error_message,
    make_event_message,
    make_processed_message,
    make_received_message,
    make_sent_message,
    make_start_message,
    make_tool_call_event,
)

from .conftest import captured_renderer as _captured_renderer
from .conftest import mock_client as _shared_mock_client
from .conftest import mock_ws as _mock_ws

_PROMPT_PATH = "prompt_toolkit.PromptSession.prompt"


def _mock_client(**overrides: Any) -> MagicMock:
    """Build a mock ApiClient with minimal defaults for REPL tests."""
    defaults: dict[str, Any] = {"get_events": MagicMock(return_value=[])}
    defaults.update(overrides)
    return _shared_mock_client(**defaults)


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
                        event=make_sent_message(content="hello"),
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
        assert "sender" in out
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

        with patch(_PROMPT_PATH, side_effect=["/quit"]):
            await session.run()

        out = buf.getvalue()
        assert "Connected to team t1" in out
        assert "Session closed." in out

    async def test_ctrl_c_exits(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)

        with patch(
            _PROMPT_PATH,
            side_effect=KeyboardInterrupt,
        ):
            await session.run()

        out = buf.getvalue()
        assert "Session closed." in out

    async def test_eof_exits(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)

        with patch(_PROMPT_PATH, side_effect=EOFError):
            await session.run()

        out = buf.getvalue()
        assert "Session closed." in out


class TestMessageSending:
    async def test_sends_message(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)

        with patch(
            _PROMPT_PATH,
            side_effect=["hello world", "/quit"],
        ):
            await session.run()

        client.send_message.assert_called_once_with("t1", "hello world")

    async def test_empty_input_skipped(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)

        with patch(
            _PROMPT_PATH,
            side_effect=["", "  ", "/quit"],
        ):
            await session.run()

        client.send_message.assert_not_called()


class TestReceiveLoop:
    async def test_renders_sent_message(self) -> None:
        renderer, buf = _captured_renderer()
        events = [
            {"event": make_sent_message(content="hi there")},
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
            _PROMPT_PATH,
            side_effect=["/quit"],
        ):
            await session.run()

        out = buf.getvalue()
        assert "sender" in out
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
            _PROMPT_PATH,
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

        with patch(_PROMPT_PATH, side_effect=["/quit"]):
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

        with patch(_PROMPT_PATH, side_effect=["/quit"]):
            await session.run()

        out = buf.getvalue()
        # Should NOT contain connection error — only system messages (Connected, Session closed)
        assert "Connection closed" not in out


class TestRenderEvent:
    def test_sent_message(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event({"event": make_sent_message(content="reply")})
        assert result is True
        out = buf.getvalue()
        assert "sender" in out
        assert "reply" in out

    def test_error_message(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event(
            {"event": make_error_message(exception_value="something broke")}
        )
        assert result is True
        out = buf.getvalue()
        assert "error" in out
        # Note: renderer extracts content via event.get("content") or event.get("error"),
        # but ErrorMessage.model_dump() uses "exception_value". The renderer renders an
        # error panel but with empty content. This is a known renderer limitation -- the
        # renderer does not extract "exception_value" from real ErrorMessage serialization.
        # A separate bug should address renderer compatibility with real ErrorMessage shape.

    def test_skip_start_message(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event({"event": make_start_message()})
        assert result is False

    def test_skip_received_message(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event({"event": make_received_message()})
        assert result is False

    def test_skip_processed_message(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event({"event": make_processed_message()})
        assert result is False

    def test_sent_message_no_content(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event({"event": make_sent_message(content="")})
        assert result is False

    def test_empty_event(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event({})
        assert result is False

    def test_sent_message_dict_sender(self) -> None:
        """Verify renderer extracts sender name from dict-format sender (ActorAddressProxy)."""
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        # make_sent_message produces sender as dict with name/role fields
        result = session._render_event({"event": make_sent_message(content="from dict sender")})
        assert result is True
        out = buf.getvalue()
        # Renderer extracts sender.get("name") from the dict-format sender
        assert "sender" in out
        assert "from dict sender" in out

    def test_event_message_tool_call(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        result = session._render_event(
            {"event": make_event_message(event=make_tool_call_event(tool_name="search"))}
        )
        assert result is True
        out = buf.getvalue()
        assert "search" in out

    def test_event_message_tool_call_with_arguments_field(self) -> None:
        """Verify renderer handles ToolCallEvent with 'arguments' field (not legacy 'args')."""
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        # make_tool_call_event produces 'arguments' key (JSON string), not 'args'
        result = session._render_event(
            {"event": make_event_message(event=make_tool_call_event(tool_name="web_search"))}
        )
        assert result is True
        out = buf.getvalue()
        assert "web_search" in out

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
        result = _print_event({"event": make_sent_message(content="reply")})
        assert result is True

    def test_skip_unknown(self) -> None:
        result = _print_event({"event": make_start_message()})
        assert result is False

    def test_empty_event(self) -> None:
        result = _print_event({})
        assert result is False


class TestSlashCompleter:
    """Test _SlashCompleter autocomplete behavior."""

    def _make_doc(self, text: str) -> MagicMock:
        """Create a mock Document with text_before_cursor."""
        doc = MagicMock()
        doc.text_before_cursor = text
        return doc

    def test_partial_slash_st_yields_status_and_stop(self) -> None:
        completer = _SlashCompleter(build_default_registry())
        completions = list(completer.get_completions(self._make_doc("/st"), None))
        texts = [c.text for c in completions]
        assert "/status" in texts
        assert "/stop" in texts

    def test_slash_help_yields_help(self) -> None:
        completer = _SlashCompleter(build_default_registry())
        completions = list(completer.get_completions(self._make_doc("/help"), None))
        texts = [c.text for c in completions]
        assert "/help" in texts

    def test_no_slash_prefix_yields_nothing(self) -> None:
        completer = _SlashCompleter(build_default_registry())
        completions = list(completer.get_completions(self._make_doc("hello"), None))
        assert len(completions) == 0

    def test_slash_alone_yields_all_commands(self) -> None:
        registry = build_default_registry()
        completer = _SlashCompleter(registry)
        completions = list(completer.get_completions(self._make_doc("/"), None))
        assert len(completions) == len(registry.commands)
