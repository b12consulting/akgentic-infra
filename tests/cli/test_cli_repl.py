"""Tests for ChatSession REPL core."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from akgentic.infra.cli.client import ApiError, EventInfo
from akgentic.infra.cli.commands import build_default_registry
from akgentic.infra.cli.formatters import OutputFormat
from akgentic.infra.cli.renderers import RichRenderer
from akgentic.infra.cli.connection import ConnectionState
from akgentic.infra.cli.repl import (
    ChatSession,
    InputMode,
    ReplyContext,
    SessionState,
    _print_event,
    _render_event_impl,
    _SlashCompleter,
)
from akgentic.infra.cli.ws_client import WsConnectionError
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
from .conftest import mock_conn as _mock_conn

_PROMPT_PATH = "prompt_toolkit.PromptSession.prompt"


def _mock_client(**overrides: Any) -> MagicMock:
    """Build a mock ApiClient with minimal defaults for REPL tests."""
    defaults: dict[str, Any] = {"get_events": MagicMock(return_value=[])}
    defaults.update(overrides)
    return _shared_mock_client(**defaults)


def _make_session(
    client: MagicMock | None = None,
    conn: AsyncMock | None = None,
    renderer: RichRenderer | None = None,
) -> ChatSession:
    """Create a ChatSession with mocked dependencies and optional captured renderer."""
    if client is None:
        client = _mock_client()
    if conn is None:
        conn = _mock_conn()
    return ChatSession(client, conn, "t1", OutputFormat.table, renderer=renderer)


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
        client.get_events.side_effect = ApiError(500, "test error")
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
        assert "Session closed." in out
        # Team info fetched for status bar
        assert session._state.team_name != "(unknown)"

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
        conn = _mock_conn()
        call_count = 0

        async def recv_events() -> dict[str, Any]:
            nonlocal call_count
            if call_count < len(events):
                evt = events[call_count]
                call_count += 1
                return evt
            await asyncio.sleep(10)
            return {}

        conn.receive_event = AsyncMock(side_effect=recv_events)
        session = _make_session(client=client, conn=conn, renderer=renderer)

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
        conn = _mock_conn()
        call_count = 0

        async def recv_events() -> dict[str, Any]:
            nonlocal call_count
            if call_count < len(events):
                evt = events[call_count]
                call_count += 1
                return evt
            await asyncio.sleep(10)
            return {}

        conn.receive_event = AsyncMock(side_effect=recv_events)
        session = _make_session(client=client, conn=conn, renderer=renderer)

        with patch(
            _PROMPT_PATH,
            side_effect=["/quit"],
        ):
            await session.run()

        out = buf.getvalue()
        assert "StateChanged" not in out


class TestReceiveLoopConnectionError:
    async def test_ws_connection_error_prints_error(self) -> None:
        """WsConnectionError (reconnection exhausted) renders error and breaks loop."""
        renderer, buf = _captured_renderer()
        client = _mock_client()
        conn = _mock_conn()
        conn.receive_event = AsyncMock(
            side_effect=WsConnectionError("Reconnection failed after 10 attempts", retryable=False)
        )
        session = _make_session(client=client, conn=conn, renderer=renderer)

        with patch(_PROMPT_PATH, side_effect=["/quit"]):
            await session.run()

        out = buf.getvalue()
        assert "Connection lost" in out
        assert "Reconnection failed" in out


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
        assert "something broke" in out

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

    def test_partial_slash_st_yields_stop(self) -> None:
        completer = _SlashCompleter(build_default_registry())
        completions = list(completer.get_completions(self._make_doc("/st"), None))
        texts = [c.text for c in completions]
        assert "/stop" in texts
        assert "/status" not in texts

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


class TestImplicitHumanInputReplyRouting:
    """Tests for story 10.2: implicit human-input reply routing."""

    def _human_input_event(
        self,
        message_id: str = "msg-uuid-123",
        agent_name: str = "AgentX",
        model: str = "HumanInputRequest",
        prompt: str = "What should I do?",
    ) -> dict[str, Any]:
        """Build a top-level event dict simulating a HumanInput WS event."""
        return {
            "id": message_id,
            "sender": {"name": agent_name, "role": "helper"},
            "event": {
                "__model__": "EventMessage",
                "event": {
                    "__model__": model,
                    "prompt": prompt,
                },
            },
        }

    # -- AC #1: pending state set on HumanInput event --

    def test_pending_set_on_human_input_event(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        data = self._human_input_event()
        session._render_event(data)
        assert session._state.input_mode == InputMode.REPLY
        assert session._state.reply_context is not None
        assert session._state.reply_context.reply_id == "msg-uuid-123"
        assert session._state.reply_context.agent_name == "AgentX"

    def test_pending_set_on_request_input_event(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        data = self._human_input_event(
            model="RequestInputSomething",
            message_id="req-456",
            agent_name="BotY",
        )
        session._render_event(data)
        assert session._state.input_mode == InputMode.REPLY
        assert session._state.reply_context is not None
        assert session._state.reply_context.reply_id == "req-456"
        assert session._state.reply_context.agent_name == "BotY"

    # -- AC #2: pending reply state --

    def test_pending_state_tracks_reply_info(self) -> None:
        session = _make_session()
        assert session._state.input_mode == InputMode.CHAT
        assert session._state.reply_context is None
        session._state = session._state.model_copy(
            update={
                "input_mode": InputMode.REPLY,
                "reply_context": ReplyContext(
                    reply_id="some-id", agent_name="AgentX", prompt=""
                ),
            }
        )
        assert session._state.input_mode == InputMode.REPLY
        assert session._state.reply_context is not None
        assert session._state.reply_context.reply_id == "some-id"
        assert session._state.reply_context.agent_name == "AgentX"

    # -- AC #3, #4: plain text consumes pending reply --

    async def test_plain_text_consumes_pending_reply(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)
        session._state = session._state.model_copy(
            update={
                "input_mode": InputMode.REPLY,
                "reply_context": ReplyContext(
                    reply_id="msg-abc", agent_name="AgentX", prompt=""
                ),
            }
        )

        with patch(_PROMPT_PATH, side_effect=["my answer", "/quit"]):
            await session.run()

        client.human_input.assert_called_once_with("t1", "my answer", "msg-abc")
        client.send_message.assert_not_called()
        # Pending cleared
        assert session._state.input_mode == InputMode.CHAT
        assert session._state.reply_context is None

    async def test_pending_preserved_on_api_error(self) -> None:
        """Safe reply clearing: pending state preserved when human_input() fails."""
        renderer, buf = _captured_renderer()
        client = _mock_client()
        client.human_input.side_effect = ApiError(500, "server error")
        session = _make_session(client=client, renderer=renderer)
        session._state = session._state.model_copy(
            update={
                "input_mode": InputMode.REPLY,
                "reply_context": ReplyContext(
                    reply_id="msg-abc", agent_name="AgentX", prompt=""
                ),
            }
        )

        with patch(_PROMPT_PATH, side_effect=["my answer", "/quit"]):
            await session.run()

        client.human_input.assert_called_once_with("t1", "my answer", "msg-abc")
        # Pending should still be set — not cleared on error
        assert session._state.input_mode == InputMode.REPLY
        assert session._state.reply_context is not None
        assert session._state.reply_context.reply_id == "msg-abc"
        assert session._state.reply_context.agent_name == "AgentX"
        out = buf.getvalue()
        assert "Error sending reply" in out

    # -- AC #5: no pending sends normal message --

    async def test_no_pending_sends_normal_message(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)

        with patch(_PROMPT_PATH, side_effect=["hello world", "/quit"]):
            await session.run()

        client.send_message.assert_called_once_with("t1", "hello world")
        client.human_input.assert_not_called()

    # -- AC #6: slash commands do not consume pending --

    async def test_slash_command_does_not_consume_pending(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)
        session._state = session._state.model_copy(
            update={
                "input_mode": InputMode.REPLY,
                "reply_context": ReplyContext(
                    reply_id="msg-pending", agent_name="AgentZ", prompt=""
                ),
            }
        )

        with patch(_PROMPT_PATH, side_effect=["/help", "/quit"]):
            await session.run()

        # Pending should still be set after slash command
        assert session._state.input_mode == InputMode.REPLY
        assert session._state.reply_context is not None
        assert session._state.reply_context.reply_id == "msg-pending"
        assert session._state.reply_context.agent_name == "AgentZ"
        client.human_input.assert_not_called()

    # -- backward compat: _render_event_impl without callback --

    def test_render_event_impl_without_callback(self) -> None:
        renderer, buf = _captured_renderer()
        data = self._human_input_event()
        result = _render_event_impl(data, renderer, on_human_input=None)
        assert result is True
        out = buf.getvalue()
        assert "Human Input Required" in out

    # -- sender as string fallback --

    def test_pending_set_with_string_sender(self) -> None:
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        data = {
            "id": "msg-str-sender",
            "sender": "SimpleAgent",
            "event": {
                "__model__": "EventMessage",
                "event": {
                    "__model__": "HumanInputRequest",
                    "prompt": "question?",
                },
            },
        }
        session._render_event(data)
        assert session._state.input_mode == InputMode.REPLY
        assert session._state.reply_context is not None
        assert session._state.reply_context.reply_id == "msg-str-sender"
        assert session._state.reply_context.agent_name == "SimpleAgent"

    def test_pending_not_set_when_id_is_none(self) -> None:
        """Verify that a None id in the outer event data does not set pending state."""
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        data = {
            "id": None,
            "sender": {"name": "AgentX"},
            "event": {
                "__model__": "EventMessage",
                "event": {
                    "__model__": "HumanInputRequest",
                    "prompt": "question?",
                },
            },
        }
        session._render_event(data)
        assert session._state.input_mode == InputMode.CHAT
        assert session._state.reply_context is None

    def test_pending_not_set_when_id_missing(self) -> None:
        """Verify that a missing id in the outer event data does not set pending state."""
        renderer, buf = _captured_renderer()
        session = _make_session(renderer=renderer)
        data = {
            "sender": {"name": "AgentX"},
            "event": {
                "__model__": "EventMessage",
                "event": {
                    "__model__": "HumanInputRequest",
                    "prompt": "question?",
                },
            },
        }
        session._render_event(data)
        assert session._state.input_mode == InputMode.CHAT
        assert session._state.reply_context is None

    def test_session_state_defaults(self) -> None:
        """Verify SessionState default values."""
        session = _make_session()
        assert session._state.team_name == "(unknown)"
        assert session._state.team_status == "?"
        assert session._state.input_mode == InputMode.CHAT
        assert session._state.reply_context is None
        assert session._state.connection_state == ConnectionState.CONNECTING


# =============================================================================
# Story 11.3: Session State Model tests
# =============================================================================


class TestSessionStateModels:
    """Tests for InputMode, ReplyContext, and SessionState models."""

    def test_input_mode_values(self) -> None:
        assert InputMode.CHAT.value == "chat"
        assert InputMode.REPLY.value == "reply"

    def test_reply_context_creation(self) -> None:
        ctx = ReplyContext(reply_id="r1", agent_name="Bot", prompt="Say hi")
        assert ctx.reply_id == "r1"
        assert ctx.agent_name == "Bot"
        assert ctx.prompt == "Say hi"

    def test_session_state_defaults(self) -> None:
        state = SessionState(team_id="t1")
        assert state.team_id == "t1"
        assert state.team_name == "(unknown)"
        assert state.team_status == "?"
        assert state.input_mode == InputMode.CHAT
        assert state.reply_context is None
        assert state.connection_state == ConnectionState.CONNECTING

    def test_session_state_model_copy_transitions(self) -> None:
        state = SessionState(team_id="t1")
        updated = state.model_copy(
            update={
                "input_mode": InputMode.REPLY,
                "reply_context": ReplyContext(reply_id="r1", agent_name="A", prompt=""),
            }
        )
        assert updated.input_mode == InputMode.REPLY
        assert updated.reply_context is not None
        assert updated.reply_context.reply_id == "r1"
        # Original unchanged
        assert state.input_mode == InputMode.CHAT

    def test_session_state_connection_state_update(self) -> None:
        state = SessionState(team_id="t1")
        updated = state.model_copy(update={"connection_state": ConnectionState.CONNECTED})
        assert updated.connection_state == ConnectionState.CONNECTED


class TestGetPrompt:
    """Tests for ChatSession._get_prompt() dynamic prompt."""

    def test_normal_chat_connected(self) -> None:
        session = _make_session()
        session._state = session._state.model_copy(
            update={"connection_state": ConnectionState.CONNECTED}
        )
        assert session._get_prompt() == "> "

    def test_disconnected(self) -> None:
        session = _make_session()
        session._state = session._state.model_copy(
            update={"connection_state": ConnectionState.DISCONNECTED}
        )
        assert session._get_prompt() == "[disconnected] > "

    def test_reconnecting(self) -> None:
        session = _make_session()
        session._state = session._state.model_copy(
            update={"connection_state": ConnectionState.RECONNECTING}
        )
        assert session._get_prompt() == "[reconnecting...] > "

    def test_reply_mode(self) -> None:
        session = _make_session()
        session._state = session._state.model_copy(
            update={
                "connection_state": ConnectionState.CONNECTED,
                "input_mode": InputMode.REPLY,
                "reply_context": ReplyContext(
                    reply_id="r1", agent_name="AgentX", prompt=""
                ),
            }
        )
        assert session._get_prompt() == "Reply to AgentX: "

    def test_disconnected_overrides_reply_mode(self) -> None:
        """Disconnected state takes priority over reply mode."""
        session = _make_session()
        session._state = session._state.model_copy(
            update={
                "connection_state": ConnectionState.DISCONNECTED,
                "input_mode": InputMode.REPLY,
                "reply_context": ReplyContext(
                    reply_id="r1", agent_name="AgentX", prompt=""
                ),
            }
        )
        assert session._get_prompt() == "[disconnected] > "


class TestConnectionAwareSending:
    """Tests for connection-aware message sending behavior."""

    async def test_connected_sends_normally(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)
        session._state = session._state.model_copy(
            update={"connection_state": ConnectionState.CONNECTED}
        )

        with patch(_PROMPT_PATH, side_effect=["hello", "/quit"]):
            await session.run()

        client.send_message.assert_called_once_with("t1", "hello")

    async def test_disconnected_blocks_send(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)
        session._state = session._state.model_copy(
            update={"connection_state": ConnectionState.DISCONNECTED}
        )

        with patch(_PROMPT_PATH, side_effect=["hello", "/quit"]):
            await session.run()

        client.send_message.assert_not_called()
        out = buf.getvalue()
        assert "Not connected" in out
        assert "/reconnect" in out

    async def test_reconnecting_buffers_message(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)
        session._state = session._state.model_copy(
            update={"connection_state": ConnectionState.RECONNECTING}
        )

        with patch(_PROMPT_PATH, side_effect=["buffered msg", "/quit"]):
            await session.run()

        client.send_message.assert_not_called()
        assert "buffered msg" in session._message_buffer
        out = buf.getvalue()
        assert "Reconnecting" in out
        assert "connection is restored" in out

    async def test_buffer_flushed_on_connected(self) -> None:
        """Buffered messages are flushed when connection state transitions to CONNECTED."""
        client = _mock_client()
        session = _make_session(client=client)
        session._message_buffer = ["msg1", "msg2"]
        # Trigger the on_state_change callback with CONNECTED
        session.conn._on_state_change(ConnectionState.CONNECTED)
        assert session._message_buffer == []
        assert client.send_message.call_count == 2
        client.send_message.assert_any_call("t1", "msg1")
        client.send_message.assert_any_call("t1", "msg2")

    async def test_disconnected_blocks_reply_send(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)
        session._state = session._state.model_copy(
            update={
                "connection_state": ConnectionState.DISCONNECTED,
                "input_mode": InputMode.REPLY,
                "reply_context": ReplyContext(
                    reply_id="r1", agent_name="Agent", prompt=""
                ),
            }
        )

        with patch(_PROMPT_PATH, side_effect=["my reply", "/quit"]):
            await session.run()

        client.human_input.assert_not_called()
        out = buf.getvalue()
        assert "Not connected" in out

    async def test_reconnecting_buffers_reply(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        session = _make_session(client=client, renderer=renderer)
        session._state = session._state.model_copy(
            update={
                "connection_state": ConnectionState.RECONNECTING,
                "input_mode": InputMode.REPLY,
                "reply_context": ReplyContext(
                    reply_id="r1", agent_name="Agent", prompt=""
                ),
            }
        )

        with patch(_PROMPT_PATH, side_effect=["my reply", "/quit"]):
            await session.run()

        client.human_input.assert_not_called()
        assert "my reply" in session._message_buffer

    def test_flush_buffer_handles_api_error(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        client.send_message.side_effect = ApiError(500, "fail")
        session = _make_session(client=client, renderer=renderer)
        session._message_buffer = ["msg1"]
        session._flush_message_buffer()
        assert session._message_buffer == []
        out = buf.getvalue()
        assert "Failed to send buffered message" in out

    def test_team_id_property_getter(self) -> None:
        session = _make_session()
        assert session.team_id == "t1"

    def test_team_id_property_setter(self) -> None:
        session = _make_session()
        session.team_id = "t2"
        assert session._state.team_id == "t2"
        assert session.team_id == "t2"
