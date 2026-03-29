"""Tests for Angular V1 adapter — WebSocket event wrapping (Story 3.2b)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import Message, ResultMessage, UserMessage
from akgentic.core.messages.orchestrator import (
    ErrorMessage,
    EventMessage,
    ProcessedMessage,
    ReceivedMessage,
    SentMessage,
    StartMessage,
    StateChangedMessage,
    StopMessage,
)
from akgentic.team.models import PersistedEvent

from akgentic.infra.server.routes.frontend_adapter import (
    LlmContextPayload,
    MessagePayload,
    StatePayload,
    ToolUpdatePayload,
    UnknownPayload,
    WrappedWsEvent,
)
from akgentic.infra.server.routes.frontend_adapter.angular_v1 import AngularV1Adapter
from akgentic.infra.server.routes.frontend_adapter.angular_v1.ws import wrap_event

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 28, 12, 0, 0, tzinfo=UTC)
_TEAM_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _make_sender(name: str = "@Agent") -> MagicMock:
    """Create a mock ActorAddress with a name attribute."""
    sender = MagicMock(spec=ActorAddress)
    sender.name = name
    return sender


def _make_persisted_event(
    event: Message,
    team_id: uuid.UUID = _TEAM_ID,
    sequence: int = 1,
) -> PersistedEvent:
    """Create a PersistedEvent fixture."""
    return PersistedEvent(
        team_id=team_id,
        sequence=sequence,
        event=event,
        timestamp=_NOW,
    )


# ---------------------------------------------------------------------------
# Task 4.2: UserMessage → type: "message", message_type: "user"
# ---------------------------------------------------------------------------


class TestWrapUserMessage:
    """Test UserMessage event wrapping."""

    def test_user_message_envelope_type(self) -> None:
        """AC #1: UserMessage produces type: 'message' envelope."""
        msg = UserMessage(content="hello world")
        msg.sender = _make_sender("@Human")
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert result.payload.type == "message"

    def test_user_message_fields(self) -> None:
        """AC #1: UserMessage envelope has V1-compatible fields."""
        msg = UserMessage(content="hello world")
        msg.sender = _make_sender("@Human")
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert isinstance(result.payload, MessagePayload)
        assert result.payload.message_type == "user"
        assert result.payload.content == "hello world"
        assert result.payload.sender == "@Human"
        assert result.payload.timestamp == _NOW.isoformat()
        assert result.payload.id


# ---------------------------------------------------------------------------
# Task 4.3: ResultMessage → type: "message", message_type: "agent"
# ---------------------------------------------------------------------------


class TestWrapResultMessage:
    """Test ResultMessage event wrapping."""

    def test_result_message_envelope_type(self) -> None:
        """AC #2: ResultMessage produces type: 'message' envelope."""
        msg = ResultMessage(content="AI response")
        msg.sender = _make_sender("@Manager")
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert result.payload.type == "message"

    def test_result_message_classified_as_agent(self) -> None:
        """AC #2: ResultMessage message_type is 'agent'."""
        msg = ResultMessage(content="AI response")
        msg.sender = _make_sender("@Manager")
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert result.payload.message_type == "agent"
        assert result.payload.content == "AI response"
        assert result.payload.sender == "@Manager"


# ---------------------------------------------------------------------------
# Task 4.4: SentMessage wrapping UserMessage
# ---------------------------------------------------------------------------


class TestWrapSentMessageUser:
    """Test SentMessage wrapping a UserMessage."""

    def test_sent_user_message_extracts_inner_content(self) -> None:
        """AC #1: SentMessage with inner UserMessage extracts content."""
        inner = UserMessage(content="routed user msg")
        inner.sender = _make_sender("@Human")
        recipient = _make_sender("@Worker")
        sent = SentMessage(message=inner, recipient=recipient)
        sent.sender = _make_sender("@Router")
        event = _make_persisted_event(sent)
        result = wrap_event(event)
        assert result.payload.type == "message"
        assert result.payload.content == "routed user msg"
        assert result.payload.sender == "@Human"
        assert result.payload.id == str(inner.id)

    def test_sent_user_message_type_is_user(self) -> None:
        """SentMessage wrapping UserMessage has message_type: 'user'."""
        inner = UserMessage(content="inner")
        recipient = _make_sender("@Worker")
        sent = SentMessage(message=inner, recipient=recipient)
        sent.sender = _make_sender("@Router")
        event = _make_persisted_event(sent)
        result = wrap_event(event)
        assert result.payload.message_type == "user"


# ---------------------------------------------------------------------------
# Task 4.5: SentMessage wrapping ResultMessage
# ---------------------------------------------------------------------------


class TestWrapSentMessageResult:
    """Test SentMessage wrapping a ResultMessage."""

    def test_sent_result_message_type_is_agent(self) -> None:
        """AC #2: SentMessage wrapping ResultMessage has message_type: 'agent'."""
        inner = ResultMessage(content="agent reply")
        inner.sender = _make_sender("@Agent")
        recipient = _make_sender("@User")
        sent = SentMessage(message=inner, recipient=recipient)
        sent.sender = _make_sender("@Router")
        event = _make_persisted_event(sent)
        result = wrap_event(event)
        assert result.payload.message_type == "agent"
        assert result.payload.content == "agent reply"
        assert result.payload.sender == "@Agent"
        assert result.payload.id == str(inner.id)


# ---------------------------------------------------------------------------
# Task 4.6: StateChangedMessage → type: "state"
# ---------------------------------------------------------------------------


class TestWrapStateChanged:
    """Test StateChangedMessage event wrapping."""

    def test_state_envelope_type(self) -> None:
        """AC #3: StateChangedMessage produces type: 'state' envelope."""
        msg = StateChangedMessage(state=BaseState())
        msg.sender = _make_sender("@Manager")
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert result.payload.type == "state"

    def test_state_envelope_fields(self) -> None:
        """AC #3: State envelope has V1-compatible fields."""
        msg = StateChangedMessage(state=BaseState())
        msg.sender = _make_sender("@Manager")
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert isinstance(result.payload, StatePayload)
        assert result.payload.agent == "@Manager"
        assert isinstance(result.payload.state, dict)
        assert result.payload.timestamp == _NOW.isoformat()


# ---------------------------------------------------------------------------
# Task 4.7: EventMessage → type: "tool_update"
# ---------------------------------------------------------------------------


class TestWrapEventMessage:
    """Test EventMessage event wrapping."""

    def test_tool_update_envelope_type(self) -> None:
        """AC #1: EventMessage produces type: 'tool_update' envelope."""
        msg = EventMessage(event={"tool_name": "search", "result": "found"})
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert result.payload.type == "tool_update"

    def test_tool_update_envelope_fields(self) -> None:
        """EventMessage envelope has event and timestamp fields."""
        msg = EventMessage(event={"tool_name": "search", "result": "found"})
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert isinstance(result.payload, ToolUpdatePayload)
        assert result.payload.timestamp == _NOW.isoformat()
        assert result.payload.event is not None

    def test_tool_update_with_pydantic_event(self) -> None:
        """EventMessage with a Pydantic model event serializes via model_dump."""
        state = BaseState()
        msg = EventMessage(event=state)
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert isinstance(result.payload.event, dict)

    def test_tool_update_with_string_event(self) -> None:
        """EventMessage with a plain string event falls back to str()."""
        msg = EventMessage(event="simple event")
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert result.payload.event == "simple event"


# ---------------------------------------------------------------------------
# Task 4.8: Non-content events (ReceivedMessage, ProcessedMessage) — fallback
# ---------------------------------------------------------------------------


class TestWrapNonContentEvents:
    """Test events without displayable content produce valid envelopes."""

    def test_received_message_fallback(self) -> None:
        """ReceivedMessage produces valid message envelope with empty content."""
        msg = ReceivedMessage(message_id=uuid.uuid4())
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert isinstance(result.payload, MessagePayload)
        assert result.payload.message_type == "system"
        assert result.payload.content is not None

    def test_processed_message_fallback(self) -> None:
        """ProcessedMessage produces valid message envelope."""
        msg = ProcessedMessage(message_id=uuid.uuid4())
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert result.payload.type == "message"
        assert result.payload.message_type == "system"

    def test_start_message_fallback(self) -> None:
        """StartMessage produces valid message envelope with system type."""
        msg = StartMessage(config=BaseConfig())
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert result.payload.type == "message"
        assert result.payload.message_type == "system"

    def test_stop_message_fallback(self) -> None:
        """StopMessage produces valid message envelope with system type."""
        msg = StopMessage()
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert result.payload.type == "message"
        assert result.payload.message_type == "system"


# ---------------------------------------------------------------------------
# Review fix: SentMessage wrapping non-content message
# ---------------------------------------------------------------------------


class TestWrapSentMessageNonContent:
    """Test SentMessage wrapping a message without displayable content."""

    def test_sent_start_message_produces_valid_envelope(self) -> None:
        """SentMessage wrapping StartMessage produces valid envelope with empty content."""
        inner = StartMessage(config=BaseConfig())
        inner.sender = _make_sender("@Orchestrator")
        recipient = _make_sender("@Worker")
        sent = SentMessage(message=inner, recipient=recipient)
        sent.sender = _make_sender("@Router")
        event = _make_persisted_event(sent)
        result = wrap_event(event)
        assert result.payload.type == "message"
        assert result.payload.message_type == "system"
        assert result.payload.sender == "@Orchestrator"
        assert result.payload.content == ""


# ---------------------------------------------------------------------------
# Task 4.9: ErrorMessage → type: "message", message_type: "system"
# ---------------------------------------------------------------------------


class TestWrapErrorMessage:
    """Test ErrorMessage event wrapping."""

    def test_error_message_type(self) -> None:
        """ErrorMessage produces type: 'message' envelope."""
        msg = ErrorMessage(
            exception_type="ValueError",
            exception_value="something went wrong",
        )
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert result.payload.type == "message"

    def test_error_message_classified_as_system(self) -> None:
        """ErrorMessage message_type is 'system'."""
        msg = ErrorMessage(
            exception_type="ValueError",
            exception_value="something went wrong",
        )
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert result.payload.message_type == "system"
        assert result.payload.content == "something went wrong"


# ---------------------------------------------------------------------------
# Task 4.10: AngularV1Adapter.wrap_ws_event() integration
# ---------------------------------------------------------------------------


class TestAdapterWrapWsEvent:
    """Test AngularV1Adapter.wrap_ws_event() delegates to ws.wrap_event()."""

    def test_adapter_returns_wrapped_event(self) -> None:
        """wrap_ws_event returns WrappedWsEvent instance."""
        adapter = AngularV1Adapter()
        msg = UserMessage(content="test")
        event = _make_persisted_event(msg)
        result = adapter.wrap_ws_event(event)
        assert isinstance(result, WrappedWsEvent)

    def test_adapter_produces_v1_envelope(self) -> None:
        """wrap_ws_event produces V1 envelope, not raw passthrough."""
        adapter = AngularV1Adapter()
        msg = UserMessage(content="test content")
        event = _make_persisted_event(msg)
        result = adapter.wrap_ws_event(event)
        assert result.payload.type == "message"
        assert result.payload.content == "test content"
        assert result.payload.message_type == "user"

    def test_adapter_state_event(self) -> None:
        """wrap_ws_event produces state envelope for StateChangedMessage."""
        adapter = AngularV1Adapter()
        msg = StateChangedMessage(state=BaseState())
        msg.sender = _make_sender("@Agent")
        event = _make_persisted_event(msg)
        result = adapter.wrap_ws_event(event)
        assert isinstance(result.payload, StatePayload)
        assert result.payload.agent == "@Agent"


# ---------------------------------------------------------------------------
# Task 4.11: JSON serialization
# ---------------------------------------------------------------------------


class TestWrappedEventSerialization:
    """Test that wrapped payloads are JSON-serializable."""

    def test_user_message_json_serializable(self) -> None:
        """WrappedWsEvent from UserMessage serializes to valid JSON."""
        msg = UserMessage(content="hello")
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        json_str = result.model_dump_json()
        assert '"type":"message"' in json_str
        assert '"content":"hello"' in json_str

    def test_state_message_json_serializable(self) -> None:
        """WrappedWsEvent from StateChangedMessage serializes to valid JSON."""
        msg = StateChangedMessage(state=BaseState())
        msg.sender = _make_sender("@Bot")
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        json_str = result.model_dump_json()
        assert '"type":"state"' in json_str

    def test_error_message_json_serializable(self) -> None:
        """WrappedWsEvent from ErrorMessage serializes to valid JSON."""
        msg = ErrorMessage(
            exception_type="RuntimeError",
            exception_value="oops",
        )
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        json_str = result.model_dump_json()
        assert '"type":"message"' in json_str
        assert '"oops"' in json_str

    def test_tool_update_json_serializable(self) -> None:
        """WrappedWsEvent from EventMessage serializes to valid JSON."""
        msg = EventMessage(event={"tool": "search"})
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        json_str = result.model_dump_json()
        assert '"type":"tool_update"' in json_str

    def test_no_sender_defaults_to_system(self) -> None:
        """Event with no sender uses 'system' as sender name."""
        msg = UserMessage(content="orphan")
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert result.payload.sender == "system"


# ---------------------------------------------------------------------------
# Story 6.8: ContextChangedMessage → type: "llm_context" (AC #4)
# ---------------------------------------------------------------------------


class ContextChangedMessage(Message):
    """Local placeholder for ContextChangedMessage (forward-compatible)."""

    context: dict[str, object] = {}


class TestWrapContextChangedMessage:
    """Test ContextChangedMessage event wrapping (AC #4)."""

    def test_llm_context_envelope_type(self) -> None:
        """AC4: ContextChangedMessage produces type: 'llm_context' envelope."""
        msg = ContextChangedMessage(context={"key": "value"})
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert result.payload.type == "llm_context"

    def test_llm_context_envelope_fields(self) -> None:
        """AC4: llm_context envelope has context and timestamp fields."""
        msg = ContextChangedMessage(context={"agent": "bot", "tokens": 100})
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert isinstance(result.payload, LlmContextPayload)
        assert result.payload.context == {"agent": "bot", "tokens": 100}
        assert result.payload.timestamp == _NOW.isoformat()

    def test_llm_context_json_serializable(self) -> None:
        """WrappedWsEvent from ContextChangedMessage serializes to valid JSON."""
        msg = ContextChangedMessage(context={"key": "val"})
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        json_str = result.model_dump_json()
        assert '"type":"llm_context"' in json_str

    def test_existing_envelope_types_unchanged(self) -> None:
        """AC4: Existing envelope types continue to work unchanged."""
        user_msg = UserMessage(content="hello")
        event = _make_persisted_event(user_msg)
        result = wrap_event(event)
        assert result.payload.type == "message"

        state_msg = StateChangedMessage(state=BaseState())
        state_msg.sender = _make_sender("@Bot")
        event2 = _make_persisted_event(state_msg)
        result2 = wrap_event(event2)
        assert result2.payload.type == "state"

        tool_msg = EventMessage(event={"tool": "test"})
        event3 = _make_persisted_event(tool_msg)
        result3 = wrap_event(event3)
        assert result3.payload.type == "tool_update"

    def test_llm_context_empty_context(self) -> None:
        """ContextChangedMessage with empty context produces valid envelope."""
        msg = ContextChangedMessage()
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        assert isinstance(result.payload, LlmContextPayload)
        assert result.payload.context == {}


# ---------------------------------------------------------------------------
# Story 6.10: Discriminated union deserialization & round-trip (AC #2, #4, #6)
# ---------------------------------------------------------------------------


class TestDiscriminatedUnionDeserialization:
    """Test that JSON with each type value deserializes to the correct payload subtype."""

    def test_message_payload_from_json(self) -> None:
        """JSON with type 'message' deserializes to MessagePayload."""
        json_str = (
            '{"payload":{"type":"message","id":"abc","sender":"@Bot",'
            '"content":"hi","timestamp":"2026-01-01T00:00:00","message_type":"user"}}'
        )
        event = WrappedWsEvent.model_validate_json(json_str)
        assert isinstance(event.payload, MessagePayload)
        assert event.payload.content == "hi"
        assert event.payload.sender == "@Bot"

    def test_state_payload_from_json(self) -> None:
        """JSON with type 'state' deserializes to StatePayload."""
        json_str = (
            '{"payload":{"type":"state","agent":"@Manager",'
            '"state":{"key":"val"},"timestamp":"2026-01-01T00:00:00"}}'
        )
        event = WrappedWsEvent.model_validate_json(json_str)
        assert isinstance(event.payload, StatePayload)
        assert event.payload.agent == "@Manager"
        assert event.payload.state == {"key": "val"}

    def test_tool_update_payload_from_json(self) -> None:
        """JSON with type 'tool_update' deserializes to ToolUpdatePayload."""
        json_str = (
            '{"payload":{"type":"tool_update","event":{"tool":"search"},'
            '"timestamp":"2026-01-01T00:00:00"}}'
        )
        event = WrappedWsEvent.model_validate_json(json_str)
        assert isinstance(event.payload, ToolUpdatePayload)
        assert event.payload.event == {"tool": "search"}

    def test_llm_context_payload_from_json(self) -> None:
        """JSON with type 'llm_context' deserializes to LlmContextPayload."""
        json_str = (
            '{"payload":{"type":"llm_context","context":{"tokens":42},'
            '"timestamp":"2026-01-01T00:00:00"}}'
        )
        event = WrappedWsEvent.model_validate_json(json_str)
        assert isinstance(event.payload, LlmContextPayload)
        assert event.payload.context == {"tokens": 42}


class TestUnknownPayloadFallback:
    """Test UnknownPayload catch-all for unrecognised event types."""

    def test_unknown_type_deserializes_to_unknown_payload(self) -> None:
        """JSON with an unrecognised type falls back to UnknownPayload."""
        json_str = '{"payload":{"type":"custom_extension","data":{"x":1}}}'
        event = WrappedWsEvent.model_validate_json(json_str)
        assert isinstance(event.payload, UnknownPayload)
        assert event.payload.type == "custom_extension"
        assert event.payload.data == {"x": 1}

    def test_unknown_type_with_empty_data(self) -> None:
        """UnknownPayload defaults data to empty dict."""
        json_str = '{"payload":{"type":"future_event"}}'
        event = WrappedWsEvent.model_validate_json(json_str)
        assert isinstance(event.payload, UnknownPayload)
        assert event.payload.data == {}


class TestSerializationRoundTrip:
    """Test model_dump_json -> model_validate_json preserves payload type and fields."""

    def test_message_round_trip(self) -> None:
        """MessagePayload survives serialization round-trip."""
        original = WrappedWsEvent(
            payload=MessagePayload(
                id="123",
                sender="@Bot",
                content="hello",
                timestamp="2026-01-01T00:00:00",
                message_type="agent",
            ),
        )
        restored = WrappedWsEvent.model_validate_json(original.model_dump_json())
        assert isinstance(restored.payload, MessagePayload)
        assert restored.payload.id == "123"
        assert restored.payload.content == "hello"

    def test_state_round_trip(self) -> None:
        """StatePayload survives serialization round-trip."""
        original = WrappedWsEvent(
            payload=StatePayload(
                agent="@Manager",
                state={"status": "idle"},
                timestamp="2026-01-01T00:00:00",
            ),
        )
        restored = WrappedWsEvent.model_validate_json(original.model_dump_json())
        assert isinstance(restored.payload, StatePayload)
        assert restored.payload.agent == "@Manager"

    def test_tool_update_round_trip(self) -> None:
        """ToolUpdatePayload survives serialization round-trip."""
        original = WrappedWsEvent(
            payload=ToolUpdatePayload(
                event={"tool": "calc"},
                timestamp="2026-01-01T00:00:00",
            ),
        )
        restored = WrappedWsEvent.model_validate_json(original.model_dump_json())
        assert isinstance(restored.payload, ToolUpdatePayload)
        assert restored.payload.event == {"tool": "calc"}

    def test_llm_context_round_trip(self) -> None:
        """LlmContextPayload survives serialization round-trip."""
        original = WrappedWsEvent(
            payload=LlmContextPayload(
                context={"tokens": 99},
                timestamp="2026-01-01T00:00:00",
            ),
        )
        restored = WrappedWsEvent.model_validate_json(original.model_dump_json())
        assert isinstance(restored.payload, LlmContextPayload)
        assert restored.payload.context == {"tokens": 99}

    def test_unknown_round_trip(self) -> None:
        """UnknownPayload survives serialization round-trip."""
        original = WrappedWsEvent(
            payload=UnknownPayload(type="custom", data={"a": "b"}),
        )
        restored = WrappedWsEvent.model_validate_json(original.model_dump_json())
        assert isinstance(restored.payload, UnknownPayload)
        assert restored.payload.type == "custom"
        assert restored.payload.data == {"a": "b"}

    def test_wire_format_matches_old_dict_shape(self) -> None:
        """model_dump(mode='json') produces the same structure as the old dicts."""
        msg = UserMessage(content="wire test")
        msg.sender = _make_sender("@Human")
        event = _make_persisted_event(msg)
        result = wrap_event(event)
        dumped = result.model_dump(mode="json")
        payload = dumped["payload"]
        assert payload["type"] == "message"
        assert payload["content"] == "wire test"
        assert payload["sender"] == "@Human"
        assert "id" in payload
        assert "timestamp" in payload
        assert "message_type" in payload
