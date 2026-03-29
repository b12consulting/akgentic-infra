"""WebSocket event wrapping for the Angular V1 frontend adapter.

Translates V2 ``PersistedEvent`` instances into V1 envelope format so
the existing Angular V1 frontend's WebSocket handling works unchanged.
"""

from __future__ import annotations

from typing import Any

from akgentic.core.messages.message import Message, UserMessage
from akgentic.core.messages.orchestrator import (
    ErrorMessage,
    EventMessage,
    SentMessage,
    StateChangedMessage,
)
from akgentic.infra.server.routes.frontend_adapter import WrappedWsEvent
from akgentic.infra.server.routes.frontend_adapter.angular_v1._helpers import (
    classify_message_type,
    extract_message_content,
    get_sender_name,
)
from akgentic.team.models import PersistedEvent


def _classify_envelope_type(event: Message) -> str:
    """Map a V2 message to its V1 envelope ``type`` discriminator.

    Args:
        event: The V2 message from a persisted event.

    Returns:
        One of ``"message"``, ``"state"``, ``"tool_update"``, or ``"llm_context"``.
    """
    if isinstance(event, StateChangedMessage):
        return "state"
    if isinstance(event, EventMessage):
        return "tool_update"
    if type(event).__name__ == "ContextChangedMessage":
        return "llm_context"
    return "message"


def _build_message_envelope(event: Message, timestamp: str) -> dict[str, Any]:
    """Build a V1 ``type: "message"`` envelope.

    For ``SentMessage`` events the inner message is inspected to extract
    content and classify the message type correctly.

    Args:
        event: The V2 message (or SentMessage wrapping one).
        timestamp: ISO-formatted event timestamp.

    Returns:
        Dict with V1 message envelope fields.
    """
    inner = event.message if isinstance(event, SentMessage) else event

    content = extract_message_content(event)
    if content is None:
        content = ""

    if isinstance(event, ErrorMessage):
        message_type = "system"
        content = event.exception_value
    elif isinstance(event, SentMessage):
        message_type = _classify_inner_message_type(inner)
    else:
        message_type = classify_message_type(event)

    return {
        "type": "message",
        "id": str(inner.id),
        "sender": get_sender_name(inner),
        "content": content,
        "timestamp": timestamp,
        "message_type": message_type,
    }


def _classify_inner_message_type(inner: Message) -> str:
    """Classify the inner message of a ``SentMessage``.

    Args:
        inner: The inner message wrapped by SentMessage.

    Returns:
        One of ``"user"``, ``"agent"``, or ``"system"``.
    """
    if isinstance(inner, UserMessage):
        return "user"
    return classify_message_type(inner)


def _build_state_envelope(event: StateChangedMessage, timestamp: str) -> dict[str, Any]:
    """Build a V1 ``type: "state"`` envelope.

    Args:
        event: The state-changed message.
        timestamp: ISO-formatted event timestamp.

    Returns:
        Dict with V1 state envelope fields.
    """
    return {
        "type": "state",
        "agent": get_sender_name(event),
        "state": event.state.model_dump(mode="json"),
        "timestamp": timestamp,
    }


def _build_tool_update_envelope(event: EventMessage, timestamp: str) -> dict[str, Any]:
    """Build a V1 ``type: "tool_update"`` envelope.

    Args:
        event: The domain/tool event message.
        timestamp: ISO-formatted event timestamp.

    Returns:
        Dict with V1 tool_update envelope fields.
    """
    if hasattr(event.event, "model_dump"):
        serialized: Any = event.event.model_dump(mode="json")
    elif isinstance(event.event, dict):
        serialized = event.event
    else:
        serialized = str(event.event)
    return {
        "type": "tool_update",
        "event": serialized,
        "timestamp": timestamp,
    }


def _build_llm_context_envelope(event: Message, timestamp: str) -> dict[str, Any]:
    """Build a V1 ``type: "llm_context"`` envelope.

    Args:
        event: The context-changed message.
        timestamp: ISO-formatted event timestamp.

    Returns:
        Dict with V1 llm_context envelope fields.
    """
    context_data: Any = {}
    if hasattr(event, "context") and hasattr(event.context, "model_dump"):
        context_data = event.context.model_dump(mode="json")
    elif hasattr(event, "context") and isinstance(event.context, dict):
        context_data = event.context
    elif hasattr(event, "context"):
        context_data = str(event.context)
    return {
        "type": "llm_context",
        "context": context_data,
        "timestamp": timestamp,
    }


def wrap_event(event: PersistedEvent) -> WrappedWsEvent:
    """Translate a V2 persisted event into a V1 WebSocket envelope.

    This is the main entry point called by ``AngularV1Adapter.wrap_ws_event``.

    Args:
        event: The V2 persisted event to translate.

    Returns:
        A ``WrappedWsEvent`` containing the V1-formatted payload.
    """
    msg = event.event
    timestamp = event.timestamp.isoformat()
    envelope_type = _classify_envelope_type(msg)

    if envelope_type == "state" and isinstance(msg, StateChangedMessage):
        payload = _build_state_envelope(msg, timestamp)
    elif envelope_type == "tool_update" and isinstance(msg, EventMessage):
        payload = _build_tool_update_envelope(msg, timestamp)
    elif envelope_type == "llm_context":
        payload = _build_llm_context_envelope(msg, timestamp)
    else:
        payload = _build_message_envelope(msg, timestamp)

    return WrappedWsEvent(payload=payload)
