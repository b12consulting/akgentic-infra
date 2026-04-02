"""Shared fixture factories for core and LLM event types.

Each factory creates a real model/dataclass instance, then returns
``model.model_dump()`` (Pydantic) or ``dataclasses.asdict()`` (dataclass).
This guarantees the dict shape always matches the real serialization contract.

Usage::

    from tests.fixtures.events import make_sent_message

    def test_something():
        event = make_sent_message(content="custom")
        # event is a plain dict matching SentMessage.model_dump()
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any

from akgentic.core.actor_address_impl import ActorAddressProxy
from akgentic.core.agent_config import BaseConfig
from akgentic.core.messages.message import UserMessage
from akgentic.core.messages.orchestrator import (
    ErrorMessage,
    EventMessage,
    ProcessedMessage,
    ReceivedMessage,
    SentMessage,
    StartMessage,
)
from akgentic.core.utils.deserializer import ActorAddressDict
from akgentic.llm.event import LlmUsageEvent, ToolCallEvent, ToolReturnEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXED_UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_FIXED_TEAM_UUID = uuid.UUID("00000000-0000-0000-0000-000000000002")


def _make_address_dict(**overrides: Any) -> ActorAddressDict:
    """Build a minimal ``ActorAddressDict`` with sensible defaults."""
    defaults: dict[str, Any] = {
        "__actor_address__": True,
        "__actor_type__": "akgentic.core.actor_address_impl.ActorAddressProxy",
        "agent_id": str(overrides.pop("agent_id", _FIXED_UUID)),
        "name": "test-agent",
        "role": "TestRole",
        "team_id": str(overrides.pop("team_id", _FIXED_TEAM_UUID)),
        "squad_id": None,
        "user_message": False,
    }
    defaults.update(overrides)
    return defaults  # type: ignore[return-value]


def _make_proxy(**overrides: Any) -> ActorAddressProxy:
    """Create an ``ActorAddressProxy`` for use as sender/recipient."""
    return ActorAddressProxy(_make_address_dict(**overrides))


# ---------------------------------------------------------------------------
# Core event factories
# ---------------------------------------------------------------------------


def make_sent_message(**overrides: Any) -> dict[str, Any]:
    """Create a ``SentMessage`` fixture dict from a real model instance.

    The inner ``message`` defaults to a ``UserMessage`` with ``content``
    taken from *overrides* (default ``"Hello from test"``).
    """
    content = overrides.pop("content", "Hello from test")
    inner = overrides.pop("message", UserMessage(content=content))
    recipient = overrides.pop("recipient", _make_proxy(name="recipient"))
    sender = overrides.pop("sender", _make_proxy(name="sender"))
    defaults: dict[str, Any] = {
        "message": inner,
        "recipient": recipient,
        "sender": sender,
    }
    defaults.update(overrides)
    return SentMessage(**defaults).model_dump()


def make_event_message(**overrides: Any) -> dict[str, Any]:
    """Create an ``EventMessage`` fixture dict from a real model instance."""
    defaults: dict[str, Any] = {
        "event": overrides.pop("event", {"type": "test-event", "data": "sample"}),
    }
    defaults.update(overrides)
    return EventMessage(**defaults).model_dump()


def make_error_message(**overrides: Any) -> dict[str, Any]:
    """Create an ``ErrorMessage`` fixture dict from a real model instance."""
    defaults: dict[str, Any] = {
        "exception_type": "ValueError",
        "exception_value": "something went wrong",
    }
    defaults.update(overrides)
    return ErrorMessage(**defaults).model_dump()


def make_start_message(**overrides: Any) -> dict[str, Any]:
    """Create a ``StartMessage`` fixture dict from a real model instance."""
    config = overrides.pop("config", BaseConfig(name="test-agent", role="tester"))
    defaults: dict[str, Any] = {
        "config": config,
    }
    defaults.update(overrides)
    return StartMessage(**defaults).model_dump()


def make_received_message(**overrides: Any) -> dict[str, Any]:
    """Create a ``ReceivedMessage`` fixture dict from a real model instance."""
    defaults: dict[str, Any] = {
        "message_id": overrides.pop("message_id", _FIXED_UUID),
    }
    defaults.update(overrides)
    return ReceivedMessage(**defaults).model_dump()


def make_processed_message(**overrides: Any) -> dict[str, Any]:
    """Create a ``ProcessedMessage`` fixture dict from a real model instance."""
    defaults: dict[str, Any] = {
        "message_id": overrides.pop("message_id", _FIXED_UUID),
    }
    defaults.update(overrides)
    return ProcessedMessage(**defaults).model_dump()


# ---------------------------------------------------------------------------
# LLM event factories (dataclasses → dataclasses.asdict)
# ---------------------------------------------------------------------------


def make_tool_call_event(**overrides: Any) -> dict[str, Any]:
    """Create a ``ToolCallEvent`` fixture dict from a real dataclass instance.

    Adds ``__model__`` to match the real serialization contract
    (``akgentic.core.utils.serializer.serialize`` adds it for dataclasses).
    """
    defaults: dict[str, Any] = {
        "run_id": "run-001",
        "tool_name": "test_tool",
        "tool_call_id": "call-001",
        "arguments": '{"key": "value"}',
    }
    defaults.update(overrides)
    result = dataclasses.asdict(ToolCallEvent(**defaults))
    result["__model__"] = "akgentic.llm.event.ToolCallEvent"
    return result


def make_tool_return_event(**overrides: Any) -> dict[str, Any]:
    """Create a ``ToolReturnEvent`` fixture dict from a real dataclass instance.

    Adds ``__model__`` to match the real serialization contract.
    """
    defaults: dict[str, Any] = {
        "run_id": "run-001",
        "tool_name": "test_tool",
        "tool_call_id": "call-001",
        "success": True,
    }
    defaults.update(overrides)
    result = dataclasses.asdict(ToolReturnEvent(**defaults))
    result["__model__"] = "akgentic.llm.event.ToolReturnEvent"
    return result


def make_llm_usage_event(**overrides: Any) -> dict[str, Any]:
    """Create an ``LlmUsageEvent`` fixture dict from a real dataclass instance."""
    defaults: dict[str, Any] = {
        "run_id": "run-001",
        "model_name": "test-model",
        "provider_name": "test-provider",
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "requests": 1,
    }
    defaults.update(overrides)
    return dataclasses.asdict(LlmUsageEvent(**defaults))
