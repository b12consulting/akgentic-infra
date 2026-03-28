"""Tests for WebSocketEventSubscriber adapter."""

from __future__ import annotations

import json
import queue
import threading

from akgentic.core.messages import Message
from akgentic.core.messages.orchestrator import SentMessage

from akgentic.infra.adapters.websocket_subscriber import WebSocketEventSubscriber


class TestWebSocketEventSubscriber:
    """Unit tests for WebSocketEventSubscriber (AC #5, #6)."""

    def test_on_message_serializes_and_enqueues(self) -> None:
        """on_message puts a JSON string on the queue."""
        subscriber = WebSocketEventSubscriber()
        msg = _make_message()

        subscriber.on_message(msg)

        item = subscriber.get_queue().get_nowait()
        assert isinstance(item, str)
        data = json.loads(item)
        assert isinstance(data, dict)

    def test_on_message_includes_model_field(self) -> None:
        """Serialized output includes __model__ for type discrimination."""
        subscriber = WebSocketEventSubscriber()
        msg = _make_message()

        subscriber.on_message(msg)

        item = subscriber.get_queue().get_nowait()
        assert item is not None
        data = json.loads(item)
        assert "__model__" in data

    def test_on_stop_puts_none_sentinel(self) -> None:
        """on_stop enqueues None to signal connection closure."""
        subscriber = WebSocketEventSubscriber()

        subscriber.on_stop()

        item = subscriber.get_queue().get_nowait()
        assert item is None

    def test_queue_is_thread_safe(self) -> None:
        """Concurrent puts from multiple threads don't lose messages."""
        subscriber = WebSocketEventSubscriber()
        msg = _make_message()
        num_threads = 10
        msgs_per_thread = 50
        total = num_threads * msgs_per_thread

        def _produce() -> None:
            for _ in range(msgs_per_thread):
                subscriber.on_message(msg)

        threads = [threading.Thread(target=_produce) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        count = 0
        q = subscriber.get_queue()
        while not q.empty():
            q.get_nowait()
            count += 1
        assert count == total

    def test_get_queue_returns_queue_instance(self) -> None:
        """get_queue returns the internal queue.Queue."""
        subscriber = WebSocketEventSubscriber()
        q = subscriber.get_queue()
        assert isinstance(q, queue.Queue)

    def test_constructor_takes_no_arguments(self) -> None:
        """WebSocketEventSubscriber() requires no constructor arguments."""
        subscriber = WebSocketEventSubscriber()
        assert subscriber.get_queue().empty()


def _make_message() -> Message:
    """Create a minimal SentMessage for testing."""
    import uuid

    from akgentic.core.actor_address_impl import ActorAddressProxy

    addr = ActorAddressProxy({
        "__actor_address__": True,
        "__actor_type__": "akgentic.core.actor_address_impl.ActorAddressProxy",
        "agent_id": str(uuid.uuid4()),
        "name": "test-agent",
        "role": "tester",
        "team_id": str(uuid.uuid4()),
        "squad_id": str(uuid.uuid4()),
        "user_message": False,
    })
    inner = Message(sender=addr)
    return SentMessage(message=inner, recipient=addr, sender=addr)
