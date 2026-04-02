"""Tests for EventStreamSubscriber adapter."""

from __future__ import annotations

import inspect
import uuid
from unittest.mock import MagicMock

from akgentic.core.messages import Message

from akgentic.infra.adapters.community.local_event_stream import LocalEventStream
from akgentic.infra.adapters.shared.event_stream_subscriber import EventStreamSubscriber


def _make_message(team_id: uuid.UUID | None = None) -> Message:
    """Create a Message with the given team_id."""
    return Message(team_id=team_id)


class TestEventStreamSubscriberProtocolCompliance:
    """AC1: EventStreamSubscriber implements EventSubscriber protocol."""

    def test_has_on_message_method(self) -> None:
        subscriber = EventStreamSubscriber(event_stream=LocalEventStream())
        assert callable(subscriber.on_message)

    def test_has_on_stop_method(self) -> None:
        subscriber = EventStreamSubscriber(event_stream=LocalEventStream())
        assert callable(subscriber.on_stop)

    def test_on_message_signature_matches_protocol(self) -> None:
        sig = inspect.signature(EventStreamSubscriber.on_message)
        assert "msg" in sig.parameters

    def test_on_stop_signature_matches_protocol(self) -> None:
        sig = inspect.signature(EventStreamSubscriber.on_stop)
        params = [p for p in sig.parameters if p != "self"]
        assert len(params) == 0


class TestOnMessage:
    """AC2, AC9: on_message wraps and appends, per-team sequences."""

    def test_valid_team_id_creates_persisted_event_and_appends(self) -> None:
        """AC2: on_message with valid team_id creates PersistedEvent and calls append."""
        stream = LocalEventStream()
        subscriber = EventStreamSubscriber(event_stream=stream)
        team_id = uuid.uuid4()
        msg = _make_message(team_id=team_id)

        subscriber.on_message(msg)

        events = stream.read_from(team_id)
        assert len(events) == 1
        assert events[0].team_id == team_id
        assert events[0].sequence == 1
        assert events[0].event.id == msg.id
        assert events[0].timestamp is not None

    def test_team_id_none_is_silently_skipped(self) -> None:
        """AC2: Messages with team_id=None are silently skipped."""
        stream = LocalEventStream()
        subscriber = EventStreamSubscriber(event_stream=stream)
        team_id = uuid.uuid4()
        msg = _make_message(team_id=None)

        subscriber.on_message(msg)

        # Verify no events were appended for any team — use public API
        assert stream.read_from(team_id) == []
        # Subscriber should not have tracked any teams
        assert len(subscriber._seen_teams) == 0

    def test_sequence_numbers_are_per_team_monotonic(self) -> None:
        """AC9: Two teams get independently incrementing sequence numbers."""
        stream = LocalEventStream()
        subscriber = EventStreamSubscriber(event_stream=stream)
        team_a = uuid.uuid4()
        team_b = uuid.uuid4()

        # Interleave messages: A, B, A, B, A
        subscriber.on_message(_make_message(team_id=team_a))
        subscriber.on_message(_make_message(team_id=team_b))
        subscriber.on_message(_make_message(team_id=team_a))
        subscriber.on_message(_make_message(team_id=team_b))
        subscriber.on_message(_make_message(team_id=team_a))

        events_a = stream.read_from(team_a)
        events_b = stream.read_from(team_b)

        assert [e.sequence for e in events_a] == [1, 2, 3]
        assert [e.sequence for e in events_b] == [1, 2]

    def test_persisted_event_fields(self) -> None:
        """AC2: PersistedEvent has correct team_id, incrementing sequence, non-None timestamp."""
        stream = LocalEventStream()
        subscriber = EventStreamSubscriber(event_stream=stream)
        team_id = uuid.uuid4()

        subscriber.on_message(_make_message(team_id=team_id))
        subscriber.on_message(_make_message(team_id=team_id))

        events = stream.read_from(team_id)
        assert len(events) == 2
        for i, ev in enumerate(events, start=1):
            assert ev.team_id == team_id
            assert ev.sequence == i
            assert ev.timestamp is not None


class TestOnStop:
    """AC3: on_stop removes streams for all seen team_ids."""

    def test_on_stop_removes_all_seen_teams(self) -> None:
        """AC3: on_stop calls remove() for all tracked team_ids."""
        stream = LocalEventStream()
        subscriber = EventStreamSubscriber(event_stream=stream)
        team_a = uuid.uuid4()
        team_b = uuid.uuid4()

        subscriber.on_message(_make_message(team_id=team_a))
        subscriber.on_message(_make_message(team_id=team_b))

        subscriber.on_stop()

        # Streams should be removed
        assert stream.read_from(team_a) == []
        assert stream.read_from(team_b) == []

    def test_on_stop_no_messages_is_noop(self) -> None:
        """AC3: on_stop with no messages received is a no-op."""
        stream = LocalEventStream()
        subscriber = EventStreamSubscriber(event_stream=stream)

        subscriber.on_stop()  # Should not raise

    def test_on_stop_handles_remove_exceptions_gracefully(self) -> None:
        """AC3: on_stop handles remove() exceptions (logged, not raised)."""
        mock_stream = MagicMock()
        mock_stream.remove.side_effect = RuntimeError("test error")
        subscriber = EventStreamSubscriber(event_stream=mock_stream)
        team_id = uuid.uuid4()

        subscriber.on_message(_make_message(team_id=team_id))

        # Should not raise even though remove() throws
        subscriber.on_stop()
        mock_stream.remove.assert_called_once_with(team_id)
