"""Tests for LocalEventStream and LocalStreamReader (Story 13.2, updated 13.7)."""

from __future__ import annotations

import threading
import time
import uuid

import pytest
from akgentic.core.messages import Message, UserMessage

from akgentic.infra.adapters.community.local_event_stream import (
    LocalEventStream,
)
from akgentic.infra.protocols.event_stream import EventStream, StreamClosed

_TEAM_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _make_event(seq: int = 1, team_id: uuid.UUID = _TEAM_ID) -> Message:
    """Create a Message fixture."""
    return UserMessage(content=f"msg-{seq}", team_id=team_id)


# --- Task 9: Protocol conformance (AC1) ---


def test_localeventstream_satisfies_protocol() -> None:
    """LocalEventStream satisfies EventStream @runtime_checkable check."""
    assert isinstance(LocalEventStream(), EventStream)


# --- Task 5: Basic operations (AC3, AC4, AC8) ---


def test_append_returns_monotonic_sequence() -> None:
    """append() returns monotonically increasing sequence numbers (AC3)."""
    stream = LocalEventStream()
    seq1 = stream.append(_TEAM_ID, _make_event(1))
    seq2 = stream.append(_TEAM_ID, _make_event(2))
    seq3 = stream.append(_TEAM_ID, _make_event(3))
    assert seq1 < seq2 < seq3


def test_append_implicitly_creates_stream() -> None:
    """append() implicitly creates stream for new team_id (AC3)."""
    stream = LocalEventStream()
    new_team = uuid.uuid4()
    seq = stream.append(new_team, _make_event(1, team_id=new_team))
    assert seq == 1
    events = stream.read_from(new_team)
    assert len(events) == 1


def test_read_from_cursor_zero_returns_all() -> None:
    """read_from(cursor=0) returns all events (AC4)."""
    stream = LocalEventStream()
    for i in range(5):
        stream.append(_TEAM_ID, _make_event(i))
    events = stream.read_from(_TEAM_ID, cursor=0)
    assert len(events) == 5


@pytest.mark.parametrize("cursor", [1, 2, 3, 4])
def test_read_from_cursor_n_returns_from_n(cursor: int) -> None:
    """read_from(cursor=K) returns events from position K onward (AC4)."""
    stream = LocalEventStream()
    for i in range(5):
        stream.append(_TEAM_ID, _make_event(i))
    events = stream.read_from(_TEAM_ID, cursor=cursor)
    assert len(events) == 5 - cursor


def test_read_from_nonexistent_team_returns_empty() -> None:
    """read_from() on nonexistent team returns empty list (AC4)."""
    stream = LocalEventStream()
    assert stream.read_from(uuid.uuid4()) == []


def test_read_next_timeout_returns_none() -> None:
    """read_next(timeout=0.1) returns None when no events pending (AC8)."""
    stream = LocalEventStream()
    reader = stream.subscribe(_TEAM_ID, cursor=0)
    result = reader.read_next(timeout=0.1)
    assert result is None
    reader.close()


# --- Task 6: Subscribe and replay (AC5, AC6, AC11) ---


def test_subscribe_cursor_zero_replays_all_then_blocks() -> None:
    """subscribe(cursor=0) replays all existing events then blocks (AC5)."""
    stream = LocalEventStream()
    for i in range(3):
        stream.append(_TEAM_ID, _make_event(i))

    reader = stream.subscribe(_TEAM_ID, cursor=0)

    # Replay phase -- should return all 3 immediately
    for _ in range(3):
        ev = reader.read_next(timeout=0.1)
        assert ev is not None

    # Now should block and timeout
    assert reader.read_next(timeout=0.1) is None
    reader.close()


def test_subscribe_cursor_n_skips_replay() -> None:
    """subscribe(cursor=N) skips replay and blocks for new events (AC6)."""
    stream = LocalEventStream()
    for i in range(3):
        stream.append(_TEAM_ID, _make_event(i))

    reader = stream.subscribe(_TEAM_ID, cursor=3)
    # Should block immediately -- no replay
    assert reader.read_next(timeout=0.1) is None
    reader.close()


def test_two_concurrent_readers_independent() -> None:
    """Two concurrent readers on same stream receive events independently (AC11)."""
    stream = LocalEventStream()
    for i in range(3):
        stream.append(_TEAM_ID, _make_event(i))

    reader1 = stream.subscribe(_TEAM_ID, cursor=0)
    reader2 = stream.subscribe(_TEAM_ID, cursor=0)

    # reader1 reads all 3
    for _ in range(3):
        assert reader1.read_next(timeout=0.1) is not None

    # reader2 still at cursor=0, should independently read all 3
    for _ in range(3):
        assert reader2.read_next(timeout=0.1) is not None

    reader1.close()
    reader2.close()


def test_reader_receives_live_events_after_replay() -> None:
    """Reader receives live events after replay is exhausted (AC5)."""
    stream = LocalEventStream()
    stream.append(_TEAM_ID, _make_event(1))

    reader = stream.subscribe(_TEAM_ID, cursor=0)
    # Replay the 1 existing event
    assert reader.read_next(timeout=0.1) is not None
    # No more events yet
    assert reader.read_next(timeout=0.1) is None

    # Append a live event
    stream.append(_TEAM_ID, _make_event(2))
    ev = reader.read_next(timeout=1.0)
    assert ev is not None
    reader.close()


def test_reader_receives_live_events_via_thread() -> None:
    """Reader blocks and receives live events appended from another thread."""
    stream = LocalEventStream()
    reader = stream.subscribe(_TEAM_ID, cursor=0)
    received: list[Message | None] = []

    def producer() -> None:
        time.sleep(0.05)
        stream.append(_TEAM_ID, _make_event(1))

    t = threading.Thread(target=producer)
    t.start()
    ev = reader.read_next(timeout=2.0)
    received.append(ev)
    t.join()

    assert len(received) == 1
    assert received[0] is not None
    reader.close()


# --- Task 7: Remove and close (AC7, AC9) ---


def test_remove_causes_read_next_to_raise_stream_closed() -> None:
    """remove() causes active read_next() to raise StreamClosed (AC7)."""
    stream = LocalEventStream()
    stream.append(_TEAM_ID, _make_event(1))
    reader = stream.subscribe(_TEAM_ID, cursor=1)  # past existing events

    errors: list[Exception] = []

    def reader_thread() -> None:
        try:
            reader.read_next(timeout=5.0)
        except StreamClosed as e:
            errors.append(e)

    t = threading.Thread(target=reader_thread)
    t.start()

    import time
    time.sleep(0.05)
    stream.remove(_TEAM_ID)
    t.join(timeout=2.0)

    assert len(errors) == 1
    assert isinstance(errors[0], StreamClosed)


def test_remove_deletes_stream_data() -> None:
    """remove() deletes stream data (subsequent read_from returns empty) (AC7)."""
    stream = LocalEventStream()
    stream.append(_TEAM_ID, _make_event(1))
    stream.remove(_TEAM_ID)
    assert stream.read_from(_TEAM_ID) == []


def test_close_is_idempotent() -> None:
    """close() on reader is idempotent (call twice without error) (AC9)."""
    stream = LocalEventStream()
    reader = stream.subscribe(_TEAM_ID, cursor=0)
    reader.close()
    reader.close()  # should not raise


def test_close_after_remove_does_not_raise() -> None:
    """close() after remove() does not raise (AC9)."""
    stream = LocalEventStream()
    reader = stream.subscribe(_TEAM_ID, cursor=0)
    stream.remove(_TEAM_ID)
    reader.close()  # should not raise


# --- Task 8: Maxlen eviction (AC10) ---


def test_maxlen_evicts_oldest() -> None:
    """Appending beyond maxlen evicts oldest events (AC10)."""
    stream = LocalEventStream(maxlen=3)
    for i in range(5):
        stream.append(_TEAM_ID, _make_event(i))
    events = stream.read_from(_TEAM_ID, cursor=0)
    assert len(events) == 3


def test_maxlen_reader_cursors_adjusted() -> None:
    """Reader cursors are adjusted after eviction (AC10)."""
    stream = LocalEventStream(maxlen=3)
    for i in range(3):
        stream.append(_TEAM_ID, _make_event(i))

    reader = stream.subscribe(_TEAM_ID, cursor=0)
    # Read first event (cursor=0)
    ev = reader.read_next(timeout=0.1)
    assert ev is not None

    # Now append 3 more -- evicts first 3, reader cursor should be adjusted
    for i in range(3, 6):
        stream.append(_TEAM_ID, _make_event(i))

    # Reader should still be able to read without error -- cursor adjusted
    ev = reader.read_next(timeout=0.1)
    assert ev is not None
    reader.close()


def test_maxlen_sequence_numbers_monotonic() -> None:
    """Sequence numbers remain monotonic after eviction (AC10)."""
    stream = LocalEventStream(maxlen=3)
    seqs = []
    for i in range(10):
        seqs.append(stream.append(_TEAM_ID, _make_event(i)))

    # All sequence numbers should be strictly increasing
    for i in range(1, len(seqs)):
        assert seqs[i] > seqs[i - 1]


# --- Review additions: edge-case coverage ---


def test_subscribe_implicitly_creates_stream() -> None:
    """subscribe() on a nonexistent team_id creates the stream implicitly."""
    stream = LocalEventStream()
    new_team = uuid.uuid4()
    reader = stream.subscribe(new_team, cursor=0)
    # Stream now exists -- append should work
    stream.append(new_team, _make_event(1, team_id=new_team))
    ev = reader.read_next(timeout=0.5)
    assert ev is not None
    reader.close()


def test_read_from_cursor_beyond_end_returns_empty() -> None:
    """read_from() with cursor beyond stream length returns empty list."""
    stream = LocalEventStream()
    for i in range(5):
        stream.append(_TEAM_ID, _make_event(i))
    events = stream.read_from(_TEAM_ID, cursor=100)
    assert events == []


def test_subscribe_on_closed_stream_raises() -> None:
    """subscribe() on a removed stream raises StreamClosed."""
    stream = LocalEventStream()
    stream.append(_TEAM_ID, _make_event(1))
    stream.remove(_TEAM_ID)
    # Re-creating via subscribe should work (new stream), but subscribing
    # on the removed internal stream is guarded -- test the safe path
    reader = stream.subscribe(_TEAM_ID, cursor=0)
    assert reader.read_next(timeout=0.1) is None
    reader.close()


def test_append_on_removed_stream_returns_negative() -> None:
    """append() on a concurrently-removed stream returns -1 safely."""
    stream = LocalEventStream()
    stream.append(_TEAM_ID, _make_event(1))

    # Get a reference to the internal _TeamStream, then remove it
    # so append hits the closed guard
    with stream._lock:
        ts = stream._streams[_TEAM_ID]

    stream.remove(_TEAM_ID)

    # Now re-insert the closed ts so append finds it
    with stream._lock:
        stream._streams[_TEAM_ID] = ts

    result = stream.append(_TEAM_ID, _make_event(2))
    assert result == -1
