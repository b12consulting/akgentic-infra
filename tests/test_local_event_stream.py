"""Tests for LocalEventStream and LocalStreamReader (Story 21.1 — per-reader Event design)."""

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


# --- Protocol conformance ---


def test_localeventstream_satisfies_protocol() -> None:
    """LocalEventStream satisfies EventStream @runtime_checkable check."""
    assert isinstance(LocalEventStream(), EventStream)


# --- Basic operations ---


def test_append_returns_monotonic_sequence() -> None:
    """append() returns monotonically increasing sequence numbers."""
    stream = LocalEventStream()
    seq1 = stream.append(_TEAM_ID, _make_event(1))
    seq2 = stream.append(_TEAM_ID, _make_event(2))
    seq3 = stream.append(_TEAM_ID, _make_event(3))
    assert seq1 < seq2 < seq3


def test_append_implicitly_creates_stream() -> None:
    """append() implicitly creates stream for new team_id."""
    stream = LocalEventStream()
    new_team = uuid.uuid4()
    seq = stream.append(new_team, _make_event(1, team_id=new_team))
    assert seq == 1
    events = stream.read_from(new_team)
    assert len(events) == 1


def test_read_from_cursor_zero_returns_all() -> None:
    """read_from(cursor=0) returns all events."""
    stream = LocalEventStream()
    for i in range(5):
        stream.append(_TEAM_ID, _make_event(i))
    events = stream.read_from(_TEAM_ID, cursor=0)
    assert len(events) == 5


@pytest.mark.parametrize("cursor", [1, 2, 3, 4])
def test_read_from_cursor_n_returns_from_n(cursor: int) -> None:
    """read_from(cursor=K) returns events from position K onward."""
    stream = LocalEventStream()
    for i in range(5):
        stream.append(_TEAM_ID, _make_event(i))
    events = stream.read_from(_TEAM_ID, cursor=cursor)
    assert len(events) == 5 - cursor


def test_read_from_nonexistent_team_returns_empty() -> None:
    """read_from() on nonexistent team returns empty list."""
    stream = LocalEventStream()
    assert stream.read_from(uuid.uuid4()) == []


def test_read_next_timeout_returns_none() -> None:
    """read_next(timeout=0.1) returns None when no events pending."""
    stream = LocalEventStream()
    reader = stream.subscribe(_TEAM_ID, cursor=0)
    result = reader.read_next(timeout=0.1)
    assert result is None
    reader.close()


# --- Subscribe and replay ---


def test_subscribe_cursor_zero_replays_all_then_blocks() -> None:
    """subscribe(cursor=0) replays all existing events then blocks."""
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
    """subscribe(cursor=N) skips replay and blocks for new events."""
    stream = LocalEventStream()
    for i in range(3):
        stream.append(_TEAM_ID, _make_event(i))

    reader = stream.subscribe(_TEAM_ID, cursor=3)
    # Should block immediately -- no replay
    assert reader.read_next(timeout=0.1) is None
    reader.close()


def test_two_concurrent_readers_independent() -> None:
    """Two concurrent readers on same stream receive events independently."""
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
    """Reader receives live events after replay is exhausted."""
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


# --- Remove and close ---


def test_remove_causes_read_next_to_raise_stream_closed() -> None:
    """remove() causes active read_next() to raise StreamClosed."""
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

    time.sleep(0.05)
    stream.remove(_TEAM_ID)
    t.join(timeout=2.0)

    assert len(errors) == 1
    assert isinstance(errors[0], StreamClosed)


def test_remove_deletes_stream_data() -> None:
    """remove() deletes stream data (subsequent read_from returns empty)."""
    stream = LocalEventStream()
    stream.append(_TEAM_ID, _make_event(1))
    stream.remove(_TEAM_ID)
    assert stream.read_from(_TEAM_ID) == []


def test_close_is_idempotent() -> None:
    """close() on reader is idempotent (call twice without error)."""
    stream = LocalEventStream()
    reader = stream.subscribe(_TEAM_ID, cursor=0)
    reader.close()
    reader.close()  # should not raise


def test_close_after_remove_does_not_raise() -> None:
    """close() after remove() does not raise."""
    stream = LocalEventStream()
    reader = stream.subscribe(_TEAM_ID, cursor=0)
    stream.remove(_TEAM_ID)
    reader.close()  # should not raise


# --- Drain before close (issue #412) ---


def test_reader_drains_unconsumed_events_after_remove() -> None:
    """A reader with an unconsumed event receives it even after remove()."""
    stream = LocalEventStream()
    ev1, ev2 = _make_event(1), _make_event(2)
    stream.append(_TEAM_ID, ev1)
    stream.append(_TEAM_ID, ev2)

    reader = stream.subscribe(_TEAM_ID, cursor=0)
    assert reader.read_next(timeout=0.1) is ev1

    stream.remove(_TEAM_ID)

    # The second event was written before the close and must still be delivered
    assert reader.read_next(timeout=0.1) is ev2
    with pytest.raises(StreamClosed):
        reader.read_next(timeout=0.1)
    reader.close()


def test_reader_drains_all_unconsumed_events_after_remove() -> None:
    """Draining is complete and ordered, not partial."""
    stream = LocalEventStream()
    events = [_make_event(i) for i in range(5)]
    for ev in events:
        stream.append(_TEAM_ID, ev)

    reader = stream.subscribe(_TEAM_ID, cursor=0)
    stream.remove(_TEAM_ID)

    drained = [reader.read_next(timeout=0.1) for _ in range(5)]
    assert drained == events
    with pytest.raises(StreamClosed):
        reader.read_next(timeout=0.1)
    reader.close()


def test_read_next_raises_stream_closed_after_drain_completes() -> None:
    """Terminal behaviour is preserved: exhausted + closed keeps raising."""
    stream = LocalEventStream()
    stream.append(_TEAM_ID, _make_event(1))

    reader = stream.subscribe(_TEAM_ID, cursor=0)
    stream.remove(_TEAM_ID)

    assert reader.read_next(timeout=0.1) is not None
    with pytest.raises(StreamClosed):
        reader.read_next(timeout=0.1)
    # Not a one-shot signal -- a closed, drained stream stays closed
    with pytest.raises(StreamClosed):
        reader.read_next(timeout=0.1)
    reader.close()


def test_reader_that_closed_itself_raises_without_draining() -> None:
    """A reader closing itself is a different condition from the stream closing."""
    stream = LocalEventStream()
    stream.append(_TEAM_ID, _make_event(1))
    stream.append(_TEAM_ID, _make_event(2))

    reader = stream.subscribe(_TEAM_ID, cursor=0)
    reader.close()

    with pytest.raises(StreamClosed):
        reader.read_next(timeout=0.1)


def test_reader_blocked_in_wait_drains_after_remove() -> None:
    """A reader blocked inside _signal.wait() still receives a late write.

    This is the path a live WebSocket pump sits in when on_stop removes the
    stream, and the checkpoint a partial fix would miss.
    """
    stream = LocalEventStream()
    reader = stream.subscribe(_TEAM_ID, cursor=0)
    # Caught up, so the next read_next() blocks in wait()
    assert reader.read_next(timeout=0.1) is None

    late = _make_event(1)
    received: list[Message | None] = []
    errors: list[Exception] = []

    def reader_thread() -> None:
        try:
            received.append(reader.read_next(timeout=5.0))
        except StreamClosed as e:  # pragma: no cover - the defect's signature
            errors.append(e)

    t = threading.Thread(target=reader_thread)
    t.start()

    time.sleep(0.05)
    stream.append(_TEAM_ID, late)
    stream.remove(_TEAM_ID)
    t.join(timeout=2.0)

    assert errors == []
    assert received == [late]
    reader.close()


def test_reader_drains_event_arriving_between_clear_and_wait() -> None:
    """Checkpoint 2: an event landing in the lost-wakeup gap is still drained.

    The write is driven from _signal.clear() so it lands deterministically
    between the clear and the wait, where only the second checkpoint can see it.
    """
    stream = LocalEventStream()
    reader = stream.subscribe(_TEAM_ID, cursor=0)
    late = _make_event(1)
    real_clear = reader._signal.clear

    def clear_then_write_and_remove() -> None:
        real_clear()
        reader._signal.clear = real_clear  # only interfere on the first pass
        stream.append(_TEAM_ID, late)
        stream.remove(_TEAM_ID)

    reader._signal.clear = clear_then_write_and_remove

    assert reader.read_next(timeout=0.1) is late
    with pytest.raises(StreamClosed):
        reader.read_next(timeout=0.1)
    reader.close()


def test_reader_drains_event_arriving_between_advance_and_closed_check() -> None:
    """An event landing after the advance comes back empty is still drained.

    The advance reports "caught up" while the stream is still open, so a write
    can land before the closed flag is read. That is the window on_stop drives:
    it appends the last events of teardown and then removes the stream.
    """
    stream = LocalEventStream()
    reader = stream.subscribe(_TEAM_ID, cursor=0)
    late = _make_event(1)
    real_advance = reader._advance

    def advance_then_write_and_remove() -> Message | None:
        result = real_advance()
        reader._advance = real_advance  # only interfere on the first pass
        stream.append(_TEAM_ID, late)
        stream.remove(_TEAM_ID)
        return result

    reader._advance = advance_then_write_and_remove

    assert reader.read_next(timeout=0.1) is late
    with pytest.raises(StreamClosed):
        reader.read_next(timeout=0.1)
    reader.close()


def test_remove_leaves_events_intact() -> None:
    """remove() must not clear events -- readers drain from it after removal."""
    stream = LocalEventStream()
    events = [_make_event(i) for i in range(3)]
    for ev in events:
        stream.append(_TEAM_ID, ev)

    with stream._lock:
        ts = stream._streams[_TEAM_ID]

    stream.remove(_TEAM_ID)

    assert ts.closed is True
    assert ts.events == events


# --- Edge-case coverage ---


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


def test_subscribe_after_remove_creates_new_stream() -> None:
    """subscribe() after remove() creates a fresh stream (remove pops the old one)."""
    stream = LocalEventStream()
    stream.append(_TEAM_ID, _make_event(1))
    stream.remove(_TEAM_ID)
    # Re-creating via subscribe should work (new stream)
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


def test_subscribe_on_concurrently_closed_stream_raises() -> None:
    """subscribe() raises StreamClosed when stream is closed between lookup and lock."""
    stream = LocalEventStream()
    stream.append(_TEAM_ID, _make_event(1))

    # Simulate the race: get a reference to the _TeamStream, close it manually,
    # but leave it in _streams so subscribe() finds it and hits the closed guard
    with stream._lock:
        ts = stream._streams[_TEAM_ID]
    with ts.lock:
        ts.closed = True

    with pytest.raises(StreamClosed):
        stream.subscribe(_TEAM_ID, cursor=0)


def test_read_from_on_closed_stream_returns_events() -> None:
    """read_from() still yields what was written when the stream is closed.

    Replaces the former assertion that a closed stream reads as empty: closed
    means no more writes, not discard what is written (see issue #412).
    """
    stream = LocalEventStream()
    event = _make_event(1)
    stream.append(_TEAM_ID, event)

    # Close the stream without removing it from _streams
    with stream._lock:
        ts = stream._streams[_TEAM_ID]
    with ts.lock:
        ts.closed = True

    assert stream.read_from(_TEAM_ID) == [event]
