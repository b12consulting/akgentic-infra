"""LocalEventStream — community-tier in-memory EventStream with per-reader signaling.

Provides a thread-safe, in-process event stream backed by a plain dict.
Designed for single-process community deployments where no external
infrastructure (Redis, Kafka) is available.

Threading model (CPython GIL assumption):
- One ``threading.Lock`` (``_lock``) on the ``LocalEventStream`` protects
  the ``_streams`` dict during ``append()``, ``subscribe()``, ``remove()``.
- One ``threading.Lock`` per ``_TeamStream`` protects its ``signals`` set
  and ``closed`` flag.
- ``read_next()`` is lock-free during replay: CPython's GIL guarantees
  atomic ``list.append()`` and ``len()``, so a reader whose cursor is
  behind the write frontier can safely index into ``events`` without
  holding any lock. The reader only blocks on its own ``threading.Event``
  when fully caught up.
- The ``closed`` flag gates **writes and exhaustion**, never reads: it stops
  ``append()`` and it makes a caught-up reader terminal, but it never hides
  events that were already written. Readers drain first, then raise.
- Safe for concurrent use from FastAPI thread-pool executors.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from akgentic.infra.protocols.event_stream import StreamClosed

if TYPE_CHECKING:
    from akgentic.core.messages import Message

logger = logging.getLogger(__name__)


@dataclass
class _TeamStream:
    """Per-team stream state: events, per-reader signals, lock, and closed flag."""

    events: list[Message] = field(default_factory=list)
    signals: set[threading.Event] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)
    closed: bool = False


class LocalStreamReader:
    """Cursor-based blocking reader for a single team's event stream.

    Each reader holds its own ``threading.Event`` for wake-up signaling
    and an absolute cursor into the parent ``_TeamStream.events`` list.

    The replay path (``cursor < len(events)``) is lock-free under CPython's
    GIL. The reader only blocks on ``_signal.wait()`` when fully caught up.
    """

    def __init__(
        self,
        team_stream: _TeamStream,
        signal: threading.Event,
        cursor: int,
    ) -> None:
        self._team_stream = team_stream
        self._signal = signal
        self._cursor = cursor
        self._closed = False

    def _advance(self) -> Message | None:
        """Advance the cursor and return the next event, or ``None`` if caught up."""
        if self._cursor < len(self._team_stream.events):
            event = self._team_stream.events[self._cursor]
            self._cursor += 1
            return event
        return None

    def _drain_or_raise(self) -> Message | None:
        """Return the next event, or raise once there is none left to give.

        Drains before honouring the stream's closed flag: ``closed`` means no
        more writes, not discard what was already written. ``append()`` refuses
        a closed stream, so closed-and-caught-up is terminal by construction.
        A reader that closed *itself* raises without draining.
        """
        if self._closed:
            raise StreamClosed()
        event = self._advance()
        if event is not None:
            return event
        if self._team_stream.closed:
            raise StreamClosed()
        return None

    def read_next(self, timeout: float = 0.5) -> Message | None:
        """Read the next event from the cursor position.

        Drains before closing: every event written to the stream before it was
        removed is delivered, and ``StreamClosed`` is raised only once the
        cursor has reached the end of ``events``. A reader that closed *itself*
        raises immediately, without draining.

        Lock-free when replaying (``cursor < len(events)``). Blocks on
        the reader's own ``threading.Event`` when caught up.

        Args:
            timeout: Maximum seconds to block.

        Returns:
            The next event, or ``None`` on timeout.

        Raises:
            StreamClosed: If this reader was closed, or if the stream was
                removed and the cursor has reached the end of ``events``.
        """
        # Lock-free replay path
        event = self._drain_or_raise()
        if event is not None:
            return event

        # Caught up — wait for signal
        self._signal.clear()

        # Re-check after clear to avoid lost-wakeup race
        event = self._drain_or_raise()
        if event is not None:
            return event

        self._signal.wait(timeout=timeout)

        return self._drain_or_raise()

    def close(self) -> None:
        """Release resources held by this reader. Idempotent."""
        if self._closed:
            return
        self._closed = True
        with self._team_stream.lock:
            self._team_stream.signals.discard(self._signal)


class LocalEventStream:
    """In-memory EventStream for the community tier.

    Satisfies the ``EventStream`` protocol (runtime-checkable). Backed by
    ``dict[UUID, _TeamStream]`` with a ``threading.Lock`` for thread safety.
    """

    def __init__(self) -> None:
        self._streams: dict[uuid.UUID, _TeamStream] = {}
        self._lock = threading.Lock()

    def append(self, team_id: uuid.UUID, event: Message) -> int:
        """Append an event to the team's stream.

        Creates the stream implicitly if it does not exist. Returns a
        monotonically increasing sequence number (per team).

        Args:
            team_id: ID of the team.
            event: The message to append.

        Returns:
            Monotonically increasing sequence number, or -1 if stream is closed.
        """
        with self._lock:
            ts = self._streams.get(team_id)
            if ts is None:
                ts = _TeamStream()
                self._streams[team_id] = ts

        with ts.lock:
            if ts.closed:
                logger.warning("append() on removed stream team_id=%s — discarding", team_id)
                return -1
            ts.events.append(event)
            seq = len(ts.events)
            for sig in ts.signals:
                sig.set()
            return seq

    def read_from(
        self, team_id: uuid.UUID, cursor: int = 0
    ) -> list[Message]:
        """Read all events from cursor position (non-blocking snapshot).

        A closed stream still yields what was written to it: ``closed`` means
        no more writes, not discard what is written. Only a team with no stream
        entry at all genuinely has no events.

        Args:
            team_id: ID of the team.
            cursor: Starting position (0 = full history).

        Returns:
            List of events from cursor to current end.
        """
        with self._lock:
            ts = self._streams.get(team_id)
            if ts is None:
                return []

        return list(ts.events[cursor:])

    def subscribe(
        self, team_id: uuid.UUID, cursor: int = 0
    ) -> LocalStreamReader:
        """Create a cursor-based blocking reader for the team's stream.

        Creates the stream implicitly if it does not exist.

        Args:
            team_id: ID of the team.
            cursor: Starting position (0 = replay full history then live).

        Returns:
            A LocalStreamReader that yields events from cursor position.
        """
        with self._lock:
            ts = self._streams.get(team_id)
            if ts is None:
                ts = _TeamStream()
                self._streams[team_id] = ts

        signal = threading.Event()
        with ts.lock:
            if ts.closed:
                raise StreamClosed()
            ts.signals.add(signal)
        return LocalStreamReader(ts, signal, cursor)

    def remove(self, team_id: uuid.UUID) -> None:
        """Remove the stream for a team.

        Pops the entry from the backing store, sets the closed flag and
        signals all active readers. Readers drain whatever is still ahead of
        their cursor and only then receive ``StreamClosed``.

        ``events`` is deliberately left intact: a reader holds its
        ``_TeamStream`` by direct reference, so the list stays readable for
        draining after the entry is gone. A later cleanup that clears it
        reinstates the loss this ordering exists to prevent (see issue #412).

        Args:
            team_id: ID of the team whose stream to remove.
        """
        with self._lock:
            ts = self._streams.pop(team_id, None)
            if ts is None:
                return

        with ts.lock:
            ts.closed = True
            for sig in ts.signals:
                sig.set()
