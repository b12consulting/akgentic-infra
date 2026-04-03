"""LocalEventStream — community-tier in-memory EventStream with replay and fan-out.

Provides a thread-safe, in-process event stream backed by a plain dict.
Designed for single-process community deployments where no external
infrastructure (Redis, Kafka) is available.

Threading model:
- One ``threading.Lock`` on the stream protects the ``_streams`` dict and all
  per-team state during ``append()``, ``read_from()``, ``subscribe()``, ``remove()``.
- One ``threading.Condition`` per team for reader notification (``read_next()``
  blocks on the condition until events are available or timeout expires).
- Safe for concurrent use from FastAPI thread-pool executors.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from akgentic.infra.protocols.event_stream import StreamClosed

if TYPE_CHECKING:
    from akgentic.core.messages import Message

logger = logging.getLogger(__name__)


@dataclass
class _TeamStream:
    """Per-team stream state grouping events, condition, readers, and bookkeeping."""

    events: list[Message] = field(default_factory=list)
    condition: threading.Condition = field(default_factory=threading.Condition)
    readers: list[LocalStreamReader] = field(default_factory=list)
    closed: bool = False
    base_offset: int = 0


class LocalStreamReader:
    """Cursor-based blocking reader for a single team's event stream.

    Holds an absolute cursor position and a reference to the parent
    ``_TeamStream``. Blocks on the team's ``Condition`` when no new
    events are available.

    Thread-safety: all mutable state is accessed under the parent
    team's ``Condition`` lock.
    """

    def __init__(
        self,
        team_stream: _TeamStream,
        cursor: int,
    ) -> None:
        self._team_stream = team_stream
        self._cursor = cursor
        self._closed = False

    def read_next(self, timeout: float = 0.5) -> Message | None:
        """Read the next event from the cursor position.

        Blocks up to *timeout* seconds waiting for a new event. Returns
        ``None`` if the timeout elapses with no event available.

        Args:
            timeout: Maximum seconds to block.

        Returns:
            The next event, or ``None`` on timeout.

        Raises:
            StreamClosed: If the stream has been removed.
        """
        with self._team_stream.condition:
            deadline = time.monotonic() + timeout
            while True:
                if self._closed or self._team_stream.closed:
                    raise StreamClosed()
                base = self._team_stream.base_offset
                end = base + len(self._team_stream.events)
                if self._cursor < base:
                    self._cursor = base
                if self._cursor < end:
                    event = self._team_stream.events[self._cursor - base]
                    self._cursor += 1
                    return event
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._team_stream.condition.wait(timeout=remaining)

    def close(self) -> None:
        """Release resources held by this reader. Idempotent."""
        if self._closed:
            return
        self._closed = True
        with self._team_stream.condition:
            try:
                self._team_stream.readers.remove(self)
            except ValueError:
                pass


class LocalEventStream:
    """In-memory EventStream for the community tier.

    Satisfies the ``EventStream`` protocol (runtime-checkable). Backed by
    ``dict[UUID, _TeamStream]`` with a ``threading.Lock`` for thread safety.

    Args:
        maxlen: Optional maximum number of events per team stream.
            When exceeded, oldest events are evicted (FIFO) and reader
            cursors are adjusted.
    """

    def __init__(self, maxlen: int | None = None) -> None:
        self._streams: dict[uuid.UUID, _TeamStream] = {}
        self._lock = threading.Lock()
        self._maxlen = maxlen

    def append(self, team_id: uuid.UUID, event: Message) -> int:
        """Append an event to the team's stream.

        Creates the stream implicitly if it does not exist. Returns a
        monotonically increasing sequence number (per team).

        Args:
            team_id: ID of the team.
            event: The message to append.

        Returns:
            Monotonically increasing sequence number.
        """
        with self._lock:
            ts = self._streams.get(team_id)
            if ts is None:
                ts = _TeamStream()
                self._streams[team_id] = ts

        with ts.condition:
            if ts.closed:
                logger.warning("append() on removed stream team_id=%s — discarding", team_id)
                return -1
            ts.events.append(event)
            seq = ts.base_offset + len(ts.events)

            if self._maxlen is not None and len(ts.events) > self._maxlen:
                evict_count = len(ts.events) - self._maxlen
                ts.events = ts.events[evict_count:]
                ts.base_offset += evict_count
                for reader in ts.readers:
                    if reader._cursor < ts.base_offset:
                        reader._cursor = ts.base_offset

            ts.condition.notify_all()
            return seq

    def read_from(
        self, team_id: uuid.UUID, cursor: int = 0
    ) -> list[Message]:
        """Read all events from cursor position (non-blocking snapshot).

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

        with ts.condition:
            if ts.closed:
                return []
            base = ts.base_offset
            if cursor < base:
                cursor = base
            start_idx = cursor - base
            return list(ts.events[start_idx:])

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

        with ts.condition:
            if ts.closed:
                raise StreamClosed()
            reader = LocalStreamReader(ts, cursor)
            ts.readers.append(reader)
            return reader

    def remove(self, team_id: uuid.UUID) -> None:
        """Remove the stream for a team.

        Sets the closed flag on all active readers and notifies them,
        then deletes the stream data from the backing store.

        Args:
            team_id: ID of the team whose stream to remove.
        """
        with self._lock:
            ts = self._streams.pop(team_id, None)
            if ts is None:
                return

        with ts.condition:
            ts.closed = True
            ts.condition.notify_all()
