"""EventStreamSubscriber -- shared subscriber that routes orchestrator events to EventStream.

Satisfies the ``EventSubscriber`` protocol from ``akgentic.core.orchestrator`` via
structural subtyping. A single instance is shared across all teams; ``team_id`` is
extracted from each ``Message`` to route events to the correct per-team stream.

Threading model:
- A ``threading.Lock`` protects ``_sequences`` and ``_seen_teams`` for safe concurrent
  access from multiple orchestrator actor threads.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from akgentic.core.messages import Message
    from akgentic.infra.protocols.event_stream import EventStream

logger = logging.getLogger(__name__)


class EventStreamSubscriber:
    """Routes orchestrator events into the shared EventStream.

    Wraps each ``Message`` into a ``PersistedEvent`` with per-team monotonic
    sequence numbers and appends it to the injected ``EventStream``.

    On stop, removes streams for all teams that were seen via ``on_message()``.
    """

    def __init__(self, event_stream: EventStream) -> None:
        self._event_stream = event_stream
        self._sequences: dict[uuid.UUID, int] = defaultdict(int)
        self._seen_teams: set[uuid.UUID] = set()
        self._lock = threading.Lock()
        logger.debug("EventStreamSubscriber initialized")

    def on_message(self, msg: Message) -> None:
        """Wrap message into PersistedEvent and append to the event stream.

        Messages with ``team_id=None`` are silently skipped (logged at DEBUG).

        Args:
            msg: Orchestrator message.
        """
        from akgentic.team.models import PersistedEvent

        team_id = msg.team_id
        if team_id is None:
            logger.debug("EventStreamSubscriber: skipping message with team_id=None")
            return

        with self._lock:
            self._sequences[team_id] += 1
            sequence = self._sequences[team_id]
            self._seen_teams.add(team_id)

        event = PersistedEvent(
            team_id=team_id,
            sequence=sequence,
            event=msg,
            timestamp=datetime.now(UTC),
        )
        self._event_stream.append(team_id, event)

    def on_stop(self) -> None:
        """Remove streams for all tracked teams (best-effort cleanup)."""
        with self._lock:
            teams = set(self._seen_teams)

        for team_id in teams:
            try:
                self._event_stream.remove(team_id)
            except Exception:
                logger.debug(
                    "EventStreamSubscriber: remove() failed for team_id=%s",
                    team_id,
                )

        logger.debug("EventStreamSubscriber stopped, cleaned up %d team streams", len(teams))
