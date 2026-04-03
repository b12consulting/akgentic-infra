"""EventStreamSubscriber -- shared subscriber that routes orchestrator events to EventStream.

Satisfies the ``EventSubscriber`` protocol from ``akgentic.core.orchestrator`` via
structural subtyping. A single instance is shared across all teams; ``team_id`` is
extracted from each ``Message`` to route events to the correct per-team stream.

Threading model:
- A ``threading.Lock`` protects ``_seen_teams`` for safe concurrent access from
  multiple orchestrator actor threads.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from akgentic.core.messages import Message
    from akgentic.infra.protocols.event_stream import EventStream

logger = logging.getLogger(__name__)


class EventStreamSubscriber:
    """Routes orchestrator events into the shared EventStream.

    Forwards each ``Message`` directly to the injected ``EventStream``.

    On stop, removes streams for all teams that were seen via ``on_message()``.
    """

    def __init__(self, event_stream: EventStream) -> None:
        self._event_stream = event_stream
        self._seen_teams: set[uuid.UUID] = set()
        self._lock = threading.Lock()
        self._restoring = False
        logger.debug("EventStreamSubscriber initialized")

    def set_restoring(self, restoring: bool) -> None:  # noqa: FBT001
        """Toggle restore-replay guard.

        When ``True``, ``on_message()`` silently skips all events so that
        the restore replay does not flood the ephemeral EventStream with
        historical events that are already available via the EventStore.

        Args:
            restoring: Whether a restore replay is in progress.
        """
        self._restoring = restoring

    def on_message(self, msg: Message) -> None:
        """Forward message directly to the event stream.

        Messages with ``team_id=None`` are silently skipped (logged at DEBUG).
        All messages are skipped during restore replay (``_restoring=True``).

        Args:
            msg: Orchestrator message.
        """
        if self._restoring:
            logger.debug("EventStreamSubscriber: skipping message during restore replay")
            return

        team_id = msg.team_id
        if team_id is None:
            logger.debug("EventStreamSubscriber: skipping message with team_id=None")
            return

        with self._lock:
            self._seen_teams.add(team_id)

        self._event_stream.append(team_id, msg)

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
