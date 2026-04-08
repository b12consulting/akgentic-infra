"""TelemetrySubscriber — shared event subscriber that traces messages via logfire."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import logfire

if TYPE_CHECKING:
    from akgentic.core.messages import Message

logger = logging.getLogger(__name__)


class TelemetrySubscriber:
    """Traces orchestrator events via logfire for observability.

    Satisfies the EventSubscriber protocol from akgentic.core.orchestrator
    via structural subtyping. Thread-safe — designed as a shared, long-lived
    subscriber across all teams.
    """

    def __init__(self) -> None:
        logger.debug("TelemetrySubscriber initialized")
        self._restoring = False

    def set_restoring(self, restoring: bool) -> None:
        """Toggle restore mode to suppress span emission during event replay."""
        self._restoring = restoring

    def on_message(self, msg: Message) -> None:
        """Log and trace an orchestrator event via logfire.

        Args:
            msg: Orchestrator telemetry message
        """
        if self._restoring:
            return

        sender = msg.sender.name if msg.sender else "unknown"
        msg_type = msg.__class__.__name__
        team_id = msg.team_id
        logfire.info(
            "{sender} event: {msg_type} - {team_id}",
            sender=sender,
            msg_type=msg_type,
            team_id=team_id,
        )
        logger.debug("Telemetry event: %s", msg_type)

    def on_stop(self) -> None:
        """Cleanup hook — logs shutdown."""
        logger.debug("TelemetrySubscriber stopped")
