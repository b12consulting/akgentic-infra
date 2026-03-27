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

    def on_message(self, msg: Message) -> None:
        """Log and trace an orchestrator event via logfire.

        Args:
            msg: Orchestrator telemetry message
        """
        msg_type = type(msg).__name__
        logfire.info(
            "orchestrator event: {msg_type}",
            msg_type=msg_type,
        )

    def on_stop(self) -> None:
        """Cleanup hook — logs shutdown."""
        logger.debug("TelemetrySubscriber stopped")
