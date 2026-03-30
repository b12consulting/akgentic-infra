"""WebSocketEventSubscriber — per-connection event subscriber for WebSocket streaming."""

from __future__ import annotations

import logging
import queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from akgentic.core.messages import Message

logger = logging.getLogger(__name__)


class WebSocketEventSubscriber:
    """Bridges Pykka actor thread events to async WebSocket via a thread-safe queue.

    Satisfies the EventSubscriber protocol from akgentic.core.orchestrator
    via structural subtyping (same pattern as TelemetrySubscriber). Each
    WebSocket connection gets its own subscriber instance.

    The producer (on_message) runs in a Pykka actor thread; the consumer
    (WebSocket route) reads from the queue in an async executor.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()

    def on_message(self, msg: Message) -> None:
        """Serialize message to JSON and enqueue for WebSocket delivery.

        Args:
            msg: Orchestrator event message.
        """
        json_str: str = msg.model_dump_json()
        self._queue.put(json_str)
        logger.debug("Event queued: %s", type(msg).__name__)

    def on_stop(self) -> None:
        """Signal connection closure by enqueuing a sentinel None."""
        logger.debug("WebSocketEventSubscriber stopped, sentinel queued")
        self._queue.put(None)

    def get_queue(self) -> queue.Queue[str | None]:
        """Return the internal queue for the WebSocket route to consume.

        Returns:
            Thread-safe queue of JSON strings (or None sentinel).
        """
        return self._queue
