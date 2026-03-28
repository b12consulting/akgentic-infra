"""InteractionChannelDispatcher — per-team outbound message dispatcher."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from akgentic.core.messages import SentMessage

if TYPE_CHECKING:
    from akgentic.core.messages import Message
    from akgentic.infra.protocols.channels import InteractionChannelAdapter


class InteractionChannelDispatcher:
    """Routes outbound SentMessage events to the first matching channel adapter.

    Satisfies the EventSubscriber protocol from akgentic.core.orchestrator
    via structural subtyping. Each team gets its own dispatcher instance
    with its own ordered adapter list.

    First-match dispatch: iterates adapters, calls matches() on each,
    and delivers to the first match only. If no adapter matches, the
    message is silently skipped (web channel handles it via WebSocket).
    """

    def __init__(
        self, adapters: list[InteractionChannelAdapter], team_id: uuid.UUID
    ) -> None:
        self._adapters = adapters
        self._team_id = team_id
        self._restoring = False

    def set_restoring(self, restoring: bool) -> None:
        """Toggle restore mode to suppress delivery during event replay."""
        self._restoring = restoring

    def on_message(self, msg: Message) -> None:
        """Dispatch a SentMessage to the first matching adapter.

        Skips delivery entirely during restore mode. Ignores non-SentMessage
        events. Silently skips if no adapter matches.

        Args:
            msg: Orchestrator event message.
        """
        if self._restoring:
            return
        if not isinstance(msg, SentMessage):
            return
        for adapter in self._adapters:
            if adapter.matches(msg):
                adapter.deliver(msg)
                break

    def on_stop(self) -> None:
        """Clean up all registered adapters when the team stops."""
        for adapter in self._adapters:
            adapter.on_stop(self._team_id)
