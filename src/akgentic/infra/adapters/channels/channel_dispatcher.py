"""InteractionChannelDispatcher — one shared, team-agnostic outbound dispatcher."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from akgentic.core.messages import SentMessage

if TYPE_CHECKING:
    from akgentic.core.messages import Message
    from akgentic.infra.protocols.channels import ChannelRegistry, InteractionChannelAdapter

logger = logging.getLogger(__name__)


class InteractionChannelDispatcher:
    """Routes outbound SentMessage events to the channel that answers them.

    Satisfies the ``EventSubscriber`` protocol from ``akgentic.core.orchestrator``
    via structural subtyping. **One instance is shared by every team** in the
    process: the team is read off the message (``SentMessage.recipient.team_id``)
    and off the lifecycle hooks, exactly as that Protocol's docstring intends.

    Delivery is **opt-in**. A message is only ever delivered to an agent the
    registry holds a ``ChannelBinding`` for; an unbound agent's message is left
    to the WebSocket, and nothing is posted anywhere. The binding is also what
    names the destination chat, which the recipient address cannot.

    Multi-channel dispatch: iterates adapters, calls ``matches()`` on each, and
    delivers to ALL matching adapters.

    Blocking:
        ``on_message`` runs in a Pykka actor thread with no event loop and must
        **never** block or await — its only registry call is
        ``find_binding_sync``, which answers from memory. A missed delivery is
        recoverable; a stalled actor thread is not. ``on_stop`` makes no
        registry call at all — it discards one set entry and fans out to the
        adapters (ADR-043 §D5, as amended by ADR-045 §D6).

    Concurrency:
        Sharing the instance also means sharing it across threads. Every team
        has its own orchestrator actor thread, so ``set_restoring`` for one team
        can run while ``on_message`` for another reads ``_restoring`` and a
        third calls ``on_stop``. ``set`` membership, ``add`` and ``discard`` are
        each a single atomic operation under CPython, and ``_adapters`` and
        ``_registry`` are never reassigned after construction, so no lock is
        needed **as this class stands**. Anything added here that mutates state
        across more than one operation — a read-modify-write, a dict built up
        over two statements — does need one.
    """

    def __init__(
        self,
        adapters: list[InteractionChannelAdapter],
        registry: ChannelRegistry,
    ) -> None:
        self._adapters = list(adapters)
        self._registry = registry
        self._restoring: set[uuid.UUID] = set()

    def set_restoring(self, team_id: uuid.UUID, restoring: bool) -> None:
        """Toggle restore mode for one team, to suppress delivery during replay.

        Suppression is per-team: the instance is shared, so a single flag would
        mute every other team for the length of one team's replay.

        Args:
            team_id: ``team_id`` from the orchestrator whose replay is starting
                or finishing.
            restoring: ``True`` while restore replay is in progress, ``False``
                to resume dispatch.
        """
        if restoring:
            self._restoring.add(team_id)
        else:
            self._restoring.discard(team_id)

    def on_message(self, msg: Message) -> None:
        """Dispatch a SentMessage to every adapter matching the recipient's binding.

        Four conditions, in order: the event is a ``SentMessage``; its team is
        not replaying; the recipient agent has a binding; and an adapter claims
        it. Absence means skip, never guess — an unbound agent's message is
        silently left to the WebSocket.

        **The binding is the whole rule, including for agent-to-agent
        traffic.** This used to drop anything whose recipient was not a user
        proxy, so that intra-team traffic could never leave the team. A binding
        is now written for an arbitrary agent only when a user asks for one
        (``/register <team-id> @Agent``), so the lookup already carries that
        consent, and the extra test only made the explicit request silently do
        nothing. A chat bound to a member therefore sees what that member is
        told — which is what binding it means.

        Performs no blocking call: ``find_binding_sync`` answers from an
        in-process index. The orchestrator already guards every subscriber
        callback, so no local ``try/except`` is needed or wanted here.

        Args:
            msg: Orchestrator event message.
        """
        if not isinstance(msg, SentMessage):
            return
        if msg.recipient.team_id in self._restoring:
            return
        binding = self._registry.find_binding_sync(msg.recipient.team_id, msg.recipient.name)
        if binding is None:
            return
        logger.debug(
            "Dispatching SentMessage to %d adapter(s): team_id=%s, channel=%s",
            len(self._adapters),
            msg.recipient.team_id,
            binding.channel,
        )
        for adapter in self._adapters:
            if adapter.matches(msg, binding):
                adapter.deliver(msg, binding)

    def on_stop_request(self, team_id: uuid.UUID) -> None:  # noqa: ARG002
        """No-op — dispatcher has no work to do on the inactivity-timer signal.

        Channel-side teardown happens on ``on_stop()`` once the team actually
        stops. Present to satisfy the ``EventSubscriber`` Protocol: a
        structurally-typed subscriber that drops the method is silently skipped,
        which reads as working code.

        Args:
            team_id: ``team_id`` from the orchestrator. Accepted to satisfy the
                Protocol but ignored — no work is performed here.
        """

    def on_stop(self, team_id: uuid.UUID) -> None:
        """Notify every adapter. The team's bindings are **kept**.

        A stop is reversible — an idle timeout the user never asked for is the
        usual cause — so the conversation survives it and the next inbound
        message still finds its binding (ADR-045 §D6). Releasing here made
        every idle timeout a lost conversation.

        Bindings are released on **delete**, by the delete route, which is the
        one lifecycle change a conversation cannot survive.

        What that next message then *does* with the binding — resume the
        stopped team and carry on, silently — is ADR-045 §D7, and the router
        does it: it resolves the bound team's state before it sends. Only
        §D7's third row is outstanding (issue #487); a binding whose team is
        gone altogether gets a notice rather than a fresh team. Neither case
        is a reason to reinstate a release here.

        Args:
            team_id: ``team_id`` from the orchestrator — the team being stopped.
        """
        self._restoring.discard(team_id)
        logger.debug("ChannelDispatcher stopped: team_id=%s", team_id)
        for adapter in self._adapters:
            adapter.on_stop(team_id)
