"""LocalTeamHandle — community-tier TeamHandle implementation wrapping TeamRuntime."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from akgentic.agent import HumanProxy
from akgentic.core.orchestrator import EventSubscriber, Orchestrator

if TYPE_CHECKING:
    from akgentic.core.messages.message import Message
    from akgentic.team.models import TeamRuntime


class LocalTeamHandle:
    """Community-tier adapter that delegates TeamHandle methods to a TeamRuntime.

    Wraps an in-process ``TeamRuntime`` and exposes the tier-agnostic
    ``TeamHandle`` protocol so that ``TeamService`` can interact with
    teams without touching actor internals directly.
    """

    def __init__(self, runtime: TeamRuntime) -> None:
        self._runtime = runtime

    @property
    def team_id(self) -> uuid.UUID:
        """The unique identifier of the team this handle points to."""
        return self._runtime.id

    def send(self, content: str) -> None:
        """Send a message to the team's default entry point."""
        self._runtime.send(content)

    def send_to(self, agent_name: str, content: str) -> None:
        """Send a message to a specific agent within the team."""
        self._runtime.send_to(agent_name, content)

    def process_human_input(self, content: str, message: Message) -> None:
        """Route human input to the team's HumanProxy agent.

        Walks the team's agent cards to find the HumanProxy, resolves its
        address, obtains a typed proxy, and delegates the call.

        Raises:
            ValueError: If no HumanProxy found in the team.
        """
        for name, card in self._runtime.team.agent_cards.items():
            if card.get_agent_class() is HumanProxy:
                addr = self._runtime.addrs.get(name)
                if addr is None:
                    msg = f"HumanProxy '{name}' found but has no resolved address"
                    raise ValueError(msg)
                proxy = self._runtime.actor_system.proxy_ask(addr, HumanProxy)
                proxy.process_human_input(content, message)
                return
        msg = "No HumanProxy found in team"
        raise ValueError(msg)

    def subscribe(self, subscriber: EventSubscriber) -> None:
        """Register an event subscriber with the team's orchestrator."""
        orch_proxy = self._runtime.actor_system.proxy_ask(
            self._runtime.orchestrator_addr,
            Orchestrator,
        )
        orch_proxy.subscribe(subscriber)

    def unsubscribe(self, subscriber: EventSubscriber) -> None:
        """Remove an event subscriber from the team's orchestrator."""
        orch_proxy = self._runtime.actor_system.proxy_ask(
            self._runtime.orchestrator_addr,
            Orchestrator,
        )
        orch_proxy.unsubscribe(subscriber)
