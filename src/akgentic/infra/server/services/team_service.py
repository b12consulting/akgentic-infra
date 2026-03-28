"""TeamService — orchestrates catalog resolution and TeamManager delegation."""

from __future__ import annotations

import uuid

from akgentic.agent import HumanProxy
from akgentic.catalog.models.errors import EntryNotFoundError
from akgentic.catalog.services import (
    AgentCatalog,
    TeamCatalog,
    TemplateCatalog,
    ToolCatalog,
)
from akgentic.core.actor_address import ActorAddress
from akgentic.core.messages.message import Message
from akgentic.infra.server.deps import CommunityServices
from akgentic.team.models import PersistedEvent, Process, TeamRuntime, TeamStatus


class TeamService:
    """Service layer bridging catalog resolution with team lifecycle management.

    Resolves catalog entry IDs to TeamCards, delegates lifecycle operations
    to TeamManager, and queries EventStore for listing. Caches TeamRuntime
    instances for action endpoints that require a live runtime handle.
    """

    def __init__(
        self,
        services: CommunityServices,
        team_catalog: TeamCatalog,
        agent_catalog: AgentCatalog,
        tool_catalog: ToolCatalog | None = None,
        template_catalog: TemplateCatalog | None = None,
    ) -> None:
        self._services = services
        self._team_catalog = team_catalog
        self._agent_catalog = agent_catalog
        self._tool_catalog = tool_catalog
        self._template_catalog = template_catalog
        self._runtimes: dict[uuid.UUID, TeamRuntime] = {}

    def create_team(self, catalog_entry_id: str, user_id: str) -> Process:
        """Resolve catalog entry and create a running team.

        Raises:
            EntryNotFoundError: If catalog_entry_id is not found.
        """
        entry = self._team_catalog.get(catalog_entry_id)
        if entry is None:
            raise EntryNotFoundError(catalog_entry_id)
        team_card = entry.to_team_card(
            self._agent_catalog, self._tool_catalog, self._template_catalog
        )
        runtime = self._services.team_manager.create_team(team_card, user_id)
        self._runtimes[runtime.id] = runtime
        process = self._services.team_manager.get_team(runtime.id)
        if process is None:  # pragma: no cover
            msg = f"Team {runtime.id} was created but not found in event store"
            raise RuntimeError(msg)
        return process

    def list_teams(self, user_id: str) -> list[Process]:
        """List all teams for a given user."""
        all_teams = self._services.event_store.list_teams()
        return [t for t in all_teams if t.user_id == user_id]

    def get_team(self, team_id: uuid.UUID) -> Process | None:
        """Get a single team by ID."""
        return self._services.team_manager.get_team(team_id)

    def delete_team(self, team_id: uuid.UUID) -> None:
        """Stop (if running) and delete a team.

        Raises:
            ValueError: If team not found or already deleted.
        """
        process = self._services.team_manager.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        if process.status == TeamStatus.RUNNING:
            self._services.team_manager.stop_team(team_id)
        self._runtimes.pop(team_id, None)
        self._services.team_manager.delete_team(team_id)

    def send_message(self, team_id: uuid.UUID, content: str) -> None:
        """Send a message to a running team.

        Raises:
            ValueError: If team not found or not running.
        """
        runtime = self._get_running_runtime(team_id)
        runtime.send(content)

    def process_human_input(
        self, team_id: uuid.UUID, content: str, message_id: str,
    ) -> None:
        """Route human input to HumanProxy for a specific message.

        Raises:
            ValueError: If team not found, not running, or message not found.
        """
        runtime = self._get_running_runtime(team_id)
        original_message = self._find_message(team_id, message_id)
        human_addr = self._find_human_proxy_addr(runtime)
        proxy = runtime.actor_system.proxy_ask(human_addr, HumanProxy)
        proxy.process_human_input(content, original_message)

    def stop_team(self, team_id: uuid.UUID) -> None:
        """Stop a running team without deleting persisted data.

        Raises:
            ValueError: If team not found or not in a stoppable state.
        """
        process = self._services.team_manager.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        if process.status == TeamStatus.STOPPED:
            msg = f"Team {team_id} is already stopped"
            raise ValueError(msg)
        if process.status == TeamStatus.DELETED:
            msg = f"Team {team_id} has been deleted"
            raise ValueError(msg)
        self._services.team_manager.stop_team(team_id)
        self._runtimes.pop(team_id, None)

    def restore_team(self, team_id: uuid.UUID) -> Process:
        """Restore a stopped team.

        Raises:
            ValueError: If team not found or not in a restorable state.
        """
        process = self._services.team_manager.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        if process.status == TeamStatus.RUNNING:
            msg = f"Team {team_id} is already running"
            raise ValueError(msg)
        if process.status == TeamStatus.DELETED:
            msg = f"Team {team_id} has been deleted"
            raise ValueError(msg)
        runtime = self._services.team_manager.resume_team(team_id)
        self._runtimes[runtime.id] = runtime
        updated = self._services.team_manager.get_team(team_id)
        if updated is None:  # pragma: no cover
            msg = f"Team {team_id} was restored but not found in event store"
            raise RuntimeError(msg)
        return updated

    def get_events(self, team_id: uuid.UUID) -> list[PersistedEvent]:
        """Get all persisted events for a team.

        Raises:
            ValueError: If team not found.
        """
        process = self._services.team_manager.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        return self._services.event_store.load_events(team_id)

    def get_runtime(self, team_id: uuid.UUID) -> TeamRuntime | None:
        """Return the cached runtime for a team, or None if not cached.

        Args:
            team_id: Team UUID.

        Returns:
            TeamRuntime if cached, else None.
        """
        return self._runtimes.get(team_id)

    def _get_running_runtime(self, team_id: uuid.UUID) -> TeamRuntime:
        """Look up a cached runtime, verifying the team is running.

        Raises:
            ValueError: If team not found or not running.
        """
        process = self._services.team_manager.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        if process.status != TeamStatus.RUNNING:
            msg = f"Team {team_id} is not running"
            raise ValueError(msg)
        runtime = self._runtimes.get(team_id)
        if runtime is None:
            msg = f"Team {team_id} runtime not cached"
            raise ValueError(msg)
        return runtime

    def _find_message(self, team_id: uuid.UUID, message_id: str) -> Message:
        """Find a message by ID in persisted events.

        Raises:
            ValueError: If message not found.
        """
        events = self._services.event_store.load_events(team_id)
        for ev in events:
            if str(ev.event.id) == message_id:
                return ev.event
        msg = f"Message {message_id} not found"
        raise ValueError(msg)

    def _find_human_proxy_addr(self, runtime: TeamRuntime) -> ActorAddress:
        """Find the HumanProxy actor address in a runtime.

        Walks the team card members to find the agent whose class is
        HumanProxy, then looks up its address in runtime.addrs.

        Raises:
            ValueError: If no HumanProxy found in team.
        """
        for name, card in runtime.team.agent_cards.items():
            if card.get_agent_class() is HumanProxy:
                addr = runtime.addrs.get(name)
                if addr is not None:
                    return addr
        msg = "No HumanProxy found in team"
        raise ValueError(msg)
