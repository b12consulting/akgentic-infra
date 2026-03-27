"""TeamService — orchestrates catalog resolution and TeamManager delegation."""

from __future__ import annotations

import uuid

from akgentic.catalog.models.errors import EntryNotFoundError
from akgentic.catalog.services import (
    AgentCatalog,
    TeamCatalog,
    TemplateCatalog,
    ToolCatalog,
)
from akgentic.infra.server.deps import CommunityServices
from akgentic.team.models import Process, TeamStatus


class TeamService:
    """Service layer bridging catalog resolution with team lifecycle management.

    Resolves catalog entry IDs to TeamCards, delegates lifecycle operations
    to TeamManager, and queries EventStore for listing.
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
        self._services.team_manager.delete_team(team_id)
