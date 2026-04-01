"""LocalWorkerHandle — community-tier WorkerHandle wrapping TeamManager."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from akgentic.infra.adapters.community.local_team_handle import LocalTeamHandle
from akgentic.team.manager import TeamManager
from akgentic.team.ports import ServiceRegistry

if TYPE_CHECKING:
    from akgentic.infra.protocols.team_handle import TeamHandle
    from akgentic.team.models import Process


logger = logging.getLogger(__name__)


class LocalWorkerHandle:
    """Community-tier adapter delegating WorkerHandle methods to TeamManager.

    Wraps an in-process ``TeamManager`` and ``ServiceRegistry`` to provide
    tier-agnostic worker lifecycle operations.
    """

    def __init__(
        self,
        team_manager: TeamManager,
        service_registry: ServiceRegistry,
    ) -> None:
        self._team_manager = team_manager
        self._service_registry = service_registry

    def stop_team(self, team_id: uuid.UUID) -> None:
        """Stop a running team by delegating to TeamManager."""
        logger.debug("Stopping team: %s", team_id)
        self._team_manager.stop_team(team_id)

    def delete_team(self, team_id: uuid.UUID) -> None:
        """Delete a team by delegating to TeamManager."""
        logger.debug("Deleting team: %s", team_id)
        self._team_manager.delete_team(team_id)

    def resume_team(self, team_id: uuid.UUID) -> TeamHandle:
        """Resume a stopped team and return a LocalTeamHandle."""
        logger.debug("Resuming team: %s", team_id)
        runtime = self._team_manager.resume_team(team_id)
        return LocalTeamHandle(runtime)

    def get_team(self, team_id: uuid.UUID) -> Process | None:
        """Get team metadata by delegating to TeamManager."""
        return self._team_manager.get_team(team_id)
