"""WorkerHandle protocol — tier-agnostic worker lifecycle operations."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from akgentic.infra.protocols.team_handle import TeamHandle
    from akgentic.team.models import Process


@runtime_checkable
class WorkerHandle(Protocol):
    """Tier-agnostic handle for worker-level team lifecycle operations.

    Abstracts stop/delete/resume/get operations so that ``TeamService``
    can manage team lifecycle without knowing the underlying tier implementation.

    Implementations: LocalWorkerHandle (community), RemoteWorkerHandle (enterprise).
    """

    def stop_team(self, team_id: uuid.UUID) -> None:
        """Stop a running team.

        Args:
            team_id: ID of the team to stop.
        """
        ...

    def delete_team(self, team_id: uuid.UUID) -> None:
        """Delete a team and its resources.

        Args:
            team_id: ID of the team to delete.
        """
        ...

    def resume_team(self, team_id: uuid.UUID) -> TeamHandle:
        """Resume a stopped team.

        Args:
            team_id: ID of the team to resume.

        Returns:
            A TeamHandle for interacting with the resumed team.
        """
        ...

    def get_team(self, team_id: uuid.UUID) -> Process | None:
        """Get team metadata.

        Args:
            team_id: ID of the team to look up.

        Returns:
            The Process metadata, or None if not found.
        """
        ...
