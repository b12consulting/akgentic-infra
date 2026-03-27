"""LocalPlacement — community-tier placement that keeps all teams in the current process."""

from __future__ import annotations

import uuid


class LocalPlacement:
    """Places all teams in the current process instance.

    Satisfies the PlacementStrategy protocol via structural subtyping.
    Always returns the same instance_id (the local process).
    """

    def __init__(self) -> None:
        self._instance_id = uuid.uuid4()

    @property
    def instance_id(self) -> uuid.UUID:
        """The worker instance ID representing this process."""
        return self._instance_id

    def select_worker(self, team_id: uuid.UUID) -> uuid.UUID:
        """Select the local process for team placement.

        Args:
            team_id: ID of the team being placed (unused — always local)

        Returns:
            The local worker instance ID
        """
        return self._instance_id
