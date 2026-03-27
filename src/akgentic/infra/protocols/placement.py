"""PlacementStrategy protocol — selects worker instances for team placement."""

from __future__ import annotations

import uuid
from typing import Protocol, runtime_checkable


@runtime_checkable
class PlacementStrategy(Protocol):
    """Selects a worker instance to host a new team.

    Implementations: LocalPlacement (community), LeastTeamsPlacement (department),
    LabelMatchPlacement / WeightedPlacement / ZoneAwarePlacement (enterprise).
    """

    def select_worker(self, team_id: uuid.UUID) -> uuid.UUID:
        """Select a worker instance for team placement.

        Args:
            team_id: ID of the team being placed

        Returns:
            Worker instance ID where the team should be created
        """
        ...
