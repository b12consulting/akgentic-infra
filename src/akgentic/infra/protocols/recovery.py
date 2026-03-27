"""RecoveryPolicy protocol — determines recovery behavior when a worker instance fails."""

from __future__ import annotations

import uuid
from typing import Protocol


class RecoveryPolicy(Protocol):
    """Determines recovery behavior when a worker instance fails.

    Implementations: MarkStoppedRecovery (department),
    AutoRestoreRecovery / NotifyOnlyRecovery (enterprise).
    Community tier has no recovery (single process).
    """

    def recover(self, instance_id: uuid.UUID, team_ids: list[uuid.UUID]) -> None:
        """Execute recovery for teams orphaned by a failed worker.

        Args:
            instance_id: The failed worker instance ID
            team_ids: Teams that were hosted on the failed worker
        """
        ...
