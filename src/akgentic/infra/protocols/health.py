"""HealthMonitor protocol — monitors worker instance liveness and detects expired workers."""

from __future__ import annotations

import uuid
from typing import Protocol


class HealthMonitor(Protocol):
    """Monitors worker instance liveness and detects expired workers.

    Implementations: RedisHealthMonitor (department),
    DaprHealthMonitor (enterprise).
    Community tier has no health monitoring (single process).
    """

    def check_health(self) -> list[uuid.UUID]:
        """Check all registered workers and return expired instance IDs.

        Returns:
            List of worker instance IDs that have expired their heartbeat
        """
        ...
