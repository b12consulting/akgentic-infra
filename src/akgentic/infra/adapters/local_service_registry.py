"""LocalServiceRegistry — in-memory service discovery for single-process deployment."""

from __future__ import annotations

import uuid


class LocalServiceRegistry:
    """Tracks team-to-instance mappings in an in-memory dict.

    Satisfies the ServiceRegistry protocol from akgentic.team.ports
    via structural subtyping. Suitable for community-tier single-process
    deployments where all teams live in the same process.
    """

    def __init__(self) -> None:
        self._instances: dict[uuid.UUID, set[uuid.UUID]] = {}

    def register_instance(self, instance_id: uuid.UUID) -> None:
        """Register a worker instance as active.

        Args:
            instance_id: The worker instance to register
        """
        if instance_id not in self._instances:
            self._instances[instance_id] = set()

    def deregister_instance(self, instance_id: uuid.UUID) -> None:
        """Remove a worker instance and all its team associations.

        Args:
            instance_id: The worker instance to deregister
        """
        self._instances.pop(instance_id, None)

    def register_team(self, instance_id: uuid.UUID, team_id: uuid.UUID) -> None:
        """Associate a team with a worker instance.

        Args:
            instance_id: The worker instance hosting the team
            team_id: The team to associate
        """
        if instance_id in self._instances:
            self._instances[instance_id].add(team_id)

    def deregister_team(self, instance_id: uuid.UUID, team_id: uuid.UUID) -> None:
        """Disassociate a team from a worker instance.

        Args:
            instance_id: The worker instance hosting the team
            team_id: The team to disassociate
        """
        if instance_id in self._instances:
            self._instances[instance_id].discard(team_id)

    def find_team(self, team_id: uuid.UUID) -> uuid.UUID | None:
        """Find the worker instance hosting a team.

        Args:
            team_id: The team to locate

        Returns:
            Worker instance ID, or None if team is not registered
        """
        for instance_id, teams in self._instances.items():
            if team_id in teams:
                return instance_id
        return None

    def get_active_instances(self) -> list[uuid.UUID]:
        """Return all currently active worker instance IDs.

        Returns:
            List of active instance IDs
        """
        return list(self._instances.keys())
