"""LocalRuntimeCache — community-tier in-process RuntimeCache implementation."""

from __future__ import annotations

import uuid

from akgentic.infra.protocols.team_handle import TeamHandle


class LocalRuntimeCache:
    """In-process dict-backed cache mapping team IDs to live TeamHandle instances.

    Starts empty on construction — teams are only cached after explicit
    ``store()`` calls (NFR2: ghost team prevention).
    """

    def __init__(self) -> None:
        self._handles: dict[uuid.UUID, TeamHandle] = {}

    def store(self, team_id: uuid.UUID, handle: TeamHandle) -> None:
        """Store a team handle in the cache."""
        self._handles[team_id] = handle

    def get(self, team_id: uuid.UUID) -> TeamHandle | None:
        """Retrieve a team handle, or None if not found."""
        return self._handles.get(team_id)

    def remove(self, team_id: uuid.UUID) -> None:
        """Remove a team handle from the cache (no-op if absent)."""
        self._handles.pop(team_id, None)
