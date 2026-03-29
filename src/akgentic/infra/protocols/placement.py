"""PlacementStrategy protocol — creates teams on worker instances."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from akgentic.infra.protocols.team_handle import TeamHandle
    from akgentic.team.models import TeamCard


@runtime_checkable
class PlacementStrategy(Protocol):
    """Creates a team on a selected worker instance and returns a handle.

    Encapsulates worker selection and team creation so that ``TeamService``
    never needs to know about ``TeamManager`` or actor internals.

    Implementations: LocalPlacement (community), LeastTeamsPlacement (department),
    LabelMatchPlacement / WeightedPlacement / ZoneAwarePlacement (enterprise).
    """

    def create_team(self, team_card: TeamCard, user_id: str) -> TeamHandle:
        """Create a team on a worker instance and return a handle.

        Args:
            team_card: Team configuration card.
            user_id: ID of the user creating the team.

        Returns:
            A TeamHandle for interacting with the newly created team.
        """
        ...
