"""LocalIngestion — community-tier InteractionChannelIngestion implementation."""

from __future__ import annotations

import uuid

from akgentic.infra.server.services.team_service import TeamService


class LocalIngestion:
    """Routes inbound channel messages directly to TeamManager in-process.

    Community-tier implementation of InteractionChannelIngestion.
    Delegates all operations to TeamService, which already encapsulates
    catalog resolution, TeamManager lifecycle, and runtime caching.
    """

    def __init__(self, team_service: TeamService) -> None:
        self._team_service = team_service

    async def route_reply(
        self,
        team_id: uuid.UUID,
        content: str,
        original_message_id: str | None = None,
    ) -> None:
        """Route an inbound reply to an existing team.

        Args:
            team_id: Target team ID.
            content: Message content from the human.
            original_message_id: Optional ID of the message being replied to.
        """
        self._team_service.send_message(team_id, content)

    async def initiate_team(
        self,
        content: str,
        channel_user_id: str,
        catalog_entry_id: str,
    ) -> uuid.UUID:
        """Create a new team and send the initial message.

        Args:
            content: Initial message content.
            channel_user_id: Channel-specific user identifier.
            catalog_entry_id: Catalog entry to use for team creation.

        Returns:
            The newly created team's ID.
        """
        process = self._team_service.create_team(
            catalog_entry_id, user_id=channel_user_id
        )
        self._team_service.send_message(process.team_id, content)
        return process.team_id
