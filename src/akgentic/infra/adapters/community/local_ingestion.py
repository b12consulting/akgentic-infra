"""LocalIngestion — community-tier InteractionChannelIngestion implementation."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from akgentic.infra.protocols.channels import InitiatedTeam

if TYPE_CHECKING:
    from akgentic.core.messages.message import Message
    from akgentic.infra.protocols.channels import JsonValue
    from akgentic.infra.server.services.team_service import TeamService

logger = logging.getLogger(__name__)


def _is_blank(content: str | Message) -> bool:
    """True when there is nothing for a team to act on.

    A channel delivers empty and whitespace-only bodies for reasons of its own —
    a caption-less photo, an edited message reduced to nothing, a stray newline.
    Handing one to a team is not harmless: the entry point forwards it, the
    supervisor spends an LLM call answering nothing, and the reply that comes
    back is an agent guessing at an empty prompt.

    A pre-formed ``Message`` is judged by its ``content`` when it has one, and
    kept otherwise — a typed message with no text field may carry its payload
    somewhere this function cannot see, and dropping it would lose more than an
    empty string.
    """
    text = content if isinstance(content, str) else getattr(content, "content", None)
    if text is None:
        return False
    return not str(text).strip()


class LocalIngestion:
    """Routes inbound channel messages directly to TeamManager in-process.

    Community-tier implementation of InteractionChannelIngestion.
    Delegates all operations to TeamService, which already encapsulates
    catalog resolution, TeamManager lifecycle, and runtime caching.

    Supports deferred wiring: ``team_service`` can be ``None`` at construction
    time and set later via the property, allowing ``wire_community`` to build
    the ingestion instance before ``TeamService`` exists.

    Deferred wiring pattern:
        ``team_service`` is ``None`` at construction because
        ``wire_community()`` must build ``CommunityServices`` before
        ``TeamService`` exists (TeamService requires CommunityServices).
        ``wire_community`` calls the setter itself once the container is
        complete, so the instance it returns is already bound.
        Enterprise implementations won't use this pattern — their ingestion
        adapters communicate over the network and are fully wired at
        construction time.
    """

    def __init__(self, team_service: TeamService | None = None) -> None:
        self._team_service = team_service

    @property
    def team_service(self) -> TeamService | None:
        """Return the wired TeamService, or None if not yet wired."""
        return self._team_service

    @team_service.setter
    def team_service(self, value: TeamService) -> None:
        """Set the TeamService for deferred wiring."""
        self._team_service = value

    def _require_team_service(self) -> TeamService:
        """Return the wired TeamService or raise if not yet wired."""
        if self._team_service is None:
            msg = "LocalIngestion.team_service has not been wired yet"
            raise RuntimeError(msg)
        return self._team_service

    async def send_message(
        self,
        team_id: uuid.UUID,
        content: str | Message,
        original_message_id: str | None = None,
    ) -> None:
        """Send an inbound human message to an existing team; blank text is dropped.

        Args:
            team_id: Target team ID.
            content: Message content from the human — bare text or a pre-formed
                ``Message``. Passed straight through: ``TeamService.send_message``
                already accepts both, so inspecting the type here would only add
                a branch that can lose what a typed message carries.
            original_message_id: Optional ID of the message being replied to.
                Accepted for Protocol conformance and not threaded any further:
                ``TeamService.send_message`` has no reply-to parameter, so the
                community tier has nowhere to put it.
        """
        logger.info("Inbound message: team_id=%s", team_id)
        if _is_blank(content):
            logger.info("Dropping blank inbound message for team %s", team_id)
            return
        self._require_team_service().send_message(team_id, content)

    async def create_team(
        self,
        channel_user_id: str,
        catalog_entry_id: str,
        metadata: dict[str, JsonValue] | None = None,
    ) -> InitiatedTeam:
        """Create a new team, and send it nothing.

        Args:
            channel_user_id: Channel-specific user identifier.
            catalog_entry_id: Catalog entry to use for team creation.
            metadata: Optional plain-JSON business metadata, forwarded
                unconditionally — including when ``None``. ``TeamService``
                validates it against the resolved card's declared contract and
                raises ``MetadataValidationError`` (422), which is not caught
                here.

        Returns:
            The created team's ID and its entry point's spawned name, both read
            off the ``Process`` ``create_team`` already returned — no second
            service call and no lookup.
        """
        logger.info("Inbound initiation: catalog=%s", catalog_entry_id)
        logger.debug("Initiation user: %s", channel_user_id)
        process = self._require_team_service().create_team(
            catalog_entry_id, user_id=channel_user_id, metadata=metadata
        )

        logger.debug(
            "Team initiated: team_id=%s, entry_point=%s",
            process.team_id,
            process.entry_point.name,
        )
        return InitiatedTeam(team_id=process.team_id, entry_point_name=process.entry_point.name)
