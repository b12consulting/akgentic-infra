"""Channel protocols — interaction channel abstractions for external communication."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from akgentic.core.messages import SentMessage

# Recursive JSON-safe type for webhook payloads — replaces dict[str, Any].
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class ChannelMessage(BaseModel):
    """Normalized message from an external interaction channel."""

    content: str = Field(description="Message content")
    channel_user_id: str = Field(description="Channel-specific user identifier")
    team_id: uuid.UUID | None = Field(default=None, description="Associated team ID")
    message_id: str | None = Field(default=None, description="Channel-specific message ID")


@runtime_checkable
class InteractionChannelAdapter(Protocol):
    """Delivers outbound messages to humans via an external channel.

    Implementations: WebAdapter, TwilioAdapter, SlackAdapter, etc.
    Called synchronously within the actor thread (see threading model note in architecture).
    """

    def matches(self, msg: SentMessage) -> bool:
        """Check if this adapter handles the given message.

        Args:
            msg: The outbound message to check.

        Returns:
            True if this adapter should deliver the message.
        """
        ...

    def deliver(self, msg: SentMessage) -> None:
        """Deliver an outbound message via this channel.

        Args:
            msg: The message to deliver.
        """
        ...

    def on_stop(self, team_id: uuid.UUID) -> None:
        """Clean up resources when a team stops.

        Args:
            team_id: The team being stopped.
        """
        ...


@runtime_checkable
class InteractionChannelIngestion(Protocol):
    """Routes inbound human replies from external channels to the correct team's UserProxy."""

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
        ...

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
        ...


@runtime_checkable
class ChannelParser(Protocol):
    """Parses channel-specific webhook payloads into a common ChannelMessage.

    Runs in FastAPI async context — uses async signatures.
    """

    @property
    def channel_name(self) -> str:
        """The channel name this parser handles (e.g. 'whatsapp', 'slack')."""
        ...

    @property
    def default_catalog_entry(self) -> str:
        """Default catalog entry ID to use when initiating a new team."""
        ...

    async def parse(self, payload: dict[str, JsonValue]) -> ChannelMessage:
        """Parse a raw webhook payload into a structured channel message.

        Args:
            payload: Raw webhook payload from the external channel.

        Returns:
            Parsed ChannelMessage with normalized fields.
        """
        ...


@runtime_checkable
class ChannelRegistry(Protocol):
    """Maps external channel users to active teams.

    Runs in FastAPI async context — uses async signatures.
    """

    async def register(
        self, channel: str, channel_user_id: str, team_id: uuid.UUID
    ) -> None:
        """Register a mapping from a channel user to a team.

        Args:
            channel: Channel name (e.g., "whatsapp", "slack").
            channel_user_id: Channel-specific user identifier.
            team_id: The team ID to associate.
        """
        ...

    async def find_team(
        self, channel: str, channel_user_id: str
    ) -> uuid.UUID | None:
        """Find the team associated with a channel user.

        Args:
            channel: Channel name (e.g., "whatsapp", "slack").
            channel_user_id: Channel-specific user identifier.

        Returns:
            Team ID if a mapping exists, None otherwise.
        """
        ...

    async def deregister(self, channel: str, channel_user_id: str) -> None:
        """Remove the mapping for a channel user.

        Args:
            channel: Channel name (e.g., "whatsapp", "slack").
            channel_user_id: Channel-specific user identifier.
        """
        ...
