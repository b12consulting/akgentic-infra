"""Channel protocols — interaction channel abstractions for external communication."""

from __future__ import annotations

import uuid
from typing import Any, Protocol

from pydantic import BaseModel, Field


class ChannelMessage(BaseModel):
    """Normalized message from an external interaction channel."""

    channel_id: str = Field(description="Channel identifier (e.g. whatsapp, slack)")
    sender_id: str = Field(description="Channel-specific sender identifier")
    content: str = Field(description="Message content")
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Channel-specific metadata"
    )


class InteractionChannelAdapter(Protocol):
    """Delivers outbound messages to humans via an external channel.

    Implementations: WebAdapter, TwilioAdapter, SlackAdapter, etc.
    Called synchronously within the actor thread (see threading model note in architecture).
    """

    def send(self, channel_id: str, message: str) -> None:
        """Send a message to a human via the external channel.

        Args:
            channel_id: Channel-specific recipient identifier
            message: Message content to deliver
        """
        ...


class InteractionChannelIngestion(Protocol):
    """Routes inbound human replies from external channels to the correct team's UserProxy."""

    def route_inbound(self, channel_id: str, content: str) -> None:
        """Route an inbound message from an external channel to the appropriate team.

        Args:
            channel_id: Channel-specific sender identifier
            content: Message content from the human
        """
        ...


class ChannelParser(Protocol):
    """Parses channel-specific webhook payloads into a common ChannelMessage.

    Runs in FastAPI async context — uses async signatures.
    """

    async def parse(self, payload: dict[str, Any]) -> ChannelMessage:
        """Parse a raw webhook payload into a structured channel message.

        Args:
            payload: Raw webhook payload from the external channel

        Returns:
            Parsed ChannelMessage with normalized fields
        """
        ...


class ChannelRegistry(Protocol):
    """Maps external channel users to active teams.

    Runs in FastAPI async context — uses async signatures.
    """

    async def find_team(self, channel_id: str, sender_id: str) -> uuid.UUID | None:
        """Find the team associated with a channel user.

        Args:
            channel_id: Channel identifier (e.g., "whatsapp", "slack")
            sender_id: Channel-specific sender identifier

        Returns:
            Team ID if a mapping exists, None otherwise
        """
        ...
