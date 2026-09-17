"""Channel protocols — interaction channel abstractions for external communication."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field

from akgentic.core.utils.serializer import SerializableBaseModel

if TYPE_CHECKING:
    from akgentic.core.messages import SentMessage
    from akgentic.core.messages.message import Message

# Recursive JSON-safe type for webhook payloads — replaces dict[str, Any].
# PEP 695 (``type`` statement) rather than a plain assignment: the alias is
# recursive, and once it annotates a Pydantic *field* (``ChannelMessage.metadata``)
# the implicit form makes schema generation recurse until it blows the stack.
# A named alias gives Pydantic a definition reference to close the cycle with.
type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


class ChannelMessage(SerializableBaseModel):
    """Normalized message from an external interaction channel."""

    content: str = Field(description="Message content")
    channel_user_id: str = Field(description="Channel-specific user identifier")
    message_id: str | None = Field(default=None, description="Channel-specific message ID")
    team_id: uuid.UUID | None = Field(default=None, description="Associated team ID")
    catalog_entry: str | None = Field(default=None, description="Catalog entry for the new team")
    metadata: dict[str, JsonValue] | None = Field(
        default=None,
        description=(
            "Plain-JSON business metadata the parser lifted from the channel payload. "
            "Carried to team creation only; validated there against the resolved card's "
            "declared contract."
        ),
    )


@runtime_checkable
class InteractionChannelAdapter(Protocol):
    """Delivers outbound messages to humans via an external channel.

    Implementations: ``TelegramChannelAdapter`` (in
    ``akgentic.infra.adapters.shared``). Channel adapters are tier-agnostic —
    the same concrete adapter is reused by the community, department, and
    enterprise profiles.

    Threading constraint:
        ``deliver()`` runs inside a Pykka actor thread (called from
        ``InteractionChannelDispatcher.on_message``). Implementations
        must not block and must not perform unguarded async I/O.
        If async delivery is needed, use a thread-safe bridge (e.g.
        enqueue to a ``queue.Queue`` consumed by an asyncio task).
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
    """Routes inbound human replies from external channels to the correct team's UserProxy.

    Implementations:

    - **Community** (``LocalIngestion``): in-process routing to the local TeamService.
    - **Department** (``HttpIngestion``): routes to the owning worker over HTTP.
    - **Enterprise** (``DaprIngestion``): routes to the owning worker via Dapr
      service invocation.

    Error contract:
        - ``initiate_team()`` raises, for a ``catalog_entry_id`` that does not
          yield a team, either ``EntryNotFoundError`` (the namespace holds
          nothing, or — as ``CatalogTeamEntryMissingError`` — holds no team
          entry) → HTTP 404, or ``CatalogValidationError`` (the namespace is
          present and its stored entries are invalid) → HTTP 409 carrying the
          catalog's own message. **Catch neither.** Both belong to the catalog's
          exception family, which the app registers handlers for, so letting
          them propagate produces those two answers for free; a local catch that
          reports every failure as 404 discards the diagnosis.
        - ``initiate_team()`` also raises ``MetadataValidationError`` → HTTP 422
          for a ``metadata`` body that fails the resolved card's declared
          contract. **Catch it nowhere**, exactly as the catalog exceptions
          above: it is a ``ServerError``, so the app-level handler already
          answers 422 carrying the validator's own message.
        - ``route_reply()`` raises ``ValueError`` if ``team_id`` does not
          correspond to a running team — ``TeamNotFoundError`` when the team is
          unknown, ``TeamStateConflictError`` when it exists in a state the
          operation forbids. Both are ``ValueError`` subclasses, so a caller
          that does not need the distinction is unaffected.
          **These two have no app-level handler.** They subclass ``ValueError``,
          not ``ServerError``, and nothing registers a ``ValueError`` handler,
          so a caller that lets them propagate gets a 500 — not a 404 or a 409.
          A caller that wants those answers must map by type itself, the way
          ``server/routes/teams.py`` does.
    """

    async def route_reply(
        self,
        team_id: uuid.UUID,
        content: str | Message,
        original_message_id: str | None = None,
    ) -> None:
        """Route an inbound reply to an existing team.

        Args:
            team_id: Target team ID.
            content: Message content from the human — either bare text or a
                pre-formed ``Message``, which the team service already accepts.
                Implementations pass it through untouched; coercing to ``str``
                here would discard everything a typed message carries.
            original_message_id: Optional ID of the message being replied to.

        Raises:
            ValueError: If team_id does not correspond to a running team.
        """
        ...

    async def initiate_team(
        self,
        content: str,
        channel_user_id: str,
        catalog_entry_id: str,
        metadata: dict[str, JsonValue] | None = None,
    ) -> uuid.UUID:
        """Create a new team and send the initial message.

        Args:
            content: Initial message content.
            channel_user_id: Channel-specific user identifier.
            catalog_entry_id: Catalog entry to use for team creation.
            metadata: Optional plain-JSON business metadata carried by the
                inbound message. Validated by ``TeamService`` against the
                ``metadata_type`` the resolved card declares — the client never
                names the type — so a channel-created team is filterable exactly
                like one created from ``POST /teams`` (ADR-24 §metadata).

        Returns:
            The newly created team's ID.

        Raises:
            EntryNotFoundError: If catalog_entry_id is not found in catalog.
            MetadataValidationError: If ``metadata`` fails the card's declared
                contract.
        """
        ...


@runtime_checkable
class ChannelParser(Protocol):
    """Parses channel-specific webhook payloads into a common ChannelMessage.

    Runs in FastAPI async context — uses async signatures.

    Implementations:

    - ``TelegramChannelParser`` (in ``akgentic.infra.adapters.shared``) — parses
      Telegram webhooks; available to all tiers.
    - **Enterprise** (``SlackParser`` plus application-supplied parsers):
      concrete parsers are registered with ``EnterpriseChannelParserRegistry``
      from the ``parser_class`` FQCNs in the channel definitions.
    """

    @property
    def channel_name(self) -> str:
        """The channel name this parser handles (e.g. 'whatsapp', 'slack')."""
        ...

    @property
    def default_catalog_entry(self) -> str:
        """Default catalog entry ID to use when initiating a new team.

        Used when ``ChannelRegistry.find_team()`` returns ``None`` (no
        existing team for this channel user). The ingestion layer passes
        this value to ``initiate_team(catalog_entry_id=...)`` to create
        a new team from the channel's default template.
        """
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

    Implementations:

    - **Community** (``YamlChannelRegistry``): YAML file on disk; channels are
      disabled when no registry path is configured.
    - **Department** (``MongoChannelRegistry``): MongoDB ``channel_users`` collection.
    - **Enterprise** (``DaprChannelRegistry``): Dapr state store.
    """

    async def register(self, channel: str, channel_user_id: str, team_id: uuid.UUID) -> None:
        """Register a mapping from a channel user to a team.

        Args:
            channel: Channel name (e.g., "whatsapp", "slack").
            channel_user_id: Channel-specific user identifier.
            team_id: The team ID to associate.
        """
        ...

    async def find_team(self, channel: str, channel_user_id: str) -> uuid.UUID | None:
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
