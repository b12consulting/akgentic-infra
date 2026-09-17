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


class ChannelCommand(SerializableBaseModel):
    """A slash command the channel's own markup identified in an inbound message.

    ``name`` is a plain ``str`` rather than an enum on purpose: the channel layer
    consumes three names and passes every other one through as ordinary text, so
    an unrecognised command must be a *value*, not a validation error at parse
    time (ADR-043 §D10).
    """

    name: str = Field(description="Command word, lowercased, without its leading slash")
    rest: str = Field(default="", description="Everything after the command word, verbatim")


class ChannelMessage(SerializableBaseModel):
    """Normalized message from an external interaction channel."""

    content: str = Field(description="Message content")
    channel_user_id: str = Field(description="Channel-specific user identifier")
    channel_message_id: str | None = Field(default=None, description="Channel-specific message ID")
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
    command: ChannelCommand | None = Field(
        default=None,
        description=(
            "The slash command the channel's own markup identified, or None. "
            "``content`` is unaffected: a command the channel layer does not "
            "consume must reach the team looking exactly as the user typed it."
        ),
    )


class ChannelAddress(SerializableBaseModel):
    """Names one chat on one channel — where a message goes, with no team attached."""

    channel: str = Field(description="Channel name (e.g., 'telegram', 'slack')")
    channel_user_id: str = Field(description="Channel-specific user identifier (the chat)")


class ChannelBinding(ChannelAddress):
    """Binds one channel conversation to one agent of one team.

    Stored **once** per ``(channel, channel_user_id)`` and read by two keys:

    - ``(channel, channel_user_id)`` on the **inbound** path — a message arrived
      from this chat, which team owns it?
    - ``(team_id, agent_name)`` on the **outbound** path — this agent produced a
      message, which chat answers it?

    Both questions are answerable from these four values, which is why there is
    one record rather than two stores kept in step (ADR-043 §D8).

    The bound agent is the team's **entry point** and only the entry point. Any
    other user-proxy member is a different human whose channel id does not exist
    until that human has messaged the bot, so binding one to the initiator's
    chat would deliver one person's questions to another.
    """

    team_id: uuid.UUID = Field(description="The team this channel conversation belongs to")
    agent_name: str = Field(
        description=(
            "Spawned name of the bound agent — the key into the team's address "
            "table, already headcount-expanded (e.g. '@HumanProxy_0'). Not a role."
        ),
    )


class InitiatedTeam(SerializableBaseModel):
    """What ``initiate_team`` created: a team, and the agent that speaks for it."""

    team_id: uuid.UUID = Field(description="The newly created team's ID")
    entry_point_name: str = Field(
        description=(
            "Spawned name of the team's entry-point agent — the key into the "
            "team's address table, unique within the team and already "
            "headcount-expanded (e.g. '@HumanProxy_0'). This is a name, not a "
            "role: a role is shared by every member hired from the same card."
        ),
    )


@runtime_checkable
class InteractionChannelAdapter(Protocol):
    """Delivers outbound messages to humans via an external channel.

    Implementations: ``TelegramChannelAdapter`` (in
    ``akgentic.infra.adapters.shared``). Channel adapters are tier-agnostic —
    the same concrete adapter is reused by the community, department, and
    enterprise profiles.

    Lifetime:
        An adapter is **process-scoped**. ``ChannelParserRegistry`` constructs
        one per configured channel at wiring time, and that single instance
        outlives every team the process ever runs. Anything an adapter holds —
        an HTTP client, a socket, a token — is shared by all of them.

    Addressing:
        Every call carries the destination chat explicitly — a
        ``ChannelBinding`` on the message path, a bare ``ChannelAddress`` on the
        notice path, which a binding also satisfies. The message's own recipient
        address cannot name it: ``ActorAddress.name`` is the TeamCard's agent
        name (``human_support``, ``@HumanProxy_0``), never a channel identifier
        (ADR-043 §D4).

    Threading constraint:
        ``deliver()`` runs inside a Pykka actor thread (called from
        ``InteractionChannelDispatcher.on_message``). Implementations
        must not block and must not perform unguarded async I/O.
        If async delivery is needed, use a thread-safe bridge (e.g.
        enqueue to a ``queue.Queue`` consumed by an asyncio task).
    """

    def matches(self, msg: SentMessage, binding: ChannelBinding) -> bool:
        """Check if this adapter handles the given message.

        Args:
            msg: The outbound message to check.
            binding: The recipient agent's channel binding. An implementation
                must compare ``binding.channel`` against the channel it serves:
                with two channels configured, a recipient-only check accepts
                another channel's binding and posts its user id to the wrong
                service.

        Returns:
            True if this adapter should deliver the message.
        """
        ...

    def deliver(self, msg: SentMessage, binding: ChannelBinding) -> None:
        """Deliver an outbound message via this channel.

        Args:
            msg: The message to deliver.
            binding: The recipient agent's channel binding. Its
                ``channel_user_id`` names the destination chat — the only place
                that value exists on the outbound path.
        """
        ...

    def deliver_notice(self, address: ChannelAddress, text: str) -> None:
        """Deliver a channel-layer acknowledgement to one chat.

        Unlike ``deliver``, this is called from the FastAPI route rather than a
        Pykka actor thread, and it needs no ``matches()``: the address names the
        destination outright, so there is no message to inspect and no recipient
        to classify.

        The parameter is a ``ChannelAddress`` and not a ``ChannelBinding``
        because the paths that need it may have no team — ``status`` can find no
        binding, and ``unregister`` has just destroyed one. A binding-typed
        parameter could only be satisfied by fabricating a team id (ADR-043
        §D10, as corrected in its revision log).

        An implementation MUST compare ``address.channel`` against the channel
        it serves and return silently otherwise, exactly as ``matches()``
        already compares ``binding.channel``: notices are fanned out to every
        configured adapter, so an unguarded implementation posts one channel's
        chat id to another's service.

        Args:
            address: The chat to answer — channel and channel user id.
            text: The acknowledgement text.
        """
        ...

    def on_stop(self, team_id: uuid.UUID) -> None:
        """Clean up resources when a team stops.

        An implementation may release state it holds **for that team** and MUST
        NOT release process-scoped resources: the adapter is constructed once
        per process and serves every team, so anything closed here is closed
        for all the others too — silently, since ``deliver()`` swallows its own
        errors and the symptom is messages that simply stop arriving.

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
    ) -> InitiatedTeam:
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
            The created team's ID together with the spawned name of its
            entry-point agent. The caller needs both to write a
            ``ChannelBinding``: the id alone cannot answer an outbound lookup,
            which starts from an agent (ADR-043 §D4).

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
class ChannelRegistryReadSync(Protocol):
    """Synchronous read face of the channel registry, keyed by agent.

    Declares the one read the **outbound** delivery path performs.
    ``EventSubscriber.on_message`` is synchronous and runs in a Pykka actor
    thread with no event loop, so that path cannot await the registry; it needs
    an answer from memory, or none (ADR-043 §D5).

    Implementations answer from an in-process index and must not perform I/O:
    a missed delivery is recoverable, a stalled actor thread is not. The
    precedent is ``akgentic-infra-department``'s ``ServiceRegistryReadSync``.
    """

    def find_binding_sync(self, team_id: uuid.UUID, agent_name: str) -> ChannelBinding | None:
        """Return the binding for one agent of one team, without blocking.

        Args:
            team_id: The team the agent belongs to.
            agent_name: The agent's spawned name (``ChannelBinding.agent_name``).

        Returns:
            The binding if the agent is bound to a channel conversation, None
            otherwise — including when the registry is disabled or the index has
            not yet seen the binding.
        """
        ...


@runtime_checkable
class ChannelRegistry(ChannelRegistryReadSync, Protocol):
    """Stores the binding between a channel conversation and a team's agent.

    The async surface runs in FastAPI context and serves the **inbound** path;
    the inherited ``find_binding_sync`` serves the **outbound** one. One stored
    ``ChannelBinding`` answers both (ADR-043 §D8) — every registry owes the sync
    read, which is why it is inherited rather than merely implemented.

    Implementations:

    - **Community** (``YamlChannelRegistry``): YAML file on disk; channels are
      disabled when no registry path is configured.
    - **Department** (``MongoChannelRegistry``): MongoDB ``channel_users`` collection.
    - **Enterprise** (``DaprChannelRegistry``): Dapr state store.
    """

    async def register(self, binding: ChannelBinding) -> None:
        """Store a binding, replacing any existing one for the same channel user.

        Args:
            binding: The complete record — channel, channel user, team and the
                bound agent's spawned name. It is passed whole rather than as
                separate values so a field added later reaches storage without
                every call site being revisited.
        """
        ...

    async def find_team(self, channel: str, channel_user_id: str) -> uuid.UUID | None:
        """Find the team associated with a channel user.

        Args:
            channel: Channel name (e.g., "whatsapp", "slack").
            channel_user_id: Channel-specific user identifier.

        Returns:
            Team ID if a binding exists, None otherwise.
        """
        ...

    async def find_binding(self, channel: str, channel_user_id: str) -> ChannelBinding | None:
        """Find the whole binding for a channel user.

        Args:
            channel: Channel name (e.g., "whatsapp", "slack").
            channel_user_id: Channel-specific user identifier.

        Returns:
            The binding if one exists, None otherwise.
        """
        ...

    async def deregister(self, channel: str, channel_user_id: str) -> None:
        """Remove the binding for a channel user.

        Args:
            channel: Channel name (e.g., "whatsapp", "slack").
            channel_user_id: Channel-specific user identifier.
        """
        ...

    async def deregister_team(self, team_id: uuid.UUID) -> None:
        """Remove every binding for a team, across all channels.

        Called when a team stops: the conversation it answered is over, and a
        binding that outlives its team can only misroute.

        Args:
            team_id: The team whose bindings are released.
        """
        ...
