"""Channel protocols — interaction channel abstractions for external communication."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field

from akgentic.core.utils.serializer import SerializableBaseModel

if TYPE_CHECKING:
    from akgentic.core.messages import SentMessage
    from akgentic.infra.adapters.shared.channel_router import ChannelRouteContext

# Recursive JSON-safe type for webhook payloads — replaces dict[str, Any].
# PEP 695 (``type`` statement) rather than a plain assignment: the alias is
# recursive, and once it annotates a Pydantic *field* (``ChannelMessage.team_metadata``)
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
    catalog_entry: str | None = Field(default=None, description="Catalog entry for the new team")
    team_id: uuid.UUID | None = Field(
        default=None,
        description=(
            "Creation key for a team this message may START — never the address of "
            "an existing team. Used only when the conversation is unbound: concurrent "
            "creations carrying the same key for the same user collapse into one team, "
            "and a key naming a team that already exists, or one being created for a "
            "different user, is refused with 409. A parser that sets it must derive it "
            "so that racing deliveries of one initiation agree on it while a new "
            "session gets a fresh one; the framework supplies no such derivation. "
            "None means no collapsing: every creation gets a fresh id."
        ),
    )
    team_metadata: dict[str, JsonValue] | None = Field(
        default=None,
        description=(
            "Plain-JSON business metadata the parser lifted from the channel payload. "
            "Carried to team creation only; validated there against the resolved card's "
            "declared contract."
        ),
    )
    binding_metadata: dict[str, JsonValue] | None = Field(
        default=None,
        description=(
            "Plain-JSON data the parser lifted from the channel payload for the "
            "router's own use, stored verbatim on the binding when a team is started. "
            "Router-owned and UNVALIDATED: unlike ``team_metadata`` it is checked "
            "against no contract, and it comes from an unauthenticated payload, so a "
            "router reading it back must treat it as untrusted input — never as proof "
            "of identity or entitlement. Nothing in the framework reads it."
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
    quoted_text: str | None = Field(
        default=None,
        description=(
            "Text of the message this one replies to, when the channel says it "
            "replies to something and carries that text — Telegram's "
            "``reply_to_message.text``. Verbatim and UNPARSED: a router reads "
            "it, the framework does not. It is the user's own quotation of an "
            "earlier message, so it is no more trustworthy than ``content``."
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
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
        description=(
            "Plain-JSON data a channel router attaches to the binding for its own "
            "later use — on the inbound path through ``find_binding`` and on the "
            "outbound one through ``find_binding_sync``. Opaque to the framework: "
            "its shape is whatever the router that wrote it defines."
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

        Used when the conversation has no binding and the router starts a
        team without naming a catalog entry of its own, i.e. when
        ``ChannelMessage.catalog_entry`` is None. ``initiate_team`` then passes
        it to ``TeamService.create_team`` as the catalog namespace to create
        from.
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
class InteractionChannelRouter(Protocol):
    """Decides what one parsed inbound message does: reply, start a team, or nothing.

    The webhook route only parses and routes; every rule about *what happens
    next* lives here, so a channel can replace them through
    ``ChannelConfig.router_fqcn``. A channel that names no router gets
    ``DefaultChannelRouter`` (in ``akgentic.infra.adapters.shared``), which
    consumes the ``new`` / ``unregister`` / ``status`` commands, replies to the
    bound team, and otherwise starts one.

    Runs in FastAPI async context. A router is **process-scoped** like the
    parser and adapter: one instance per configured channel, constructed with
    the channel's ``config`` kwargs, so it must hold no per-request state.

    The binding is the authorization — and it holds by construction:
        The webhook is unauthenticated, so the payload is untrusted. The context
        a router receives exposes no service that addresses a team by id: it can
        send to, release, notify and report on **this** conversation's team
        only. The one method that accepts a team id, ``initiate_team``, takes it
        as a *creation key* — the placement contract refuses a key naming an
        existing team, so it can start a team but never reach one. A router
        acting through the context therefore cannot reach another chat's team.
        What review must still check is that it does not reach *past* the
        context — into its private attributes, ``app.state`` or a module
        global — for a service that takes a team id.

    Errors propagate: the route maps the team service's and catalog's exceptions
    exactly as it did before routers existed.
    """

    async def route(self, message: ChannelMessage, ctx: ChannelRouteContext) -> None:
        """Act on one inbound message.

        Args:
            message: The parsed, channel-agnostic message.
            ctx: This request's view of the channel services, already scoped to
                the conversation the message came from.
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

    async def find_binding(self, address: ChannelAddress) -> ChannelBinding | None:
        """Find the whole binding for a channel conversation.

        Args:
            address: The conversation — channel and channel user id.

        Returns:
            The binding if one exists, None otherwise.
        """
        ...

    async def deregister(self, address: ChannelAddress) -> None:
        """Remove the binding for a channel conversation.

        Args:
            address: The conversation — channel and channel user id.
        """
        ...

    async def deregister_team(self, team_id: uuid.UUID) -> None:
        """Remove every binding for a team, across all channels.

        Called when a team is **deleted** — not when it stops. A stop is
        reversible, so the conversation survives it; a delete is final, and a
        binding that outlives its team can only misroute (ADR-045 §D6).

        Resuming the stopped team on the conversation's next message is
        ADR-045 §D7, and the router does it: a surviving binding whose team is
        stopped brings that team back, silently, and the message goes through.
        What §D7 still owes is its third row (issue #487) — starting a fresh
        team and rebinding when the bound team is gone altogether. A chat in
        that state is told so rather than healed.

        Args:
            team_id: The team whose bindings are released.
        """
        ...
