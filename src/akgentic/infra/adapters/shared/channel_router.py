"""Channel routing — what one parsed inbound message does once the route has parsed it."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from akgentic.infra.protocols.channels import (
    ChannelAddress,
    ChannelBinding,
    ChannelCommand,
    ChannelMessage,
    ChannelRegistry,
    InitiatedTeam,
    InteractionChannelAdapter,
    InteractionChannelIngestion,
    JsonValue,
)

if TYPE_CHECKING:
    from akgentic.core.messages.message import Message
    from akgentic.infra.server.services.team_service import TeamService

logger = logging.getLogger(__name__)

# The three command words the default router consumes. Every other name falls
# through to the routing branches as ordinary text, where akgentic-tool's
# in-team slash mechanism may claim it (ADR-043 §D10).
_COMMAND_NEW = "new"
_COMMAND_UNREGISTER = "unregister"
_COMMAND_STATUS = "status"


class ChannelRouteContext:
    """One request's view of the channel services, scoped to one conversation.

    Built by the webhook route per request and handed to the channel's router.
    It holds runtime services, not data, which is why it is a plain class and
    not a model.

    Every helper acts on ``address`` — the conversation the message came from —
    so a router using them cannot reach another chat's team by construction.
    ``ingestion`` stays reachable for ``send_message``; see the router Protocol's
    authorization obligation before calling it with anything but the bound team.
    """

    def __init__(
        self,
        *,
        address: ChannelAddress,
        registry: ChannelRegistry,
        ingestion: InteractionChannelIngestion,
        team_service: TeamService,
        adapters: list[InteractionChannelAdapter],
        default_catalog_entry: str,
    ) -> None:
        self.address = address
        self.registry = registry
        self.ingestion = ingestion
        self.team_service = team_service
        self.default_catalog_entry = default_catalog_entry
        self._adapters = list(adapters)

    async def find_binding(self) -> ChannelBinding | None:
        """Return this conversation's binding, or None when it has no team."""
        return await self.registry.find_binding(self.address)

    async def release(self) -> ChannelBinding | None:
        """Release this conversation's binding, returning what was released.

        The team itself is **not** stopped: abandoning a running team is the
        affordance (ADR-043 §D8), stopping one is a lifecycle change.
        """
        binding = await self.find_binding()
        if binding is not None:
            await self.registry.deregister(self.address)
        return binding

    async def initiate_team(
        self,
        content: str | Message | None,
        *,
        catalog_entry: str | None = None,
        team_metadata: dict[str, JsonValue] | None = None,
        binding_metadata: dict[str, JsonValue] | None = None,
    ) -> InitiatedTeam:
        """Create a team for this conversation and bind the conversation to it.

        Three steps, in this order: create, bind, then send the first message.
        The first message is sent last because the team may answer it at once,
        and outbound delivery can only find the chat once the binding exists.

        Always binds. A team started from a channel without a binding could
        never answer it — its entry point's messages would reach no chat and
        nothing would say so — so there is no unbound variant.

        The binding replaces any existing one for this conversation, so a router
        that wants the old team released first calls ``release()`` before this.

        Args:
            content: The first message, or None to create the team silently.
            catalog_entry: Catalog entry to create from; the parser's default
                when None.
            team_metadata: Team metadata, validated against the resolved card at
                creation — the only point it is validated.
            binding_metadata: Router-owned data stored on the binding.

        Returns:
            The created team and the spawned name of its entry-point agent.
        """
        initiated = await self.ingestion.create_team(
            self.address.channel_user_id,
            catalog_entry or self.default_catalog_entry,
            metadata=team_metadata,
        )
        logger.debug(
            "Channel initiation: channel=%s, user=%s, new_team=%s, entry_point=%s",
            self.address.channel,
            self.address.channel_user_id,
            initiated.team_id,
            initiated.entry_point_name,
        )
        # Bind BEFORE the first message: a team answering it before the binding
        # exists would find no chat on the outbound path, and that reply is lost.
        await self.registry.register(
            ChannelBinding(
                channel=self.address.channel,
                channel_user_id=self.address.channel_user_id,
                team_id=initiated.team_id,
                agent_name=initiated.entry_point_name,
                metadata=binding_metadata or {},
            )
        )
        if content is not None:
            await self.ingestion.send_message(initiated.team_id, content)
        return initiated

    def notify(self, text: str) -> None:
        """Send a channel-layer acknowledgement to this conversation.

        Fanned out to every configured adapter; each compares the address's
        channel against its own and returns silently otherwise.
        """
        for adapter in self._adapters:
            adapter.deliver_notice(self.address, text)


class DefaultChannelRouter:
    """The routing rules a channel gets when its config names no router.

    1. A ``new`` / ``unregister`` / ``status`` command is consumed (ADR-043
       §D10). Any other command name is ordinary text and falls through.
    2. A bound conversation's message is a reply to its team.
    3. An unbound conversation's message starts a team and binds to it.

    Subclass it to keep part of this: each step is an overridable hook, so a
    router that only changes initiation overrides ``on_unbound`` and keeps the
    commands.
    """

    def __init__(self, **config: str) -> None:
        """Accept the channel's shared ``config`` kwargs, which it does not use.

        ``ChannelConfig.config`` is passed to the parser, the adapter and the
        router alike, so a router constructor must tolerate keys meant for the
        other two.
        """

    async def route(self, message: ChannelMessage, ctx: ChannelRouteContext) -> None:
        """Consume a command, else reply to the bound team, else start one."""
        if message.command is not None and await self.on_command(message, message.command, ctx):
            logger.debug(
                "Channel command consumed: channel=%s, user=%s, command=%s",
                ctx.address.channel,
                ctx.address.channel_user_id,
                message.command.name,
            )
            return
        binding = await ctx.find_binding()
        if binding is None:
            await self.on_unbound(message, ctx)
        else:
            await self.on_bound(message, binding, ctx)

    async def on_command(
        self, message: ChannelMessage, command: ChannelCommand, ctx: ChannelRouteContext
    ) -> bool:
        """Consume ``new`` / ``unregister`` / ``status``, or decline the message.

        No command reads anything from the payload beyond its own text: each
        resolves its subject from the conversation's binding.

        Returns:
            True when consumed. False for any other name, which then reaches
            the team as ordinary text — no 4xx, no notice, no log-and-drop.
        """
        if command.name == _COMMAND_NEW:
            await self._command_new(message, command.rest, ctx)
        elif command.name == _COMMAND_UNREGISTER:
            await self._command_unregister(ctx)
        elif command.name == _COMMAND_STATUS:
            await self._command_status(ctx)
        else:
            return False
        return True

    async def on_bound(
        self, message: ChannelMessage, binding: ChannelBinding, ctx: ChannelRouteContext
    ) -> None:
        """Deliver the message to the team this conversation is bound to."""
        logger.debug(
            "Channel continuation: channel=%s, user=%s, team_id=%s",
            ctx.address.channel,
            ctx.address.channel_user_id,
            binding.team_id,
        )
        await ctx.ingestion.send_message(
            binding.team_id, message.content, message.channel_message_id
        )

    async def on_unbound(self, message: ChannelMessage, ctx: ChannelRouteContext) -> None:
        """Start a team for this conversation, with the message as its first."""
        await ctx.initiate_team(
            message.content,
            catalog_entry=message.catalog_entry,
            team_metadata=message.metadata,
        )

    async def _command_new(
        self, message: ChannelMessage, rest: str, ctx: ChannelRouteContext
    ) -> None:
        """Abandon this conversation's team, if any, and start a fresh one.

        The order is load-bearing: both steps write the same
        ``(channel, channel_user_id)`` key, so a release running *after* the
        initiation would leave the conversation with no binding at all.

        ``rest`` is the first message and is never parsed for a namespace; an
        empty one sends nothing. ``message.catalog_entry`` is honoured exactly
        as ``on_unbound`` honours it.
        """
        await ctx.release()
        initiated = await ctx.initiate_team(
            rest or None,
            catalog_entry=message.catalog_entry,
            team_metadata=message.metadata,
        )
        # ``new`` acknowledges even though the team usually answers for itself:
        # ``/new`` with no text produces no team reply at all, so without this
        # the user who just abandoned a conversation would see nothing.
        ctx.notify(f"Started a new session — team {initiated.team_id}.")

    async def _command_unregister(self, ctx: ChannelRouteContext) -> None:
        """Release this conversation's binding, acknowledging either way.

        The unbound case still answers — a different text, not silence. A caller
        cannot otherwise tell "released" from "nothing happened".
        """
        released = await ctx.release()
        if released is None:
            ctx.notify("No active session to release.")
            return
        ctx.notify(f"Released team {released.team_id}.")

    async def _command_status(self, ctx: ChannelRouteContext) -> None:
        """Report the bound team and its lifecycle state.

        The binding alone would lose the dead-binding diagnosis: a record can
        outlive the team it names, and only the team service can say so.
        """
        binding = await ctx.find_binding()
        if binding is None:
            ctx.notify("No active session.")
            return
        process = ctx.team_service.get_team(binding.team_id)
        if process is None:
            ctx.notify(f"Bound to team {binding.team_id}, which is no longer known.")
            return
        ctx.notify(f"Bound to team {binding.team_id} — {process.status.value}.")
