"""Channel routing — what one parsed inbound message does once the route has parsed it."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import TYPE_CHECKING

from akgentic.infra.protocols.channels import (
    ChannelAddress,
    ChannelBinding,
    ChannelCommand,
    ChannelMessage,
    ChannelRegistry,
    InteractionChannelAdapter,
    InteractionChannelRouter,
    JsonValue,
)

if TYPE_CHECKING:
    from akgentic.core.messages.message import Message
    from akgentic.infra.server.services.team_service import TeamService
    from akgentic.team.models import Process

logger = logging.getLogger(__name__)

# The command words the default router consumes. ``register`` is consumed only
# where the channel config enables it, and answers with a refusal otherwise —
# never silently, since a user who typed it is waiting for an answer. Every
# other name falls through to the routing branches as ordinary text, where
# akgentic-tool's in-team slash mechanism may claim it (ADR-043 §D10).
_COMMAND_NEW = "new"
_COMMAND_UNREGISTER = "unregister"
_COMMAND_STATUS = "status"
_COMMAND_REGISTER = "register"

# ``/register`` reads its two ids out of free text — the command's own rest, or
# the message it replies to, which is how the bot's own notices ("Started a new
# session — team <id>.") hand a team id back to the user. Both patterns are
# anchored on word boundaries so an id embedded in a sentence is still found.
_TEAM_ID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
# A spawned agent name as ``AgentRef.name`` carries it: '@' then the card's name,
# headcount-expanded ('@HumanProxy_0'). Telegram @-mentions of *people* exist in
# the same text, which is why a match is verified against the team's own agents
# rather than trusted.
_AGENT_NAME_RE = re.compile(r"@[A-Za-z0-9_]+")


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


class ChannelRouteContext:
    """One request's view of the channel services, scoped to one conversation.

    Built by the webhook route per request and handed to the channel's router.
    It holds runtime services, not data, which is why it is a plain class and
    not a model.

    Every method acts on ``address`` — the conversation the message came from.
    The registry and the team service are held privately: each has a method
    taking an arbitrary team id (``register``, ``send_message``, ``get_team``),
    and exposing either would turn the authorization rule into a request to be
    careful about every call a router makes.

    ``initiate_team`` accepts a ``team_id`` and is not an exception: the id is a
    *creation key*. The placement contract (``PlacementStrategy.create_team``)
    refuses one naming a team that already exists, so the method can start a
    team but can never reach one.

    **``bind_existing_team`` is the one real exception, and it is deliberate.**
    It binds this conversation to a team named by id, which is precisely what
    the rest of the class prevents — there is no ownership check available,
    because the channel layer has no verified identity for the chat user. It
    exists so a user can re-attach a chat to a team they already know about,
    and a router that exposes it to users must gate it: ``DefaultChannelRouter``
    ships it off, behind ``allow_register`` on the channel config. It is one
    method, named for what it does, so a review of a custom router has exactly
    one call to look for rather than a class of them.

    There is no ingestion layer between this context and ``TeamService``.
    ``TeamService`` is already the tier-agnostic seam — creation goes through
    the tier's placement, delivery through the tier's team handle — so a
    further bridge would only forward calls.

    **What is offloaded, and what is not.** Every ``TeamService`` call runs on
    a worker thread via ``asyncio.to_thread``: they are synchronous and some are
    slow (creation spawns actors; lookups read the event store), and on the
    event loop they would stall every other request, WebSocket and background
    task in the process. Registry calls are awaited directly. The registry
    Protocol is async, so each implementation decides how to do its I/O, and
    ``YamlChannelRegistry`` must not be pushed onto threads from here: its
    read-modify-write is safe only because the loop serialises it, and two
    threads could each read the old file and the second write would silently
    discard the first.
    """

    def __init__(
        self,
        *,
        address: ChannelAddress,
        registry: ChannelRegistry,
        team_service: TeamService,
        adapters: list[InteractionChannelAdapter],
        default_catalog_entry: str,
    ) -> None:
        self.address = address
        self.default_catalog_entry = default_catalog_entry
        self._registry = registry
        self._team_service = team_service
        self._adapters = list(adapters)

    async def find_binding(self) -> ChannelBinding | None:
        """Return this conversation's binding, or None when it has no team."""
        return await self._registry.find_binding(self.address)

    async def release(self) -> ChannelBinding | None:
        """Release this conversation's binding, returning what was released.

        The team itself is **not** stopped: abandoning a running team is the
        affordance (ADR-043 §D8), stopping one is a lifecycle change.
        """
        binding = await self.find_binding()
        if binding is not None:
            await self._registry.deregister(self.address)
        return binding

    async def initiate_team(
        self,
        content: str | Message | None,
        *,
        catalog_entry: str | None = None,
        team_id: uuid.UUID | None = None,
        team_metadata: dict[str, JsonValue] | None = None,
        binding_metadata: dict[str, JsonValue] | None = None,
    ) -> Process:
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
            team_id: Optional creation key. Concurrent initiations of this
                conversation carrying the same key yield one team — the losers
                receive the winner's team, bind to it (an identical binding, so
                the write is harmless) and send their message to it, so no
                message is lost. A key naming an existing team is refused with
                409 by the placement; see ``PlacementStrategy.create_team``.
            team_metadata: Team metadata, validated against the resolved card at
                creation — the only point it is validated.
            binding_metadata: Router-owned data stored on the binding.

        Returns:
            The created team's ``Process``, which names both the team and the
            spawned entry-point agent the binding was written for.
        """
        process = await asyncio.to_thread(
            self._team_service.create_team,
            catalog_entry or self.default_catalog_entry,
            user_id=self.address.channel_user_id,
            team_id=team_id,
            metadata=team_metadata,
        )
        logger.debug(
            "Channel initiation: channel=%s, user=%s, new_team=%s, entry_point=%s",
            self.address.channel,
            self.address.channel_user_id,
            process.team_id,
            process.entry_point.name,
        )
        # Bind BEFORE the first message: a team answering it before the binding
        # exists would find no chat on the outbound path, and that reply is lost.
        await self._registry.register(
            ChannelBinding(
                channel=self.address.channel,
                channel_user_id=self.address.channel_user_id,
                team_id=process.team_id,
                agent_name=process.entry_point.name,
                metadata=binding_metadata or {},
            )
        )
        # The team is created and bound even when the first message is blank:
        # the binding is what lets the next message continue this conversation
        # rather than start yet another team. Only the empty prompt is withheld.
        if content is not None and not _is_blank(content):
            await asyncio.to_thread(self._team_service.send_message, process.team_id, content)
        return process

    async def send(self, content: str | Message) -> bool:
        """Send a message to this conversation's team.

        The team is resolved from the binding here, never passed in: that is
        what makes the authorization rule hold by construction.

        There is no reply-to parameter. A channel's own message id
        (``ChannelMessage.channel_message_id``) is not an akgentic message id,
        so it cannot name the message being answered, and
        ``TeamService.send_message`` has nowhere to put one.

        Returns:
            True when sent. False when nothing was sent — the conversation has
            no team, or the content is blank.
        """
        if _is_blank(content):
            logger.info("Dropping blank inbound message for %s", self.address)
            return False
        binding = await self.find_binding()
        if binding is None:
            return False
        await asyncio.to_thread(self._team_service.send_message, binding.team_id, content)
        return True

    async def bind_existing_team(
        self, team_id: uuid.UUID, agent_name: str | None = None
    ) -> Process | None:
        """Bind this conversation to a team that already exists.

        **This is the one method on the context that reaches a team by id, and
        it is a deliberate hole in the rule the rest of the class keeps.** A
        webhook payload is unauthenticated, so a chat calling this can name any
        team whose id it knows and will then receive that team's messages. There
        is no ownership check to make: the channel layer has no verified
        identity for the chat user to compare against ``Process.user_id``. Any
        router exposing it to users must gate it — ``DefaultChannelRouter``
        requires ``allow_register`` on the channel config, off by default.

        The binding replaces any existing one for this conversation, exactly as
        ``initiate_team``'s does, so a chat can move between teams but never
        hold two.

        Args:
            team_id: The team to bind to.
            agent_name: The agent to bind, or None for the team's entry point.
                Only the entry point and the first-layer supervisors are
                accepted: every other name is either unaddressable or a
                different human's proxy.

        Returns:
            The team's ``Process``, or None when the team service does not know
            it — deleted, purged, or never existed.

        Raises:
            ValueError: The team is known but ``agent_name`` names no agent of
                it. The message names the entry point, which is what the caller
                should have asked for.
        """
        process = await asyncio.to_thread(self._team_service.get_team, team_id)
        if process is None:
            return None
        bound_name = self._resolve_agent_name(process, agent_name)
        await self._registry.register(
            ChannelBinding(
                channel=self.address.channel,
                channel_user_id=self.address.channel_user_id,
                team_id=process.team_id,
                agent_name=bound_name,
            )
        )
        logger.info(
            "Channel bound to an existing team: channel=%s, user=%s, team_id=%s, agent=%s",
            self.address.channel,
            self.address.channel_user_id,
            process.team_id,
            bound_name,
        )
        return process

    @staticmethod
    def _resolve_agent_name(process: Process, agent_name: str | None) -> str:
        """Return the spawned name to bind, defaulting to the entry point.

        Raises:
            ValueError: ``agent_name`` names no agent of this team.
        """
        if agent_name is None:
            return process.entry_point.name
        addressable = {process.entry_point.name} | {ref.name for ref in process.supervisors}
        if agent_name not in addressable:
            msg = f"{agent_name} is not an agent of team {process.team_id}"
            raise ValueError(msg)
        return agent_name

    async def bound_process(self) -> Process | None:
        """Return the bound team's ``Process``, or None when unbound or unknown.

        None covers two cases a caller may want to tell apart — no binding, and a
        binding naming a team the team service no longer knows. Call
        ``find_binding()`` first to distinguish them.
        """
        binding = await self.find_binding()
        if binding is None:
            return None
        return await asyncio.to_thread(self._team_service.get_team, binding.team_id)

    def notify(self, text: str) -> None:
        """Send a channel-layer acknowledgement to this conversation.

        Fanned out to every configured adapter; each compares the address's
        channel against its own and returns silently otherwise.
        """
        for adapter in self._adapters:
            adapter.deliver_notice(self.address, text)


class DefaultChannelRouter(InteractionChannelRouter):
    """The routing rules a channel gets when its config names no router.

    1. A ``new`` / ``unregister`` / ``status`` command is consumed (ADR-043
       §D10), and ``register`` too where the channel config enables it. Any
       other command name is ordinary text and falls through.
    2. A bound conversation's message is a reply to its team.
    3. An unbound conversation's message starts a team and binds to it.

    Subclass it to keep part of this: each step is an overridable hook, so a
    router that only changes initiation overrides ``on_unbound`` and keeps the
    commands.
    """

    def __init__(self, **config: str) -> None:
        """Read ``allow_register``, ignoring every other key.

        ``ChannelConfig.config`` is passed to the parser, the adapter and the
        router alike, so a router constructor must tolerate keys meant for the
        other two.

        ``allow_register`` enables the ``register`` command and defaults to
        **off**. The command binds a chat to a team named by an unauthenticated
        payload, so a deployment turns it on only where chat users are trusted
        or team ids are not obtainable by anyone who should not have them.
        Anything but ``"true"`` (case-insensitively) leaves it off: a typo must
        fail closed.
        """
        self._allow_register = config.get("allow_register", "").strip().lower() == "true"

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
        """Consume ``new`` / ``unregister`` / ``status`` / ``register``, or decline.

        Three of the four read nothing from the payload beyond their own text:
        each resolves its subject from the conversation's binding. ``register``
        is the exception — it takes a team id from the message — and it answers
        only when the channel config enables it.

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
        elif command.name == _COMMAND_REGISTER:
            await self._command_register(message, command.rest, ctx)
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
        await ctx.send(message.content)

    async def on_unbound(self, message: ChannelMessage, ctx: ChannelRouteContext) -> None:
        """Start a team for this conversation, with the message as its first."""
        await ctx.initiate_team(
            message.content,
            catalog_entry=message.catalog_entry,
            team_id=message.team_id,
            team_metadata=message.team_metadata,
            binding_metadata=message.binding_metadata,
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
        process = await ctx.initiate_team(
            rest or None,
            catalog_entry=message.catalog_entry,
            team_id=message.team_id,
            team_metadata=message.team_metadata,
            binding_metadata=message.binding_metadata,
        )
        # ``new`` acknowledges even though the team usually answers for itself:
        # ``/new`` with no text produces no team reply at all, so without this
        # the user who just abandoned a conversation would see nothing.
        ctx.notify(f"Started a new session — team {process.team_id}.")

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

    async def _command_register(
        self, message: ChannelMessage, rest: str, ctx: ChannelRouteContext
    ) -> None:
        """Bind this conversation to a team that already exists.

        Two ways to name the team, in this order: the command's own text, and
        failing that the message it replies to. The second is what makes the
        command usable — the bot's notices carry a team id back to the chat, so
        replying to one with ``/register`` needs no copying.

        Every outcome answers the user. A command that silently does nothing is
        indistinguishable from one that worked, and this one changes which team
        the next message reaches.

        Disabled unless the channel config sets ``allow_register``: the id comes
        from an unauthenticated payload and is checked against nothing (see
        ``ChannelRouteContext.bind_existing_team``).
        """
        if not self._allow_register:
            logger.warning(
                "Rejected /register on a channel that has not enabled it: channel=%s, user=%s",
                ctx.address.channel,
                ctx.address.channel_user_id,
            )
            ctx.notify("Registering to an existing team is not enabled on this channel.")
            return
        source = rest if _TEAM_ID_RE.search(rest) else (message.quoted_text or "")
        team_id_match = _TEAM_ID_RE.search(source)
        if team_id_match is None:
            ctx.notify(
                "No team id found. Send '/register <team-id> [@Agent]', "
                "or reply to a message naming the team with '/register'."
            )
            return
        team_id = uuid.UUID(team_id_match.group())
        agent_match = _AGENT_NAME_RE.search(source)
        agent_name = agent_match.group() if agent_match is not None else None
        try:
            process = await ctx.bind_existing_team(team_id, agent_name)
        except ValueError as exc:
            logger.info("Rejected /register naming an unknown agent: %s", exc)
            ctx.notify(f"{agent_name} is not an agent of team {team_id}.")
            return
        if process is None:
            ctx.notify(f"Team {team_id} is not known.")
            return
        bound_name = agent_name or process.entry_point.name
        ctx.notify(f"Bound to team {process.team_id} as {bound_name} — {process.status.value}.")

    async def _command_status(self, ctx: ChannelRouteContext) -> None:
        """Report the bound team and its lifecycle state.

        The binding alone would lose the dead-binding diagnosis: a record can
        outlive the team it names, and only the team service can say so.
        """
        binding = await ctx.find_binding()
        if binding is None:
            ctx.notify("No active session.")
            return
        process = await ctx.bound_process()
        if process is None:
            ctx.notify(f"Bound to team {binding.team_id}, which is no longer known.")
            return
        ctx.notify(f"Bound to team {binding.team_id} — {process.status.value}.")
