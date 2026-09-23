"""Channel routing — what one parsed inbound message does once the route has parsed it."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import TYPE_CHECKING

from akgentic.infra.errors import TeamNotFoundError, TeamStateConflictError
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
from akgentic.team.models import TeamStatus

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
# Addressing an agent is a name at the START of what the user typed. A name
# anywhere else is part of the sentence and belongs to the recipient, not to
# the routing: "@Expert, ask a joke to @Support" is for the Expert, and naming
# the Support is the Expert's instruction to carry out.
_LEADING_AGENT_RE = re.compile(r"^\s*(@[A-Za-z0-9_]+)")


class NoDefaultRecipientError(ValueError):
    """The bound agent has nobody to address, and the message named nobody.

    Raised when the team holds no agent other than the bound one — it is the
    team's own entry point and declares no supervisors. Every fallback
    available at that point changes who is speaking, so the user is asked to
    name a recipient instead.

    A ``ValueError`` subclass for the same additive reason as
    ``TeamNotFoundError``: a caller that catches ``ValueError`` around the
    channel path keeps catching it.
    """


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

    **``bind_team`` is the one real exception, and it is deliberate.** It binds
    this conversation to a team named by id, which is precisely what the rest
    of the class prevents — there is no ownership check available, because the
    channel layer has no verified identity for the chat user. It exists so a
    user can re-attach a chat to a team they already know about, and a router
    that exposes it to users must gate it: ``DefaultChannelRouter`` ships it
    off, behind ``allow_register`` on the channel config. It is one method,
    named for what it does, so a review of a custom router has exactly one call
    to look for rather than a class of them. It verifies neither name, which is
    what keeps it from becoming an existence oracle.

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
        """Send as the bound agent, to the team's first supervisor.

        The chat *is* the bound agent — that is what the binding says — so an
        inbound message is that agent speaking, not an anonymous injection.
        ``TeamRuntime.send_from_to`` takes a proxy for the sender and calls
        ``send()`` on it, so the message's ``sender`` is the bound agent and
        the team answers it as it would any member.

        The recipient is the first supervisor that is not the sender itself,
        else the entry point — a chat bound to the only supervisor still has
        the team's own seat to address, and nobody is ever made to talk to
        themselves.

        There is no fallback to the team's default entry. That path sends
        through the entry proxy, which stamps the **entry point** as the
        sender, so a chat bound to any other member would speak with someone
        else's voice — and no reader of the transcript could untangle it. A
        team with nobody else to address is a broken team, and saying so beats
        delivering a message attributed to the wrong member.

        Returns:
            True when sent. False when nothing was sent — the conversation has
            no team, or the content is blank.

        Raises:
            TeamNotFoundError: The binding names a team the service does not
                know.
            NoDefaultRecipientError: The team has no agent for the bound one
                to address — it is its own entry point and has no supervisors.
                The caller names a recipient, or tells the user to.
        """
        if _is_blank(content):
            logger.info("Dropping blank inbound message for %s", self.address)
            return False

        binding = await self.find_binding()
        if binding is None:
            return False

        supervisor = await self._default_recipient(binding)
        return await self._send_to(binding, supervisor, content)

    async def send_to(self, recipient_name: str, content: str | Message) -> bool:
        """Send as the bound agent, to the agent the message named.

        The recipient is taken as given. The team raises for a name it does
        not have, and that error propagates: unlike a channel id, an agent
        name the user typed is worth reporting back rather than silently
        redirecting.

        Args:
            recipient_name: Spawned name of the agent to address.
            content: What to send.

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

        return await self._send_to(binding, recipient_name, content)

    async def _send_to(
        self, binding: ChannelBinding, recipient_name: str, content: str | Message
    ) -> bool:
        """Perform the offloaded send, from the bound agent to one recipient."""
        await asyncio.to_thread(
            self._team_service.send_message_from_to,
            binding.team_id,
            binding.agent_name,
            recipient_name,
            content,
        )
        return True

    async def _default_recipient(self, binding: ChannelBinding) -> str:
        """Return whom the bound agent addresses when the message names nobody.

        The first supervisor that is not the bound agent, else the entry point
        if that is not the bound agent either. Read from the persisted team
        rather than configured: the supervisors are whoever this team was built
        with, and a chat bound to one of them must not be made to talk to
        itself.

        Raises rather than falling back, because every fallback available here
        changes who is speaking: sending through the team's default entry
        stamps the entry point as the sender, and addressing the bound agent
        itself makes it answer its own message.

        Raises:
            TeamNotFoundError: The team is not in the system of record — the
                binding outlived it, or named one that never existed.
            NoDefaultRecipientError: The team exists and holds nobody else: it
                is its own entry point and declares no supervisors.
        """
        process = await asyncio.to_thread(self._team_service.get_team, binding.team_id)
        if process is None:
            msg = f"Team {binding.team_id} not found"
            raise TeamNotFoundError(msg)
        candidates = [ref.name for ref in process.supervisors] + [process.entry_point.name]
        recipient = next((name for name in candidates if name != binding.agent_name), None)
        if recipient is None:
            msg = (
                f"Team {binding.team_id} has no agent for {binding.agent_name} to address: "
                "it is the team's entry point and the team declares no supervisors. "
                "Name the recipient in the message"
            )
            raise NoDefaultRecipientError(msg)
        return recipient

    async def bind_team(self, team_id: uuid.UUID, agent_name: str) -> None:
        """Bind this conversation to a team, taking both names as given.

        **Nothing is verified — deliberately, and it is the safer of the two
        designs.** Looking the team up would make this an oracle: a chat could
        ask "does this id exist?" and read the answer off the reply, and the
        same for an agent name. Since nothing can be checked anyway — the
        payload is unauthenticated, so there is no identity to compare against
        ``Process.user_id`` — the lookup would buy a better error message at
        the cost of confirming what exists. It also keeps this method free of
        the team service entirely: it writes a binding, and that is all.

        A binding naming a team that does not exist is inert rather than
        harmful: the next message fails to reach it, and the outbound path
        never matches it. The cost of a mistyped id is the user's own
        conversation, which is the only thing they could have broken anyway.

        The binding replaces any existing one for this conversation, exactly as
        ``initiate_team``'s does, so a chat can move between teams but never
        hold two.

        Args:
            team_id: The team to bind to, unverified.
            agent_name: The agent's spawned name, unverified. There is no
                default: the entry point's name could only be learned by
                looking the team up, which is what this method does not do.
        """
        await self._registry.register(
            ChannelBinding(
                channel=self.address.channel,
                channel_user_id=self.address.channel_user_id,
                team_id=team_id,
                agent_name=agent_name,
            )
        )
        logger.info(
            "Channel bound to a named team: channel=%s, user=%s, team_id=%s, agent=%s",
            self.address.channel,
            self.address.channel_user_id,
            team_id,
            agent_name,
        )

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

    async def resume_bound_team(self) -> Process | None:
        """Bring this conversation's bound team back up, and return it.

        Takes **no argument**, like ``release()``, ``send()`` and
        ``bound_process()``: the subject is the conversation's own binding, so
        this cannot reach a team the chat is not bound to. That is the whole
        of why it is safe to expose (see this class's docstring).

        Goes through ``TeamService.restore_team`` rather than the placement
        directly, and that is load-bearing: ``restore_team`` is what stores the
        new handle in the runtime cache. A resume that bypassed it would leave
        the cache cold, and the very next send would fail with
        ``Team {id} handle not cached`` — a bare ``ValueError`` the webhook
        surfaces as a 500 and the channel then redelivers (ADR-045 §D7).

        Returns:
            The restored ``Process``, or None when the conversation has no
            binding — there is nothing to resume.

        Raises:
            TeamNotFoundError: The binding names a team the service does not
                know.
            TeamStateConflictError: The team was already running by the time
                this ran — something else brought it up.
            ValueError: The team has been deleted.
        """
        binding = await self.find_binding()
        if binding is None:
            return None
        return await asyncio.to_thread(self._team_service.restore_team, binding.team_id)

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
        """Deliver the message as the bound agent, to whoever it addresses.

        The sender is never in question: the binding says which agent this
        chat is, so the message is that agent speaking.

        The recipient is the ``@Name`` the message **starts** with, else the
        first one in the message being replied to — answering an agent is how a
        user addresses it — and otherwise the team's default recipient. A name
        anywhere but the front of the user's own text is left alone: it is part
        of what they are saying. ``content`` is passed verbatim either way.

        **The bound team's state is resolved first**, because since the
        dispatcher stopped releasing bindings on a stop (ADR-045 §D6) this hook
        meets teams that are not running. Three rows (ADR-045 §D7), resolved
        once for **both** dispatch paths below:

        - **running** — sent, exactly as it always was;
        - **anything else** — resumed, then sent, and *silently*. The
          conversation simply continued; a notice on every post-idle message
          would make the idle timeout a user-visible rule again, which is what
          §D6 exists to remove.
        - **unknown** — the binding names a team the service no longer has.
          The chat is told, and nothing is sent.

        The third row **reports** the loss rather than healing it. §D7 has it
        start a fresh team and rebind the chat, which is a larger behavioural
        decision than saying so; that rebind is what issue #487 still owes.
        The binding is left in place in every row.
        """
        if not await self._resolve_bound_team(binding, ctx):
            return
        recipient = self._recipient_named_in(message)
        logger.debug(
            "Channel continuation: channel=%s, user=%s, team_id=%s, from=%s, to=%s",
            ctx.address.channel,
            ctx.address.channel_user_id,
            binding.team_id,
            binding.agent_name,
            recipient or "<supervisor>",
        )
        if recipient is not None:
            await ctx.send_to(recipient, message.content)
            return
        try:
            await ctx.send(message.content)
        except NoDefaultRecipientError as exc:
            logger.info("No default recipient for %s: %s", ctx.address, exc)
            ctx.notify(
                "This team has nobody for me to pass that to. "
                "Name the agent, for example '@Agent your question'."
            )

    async def _resolve_bound_team(
        self, binding: ChannelBinding, ctx: ChannelRouteContext
    ) -> bool:
        """Make the bound team reachable, or tell the chat it is gone.

        Called once at the top of ``on_bound``, before the recipient is
        resolved, so the named-recipient and default-recipient paths are
        governed by the same check — half of it in one branch would leave half
        the regression it closes.

        ``bound_process()`` returns None for a binding naming a team the
        service no longer knows. It also returns None for an unbound
        conversation, but this hook is only ever reached with a binding in
        hand, so None here is unambiguously the third row; do not call
        ``find_binding()`` a second time to tell them apart.

        Returns:
            True when the message can be dispatched — the team was running, or
            has just been resumed. False when it cannot: the team is gone and
            the chat has been told so.
        """
        process = await ctx.bound_process()
        if process is None:
            return self._report_team_gone(binding, ctx)
        if process.status == TeamStatus.RUNNING:
            return True
        try:
            await ctx.resume_bound_team()
        except TeamStateConflictError:
            # Between the state read and the resume, something else brought
            # the team up — a concurrent message, or an operator. Nothing to
            # do but carry on with the send. Ordered before ``ValueError``
            # deliberately: it is a subclass of it.
            logger.debug("Bound team %s was already running by the resume", binding.team_id)
        except ValueError:
            # TeamNotFoundError, and the bare ValueError a deleted team
            # raises. The team is not coming back, so this is the third row.
            return self._report_team_gone(binding, ctx)
        else:
            logger.info(
                "Resumed a bound team for an inbound message: channel=%s, user=%s, team_id=%s",
                ctx.address.channel,
                ctx.address.channel_user_id,
                binding.team_id,
            )
        return True

    @staticmethod
    def _report_team_gone(binding: ChannelBinding, ctx: ChannelRouteContext) -> bool:
        """Tell the chat its team is gone, send nothing, and always answer False.

        The notice promises nothing: the team is not coming back on its own,
        and the only remedy this router has is one the user can act on now.
        Starting a fresh team and rebinding the chat here is ADR-045 §D7's
        third row and issue #487's — so the binding is left exactly as it is,
        naming the team that was lost.
        """
        logger.info(
            "Bound team is gone: channel=%s, user=%s, team_id=%s",
            ctx.address.channel,
            ctx.address.channel_user_id,
            binding.team_id,
        )
        ctx.notify(
            f"Team {binding.team_id} is no longer available, and I cannot bring it back. "
            "Say /new to start a fresh session."
        )
        return False

    @staticmethod
    def _recipient_named_in(message: ChannelMessage) -> str | None:
        """Return the agent this message addresses, or None for the default.

        Two texts, two rules, because they are written by different people:

        - **What the user typed** addresses an agent only when the name comes
          *first*. Anywhere else it is part of the sentence, and belongs to the
          recipient rather than to the routing — "@Expert, ask a joke to
          @Support" is for the Expert, and the Support is what the Expert is
          being asked to do.
        - **The message being replied to** is the bot's own text, where a name
          appears wherever the adapter put it ("You received a message from
          @Expert_1: …"). There the first name found identifies who is being
          answered.

        Typing a name wins over replying to one: a reply says who you were
        reading, the leading name says whom you mean now.
        """
        typed = _LEADING_AGENT_RE.match(message.content)
        if typed is not None:
            return typed.group(1)
        quoted = _AGENT_NAME_RE.search(message.quoted_text or "")
        return quoted.group() if quoted is not None else None

    async def on_unbound(self, message: ChannelMessage, ctx: ChannelRouteContext) -> None:
        """Start a team for this conversation, with the message as its first."""
        await self._start_session(message, message.content, ctx)

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
        await self._start_session(message, rest or None, ctx)

    async def _start_session(
        self, message: ChannelMessage, content: str | None, ctx: ChannelRouteContext
    ) -> Process:
        """Create the team, bind the chat to it, and tell the user which team it is.

        Every creation announces itself, whether the user asked for one with
        ``new`` or simply spoke to an unbound chat. Two reasons, and the second
        is why this is not merely a nicety:

        - ``new`` with no text produces no team reply at all, so the user who
          just abandoned a conversation would otherwise see nothing happen;
        - the notice is the only place the chat ever learns its team id and
          bound agent. ``register`` reads both out of a replied-to message, so
          this is what makes re-attaching a chat possible without going to the
          web UI for the id.

        Both callers go through here so the two notices cannot drift apart.

        Args:
            message: The inbound message, for the creation parameters it carries.
            content: The team's first message, or None to create it silently.
            ctx: This request's context.

        Returns:
            The created team's ``Process``.
        """
        process = await ctx.initiate_team(
            content,
            catalog_entry=message.catalog_entry,
            team_id=message.team_id,
            team_metadata=message.team_metadata,
            binding_metadata=message.binding_metadata,
        )
        ctx.notify(f"Started a new session — team {process.team_id} as {process.entry_point.name}.")
        return process

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
        """Bind this conversation to the team and agent the message names.

        Both names are read out of free text, and **each falls back on its own**:
        a name the command's own words do not carry is looked for in the
        message being replied to. They routinely arrive from different places —
        a notice names the team but no agent, so replying to one with
        ``/register @HumanProxy_0`` supplies the missing half by hand. Taking
        both from whichever text happened to carry the team id would reject
        that, which is the usable case.

        Neither name is verified (``ChannelRouteContext.bind_team``). The reply
        therefore says what was bound, never whether it exists: a user who
        mistypes finds out when their next message goes unanswered, and a chat
        probing for live team ids learns nothing from the difference.

        Every outcome answers the user, because this decides where their next
        message goes.

        Disabled unless the channel config sets ``allow_register``.
        """
        if not self._allow_register:
            logger.warning(
                "Rejected /register on a channel that has not enabled it: channel=%s, user=%s",
                ctx.address.channel,
                ctx.address.channel_user_id,
            )
            ctx.notify("Registering to an existing team is not enabled on this channel.")
            return
        quoted = message.quoted_text or ""
        team_id_match = _TEAM_ID_RE.search(rest) or _TEAM_ID_RE.search(quoted)
        agent_match = _AGENT_NAME_RE.search(rest) or _AGENT_NAME_RE.search(quoted)
        if team_id_match is None or agent_match is None:
            ctx.notify(
                "Send '/register <team-id> @Agent', or reply to a message naming both "
                "with '/register'. The agent is its spawned name, e.g. @HumanProxy_0."
            )
            return
        team_id = uuid.UUID(team_id_match.group())
        agent_name = agent_match.group()
        await ctx.bind_team(team_id, agent_name)
        ctx.notify(f"Bound to team {team_id} as {agent_name}.")

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

        team_id = binding.team_id
        agent_name = binding.agent_name
        status = process.status.value
        ctx.notify(f"Bound to team {team_id} as {agent_name} — {status}.")
