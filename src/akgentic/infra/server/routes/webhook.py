"""Webhook route — inbound message ingestion from external interaction channels."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from akgentic.infra.adapters.shared.channel_parser_registry import ChannelParserRegistry
from akgentic.infra.protocols.channels import (
    ChannelAddress,
    ChannelBinding,
    ChannelMessage,
    ChannelParser,
    ChannelRegistry,
    InitiatedTeam,
    InteractionChannelIngestion,
    JsonValue,
)
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.state_keys import (
    CHANNEL_PARSERS,
    CHANNEL_REGISTRY,
    INGESTION,
    TEAM_SERVICE,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])

# The three command words the channel layer consumes. Every other name falls
# through to the routing branches as ordinary text, where akgentic-tool's
# in-team slash mechanism may claim it (ADR-043 §D10).
_COMMAND_NEW = "new"
_COMMAND_UNREGISTER = "unregister"
_COMMAND_STATUS = "status"


def get_channel_parser_registry(request: Request) -> ChannelParserRegistry:
    """FastAPI dependency: extract ChannelParserRegistry from app.state.

    ``channel_parser_registry`` is a required slot on any webhook-serving tier:
    a request reaching this route on a deployment that never wired a registry is
    a misconfiguration, so ``.require()`` raises ``LookupError`` (surfaced as a
    500) rather than returning ``None`` and crashing later. The non-optional
    ``-> ChannelParserRegistry`` return means the handler needs no null-check.
    """
    return CHANNEL_PARSERS.require(request)


def get_channel_registry(request: Request) -> ChannelRegistry:
    """FastAPI dependency: extract ChannelRegistry from app.state."""
    return CHANNEL_REGISTRY.require(request)


def get_ingestion(request: Request) -> InteractionChannelIngestion:
    """FastAPI dependency: extract InteractionChannelIngestion from app.state."""
    return INGESTION.require(request)


def get_team_service(request: Request) -> TeamService:
    """FastAPI dependency: extract TeamService from app.state.

    ``status`` reports the bound team *and its lifecycle state*, and the state
    lives with the team, not with the binding. This is not a new tier coupling:
    ``team_service`` is a required slot on every tier, and ``get_team``
    delegates to ``worker_handle.get_team``, a Protocol each tier already
    implements.
    """
    return TEAM_SERVICE.require(request)


async def _verify_claimed_team(
    channel_registry: ChannelRegistry,
    channel: str,
    message: ChannelMessage,
) -> None:
    """Reject a payload-supplied ``team_id`` that this conversation is not bound to.

    The webhook is unauthenticated by design: signature verification, where a
    tier has one, proves the payload came from the platform — not that whoever
    sent it owns the team named inside it. So ``message.team_id`` is a claim,
    and the binding held for ``(channel, channel_user_id)`` is what the claim is
    checked against (ADR-043 §D9).

    **403, not 404**: the team exists and this caller may not reach it. A 404
    would additionally answer whether an arbitrary team id is real.

    Raises:
        HTTPException: 403 when no binding exists for this conversation, or when
            the binding names a different team.
    """
    binding = await channel_registry.find_binding(channel, message.channel_user_id)
    if binding is None or binding.team_id != message.team_id:
        logger.warning(
            "Webhook reply rejected: channel=%s, user=%s, claimed_team_id=%s — "
            "the claim is not the team this conversation is bound to",
            channel,
            message.channel_user_id,
            message.team_id,
        )
        raise HTTPException(status_code=403, detail="team_id does not belong to this conversation")


def _deliver_notice(
    parser_registry: ChannelParserRegistry,
    address: ChannelAddress,
    text: str,
) -> None:
    """Fan one acknowledgement out to every configured adapter.

    Each adapter compares the address's channel against the one it serves and
    returns silently otherwise, so the fan-out costs at most one no-op per
    unrelated channel. A deployment with no channel configured delivers no
    notice, which is correct: nothing could have reached this route.
    """
    for adapter in parser_registry.get_adapters():
        adapter.deliver_notice(address, text)


async def _create_and_bind(
    *,
    channel_registry: ChannelRegistry,
    ingestion: InteractionChannelIngestion,
    parser: ChannelParser,
    channel: str,
    message: ChannelMessage,
    content: str,
) -> InitiatedTeam:
    """Create a team for this conversation and write its binding.

    Shared by the initiation branch and by ``new`` so that "create a team and
    bind the chat to it" exists once. The two differ only in what they send as
    the first message: the whole inbound text, or the command's ``rest``.

    Args:
        channel_registry: Where the binding is written.
        ingestion: The tier's ingestion, which creates the team.
        parser: The channel's parser, for its default catalog entry.
        message: The inbound message — its channel user, catalog entry and
            metadata are what creation is given.
        channel: The route's path parameter. This is the only component that
            knows which channel the message arrived on, which is why the binding
            is written here rather than inside the ingestion.
        content: The first message's text.

    Returns:
        The created team and the spawned name of its entry-point agent, so a
        caller can name the team in an acknowledgement.
    """
    # Metadata rides the initiation call only: it is fixed at team creation,
    # and only creation validates it against the card.
    initiated = await ingestion.initiate_team(
        content,
        message.channel_user_id,
        message.catalog_entry or parser.default_catalog_entry,
        metadata=message.metadata,
    )
    logger.debug(
        "Webhook initiation: channel=%s, user=%s, new_team=%s, entry_point=%s",
        channel,
        message.channel_user_id,
        initiated.team_id,
        initiated.entry_point_name,
    )
    await channel_registry.register(
        ChannelBinding(
            channel=channel,
            channel_user_id=message.channel_user_id,
            team_id=initiated.team_id,
            agent_name=initiated.entry_point_name,
        )
    )
    return initiated


async def _command_new(
    *,
    parser_registry: ChannelParserRegistry,
    channel_registry: ChannelRegistry,
    ingestion: InteractionChannelIngestion,
    parser: ChannelParser,
    channel: str,
    message: ChannelMessage,
    rest: str,
) -> None:
    """Abandon this conversation's team, if any, and start a fresh one.

    The order is load-bearing: both steps write the same
    ``(channel, channel_user_id)`` key, so a deregister running *after* the
    register would leave the conversation with no binding at all.

    The old team is **not** stopped. Abandoning a running team is the affordance
    (ADR-043 §D8); stopping one is a lifecycle change nothing asked for.

    ``rest`` is sent verbatim as the first message, empty included, and is never
    parsed for a namespace. ``message.catalog_entry`` is honoured exactly as the
    initiation branch honours it — the concern is the chat *text* choosing a
    namespace, not a parser that names an entry on every inbound message.
    """
    address = ChannelAddress(channel=channel, channel_user_id=message.channel_user_id)
    binding = await channel_registry.find_binding(channel, message.channel_user_id)
    if binding is not None:
        await channel_registry.deregister(channel, message.channel_user_id)
    initiated = await _create_and_bind(
        channel_registry=channel_registry,
        ingestion=ingestion,
        parser=parser,
        channel=channel,
        message=message,
        content=rest,
    )
    # ``new`` acknowledges even though the team usually answers for itself:
    # ``/new`` with an empty ``rest`` produces no team reply at all, so without
    # this the user who just abandoned a conversation would see nothing.
    _deliver_notice(parser_registry, address, f"Started a new session — team {initiated.team_id}.")


async def _command_unregister(
    *,
    parser_registry: ChannelParserRegistry,
    channel_registry: ChannelRegistry,
    channel: str,
    message: ChannelMessage,
) -> None:
    """Release this conversation's binding, acknowledging either way.

    The unbound case still answers — a different text, not silence. A caller
    cannot otherwise tell "released" from "nothing happened".
    """
    address = ChannelAddress(channel=channel, channel_user_id=message.channel_user_id)
    binding = await channel_registry.find_binding(channel, message.channel_user_id)
    if binding is None:
        _deliver_notice(parser_registry, address, "No active session to release.")
        return
    await channel_registry.deregister(channel, message.channel_user_id)
    _deliver_notice(parser_registry, address, f"Released team {binding.team_id}.")


async def _command_status(
    *,
    parser_registry: ChannelParserRegistry,
    channel_registry: ChannelRegistry,
    team_service: TeamService,
    channel: str,
    message: ChannelMessage,
) -> None:
    """Report the bound team and its lifecycle state.

    The binding alone would drop the word "state" and lose the dead-binding
    diagnosis: a record can outlive the team it names, and only the team service
    can say so.
    """
    address = ChannelAddress(channel=channel, channel_user_id=message.channel_user_id)
    binding = await channel_registry.find_binding(channel, message.channel_user_id)
    if binding is None:
        _deliver_notice(parser_registry, address, "No active session.")
        return
    process = team_service.get_team(binding.team_id)
    if process is None:
        _deliver_notice(
            parser_registry,
            address,
            f"Bound to team {binding.team_id}, which is no longer known.",
        )
        return
    _deliver_notice(
        parser_registry,
        address,
        f"Bound to team {binding.team_id} — {process.status.value}.",
    )


async def _dispatch_command(
    *,
    parser_registry: ChannelParserRegistry,
    channel_registry: ChannelRegistry,
    ingestion: InteractionChannelIngestion,
    team_service: TeamService,
    parser: ChannelParser,
    channel: str,
    message: ChannelMessage,
) -> bool:
    """Consume ``new`` / ``unregister`` / ``status``, or decline the message.

    **No command reads ``message.team_id``.** Each resolves its subject from the
    binding held for ``(channel, channel_user_id)`` — the conversation the
    message demonstrably came from — so a payload carrying both a command and a
    team id cannot address anything but the caller's own session (ADR-043 §D10).
    That is also why this runs above 74-4's claim check: there is nothing for a
    claim to influence.

    Returns:
        True when the message was consumed and the handler must return. False
        for no command and for any name outside the three, which then reaches
        the team as ordinary text — no 4xx, no notice, no log-and-drop.
    """
    command = message.command
    if command is None:
        return False
    if command.name == _COMMAND_NEW:
        await _command_new(
            parser_registry=parser_registry,
            channel_registry=channel_registry,
            ingestion=ingestion,
            parser=parser,
            channel=channel,
            message=message,
            rest=command.rest,
        )
    elif command.name == _COMMAND_UNREGISTER:
        await _command_unregister(
            parser_registry=parser_registry,
            channel_registry=channel_registry,
            channel=channel,
            message=message,
        )
    elif command.name == _COMMAND_STATUS:
        await _command_status(
            parser_registry=parser_registry,
            channel_registry=channel_registry,
            team_service=team_service,
            channel=channel,
            message=message,
        )
    else:
        return False
    logger.debug(
        "Webhook command consumed: channel=%s, user=%s, command=%s",
        channel,
        message.channel_user_id,
        command.name,
    )
    return True


@router.post("/{channel}", status_code=204)
async def webhook(
    channel: str,
    request: Request,
    parser_registry: ChannelParserRegistry = Depends(get_channel_parser_registry),
    channel_registry: ChannelRegistry = Depends(get_channel_registry),
    ingestion: InteractionChannelIngestion = Depends(get_ingestion),
    team_service: TeamService = Depends(get_team_service),
) -> None:
    """Process an inbound webhook from an external interaction channel.

    A ``new`` / ``unregister`` / ``status`` command is consumed first, acting on
    the caller's own conversation and answering 204 (ADR-043 §D10). Any other
    command name is ordinary text and falls through.

    Three routing flows based on parsed ChannelMessage:
    1. Reply: team_id is set → verified against the conversation's binding
       (403 if it names another team), then route_reply
    2. Continuation: no team_id but existing team found → route_reply
    3. Initiation: no existing team → initiate_team + register
    """
    content_type = request.headers.get("content-type", "")
    logger.info("POST /webhook/%s — content_type=%s", channel, content_type)

    parser = parser_registry.get_parser(channel)
    if parser is None:
        logger.warning("Unknown channel: %s", channel)
        raise HTTPException(status_code=404, detail=f"Unknown channel: {channel}")

    if "application/json" in content_type:
        payload: dict[str, JsonValue] = await request.json()
    elif "application/x-www-form-urlencoded" in content_type:
        form_data = await request.form()
        payload = {k: str(v) for k, v in form_data.items()}
    else:
        logger.warning("Unsupported content type: %s", content_type)
        raise HTTPException(status_code=415, detail="Unsupported content type")
    try:
        message = await parser.parse(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if await _dispatch_command(
        parser_registry=parser_registry,
        channel_registry=channel_registry,
        ingestion=ingestion,
        team_service=team_service,
        parser=parser,
        channel=channel,
        message=message,
    ):
        return

    if message.team_id is not None:
        # Reply flow — the claimed team is verified before anything is delivered.
        await _verify_claimed_team(channel_registry, channel, message)
        logger.debug("Webhook reply: channel=%s, team_id=%s", channel, message.team_id)
        await ingestion.route_reply(message.team_id, message.content, message.message_id)
    else:
        existing_team = await channel_registry.find_team(channel, message.channel_user_id)
        if existing_team is not None:
            # Continuation flow
            logger.debug(
                "Webhook continuation: channel=%s, user=%s, team_id=%s",
                channel,
                message.channel_user_id,
                existing_team,
            )
            await ingestion.route_reply(existing_team, message.content, message.message_id)
        else:
            # Initiation flow — shared with ``new``, which otherwise carries a
            # second, independent copy of "create a team and write its binding".
            await _create_and_bind(
                channel_registry=channel_registry,
                ingestion=ingestion,
                parser=parser,
                channel=channel,
                message=message,
                content=message.content,
            )
