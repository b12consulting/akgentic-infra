"""Webhook route — inbound message ingestion from external interaction channels."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from akgentic.infra.adapters.shared.channel_parser_registry import ChannelParserRegistry
from akgentic.infra.protocols.channels import (
    ChannelBinding,
    ChannelMessage,
    ChannelRegistry,
    InteractionChannelIngestion,
    JsonValue,
)
from akgentic.infra.server.state_keys import CHANNEL_PARSERS, CHANNEL_REGISTRY, INGESTION

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])


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


@router.post("/{channel}", status_code=204)
async def webhook(
    channel: str,
    request: Request,
    parser_registry: ChannelParserRegistry = Depends(get_channel_parser_registry),
    channel_registry: ChannelRegistry = Depends(get_channel_registry),
    ingestion: InteractionChannelIngestion = Depends(get_ingestion),
) -> None:
    """Process an inbound webhook from an external interaction channel.

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
            # Initiation flow
            # Metadata rides the initiation branch only: it is fixed at team
            # creation, and only creation validates it against the card.
            initiated = await ingestion.initiate_team(
                message.content,
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
            # The binding is written here rather than inside the ingestion
            # because ``channel`` is the route's path parameter — this is the
            # only component that knows which channel the message arrived on.
            await channel_registry.register(
                ChannelBinding(
                    channel=channel,
                    channel_user_id=message.channel_user_id,
                    team_id=initiated.team_id,
                    agent_name=initiated.entry_point_name,
                )
            )
