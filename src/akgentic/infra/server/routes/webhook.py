"""Webhook route — inbound messages from external interaction channels, parsed and routed."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from akgentic.infra.adapters.channels.channel_parser_registry import ChannelParserRegistry
from akgentic.infra.adapters.channels.channel_router import ChannelRouteContext
from akgentic.infra.protocols.channels import (
    ChannelAddress,
    ChannelRegistry,
    JsonValue,
)
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.state_keys import (
    CHANNEL_PARSERS,
    CHANNEL_REGISTRY,
    TEAM_SERVICE,
)

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


def get_team_service(request: Request) -> TeamService:
    """FastAPI dependency: extract TeamService from app.state.

    The channel router creates, messages and reports on teams through it, held
    privately by ``ChannelRouteContext``. It is the tier-agnostic seam already —
    creation goes through the tier's placement, delivery through the tier's team
    handle — and ``team_service`` is a required slot on every tier.
    """
    return TEAM_SERVICE.require(request)


@router.post("/{channel}", status_code=204)
async def webhook(
    channel: str,
    request: Request,
    parser_registry: ChannelParserRegistry = Depends(get_channel_parser_registry),
    channel_registry: ChannelRegistry = Depends(get_channel_registry),
    team_service: TeamService = Depends(get_team_service),
) -> None:
    """Process an inbound webhook from an external interaction channel.

    The route parses and routes, and nothing else. What a message *does* —
    consume a command, reply to the bound team, start a team, or nothing — is
    the channel router's decision (``ChannelConfig.router_fqcn``, else
    ``DefaultChannelRouter``).
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
        # Acknowledged and dropped, NOT refused. A channel treats any non-2xx as
        # "delivery failed" and redelivers with backoff, so answering 4xx to an
        # update the parser will never accept queues it forever: observed on
        # Telegram as two undeliverable updates retried at 1s, 2s, 7s, 15s, 31s,
        # 63s … with `pending_update_count` never reaching zero. Nothing a retry
        # can change is a client error worth reporting — a photo, a sticker or a
        # service message simply carries no text this parser can route.
        logger.warning("Dropping unparseable %s update: %s", channel, exc)
        return

    ctx = ChannelRouteContext(
        address=ChannelAddress(
            channel=channel,
            channel_user_id=message.channel_user_id,
            # Carried onto the address, not just the binding a router may later
            # write: the notice path has no binding, so an adapter needing
            # per-conversation routing data would otherwise have it for agent
            # messages and not for acknowledgements.
            metadata=message.binding_metadata or {},
        ),
        registry=channel_registry,
        team_service=team_service,
        adapters=parser_registry.get_adapters(),
        default_catalog_entry=parser.default_catalog_entry,
    )
    await parser_registry.get_router(channel).route(message, ctx)
