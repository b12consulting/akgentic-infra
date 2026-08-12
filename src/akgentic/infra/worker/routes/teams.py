"""Worker team operation routes.

create, message, send_to, send_from_to, notification, human-input, metadata, stop,
delete, resume.

There is deliberately **no read route** here. Verbs on the live actor go to the
worker, because only the worker holds it; reads of persisted state go to the
event store, which is the source of truth. A read routed through a worker would
let a momentarily-unreachable worker report a team that plainly exists as "not
found", and a worker response cannot carry a full ``Process`` anyway.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from akgentic.core.messages.orchestrator import SentMessage
from akgentic.infra.adapters.community.local_team_handle import LocalTeamHandle
from akgentic.infra.errors import MetadataValidationError
from akgentic.infra.server.models import (
    EmitMessageRequest,
    HumanInputRequest,
    SendMessageRequest,
    TeamResponse,
    UpdateTeamMetadataRequest,
)
from akgentic.infra.server.routes._message_payload import decode_message, resolve_send_payload
from akgentic.infra.server.services._metadata_payload import dump_metadata, validate_metadata
from akgentic.infra.worker.deps import WorkerServices
from akgentic.infra.worker.state_keys import SERVICES
from akgentic.team.models import Process, TeamCard, TeamRuntime
from akgentic.team.ports import EventStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/teams", tags=["teams"])


class WorkerCreateTeamRequest(BaseModel):
    """Request body for POST /teams on the worker.

    The worker receives the already-resolved TeamCard and user identity from
    the server — catalog resolution happens server-side.

    ``metadata`` travels as plain JSON and is **revalidated here**, against the
    ``metadata_type`` the carried ``team_card`` declares — the worker does not
    take the server's word for it. A worker is reachable by anything holding its
    address, so "the server already checked" is a deployment assumption, not a
    security property; the server-side check protects the server's callers and
    says nothing about who else can reach this route. The validation is the same
    shared helper the server calls, never a second copy.
    """

    team_card: TeamCard = Field(description="Pre-resolved TeamCard for team creation")
    user_id: str = Field(description="Authenticated user identifier (from server)")
    user_email: str = Field(default="", description="Authenticated user email (from server)")
    team_id: uuid.UUID | None = Field(
        default=None,
        description="Caller-supplied team identifier; worker auto-generates a UUID when None",
    )
    catalog_namespace: str | None = Field(
        default=None,
        description="Catalog namespace the team was instantiated from; None if not catalog-sourced",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Plain-JSON business metadata, revalidated here against the metadata_type "
            "team_card declares. Names no type: a __model__ key at any depth is "
            "rejected with 422. Raw wire body, deserialized immediately into a "
            "validated model and never held as state, so it is NOT the "
            "dict[str, Any] anti-pattern (Golden Rule #1) — the same documented "
            "exception CreateTeamRequest.metadata carries."
        ),
    )


def get_services(request: Request) -> WorkerServices:
    """FastAPI dependency: extract WorkerServices from app.state."""
    return SERVICES.require(request)


def _process_to_response(process: Process) -> TeamResponse:
    """Convert a Process model to a TeamResponse.

    Populates ``metadata`` through the same ``dump_metadata`` helper the server
    router uses: ``TeamResponse`` is one shared model with two producers, and a
    producer that left the field at its default would report ``null`` for a team
    that carries metadata. The tag strip comes along for free, so the worker can
    never emit a ``__model__`` the server-side API would refuse back in.
    """
    team_name = process.team_card.name or process.catalog_namespace or str(process.team_id)
    return TeamResponse(
        team_id=process.team_id,
        name=team_name,
        status=process.status.value,
        user_id=process.user_id,
        created_at=process.created_at,
        updated_at=process.updated_at,
        metadata=dump_metadata(process.metadata),
    )


@router.post("", status_code=201, response_model=TeamResponse)
def create_team(
    body: WorkerCreateTeamRequest,
    services: WorkerServices = Depends(get_services),
) -> TeamResponse:
    """Create a new team from a pre-resolved TeamCard, optionally with metadata.

    The server resolves the catalog namespace to a TeamCard and forwards it
    to the worker. The worker calls team_manager.create_team() directly.

    ``body.metadata`` is revalidated here against the card's own
    ``metadata_type`` — the worker never trusts an upstream check it cannot see.
    A rejected body is a 422 and creates nothing.
    """
    logger.info("POST /teams — user_id=%s", body.user_id)
    try:
        metadata = validate_metadata(body.team_card.metadata_type, body.metadata)
    except MetadataValidationError as exc:
        # Deliberately not _raise_action_error: that helper string-matches the
        # message to 404/409 and would report a validation failure as a conflict.
        logger.warning("Team creation rejected — invalid metadata: %s", exc.detail)
        raise HTTPException(status_code=422, detail=exc.detail) from None

    runtime: TeamRuntime = services.team_manager.create_team(
        team_card=body.team_card,
        user_id=body.user_id,
        user_email=body.user_email,
        team_id=body.team_id,
        catalog_namespace=body.catalog_namespace,
        metadata=metadata,
    )

    # Make the team reachable by this router's message / human-input routes,
    # which all resolve the live handle via runtime_cache.get (ADR-001 Decision 1;
    # mirrors resume_team's store). Without this, every post-create call 404s.
    handle = LocalTeamHandle(runtime)
    services.runtime_cache.store(runtime.id, handle)

    process = services.worker_handle.get_team(runtime.id)
    if process is None:  # pragma: no cover
        msg = f"Team {runtime.id} was created but not found in event store"
        raise RuntimeError(msg)
    return _process_to_response(process)


@router.post("/{team_id}/message", status_code=204)
def send_message(
    team_id: uuid.UUID,
    body: SendMessageRequest,
    services: WorkerServices = Depends(get_services),
) -> None:
    """Send a message to a running team on this worker."""
    logger.info("POST /teams/%s/message", team_id)
    handle = services.runtime_cache.get(team_id)
    if handle is None:
        raise HTTPException(status_code=404, detail="Team not found in worker cache")
    try:
        handle.send(resolve_send_payload(body))
    except ValueError as exc:
        _raise_action_error(exc)


@router.post("/{team_id}/message/{agent_name}", status_code=204)
def send_message_to_agent(
    team_id: uuid.UUID,
    agent_name: str,
    body: SendMessageRequest,
    services: WorkerServices = Depends(get_services),
) -> None:
    """Send a message to a specific agent within a running team."""
    logger.info("POST /teams/%s/message/%s", team_id, agent_name)
    handle = services.runtime_cache.get(team_id)
    if handle is None:
        raise HTTPException(status_code=404, detail="Team not found in worker cache")
    try:
        handle.send_to(agent_name, resolve_send_payload(body))
    except ValueError as exc:
        _raise_action_error(exc)


@router.post(
    "/{team_id}/message/from/{sender_name}/to/{recipient_name}",
    status_code=204,
)
def send_message_from_to(
    team_id: uuid.UUID,
    sender_name: str,
    recipient_name: str,
    body: SendMessageRequest,
    services: WorkerServices = Depends(get_services),
) -> None:
    """Send a message from a specific agent to another agent."""
    logger.info("POST /teams/%s/message/from/%s/to/%s", team_id, sender_name, recipient_name)
    handle = services.runtime_cache.get(team_id)
    if handle is None:
        raise HTTPException(status_code=404, detail="Team not found in worker cache")
    try:
        handle.send_from_to(sender_name, recipient_name, resolve_send_payload(body))
    except ValueError as exc:
        _raise_action_error(exc)


@router.post("/{team_id}/notification", status_code=204)
def emit_notification(
    team_id: uuid.UUID,
    body: EmitMessageRequest,
    services: WorkerServices = Depends(get_services),
) -> None:
    """Publish a pre-formed notification Message to a running team on this worker.

    Mirrors the server-side ``/notification`` route (ADR-22) on the worker tier so
    the department/enterprise deployments stop duplicating it. The ``__model__``-
    tagged ``message`` envelope is decoded into the concrete typed ``Message`` (400
    on a bad envelope) and handed to ``handle.emitMessage`` — durable store + live
    stream, no agent processing. The wire envelope is the DICT form the merged
    ``/message*`` routes use (``EmitMessageRequest.message`` is a serialized
    ``model_dump(mode="json")`` dict; ``decode_message`` takes that dict).
    """
    logger.info("POST /teams/%s/notification", team_id)
    message = decode_message(body.message)
    handle = services.runtime_cache.get(team_id)
    if handle is None:
        raise HTTPException(status_code=404, detail="Team not found in worker cache")
    try:
        handle.emitMessage(message)
    except ValueError as exc:
        _raise_action_error(exc)


def _find_message(event_store: EventStore, team_id: uuid.UUID, message_id: str) -> SentMessage:
    """Find a SentMessage by its inner ``message.id`` in persisted events.

    Resolution is by the **inner** ``SentMessage.message.id`` — the id a
    reply's ``parent_id`` references and the id every distributed tier
    (``HttpTeamHandle``, ``RemoteTeamHandle``) puts on the wire — not the
    outer event-envelope ``SentMessage.id`` (ADR-027 §Decision 1).

    Raises:
        ValueError: If no matching SentMessage is found. The ``not found``
            substring is load-bearing: ``_raise_action_error`` maps it to
            HTTP 404.
    """
    events = event_store.load_events(team_id)
    for ev in events:
        if isinstance(ev.event, SentMessage) and str(ev.event.message.id) == message_id:
            return ev.event
    msg = f"Message {message_id} not found"
    raise ValueError(msg)


@router.post("/{team_id}/human-input", status_code=204)
def human_input(
    team_id: uuid.UUID,
    body: HumanInputRequest,
    services: WorkerServices = Depends(get_services),
) -> None:
    """Process human input for a running team."""
    logger.info("POST /teams/%s/human-input", team_id)
    handle = services.runtime_cache.get(team_id)
    if handle is None:
        raise HTTPException(status_code=404, detail="Team not found in worker cache")
    try:
        # _find_message resolves by inner id and returns only SentMessage, so
        # event.message is the inner Message to route (ADR-027 §Decision 1).
        event = _find_message(services.event_store, team_id, body.message_id)
        handle.process_human_input(body.content, event.message)
    except ValueError as exc:
        _raise_action_error(exc)


@router.post("/{team_id}/stop", status_code=204)
def stop_team(
    team_id: uuid.UUID,
    services: WorkerServices = Depends(get_services),
) -> None:
    """Stop a running team without deleting persisted data."""
    logger.info("POST /teams/%s/stop", team_id)
    try:
        services.worker_handle.stop_team(team_id)
        services.runtime_cache.remove(team_id)
    except ValueError as exc:
        _raise_action_error(exc)


@router.delete("/{team_id}", status_code=204)
def delete_team(
    team_id: uuid.UUID,
    services: WorkerServices = Depends(get_services),
) -> None:
    """Delete a team and its resources."""
    logger.info("DELETE /teams/%s", team_id)
    try:
        services.worker_handle.delete_team(team_id)
        services.runtime_cache.remove(team_id)
    except ValueError as exc:
        _raise_action_error(exc)


@router.patch("/{team_id}/metadata", status_code=200, response_model=None)
def update_team_metadata(
    team_id: uuid.UUID,
    body: UpdateTeamMetadataRequest,
    services: WorkerServices = Depends(get_services),
) -> Process:
    """Replace a team's business metadata and return the persisted ``Process``.

    Mirrors the server's ``PATCH /teams/{team_id}/metadata`` in path and verb —
    this module's first ``PATCH``, deliberately, so the operation has one shape
    on both surfaces. The body is a COMPLETE document: a field omitted here is
    gone from the stored value and from the derived filter index alike.

    ``body.metadata`` is revalidated against the ``metadata_type`` the
    **persisted** card declares, never a fresh catalog lookup — the type cannot
    change for a live team (ADR-24 §D7), and re-resolving would let a catalog
    edit silently change what an existing team accepts.

    The response is the persisted ``Process`` **unmodified**, with its
    ``__model__`` tag intact. That is the exception in this module, and the
    reason is structural: this is a worker-to-server internal hop, not a client
    response, and the caller is a tier adapter that must reconstruct a typed
    ``Process`` — including a ``metadata`` value of the team's concrete declared
    class — to satisfy the ``WorkerHandle`` protocol's ``-> Process``. So it is
    neither passed through ``dump_metadata`` nor through ``_process_to_response``
    (which is flat, carries no ``team_card`` and no ``metadata_indexes``).
    Returning it is not a courtesy either: the write path re-derives
    ``metadata_indexes``, and this response is the only place that re-derivation
    becomes observable to the caller.

    A FAILED best-effort push of the new value to a live orchestrator is
    deliberately NOT reflected here (ADR-24 §D7/§D8): the database is the system
    of record and the actor re-reads on its next resume, so reporting an error
    would misdescribe a write that stands. Hence no branch on the push outcome
    and no re-read after the update call.

    ``services.runtime_cache`` is deliberately untouched: it maps team ids to
    live ``TeamHandle``s and holds no metadata, so there is nothing to
    invalidate — a write or eviction here would 404 the cache-reading routes.
    """
    logger.info("PATCH /teams/%s/metadata", team_id)  # never the body: caller business context
    process = services.worker_handle.get_team(team_id)
    if process is None:
        raise HTTPException(status_code=404, detail="Team not found")

    try:
        metadata = validate_metadata(process.team_card.metadata_type, body.metadata)
    except MetadataValidationError as exc:
        # Deliberately not _raise_action_error: that helper string-matches the
        # message to 404/409 and would report a validation failure as a conflict.
        logger.warning("Metadata update rejected — invalid metadata: %s", exc.detail)
        raise HTTPException(status_code=422, detail=exc.detail) from None

    try:
        return services.worker_handle.update_team_metadata(team_id, metadata)
    except ValueError as exc:
        # Lifecycle failures only (unknown / deleted team) — the mapper's
        # "not found" / "deleted" match is exactly right for those.
        _raise_action_error(exc)


@router.post("/{team_id}/resume", status_code=200, response_model=TeamResponse)
def resume_team(
    team_id: uuid.UUID,
    services: WorkerServices = Depends(get_services),
) -> TeamResponse:
    """Resume a stopped team and return its metadata."""
    logger.info("POST /teams/%s/resume", team_id)
    try:
        handle = services.worker_handle.resume_team(team_id)
    except ValueError as exc:
        _raise_action_error(exc)
    services.runtime_cache.store(team_id, handle)
    process = services.worker_handle.get_team(team_id)
    if process is None:
        raise HTTPException(status_code=404, detail="Team not found after resume")
    return _process_to_response(process)


def _raise_action_error(exc: ValueError) -> NoReturn:
    """Map ValueError messages to appropriate HTTP status codes.

    Raises:
        HTTPException: 404 for not-found/deleted errors, 409 for state conflicts.
    """
    detail = str(exc)
    if "not found" in detail or "deleted" in detail:
        raise HTTPException(status_code=404, detail=detail) from None
    raise HTTPException(status_code=409, detail=detail) from None
