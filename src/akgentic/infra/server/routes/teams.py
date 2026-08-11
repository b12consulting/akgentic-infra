"""Team CRUD and action endpoints — create, list, get, delete, message, stop, restore, events."""

from __future__ import annotations

import logging
import uuid
from typing import NoReturn

from fastapi import APIRouter, Depends, HTTPException, Request

from akgentic.catalog.models.errors import EntryNotFoundError
from akgentic.infra.errors import MetadataValidationError
from akgentic.infra.server.auth import RequestUser, get_request_user
from akgentic.infra.server.models import (
    AgentStateListResponse,
    AgentStateResponse,
    CreateTeamRequest,
    EmitMessageRequest,
    EventListResponse,
    EventResponse,
    HumanInputRequest,
    SendMessageRequest,
    TeamListResponse,
    TeamResponse,
)
from akgentic.infra.server.routes._message_payload import decode_message, resolve_send_payload
from akgentic.infra.server.routes._team_access import get_team_service, require_team_access
from akgentic.infra.server.services._metadata_payload import dump_metadata
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.state_keys import CONNECTION_MANAGER
from akgentic.team import EventNotFoundError
from akgentic.team.models import Process, TeamStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/teams", tags=["teams"])


def _process_to_response(process: Process) -> TeamResponse:
    """Convert a Process model to a TeamResponse.

    The single conversion point for every route returning a ``TeamResponse``,
    which is why the metadata ``__model__`` strip belongs here: the persisted
    value carries the tag for the store's benefit, the wire never does, and one
    site means no route can forget.
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
    body: CreateTeamRequest,
    user: RequestUser = Depends(get_request_user),
    service: TeamService = Depends(get_team_service),
) -> TeamResponse:
    """Create a new team from a catalog namespace, optionally with metadata.

    ``body.metadata`` is plain JSON validated server-side against the type the
    team's catalog entry declares; a rejected body is a 422 and creates nothing.
    """
    logger.info("POST /teams — catalog_namespace=%s", body.catalog_namespace)
    try:
        process = service.create_team(
            catalog_namespace=body.catalog_namespace,
            user_id=user.user_id,
            user_email=user.email,
            metadata=body.metadata,
        )
    except MetadataValidationError as exc:
        # Deliberately not _raise_action_error: that helper string-matches the
        # message to 404/409 and would report a validation failure as a conflict.
        logger.warning("Team creation rejected — invalid metadata: %s", exc.detail)
        raise HTTPException(status_code=422, detail=exc.detail) from None
    except EntryNotFoundError:
        logger.warning(
            "Team creation failed: catalog namespace %s not found",
            body.catalog_namespace,
        )
        raise HTTPException(status_code=404, detail="Catalog namespace not found") from None
    return _process_to_response(process)


METADATA_FILTER_PREFIX = "meta."
"""Query-parameter prefix that marks a business-metadata equality filter."""


def _parse_metadata_filter(request: Request) -> dict[str, str] | None:
    """Collect repeated ``?meta.<key>=<value>`` parameters into a filter.

    Read from the raw multi-item query string rather than declared as a route
    parameter because the key set is open: it is whatever the team's metadata
    model declares, which the HTTP layer neither knows nor needs to know.

    Returns:
        The ``key -> value`` filter, or ``None`` when no ``meta.`` parameter was
        given. ``None`` rather than ``{}``: an empty dict is an empty
        conjunction that some backends would still translate into a query, and
        "no filter" is not a filter that matches everything by coincidence.

    Raises:
        HTTPException: 422 naming the offending parameter, when a key is empty
            or repeated. Two values for one key can never both hold under
            equality matching, so first-wins or last-wins would answer a
            question the client did not ask, silently.
    """
    filters: dict[str, str] = {}
    for name, value in request.query_params.multi_items():
        if not name.startswith(METADATA_FILTER_PREFIX):
            continue
        key = name[len(METADATA_FILTER_PREFIX) :]
        if not key:
            raise HTTPException(
                status_code=422,
                detail=f"query parameter '{name}' names no metadata key",
            )
        if key in filters:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"query parameter '{name}' is repeated; metadata filtering is "
                    "equality-only, so one key cannot carry two values"
                ),
            )
        filters[key] = value
    return filters or None


@router.get("", response_model=TeamListResponse)
def list_teams(
    request: Request,
    user: RequestUser = Depends(get_request_user),
    service: TeamService = Depends(get_team_service),
    status: TeamStatus | None = None,
    page: int = 1,
    size: int = 250,
) -> TeamListResponse:
    """List one page of the current user's teams, plus the total count.

    ``status`` is validated by FastAPI against ``TeamStatus``; an unknown value
    is a 422 raised by the framework, not handled here. Omitting it returns
    every status, ``DELETED`` included.

    Repeated ``?meta.<key>=<value>`` parameters add an equality filter on the
    team's business metadata; distinct keys AND-combine. Values travel to the
    store verbatim — deriving the index entry and escaping the ``|`` separator
    happen exactly once, inside ``akgentic-team``, and duplicating either here
    would double-escape and match nothing.

    No filter widens the set beyond the caller's own teams: ``user_id`` comes
    from the request identity seam and is always pushed down alongside, so a
    metadata filter can never reach — or count — another owner's team.
    ``total_count`` counts the filtered set, so it stays consistent with the
    page slice it accompanies.
    """
    metadata = _parse_metadata_filter(request)
    logger.debug(
        "GET /teams — status=%s meta_keys=%s page=%s size=%s",
        status,
        sorted(metadata) if metadata else None,
        page,
        size,
    )
    page_slice, total = service.list_teams(
        user_id=user.user_id, status=status, metadata=metadata, page=page, size=size
    )
    return TeamListResponse(
        teams=[_process_to_response(p) for p in page_slice],
        total_count=total,
    )


@router.get(
    "/{team_id}",
    response_model=TeamResponse,
    dependencies=[Depends(require_team_access)],
)
def get_team(
    team_id: uuid.UUID,
    service: TeamService = Depends(get_team_service),
) -> TeamResponse:
    """Get a single team by ID."""
    logger.debug("GET /teams/%s", team_id)
    process = service.get_team(team_id)
    if process is None:
        raise HTTPException(status_code=404, detail="Team not found")
    return _process_to_response(process)


@router.delete(
    "/{team_id}",
    status_code=204,
    dependencies=[Depends(require_team_access)],
)
def delete_team(
    team_id: uuid.UUID,
    service: TeamService = Depends(get_team_service),
) -> None:
    """Stop and delete a team."""
    logger.info("DELETE /teams/%s", team_id)
    try:
        service.delete_team(team_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Team not found") from None


# --- Action Endpoints ---


@router.post(
    "/{team_id}/message",
    status_code=204,
    dependencies=[Depends(require_team_access)],
)
def send_message(
    team_id: uuid.UUID,
    body: SendMessageRequest,
    service: TeamService = Depends(get_team_service),
) -> None:
    """Send a message to a running team.

    Accepts either a plain ``content`` string or a pre-formed typed ``Message``
    (the ``message`` wire envelope, deserialized to the concrete type for agent
    processing); a bad envelope is a 400 (ADR-22). Team-state errors map through
    ``_raise_action_error``.
    """
    logger.info("POST /teams/%s/message", team_id)
    try:
        service.send_message(team_id, resolve_send_payload(body))
    except ValueError as exc:
        _raise_action_error(exc)


@router.post(
    "/{team_id}/message/{agent_name}",
    status_code=204,
    dependencies=[Depends(require_team_access)],
)
def send_message_to_agent(
    team_id: uuid.UUID,
    agent_name: str,
    body: SendMessageRequest,
    service: TeamService = Depends(get_team_service),
) -> None:
    """Send a message (plain ``content`` or typed ``message`` envelope) to a specific agent."""
    logger.info("POST /teams/%s/message/%s", team_id, agent_name)
    try:
        service.send_message_to(team_id, agent_name, resolve_send_payload(body))
    except ValueError as exc:
        _raise_action_error(exc)


@router.post(
    "/{team_id}/message/from/{sender_name}/to/{recipient_name}",
    status_code=204,
    dependencies=[Depends(require_team_access)],
)
def send_message_from_to(
    team_id: uuid.UUID,
    sender_name: str,
    recipient_name: str,
    body: SendMessageRequest,
    service: TeamService = Depends(get_team_service),
) -> None:
    """Send a message (plain ``content`` or typed ``message``) from one agent to another."""
    logger.info("POST /teams/%s/message/from/%s/to/%s", team_id, sender_name, recipient_name)
    try:
        service.send_message_from_to(
            team_id, sender_name, recipient_name, resolve_send_payload(body)
        )
    except ValueError as exc:
        _raise_action_error(exc)


@router.post(
    "/{team_id}/notification",
    status_code=204,
    dependencies=[Depends(require_team_access)],
)
def emit_notification(
    team_id: uuid.UUID,
    body: EmitMessageRequest,
    service: TeamService = Depends(get_team_service),
) -> None:
    """Inject a pre-formed notification Message into a running team.

    Deserializes the ``__model__``-tagged payload back into the concrete
    ``Message`` and publishes it to the team's subscribers with no agent
    processing (ADR-22). A decode failure or a non-``Message`` payload is a
    client error (400); team-state errors map through ``_raise_action_error``.
    """
    logger.info("POST /teams/%s/notification", team_id)
    message = decode_message(body.message)
    try:
        service.emit_message(team_id, message)
    except ValueError as exc:
        _raise_action_error(exc)


@router.post(
    "/{team_id}/human-input",
    status_code=204,
    dependencies=[Depends(require_team_access)],
)
def human_input(
    team_id: uuid.UUID,
    body: HumanInputRequest,
    service: TeamService = Depends(get_team_service),
) -> None:
    """Provide human input in response to an agent request."""
    logger.info("POST /teams/%s/human-input", team_id)
    try:
        service.process_human_input(team_id, body.content, body.message_id)
    except ValueError as exc:
        _raise_action_error(exc)


@router.post(
    "/{team_id}/stop",
    status_code=204,
    dependencies=[Depends(require_team_access)],
)
def stop_team(
    team_id: uuid.UUID,
    service: TeamService = Depends(get_team_service),
) -> None:
    """Stop a running team without deleting persisted data."""
    logger.info("POST /teams/%s/stop", team_id)
    try:
        service.stop_team(team_id)
    except ValueError as exc:
        _raise_action_error(exc)


@router.post(
    "/{team_id}/restore",
    status_code=200,
    response_model=TeamResponse,
    dependencies=[Depends(require_team_access)],
)
def restore_team(
    team_id: uuid.UUID,
    request: Request,
    service: TeamService = Depends(get_team_service),
) -> TeamResponse:
    """Restore a stopped team and notify waiting WebSocket connections."""
    logger.info("POST /teams/%s/restore", team_id)
    try:
        process = service.restore_team(team_id)
    except ValueError as exc:
        _raise_action_error(exc)

    conn_mgr = CONNECTION_MANAGER.get(request)
    if conn_mgr is not None:
        from akgentic.infra.server.routes.ws import notify_restore

        notify_restore(conn_mgr, service, team_id)

    return _process_to_response(process)


@router.get(
    "/{team_id}/events",
    response_model=EventListResponse,
    dependencies=[Depends(require_team_access)],
)
def get_events(
    team_id: uuid.UUID,
    after_event_id: uuid.UUID | None = None,
    service: TeamService = Depends(get_team_service),
) -> EventListResponse:
    """Get persisted events for a team, or only those after ``after_event_id``."""
    logger.debug("GET /teams/%s/events after_event_id=%s", team_id, after_event_id)
    try:
        events = service.get_events(team_id, after_event_id=after_event_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Team not found") from None
    except EventNotFoundError:
        # A stale cursor is a bad cursor, not a missing team — 404 here would tell
        # the client the team is gone and tear down a live team's view.
        raise HTTPException(
            status_code=400, detail=f"Unknown after_event_id cursor: {after_event_id}"
        ) from None
    return EventListResponse(
        events=[
            EventResponse(
                team_id=ev.team_id,
                sequence=ev.sequence,
                event=ev.event.model_dump(mode="json"),
                timestamp=ev.timestamp,
            )
            for ev in events
        ]
    )


@router.get(
    "/{team_id}/agent-states",
    response_model=AgentStateListResponse,
    dependencies=[Depends(require_team_access)],
)
def get_agent_states(
    team_id: uuid.UUID,
    service: TeamService = Depends(get_team_service),
) -> AgentStateListResponse:
    """Get the latest persisted state snapshot for each agent of a team."""
    logger.debug("GET /teams/%s/agent-states", team_id)
    try:
        snapshots = service.get_agent_states(team_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Team not found") from None
    return AgentStateListResponse(
        states=[
            AgentStateResponse(
                agent_id=s.agent_id,
                name=s.name,
                state=s.state.model_dump(mode="json"),
                updated_at=s.updated_at,
            )
            for s in snapshots
        ]
    )


def _raise_action_error(exc: ValueError) -> NoReturn:
    """Map ValueError messages to appropriate HTTP status codes.

    Raises:
        HTTPException: 404 for not-found/deleted errors, 409 for state conflicts.

    Note:
        String matching on exception messages is fragile. Department/enterprise
        tiers should replace this with structured error codes (e.g. typed
        exception subclasses with an ``http_status`` attribute).
    """
    detail = str(exc)
    if "not found" in detail or "deleted" in detail:
        logger.debug("Action error (not found): %s", detail)
        raise HTTPException(status_code=404, detail=detail) from None
    logger.warning("Action error: %s", detail)
    raise HTTPException(status_code=409, detail=detail) from None
