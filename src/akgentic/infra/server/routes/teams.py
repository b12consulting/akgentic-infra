"""Team CRUD and action endpoints — create, list, get, delete, message, stop, restore, events."""

from __future__ import annotations

import uuid
from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Request

from akgentic.catalog.models.errors import EntryNotFoundError
from akgentic.infra.server.models import (
    CreateTeamRequest,
    EventListResponse,
    EventResponse,
    HumanInputRequest,
    SendMessageRequest,
    TeamListResponse,
    TeamResponse,
)
from akgentic.infra.server.services.team_service import TeamService
from akgentic.team.models import Process

router = APIRouter(prefix="/teams", tags=["teams"])


def get_team_service(request: Request) -> TeamService:
    """FastAPI dependency: extract TeamService from app.state."""
    return cast(TeamService, request.app.state.team_service)


def _process_to_response(process: Process) -> TeamResponse:
    """Convert a Process model to a TeamResponse."""
    return TeamResponse(
        team_id=process.team_id,
        name=process.team_card.name,
        status=process.status.value,
        user_id=process.user_id,
        created_at=process.created_at,
        updated_at=process.updated_at,
    )


@router.post("/", status_code=201, response_model=TeamResponse)
def create_team(
    body: CreateTeamRequest,
    service: TeamService = Depends(get_team_service),
) -> TeamResponse:
    """Create a new team from a catalog entry."""
    try:
        process = service.create_team(
            catalog_entry_id=body.catalog_entry_id,
            user_id="anonymous",  # Community tier: no auth, single-user
        )
    except EntryNotFoundError:
        raise HTTPException(status_code=404, detail="Catalog entry not found") from None
    return _process_to_response(process)


@router.get("/", response_model=TeamListResponse)
def list_teams(
    service: TeamService = Depends(get_team_service),
) -> TeamListResponse:
    """List all teams for the current user."""
    processes = service.list_teams(user_id="anonymous")  # Community tier: no auth
    return TeamListResponse(teams=[_process_to_response(p) for p in processes])


@router.get("/{team_id}", response_model=TeamResponse)
def get_team(
    team_id: uuid.UUID,
    service: TeamService = Depends(get_team_service),
) -> TeamResponse:
    """Get a single team by ID."""
    process = service.get_team(team_id)
    if process is None:
        raise HTTPException(status_code=404, detail="Team not found")
    return _process_to_response(process)


@router.delete("/{team_id}", status_code=204)
def delete_team(
    team_id: uuid.UUID,
    service: TeamService = Depends(get_team_service),
) -> None:
    """Stop and delete a team."""
    try:
        service.delete_team(team_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Team not found") from None


# --- Action Endpoints ---


@router.post("/{team_id}/message", status_code=204)
def send_message(
    team_id: uuid.UUID,
    body: SendMessageRequest,
    service: TeamService = Depends(get_team_service),
) -> None:
    """Send a message to a running team."""
    try:
        service.send_message(team_id, body.content)
    except ValueError as exc:
        _raise_action_error(exc)


@router.post("/{team_id}/human-input", status_code=204)
def human_input(
    team_id: uuid.UUID,
    body: HumanInputRequest,
    service: TeamService = Depends(get_team_service),
) -> None:
    """Provide human input in response to an agent request."""
    try:
        service.process_human_input(team_id, body.content, body.message_id)
    except ValueError as exc:
        _raise_action_error(exc)


@router.post("/{team_id}/stop", status_code=204)
def stop_team(
    team_id: uuid.UUID,
    service: TeamService = Depends(get_team_service),
) -> None:
    """Stop a running team without deleting persisted data."""
    try:
        service.stop_team(team_id)
    except ValueError as exc:
        _raise_action_error(exc)


@router.post("/{team_id}/restore", status_code=200, response_model=TeamResponse)
def restore_team(
    team_id: uuid.UUID,
    service: TeamService = Depends(get_team_service),
) -> TeamResponse:
    """Restore a stopped team."""
    try:
        process = service.restore_team(team_id)
    except ValueError as exc:
        _raise_action_error(exc)
        raise  # unreachable, satisfies type checker
    return _process_to_response(process)


@router.get("/{team_id}/events", response_model=EventListResponse)
def get_events(
    team_id: uuid.UUID,
    service: TeamService = Depends(get_team_service),
) -> EventListResponse:
    """Get all persisted events for a team."""
    try:
        events = service.get_events(team_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Team not found") from None
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


def _raise_action_error(exc: ValueError) -> None:
    """Map ValueError messages to appropriate HTTP status codes.

    Raises:
        HTTPException: 404 for not-found errors, 409 for state conflicts.
    """
    detail = str(exc)
    if "not found" in detail:
        raise HTTPException(status_code=404, detail=detail) from None
    raise HTTPException(status_code=409, detail=detail) from None
