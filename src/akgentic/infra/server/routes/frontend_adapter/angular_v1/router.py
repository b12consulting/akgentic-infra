"""V1 REST route translations for the Angular V1 frontend adapter.

Every route delegates to TeamService — zero business logic in the adapter.
"""

from __future__ import annotations

import uuid
from typing import NoReturn, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from akgentic.core.messages.message import Message
from akgentic.core.messages.orchestrator import (
    StateChangedMessage,
)
from akgentic.infra.server.routes.frontend_adapter.angular_v1._helpers import (
    classify_message_type,
    extract_message_content,
    get_sender_name,
)
from akgentic.infra.server.routes.frontend_adapter.angular_v1.models import (
    V1LlmContextEntry,
    V1MessageEntry,
    V1ProcessContext,
    V1ProcessList,
    V1StateEntry,
    V1StatusResponse,
)
from akgentic.infra.server.services.team_service import TeamService
from akgentic.team.models import PersistedEvent, Process

process_router = APIRouter(prefix="/process", tags=["v1-compat"])
human_input_router = APIRouter(prefix="/process_human_input", tags=["v1-compat"])
messages_router = APIRouter(prefix="/messages", tags=["v1-compat"])
llm_context_router = APIRouter(prefix="/llm_context", tags=["v1-compat"])
states_router = APIRouter(prefix="/states", tags=["v1-compat"])


class V1MessageBody(BaseModel):
    """Request body for PATCH /process/{id}."""

    content: str = Field(description="Message content to send")


class V1HumanInputBody(BaseModel):
    """Request body for POST /process_human_input/{id}/human/{proxy}."""

    content: str = Field(description="Human input content")
    message_id: str = Field(description="ID of the original message being answered")


def get_team_service(request: Request) -> TeamService:
    """FastAPI dependency: extract TeamService from app.state."""
    return cast(TeamService, request.app.state.team_service)


def _to_v1_process_context(process: Process) -> V1ProcessContext:
    """Convert a V2 Process to a V1ProcessContext."""
    return V1ProcessContext(
        id=str(process.team_id),
        type=process.team_card.name,
        status=process.status.value,
        created_at=process.created_at.isoformat(),
        updated_at=process.updated_at.isoformat(),
        params={},
    )


def _raise_action_error(exc: ValueError) -> NoReturn:
    """Map ValueError messages to appropriate HTTP status codes."""
    detail = str(exc)
    if "not found" in detail or "deleted" in detail:
        raise HTTPException(status_code=404, detail=detail) from None
    raise HTTPException(status_code=409, detail=detail) from None


def _parse_uuid(id_str: str) -> uuid.UUID:
    """Parse a string to UUID, raising 422 on failure."""
    try:
        return uuid.UUID(id_str)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid UUID: {id_str}") from None


def _extract_message_content(event: Message) -> str | None:
    """Extract displayable content from a message event."""
    return extract_message_content(event)


def _classify_message_type(event: Message) -> str:
    """Classify a message event as user/agent/system."""
    return classify_message_type(event)


def _get_sender_name(event: Message) -> str:
    """Extract sender name from a message event."""
    return get_sender_name(event)


def _events_to_v1_messages(events: list[PersistedEvent]) -> list[V1MessageEntry]:
    """Filter and transform persisted events to V1 message entries."""
    result: list[V1MessageEntry] = []
    for ev in events:
        content = _extract_message_content(ev.event)
        if content is None:
            continue
        result.append(
            V1MessageEntry(
                id=str(ev.event.id),
                sender=_get_sender_name(ev.event),
                content=content,
                timestamp=ev.timestamp.isoformat(),
                type=_classify_message_type(ev.event),
            )
        )
    return result


# --- Process routes ---


@process_router.post("/{type}", response_model=V1ProcessContext)
def create_process(
    type: str,
    service: TeamService = Depends(get_team_service),
) -> V1ProcessContext:
    """POST /process/{type} -> create team via V2 service."""
    from akgentic.catalog.models.errors import EntryNotFoundError

    try:
        process = service.create_team(catalog_entry_id=type, user_id="anonymous")
    except EntryNotFoundError:
        raise HTTPException(status_code=404, detail="Catalog entry not found") from None
    return _to_v1_process_context(process)


@process_router.get("/", response_model=V1ProcessList)
def list_processes(
    service: TeamService = Depends(get_team_service),
) -> V1ProcessList:
    """GET /process -> list teams with V1 response shape."""
    processes = service.list_teams(user_id="anonymous")
    return V1ProcessList(processes=[_to_v1_process_context(p) for p in processes])


@process_router.get("/{id}", response_model=V1ProcessContext)
def get_process(
    id: str,
    service: TeamService = Depends(get_team_service),
) -> V1ProcessContext:
    """GET /process/{id} -> get team with V1 response shape."""
    team_id = _parse_uuid(id)
    process = service.get_team(team_id)
    if process is None:
        raise HTTPException(status_code=404, detail="Team not found")
    return _to_v1_process_context(process)


@process_router.patch("/{id}", status_code=200, response_model=V1StatusResponse)
def send_process_message(
    id: str,
    body: V1MessageBody,
    service: TeamService = Depends(get_team_service),
) -> V1StatusResponse:
    """PATCH /process/{id} -> send message via V2 service."""
    team_id = _parse_uuid(id)
    try:
        service.send_message(team_id, body.content)
    except ValueError as exc:
        _raise_action_error(exc)
    return V1StatusResponse(status="ok")


@process_router.delete("/{id}", status_code=200, response_model=V1StatusResponse)
def delete_process(
    id: str,
    service: TeamService = Depends(get_team_service),
) -> V1StatusResponse:
    """DELETE /process/{id} -> delete team via V2 service."""
    team_id = _parse_uuid(id)
    try:
        service.delete_team(team_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Team not found") from None
    return V1StatusResponse(status="ok")


@process_router.delete("/{id}/archive", status_code=200, response_model=V1StatusResponse)
def archive_process(
    id: str,
    service: TeamService = Depends(get_team_service),
) -> V1StatusResponse:
    """DELETE /process/{id}/archive -> stop team via V2 service."""
    team_id = _parse_uuid(id)
    try:
        service.stop_team(team_id)
    except ValueError as exc:
        _raise_action_error(exc)
    return V1StatusResponse(status="ok")


@process_router.post("/{id}/restore", response_model=V1ProcessContext)
def restore_process(
    id: str,
    service: TeamService = Depends(get_team_service),
) -> V1ProcessContext:
    """POST /process/{id}/restore -> restore team via V2 service."""
    team_id = _parse_uuid(id)
    try:
        process = service.restore_team(team_id)
    except ValueError as exc:
        _raise_action_error(exc)
    return _to_v1_process_context(process)


# --- Human input route ---


@human_input_router.post("/{id}/human/{proxy}", status_code=200, response_model=V1StatusResponse)
def process_human_input(
    id: str,
    proxy: str,
    body: V1HumanInputBody,
    service: TeamService = Depends(get_team_service),
) -> V1StatusResponse:
    """POST /process_human_input/{id}/human/{proxy} -> route human input.

    V1 had `proxy` in path but V2 finds HumanProxy automatically — ignore proxy param.
    """
    team_id = _parse_uuid(id)
    try:
        service.process_human_input(team_id, body.content, body.message_id)
    except ValueError as exc:
        _raise_action_error(exc)
    return V1StatusResponse(status="ok")


# --- Messages route ---


@messages_router.get("/{id}", response_model=list[V1MessageEntry])
def get_messages(
    id: str,
    service: TeamService = Depends(get_team_service),
) -> list[V1MessageEntry]:
    """GET /messages/{id} -> event-sourced messages in V1 format."""
    team_id = _parse_uuid(id)
    try:
        events = service.get_events(team_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Team not found") from None
    return _events_to_v1_messages(events)


# --- LLM context route ---


@llm_context_router.get("/{id}", response_model=list[V1LlmContextEntry])
def get_llm_context(
    id: str,
    service: TeamService = Depends(get_team_service),
) -> list[V1LlmContextEntry]:
    """GET /llm_context/{id} -> LLM context events in V1 format."""
    team_id = _parse_uuid(id)
    try:
        events = service.get_events(team_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Team not found") from None
    result: list[V1LlmContextEntry] = []
    for ev in events:
        content = _extract_message_content(ev.event)
        if content is None:
            continue
        result.append(
            V1LlmContextEntry(
                role=_classify_message_type(ev.event),
                content=content,
                timestamp=ev.timestamp.isoformat(),
            )
        )
    return result


# --- States route ---


@states_router.get("/{id}", response_model=list[V1StateEntry])
def get_states(
    id: str,
    service: TeamService = Depends(get_team_service),
) -> list[V1StateEntry]:
    """GET /states/{id} -> state change events in V1 format."""
    team_id = _parse_uuid(id)
    try:
        events = service.get_events(team_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Team not found") from None
    result: list[V1StateEntry] = []
    for ev in events:
        if isinstance(ev.event, StateChangedMessage):
            result.append(
                V1StateEntry(
                    agent=_get_sender_name(ev.event),
                    state=ev.event.state.model_dump(mode="json"),
                    timestamp=ev.timestamp.isoformat(),
                )
            )
    return result
