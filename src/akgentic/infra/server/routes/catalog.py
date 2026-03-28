"""Catalog browsing endpoints — read-only access to team templates."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, HTTPException, Request

from akgentic.catalog.models.team import TeamEntry, TeamMemberSpec
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.models import (
    CatalogTeamListResponse,
    CatalogTeamMember,
    CatalogTeamResponse,
)

router = APIRouter(prefix="/catalog", tags=["catalog"])


def _member_to_response(spec: TeamMemberSpec) -> CatalogTeamMember:
    """Recursively convert a TeamMemberSpec tree to CatalogTeamMember."""
    return CatalogTeamMember(
        agent_id=spec.agent_id,
        children=[_member_to_response(child) for child in spec.members],
    )


def _entry_to_response(entry: TeamEntry) -> CatalogTeamResponse:
    """Convert a TeamEntry to CatalogTeamResponse."""
    return CatalogTeamResponse(
        id=entry.id,
        name=entry.name,
        description=entry.description,
        entry_point=entry.entry_point,
        members=[_member_to_response(m) for m in entry.members],
        profiles=entry.profiles,
    )


@router.get("/teams", response_model=CatalogTeamListResponse)
def list_team_templates(request: Request) -> CatalogTeamListResponse:
    """List all available team templates from the catalog."""
    services = cast(CommunityServices, request.app.state.services)
    entries = services.team_catalog.list()
    return CatalogTeamListResponse(teams=[_entry_to_response(e) for e in entries])


@router.get("/teams/{name}", response_model=CatalogTeamResponse)
def get_team_template(name: str, request: Request) -> CatalogTeamResponse:
    """Get a specific team template by name."""
    services = cast(CommunityServices, request.app.state.services)
    entry = services.team_catalog.get(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Team template '{name}' not found")
    return _entry_to_response(entry)
