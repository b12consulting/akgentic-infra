"""Pydantic request/response models for the REST API."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class CreateTeamRequest(BaseModel):
    """Request body for POST /teams."""

    catalog_entry_id: str = Field(description="Catalog entry ID to resolve into a TeamCard")
    params: dict[str, str] = Field(
        default_factory=dict,
        description="Pass-through configuration parameters",
    )


class TeamResponse(BaseModel):
    """Serialized team metadata returned by team endpoints."""

    team_id: uuid.UUID = Field(description="Unique team identifier")
    name: str = Field(description="Human-readable team name")
    status: str = Field(description="Current team lifecycle status")
    user_id: str = Field(description="Owner user identifier")
    created_at: datetime = Field(description="Team creation timestamp")
    updated_at: datetime = Field(description="Last status change timestamp")


class TeamListResponse(BaseModel):
    """Response body for GET /teams."""

    teams: list[TeamResponse] = Field(description="List of team metadata entries")
