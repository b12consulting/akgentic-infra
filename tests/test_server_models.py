"""Tests for REST API request/response models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from akgentic.infra.server.models import (
    CreateTeamRequest,
    TeamListResponse,
    TeamResponse,
)


def test_create_team_request_minimal() -> None:
    """CreateTeamRequest requires only catalog_entry_id."""
    req = CreateTeamRequest(catalog_entry_id="test-team")
    assert req.catalog_entry_id == "test-team"
    assert req.params == {}


def test_create_team_request_with_params() -> None:
    """CreateTeamRequest accepts optional params."""
    req = CreateTeamRequest(
        catalog_entry_id="test-team",
        params={"key": "value"},
    )
    assert req.params == {"key": "value"}


def test_team_response_serialization() -> None:
    """TeamResponse serializes all fields correctly."""
    tid = uuid.uuid4()
    now = datetime.now(tz=UTC)
    resp = TeamResponse(
        team_id=tid,
        name="Test",
        status="running",
        user_id="anonymous",
        created_at=now,
        updated_at=now,
    )
    data = resp.model_dump(mode="json")
    assert data["team_id"] == str(tid)
    assert data["status"] == "running"


def test_team_list_response_empty() -> None:
    """TeamListResponse can hold an empty list."""
    resp = TeamListResponse(teams=[])
    assert resp.teams == []


def test_team_list_response_with_items() -> None:
    """TeamListResponse serializes a list of TeamResponses."""
    tid = uuid.uuid4()
    now = datetime.now(tz=UTC)
    item = TeamResponse(
        team_id=tid,
        name="A",
        status="running",
        user_id="u",
        created_at=now,
        updated_at=now,
    )
    resp = TeamListResponse(teams=[item])
    assert len(resp.teams) == 1
    assert resp.teams[0].team_id == tid
