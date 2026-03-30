"""Tests for TeamService — service layer with real in-memory adapters."""

from __future__ import annotations

import logging
import uuid

import pytest
from akgentic.catalog.models.errors import EntryNotFoundError
from akgentic.team.models import TeamStatus

from akgentic.infra.server.services.team_service import TeamService


def test_create_team_returns_process(team_service: TeamService) -> None:
    """Creating a team with a valid catalog entry returns a Process."""
    process = team_service.create_team("test-team", user_id="anonymous")
    assert process.team_id is not None
    assert process.status == TeamStatus.RUNNING
    assert process.user_id == "anonymous"
    assert process.team_card.name == "Test Team"


def test_create_team_invalid_entry_raises(team_service: TeamService) -> None:
    """Creating a team with an invalid catalog entry raises EntryNotFoundError."""
    with pytest.raises(EntryNotFoundError):
        team_service.create_team("nonexistent", user_id="anonymous")


def test_list_teams_empty(team_service: TeamService) -> None:
    """Listing teams when none exist returns empty list."""
    result = team_service.list_teams(user_id="anonymous")
    assert result == []


def test_list_teams_filters_by_user(team_service: TeamService) -> None:
    """list_teams returns only teams belonging to the given user."""
    team_service.create_team("test-team", user_id="alice")
    team_service.create_team("test-team", user_id="bob")
    alice_teams = team_service.list_teams(user_id="alice")
    bob_teams = team_service.list_teams(user_id="bob")
    assert len(alice_teams) == 1
    assert len(bob_teams) == 1
    assert alice_teams[0].user_id == "alice"


def test_get_team_found(team_service: TeamService) -> None:
    """get_team returns the Process for an existing team."""
    process = team_service.create_team("test-team", user_id="anonymous")
    found = team_service.get_team(process.team_id)
    assert found is not None
    assert found.team_id == process.team_id


def test_get_team_not_found(team_service: TeamService) -> None:
    """get_team returns None for a nonexistent team ID."""
    result = team_service.get_team(uuid.uuid4())
    assert result is None


def test_delete_team_stops_and_deletes(team_service: TeamService) -> None:
    """delete_team stops a running team and purges it from the event store."""
    process = team_service.create_team("test-team", user_id="anonymous")
    team_service.delete_team(process.team_id)
    # After deletion, the team is fully purged from the event store
    after = team_service.get_team(process.team_id)
    assert after is None


def test_delete_stopped_team(team_service: TeamService) -> None:
    """delete_team handles an already-stopped team without calling stop_team."""
    process = team_service.create_team("test-team", user_id="anonymous")
    team_service._services.worker_handle.stop_team(process.team_id)
    team_service.delete_team(process.team_id)
    after = team_service.get_team(process.team_id)
    assert after is None


def test_delete_team_not_found_raises(team_service: TeamService) -> None:
    """delete_team raises ValueError for a nonexistent team ID."""
    with pytest.raises(ValueError, match="not found"):
        team_service.delete_team(uuid.uuid4())


# ---------------------------------------------------------------------------
# Reclassified from integration/test_adr003_tier_agnostic.py
# Source inspection; no real app needed.
# ---------------------------------------------------------------------------


class TestTeamServiceImports:
    """Verify TeamService module does not import actor internals."""

    def test_team_service_has_no_actor_internal_imports(self) -> None:
        """TeamService module does not import actor internals."""
        import inspect

        from akgentic.infra.server.services import team_service as ts_module

        source = inspect.getsource(ts_module)
        forbidden = ["TeamManager", "ActorSystem", "LocalTeamHandle", "CommunityServices"]
        for name in forbidden:
            assert name not in source, f"TeamService module must not import {name}"


class TestTeamServiceLogging:
    """TeamService emits expected log messages."""

    def test_create_team_emits_info_log(
        self,
        team_service: TeamService,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """create_team() emits INFO log with team_id and catalog_entry."""
        with caplog.at_level(logging.INFO, logger="akgentic.infra.server.services.team_service"):
            team_service.create_team("test-team", user_id="anonymous")
        assert any("Team created" in r.message for r in caplog.records)

    def test_delete_team_emits_info_log(
        self,
        team_service: TeamService,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """delete_team() emits INFO log with team_id."""
        process = team_service.create_team("test-team", user_id="anonymous")
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="akgentic.infra.server.services.team_service"):
            team_service.delete_team(process.team_id)
        assert any("Team deleted" in r.message for r in caplog.records)
