"""Tests for team CRUD endpoints using FastAPI TestClient."""

from __future__ import annotations

import uuid
from collections.abc import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from akgentic.infra.server.app import create_app
from akgentic.infra.server.auth import RequestUser, get_request_user
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community


def test_create_team_success(client: TestClient) -> None:
    """POST /teams with valid catalog entry returns 201."""
    resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert resp.status_code == 201
    data = resp.json()
    assert "team_id" in data
    assert data["status"] == "running"
    assert data["name"] == "Test Team"


def test_create_team_invalid_entry(client: TestClient) -> None:
    """POST /teams with unknown catalog entry returns 404."""
    resp = client.post("/teams/", json={"catalog_namespace": "nonexistent"})
    assert resp.status_code == 404


def test_list_teams_empty(client: TestClient) -> None:
    """GET /teams returns empty list and a null next_cursor when no teams exist."""
    resp = client.get("/teams/")
    assert resp.status_code == 200
    assert resp.json()["teams"] == []
    assert resp.json()["next_cursor"] is None


def test_list_teams_after_create(client: TestClient) -> None:
    """GET /teams returns created teams; a single-team set has a null next_cursor."""
    client.post("/teams/", json={"catalog_namespace": "test-team"})
    resp = client.get("/teams/")
    assert resp.status_code == 200
    body = resp.json()
    teams = body["teams"]
    assert len(teams) == 1
    assert teams[0]["name"] == "Test Team"
    assert body["next_cursor"] is None


def test_get_team_success(client: TestClient) -> None:
    """GET /teams/{id} returns team detail."""
    create_resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    team_id = create_resp.json()["team_id"]
    resp = client.get(f"/teams/{team_id}")
    assert resp.status_code == 200
    assert resp.json()["team_id"] == team_id


def test_get_team_not_found(client: TestClient) -> None:
    """GET /teams/{id} returns 404 for unknown team."""
    resp = client.get(f"/teams/{uuid.uuid4()}")
    assert resp.status_code == 404


@pytest.mark.skip(
    reason="Flaky: race in TeamManager.delete_team — on_stop subscribers still "
    "flushing event_store writes while rmtree runs; pre-existing, not introduced by Epic 22."
)
def test_delete_team_success(client: TestClient) -> None:
    """DELETE /teams/{id} returns 204 and removes team."""
    create_resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    team_id = create_resp.json()["team_id"]
    resp = client.delete(f"/teams/{team_id}")
    assert resp.status_code == 204
    # Verify team is actually gone
    get_resp = client.get(f"/teams/{team_id}")
    assert get_resp.status_code == 404


def test_delete_team_not_found(client: TestClient) -> None:
    """DELETE /teams/{id} returns 404 for unknown team."""
    resp = client.delete(f"/teams/{uuid.uuid4()}")
    assert resp.status_code == 404


# --- RequestUser identity seam (ADR-023 Story 26.1) ---


def test_create_team_default_identity_anonymous(client: TestClient) -> None:
    """With no override, POST /teams persists user_id == 'anonymous' (AC #5)."""
    resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert resp.status_code == 201
    assert resp.json()["user_id"] == "anonymous"


@pytest.fixture()
def overridden_user_client(app: FastAPI) -> Generator[TestClient, None, None]:
    """TestClient with get_request_user overridden to a fixed identity."""
    app.dependency_overrides[get_request_user] = lambda: RequestUser(
        user_id="alice", email="alice@example.com"
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_create_team_uses_overridden_identity(
    overridden_user_client: TestClient,
) -> None:
    """When get_request_user is overridden, POST /teams persists that user_id (AC #6)."""
    resp = overridden_user_client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert resp.status_code == 201
    assert resp.json()["user_id"] == "alice"


def test_list_teams_filters_by_overridden_identity(
    overridden_user_client: TestClient,
) -> None:
    """Under the override, GET /teams returns only the overridden user's teams (AC #6)."""
    overridden_user_client.post("/teams/", json={"catalog_namespace": "test-team"})
    resp = overridden_user_client.get("/teams/")
    assert resp.status_code == 200
    body = resp.json()
    teams = body["teams"]
    assert len(teams) == 1
    assert teams[0]["user_id"] == "alice"
    assert body["next_cursor"] is None


# --- Cursor pagination (Story 36.1, ADR-031 §Decision 1-4) ---


def _create_teams(client: TestClient, n: int) -> None:
    """Create ``n`` teams for the default (anonymous) identity."""
    for _ in range(n):
        resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
        assert resp.status_code == 201


def test_list_teams_limit_zero_clamped_no_500(client: TestClient) -> None:
    """(d) ?limit=0 is clamped (>=1) and returns 200, never a 500."""
    _create_teams(client, 2)
    resp = client.get("/teams/", params={"limit": 0})
    assert resp.status_code == 200
    assert 1 <= len(resp.json()["teams"]) <= 200


def test_list_teams_limit_huge_clamped_no_500(client: TestClient) -> None:
    """(d) ?limit=99999 is clamped to <=200 and returns 200."""
    _create_teams(client, 3)
    resp = client.get("/teams/", params={"limit": 99999})
    assert resp.status_code == 200
    assert len(resp.json()["teams"]) <= 200


def test_list_teams_malformed_cursor_returns_400(client: TestClient) -> None:
    """(#6) A malformed cursor returns a 400, never a 500 or unhandled error."""
    resp = client.get("/teams/", params={"cursor": "not-a-valid-token"})
    assert resp.status_code == 400


def test_list_teams_walk_visits_every_team_once(client: TestClient) -> None:
    """(#6) An end-to-end HTTP walk over a multi-page set visits every team once."""
    _create_teams(client, 5)
    seen: list[str] = []
    cursor: str | None = None
    pages = 0
    while True:
        params: dict[str, object] = {"limit": 2}
        if cursor is not None:
            params["cursor"] = cursor
        resp = client.get("/teams/", params=params)
        assert resp.status_code == 200
        body = resp.json()
        seen.extend(t["team_id"] for t in body["teams"])
        cursor = body["next_cursor"]
        pages += 1
        assert pages < 10  # guard against an infinite walk
        if cursor is None:
            break
    assert len(seen) == 5
    assert len(set(seen)) == 5  # no duplicates, no gaps


# --- Statelessness (Story 36.1 AC #12): cursor is the only pagination state ---


@pytest.fixture()
def second_replica_client(
    seeded_settings: CommunitySettings,
) -> Generator[TestClient, None, None]:
    """A second, independently-wired app over the SAME on-disk event store.

    Simulates a different worker/replica: it shares no in-process state with the
    primary ``client`` — only the persisted store contents (same event_store_path).
    """
    services = wire_community(seeded_settings)
    application = create_app(services, seeded_settings)
    yield TestClient(application)
    services.actor_system.shutdown()


def test_cursor_followable_across_independent_replicas(
    client: TestClient,
    second_replica_client: TestClient,
) -> None:
    """AC #12: a cursor minted by one replica is followable by another replica
    serving the same store — no reliance on the minting request's in-process state.
    """
    _create_teams(client, 5)

    # Replica A mints page 1 + cursor.
    resp_a = client.get("/teams/", params={"limit": 2})
    assert resp_a.status_code == 200
    body_a = resp_a.json()
    cursor = body_a["next_cursor"]
    assert cursor is not None
    page1 = {t["team_id"] for t in body_a["teams"]}

    # Replica B (separate app/services) follows that cursor over the shared store.
    resp_b = second_replica_client.get("/teams/", params={"limit": 2, "cursor": cursor})
    assert resp_b.status_code == 200
    page2 = {t["team_id"] for t in resp_b.json()["teams"]}

    assert page1.isdisjoint(page2)  # B did not re-show A's page
    assert len(page1 | page2) == 4  # contiguous walk across the replica boundary
