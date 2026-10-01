"""Tests for GET /teams/{id}/agent-states using FastAPI TestClient.

Covers the thin DB read of the per-agent snapshot store: snapshots are
returned exactly as persisted (no liveness filtering, no name->UUID
resolution), for running and stopped teams alike (Epic 35 / story 35-1).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from akgentic.agent.config import AgentState
from akgentic.team.models import AgentStateSnapshot
from fastapi.testclient import TestClient

from akgentic.infra.server.deps import CommunityServices


def _seed_snapshot(
    services: CommunityServices,
    team_id: uuid.UUID,
    *,
    agent_id: str,
    name: str | None,
    backstory: str,
) -> AgentStateSnapshot:
    """Persist one agent-state snapshot directly into the team's snapshot store."""
    snapshot = AgentStateSnapshot(
        team_id=team_id,
        agent_id=agent_id,
        name=name,
        state=AgentState(backstory=backstory),
        updated_at=datetime.now(UTC),
    )
    services.event_store.save_agent_state(snapshot)
    return snapshot


def test_get_agent_states_success(
    client: TestClient, community_services: CommunityServices
) -> None:
    """Returns one entry per persisted snapshot with fields echoed as stored."""
    create_resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    team_id = uuid.UUID(create_resp.json()["team_id"])
    agent_uuid = str(uuid.uuid4())
    _seed_snapshot(
        community_services,
        team_id,
        agent_id=agent_uuid,
        name="@Manager",
        backstory="You coordinate the team.",
    )

    resp = client.get(f"/teams/{team_id}/agent-states")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["states"]) == 1
    entry = data["states"][0]
    assert entry["agent_id"] == agent_uuid
    assert entry["name"] == "@Manager"
    assert entry["state"]["backstory"] == "You coordinate the team."
    assert entry["updated_at"] is not None


def test_get_agent_states_legacy_name_keyed_snapshot(
    client: TestClient, community_services: CommunityServices
) -> None:
    """A pre-Epic-23 snapshot (agent_id holds a name, name is None) is passed through as-is."""
    create_resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    team_id = uuid.UUID(create_resp.json()["team_id"])
    _seed_snapshot(
        community_services,
        team_id,
        agent_id="@Manager",
        name=None,
        backstory="legacy backstory",
    )

    resp = client.get(f"/teams/{team_id}/agent-states")

    assert resp.status_code == 200
    entry = resp.json()["states"][0]
    assert entry["agent_id"] == "@Manager"
    assert entry["name"] is None


def test_get_agent_states_stopped_team_returns_snapshots(
    client: TestClient, community_services: CommunityServices
) -> None:
    """The load-bearing case: snapshots are returned regardless of agent liveness.

    A stopped team has no live (Start - Stop) agent set, yet the endpoint must
    still return its persisted snapshots — there is no live-set filtering.
    """
    create_resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    team_id = uuid.UUID(create_resp.json()["team_id"])
    agent_uuid = str(uuid.uuid4())
    _seed_snapshot(
        community_services,
        team_id,
        agent_id=agent_uuid,
        name="@Manager",
        backstory="still here after stop",
    )
    client.post(f"/teams/{team_id}/stop")

    resp = client.get(f"/teams/{team_id}/agent-states")

    assert resp.status_code == 200
    states = resp.json()["states"]
    assert [s["agent_id"] for s in states] == [agent_uuid]
    assert states[0]["state"]["backstory"] == "still here after stop"


def test_get_agent_states_empty_is_200(client: TestClient) -> None:
    """An existing team with no persisted snapshots returns 200 with an empty list."""
    create_resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    team_id = create_resp.json()["team_id"]

    resp = client.get(f"/teams/{team_id}/agent-states")

    assert resp.status_code == 200
    assert resp.json()["states"] == []


def test_get_agent_states_unknown_team_is_404(client: TestClient) -> None:
    """An unknown team_id returns 404 with detail 'Team not found'."""
    resp = client.get(f"/teams/{uuid.uuid4()}/agent-states")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Team not found"


def test_get_agent_states_without_agent_id_returns_every_agent(
    client: TestClient, community_services: CommunityServices
) -> None:
    """With two agents seeded and no agent_id, both snapshots come back."""
    create_resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    team_id = uuid.UUID(create_resp.json()["team_id"])
    agent_a, agent_b = str(uuid.uuid4()), str(uuid.uuid4())
    _seed_snapshot(community_services, team_id, agent_id=agent_a, name="@Manager", backstory="a")
    _seed_snapshot(community_services, team_id, agent_id=agent_b, name="@Worker", backstory="b")

    resp = client.get(f"/teams/{team_id}/agent-states")

    assert resp.status_code == 200
    assert sorted(s["agent_id"] for s in resp.json()["states"]) == sorted([agent_a, agent_b])


def test_get_agent_states_agent_id_returns_only_that_agent(
    client: TestClient, community_services: CommunityServices
) -> None:
    """With two agents seeded, ?agent_id= narrows the list to that agent's entry."""
    create_resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    team_id = uuid.UUID(create_resp.json()["team_id"])
    agent_a, agent_b = str(uuid.uuid4()), str(uuid.uuid4())
    seeded = _seed_snapshot(
        community_services, team_id, agent_id=agent_a, name="@Manager", backstory="A's backstory"
    )
    _seed_snapshot(
        community_services, team_id, agent_id=agent_b, name="@Worker", backstory="B's backstory"
    )

    resp = client.get(f"/teams/{team_id}/agent-states", params={"agent_id": agent_a})

    assert resp.status_code == 200
    states = resp.json()["states"]
    assert len(states) == 1
    entry = states[0]
    assert entry["agent_id"] == agent_a
    assert entry["name"] == "@Manager"
    assert entry["state"]["backstory"] == "A's backstory"
    assert datetime.fromisoformat(entry["updated_at"]) == seeded.updated_at


def test_get_agent_states_agent_id_without_snapshot_is_200_empty(
    client: TestClient, community_services: CommunityServices
) -> None:
    """An agent_id with no snapshot answers 200 and an empty list, not 404."""
    create_resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    team_id = uuid.UUID(create_resp.json()["team_id"])
    _seed_snapshot(
        community_services, team_id, agent_id=str(uuid.uuid4()), name="@Manager", backstory="x"
    )

    resp = client.get(f"/teams/{team_id}/agent-states", params={"agent_id": str(uuid.uuid4())})

    assert resp.status_code == 200
    assert resp.json()["states"] == []


def test_get_agent_states_agent_id_unknown_team_is_404(client: TestClient) -> None:
    """An unknown team is 404 with an agent_id too, not 200 and an empty list.

    The 404 comes from ``require_team_access`` before the service is called, so
    this proves the contract, not the service's guard order — that is pinned by
    ``test_get_agent_states_with_agent_id_unknown_team_raises``.
    """
    resp = client.get(f"/teams/{uuid.uuid4()}/agent-states", params={"agent_id": str(uuid.uuid4())})

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Team not found"


def test_get_agent_states_agent_id_legacy_name_is_422(
    client: TestClient, community_services: CommunityServices
) -> None:
    """A non-UUID agent_id ("@Manager") is rejected with 422, even with a legacy snapshot.

    A legacy name-keyed snapshot is reachable only through the unfiltered list.
    """
    create_resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    team_id = uuid.UUID(create_resp.json()["team_id"])
    _seed_snapshot(community_services, team_id, agent_id="@Manager", name=None, backstory="legacy")

    resp = client.get(f"/teams/{team_id}/agent-states", params={"agent_id": "@Manager"})

    assert resp.status_code == 422
    assert [e["loc"] for e in resp.json()["detail"]] == [["query", "agent_id"]]


def test_get_agent_states_agent_id_is_documented_in_openapi(client: TestClient) -> None:
    """The OpenAPI schema lists agent_id as an optional, described, string query parameter."""
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/teams/{team_id}/agent-states"]["get"]

    params = [p for p in operation["parameters"] if p["name"] == "agent_id"]
    assert len(params) == 1
    param = params[0]
    assert param["in"] == "query"
    assert param.get("required", False) is False
    assert param["description"].strip()
    types = {branch.get("type") for branch in param["schema"].get("anyOf", [param["schema"]])}
    assert "string" in types


@pytest.mark.parametrize("prefix", ["../", "x/"], ids=["dotdot", "slash"])
def test_get_agent_states_path_shaped_agent_id_is_422(
    client: TestClient, community_services: CommunityServices, prefix: str
) -> None:
    """A path-shaped agent_id is a 422: it never reaches the store or another agent's snapshot."""
    create_resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    team_id = uuid.UUID(create_resp.json()["team_id"])
    agent_b = str(uuid.uuid4())
    _seed_snapshot(community_services, team_id, agent_id=agent_b, name="@Worker", backstory="b")

    resp = client.get(f"/teams/{team_id}/agent-states", params={"agent_id": f"{prefix}{agent_b}"})

    assert resp.status_code == 422
    assert [e["loc"] for e in resp.json()["detail"]] == [["query", "agent_id"]]
