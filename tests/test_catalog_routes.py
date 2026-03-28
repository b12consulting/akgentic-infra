"""Tests for catalog browsing endpoints (GET /catalog/teams, GET /catalog/teams/{name})."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_list_team_templates(client: TestClient) -> None:
    """GET /catalog/teams returns seeded team templates."""
    resp = client.get("/catalog/teams")
    assert resp.status_code == 200
    body = resp.json()
    assert "teams" in body
    assert len(body["teams"]) >= 1
    ids = [t["id"] for t in body["teams"]]
    assert "test-team" in ids


def test_get_team_template(client: TestClient) -> None:
    """GET /catalog/teams/{name} returns specific team details."""
    resp = client.get("/catalog/teams/test-team")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "test-team"
    assert body["name"] == "Test Team"
    assert body["entry_point"] == "human-proxy"
    assert isinstance(body["members"], list)
    assert len(body["members"]) == 2


def test_get_team_template_members_have_expected_fields(client: TestClient) -> None:
    """GET /catalog/teams/{name} members contain agent_id and children fields."""
    resp = client.get("/catalog/teams/test-team")
    assert resp.status_code == 200
    body = resp.json()
    for member in body["members"]:
        assert "agent_id" in member
        assert "children" in member
        assert isinstance(member["children"], list)


def test_get_team_template_not_found(client: TestClient) -> None:
    """GET /catalog/teams/{name} returns 404 for non-existent template."""
    resp = client.get("/catalog/teams/nonexistent-template")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_list_team_templates_response_shape(client: TestClient) -> None:
    """GET /catalog/teams response has all expected fields per CatalogTeamResponse."""
    resp = client.get("/catalog/teams")
    assert resp.status_code == 200
    team = resp.json()["teams"][0]
    expected_fields = {"id", "name", "description", "entry_point", "members", "profiles"}
    assert expected_fields.issubset(set(team.keys()))
