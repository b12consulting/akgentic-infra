"""Tests for the mounted v2 unified /catalog router."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_catalog_team_list_mounted(client: TestClient) -> None:
    """GET /catalog/team returns list of team entries in the seeded namespace."""
    resp = client.get("/catalog/team", params={"namespace": "test-team"})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    ids = [t["id"] for t in body]
    assert "team" in ids


def test_catalog_team_resolve(client: TestClient) -> None:
    """GET /catalog/team/{namespace}/resolve returns the dumped TeamCard."""
    resp = client.get("/catalog/team/test-team/resolve")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Test Team"
    assert isinstance(body.get("members"), list)


def test_catalog_team_get_by_id(client: TestClient) -> None:
    """GET /catalog/team/{id} returns the specific team entry."""
    resp = client.get("/catalog/team/team", params={"namespace": "test-team"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "team"
    assert body["namespace"] == "test-team"
    assert body["kind"] == "team"


def test_catalog_team_crud_create_and_delete(client: TestClient) -> None:
    """POST then DELETE /catalog/team/{id} works through the unified router."""
    entry = {
        "id": "crud-team",
        "kind": "team",
        "namespace": "crud-ns",
        "model_type": "akgentic.team.models.TeamCard",
        "description": "",
        "payload": {
            "name": "CRUD Team",
            "description": "crud",
            "entry_point": {
                "card": {
                    "role": "Human",
                    "description": "Human",
                    "skills": [],
                    "agent_class": "akgentic.core.agent.Akgent",
                    "config": {"name": "@Human", "role": "Human"},
                    "routes_to": [],
                },
                "headcount": 1,
                "members": [],
            },
            "members": [],
            "message_types": [{"__type__": "akgentic.core.messages.UserMessage"}],
            "agent_profiles": [],
        },
    }
    # Create
    resp = client.post("/catalog/team", json=entry)
    assert resp.status_code == 201
    assert resp.json()["id"] == "crud-team"

    # Verify it exists
    resp = client.get("/catalog/team/crud-team", params={"namespace": "crud-ns"})
    assert resp.status_code == 200
    assert resp.json()["namespace"] == "crud-ns"

    # Delete
    resp = client.delete("/catalog/team/crud-team", params={"namespace": "crud-ns"})
    assert resp.status_code == 204

    # Verify entry is gone
    resp = client.get("/catalog/team/crud-team", params={"namespace": "crud-ns"})
    assert resp.status_code == 404


def test_catalog_exception_handler_404(client: TestClient) -> None:
    """GET /catalog/team/{id} returns 404 for missing entry via exception handler."""
    resp = client.get("/catalog/team/nonexistent-entry", params={"namespace": "test-team"})
    assert resp.status_code == 404
    body = resp.json()
    assert "detail" in body
    assert "not found" in body["detail"].lower()


def test_catalog_exception_handler_409(client: TestClient) -> None:
    """POST duplicate entry returns 409 via CatalogValidationError exception handler."""
    entry = {
        "id": "team",
        "kind": "team",
        "namespace": "test-team",
        "model_type": "akgentic.team.models.TeamCard",
        "description": "",
        "payload": {
            "name": "Duplicate Team",
            "description": "dup",
            "entry_point": {
                "card": {
                    "role": "Human",
                    "description": "Human",
                    "skills": [],
                    "agent_class": "akgentic.core.agent.Akgent",
                    "config": {"name": "@Human", "role": "Human"},
                    "routes_to": [],
                },
                "headcount": 1,
                "members": [],
            },
            "members": [],
            "message_types": [{"__type__": "akgentic.core.messages.UserMessage"}],
            "agent_profiles": [],
        },
    }
    resp = client.post("/catalog/team", json=entry)
    assert resp.status_code == 409
    body = resp.json()
    assert "detail" in body


def test_catalog_exception_handler_registered(app: FastAPI) -> None:
    """Exception handlers for EntryNotFoundError and CatalogValidationError are registered."""
    from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError

    assert EntryNotFoundError in app.exception_handlers
    assert CatalogValidationError in app.exception_handlers


def test_v1_catalog_routes_removed(client: TestClient) -> None:
    """Legacy v1 per-kind /catalog/api/* endpoints no longer exist.

    Under the unified v2 router, /catalog/api/teams either does not match
    any path (404/405) or matches /catalog/{kind}/{id} with kind="api",
    which is rejected by EntryKind validation (422). All three are
    acceptable "v1 is dead" signals.
    """
    resp = client.get("/catalog/api/teams")
    assert resp.status_code in (404, 405, 422)


def test_core_routes_coexist_with_catalog(client: TestClient) -> None:
    """Core app routes still work after mounting the unified catalog router."""
    resp = client.get("/teams/")
    assert resp.status_code == 200
