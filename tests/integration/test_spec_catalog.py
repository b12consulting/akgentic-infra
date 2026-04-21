"""Integration tests — v2 unified /catalog router CRUD.

Validates the unified catalog router exposed by infra after Story 18.3.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.integration, pytest.mark.llm]


_CRUD_TEAM_ENTRY = {
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


class TestCatalogTeamCrud:
    """Exercise the unified /catalog/team router CRUD operations."""

    def test_list_teams(self, integration_client: TestClient) -> None:
        """AC #6: GET /catalog/team returns seeded entries."""
        resp = integration_client.get(
            "/catalog/team", params={"namespace": "test-team"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        ids = [t["id"] for t in data]
        assert "team" in ids

    def test_resolve_team(self, integration_client: TestClient) -> None:
        """AC #6: GET /catalog/team/test-team/resolve returns TeamCard."""
        resp = integration_client.get("/catalog/team/test-team/resolve")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Integration Test Team"

    def test_create_update_delete_team(
        self,
        integration_client: TestClient,
    ) -> None:
        """AC #6: POST/PUT/DELETE /catalog/team via the unified router."""
        # Create
        resp = integration_client.post("/catalog/team", json=_CRUD_TEAM_ENTRY)
        assert resp.status_code == 201
        assert resp.json()["id"] == "crud-team"

        # Read back
        resp = integration_client.get(
            "/catalog/team/crud-team", params={"namespace": "crud-ns"}
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "crud-team"

        # Update
        updated = dict(_CRUD_TEAM_ENTRY)
        updated_payload = dict(_CRUD_TEAM_ENTRY["payload"])
        updated_payload["name"] = "Updated CRUD Team"
        updated["payload"] = updated_payload
        resp = integration_client.put(
            "/catalog/team/crud-team",
            params={"namespace": "crud-ns"},
            json=updated,
        )
        assert resp.status_code == 200
        assert resp.json()["payload"]["name"] == "Updated CRUD Team"

        # Delete
        resp = integration_client.delete(
            "/catalog/team/crud-team", params={"namespace": "crud-ns"}
        )
        assert resp.status_code == 204

    def test_get_nonexistent_team_404(
        self,
        integration_client: TestClient,
    ) -> None:
        """AC #6: GET /catalog/team/nonexistent returns 404."""
        resp = integration_client.get(
            "/catalog/team/nonexistent-xyz", params={"namespace": "test-team"}
        )
        assert resp.status_code == 404


class TestCatalogAgentList:
    """Exercise the unified /catalog/agent router listing."""

    def test_list_agents(self, integration_client: TestClient) -> None:
        """AC #6: GET /catalog/agent returns list (may be empty under v2 layout)."""
        resp = integration_client.get(
            "/catalog/agent", params={"namespace": "test-team"}
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_nonexistent_agent_404(
        self,
        integration_client: TestClient,
    ) -> None:
        """AC #6: GET /catalog/agent/nonexistent returns 404."""
        resp = integration_client.get(
            "/catalog/agent/nonexistent-xyz", params={"namespace": "test-team"}
        )
        assert resp.status_code == 404


class TestCatalogToolsTemplates:
    """Exercise the unified /catalog/tool and /catalog/prompt list routes."""

    def test_list_tools(self, integration_client: TestClient) -> None:
        """AC #6: GET /catalog/tool returns 200 (may be empty)."""
        resp = integration_client.get(
            "/catalog/tool", params={"namespace": "test-team"}
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_prompts(self, integration_client: TestClient) -> None:
        """AC #6: GET /catalog/prompt returns 200 (may be empty)."""
        resp = integration_client.get(
            "/catalog/prompt", params={"namespace": "test-team"}
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
