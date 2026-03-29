"""Integration tests — V1 frontend adapter spec compliance for story 6.8 endpoints.

Validates all V1 endpoints added in story 6.8 and the llm_context WS envelope type.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

CATALOG_ENTRY_ID = "test-team"


def _create_v1_team(client: TestClient) -> str:
    """POST /process/{type} (V1) and return the team_id."""
    resp = client.post(f"/process/{CATALOG_ENTRY_ID}")
    assert resp.status_code == 200
    data = resp.json()
    return data["id"]


# ---------------------------------------------------------------------------
# AC #5 — V1 story-6.8 endpoint translations
# ---------------------------------------------------------------------------


class TestV1Story68Endpoints:
    """Integration tests for V1 endpoints added in story 6.8."""

    def test_patch_description(self, v1_adapter_client: TestClient) -> None:
        """AC #5: PATCH /process/{id}/description returns ok."""
        team_id = _create_v1_team(v1_adapter_client)

        resp = v1_adapter_client.patch(
            f"/process/{team_id}/description",
            json={"description": "Updated description"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        v1_adapter_client.delete(f"/process/{team_id}/archive")
        time.sleep(0.5)

    def test_relaunch_message_not_found(self, v1_adapter_client: TestClient) -> None:
        """AC #5: POST /relaunch/{id}/message/{msgId} returns 404 for unknown msg."""
        team_id = _create_v1_team(v1_adapter_client)

        resp = v1_adapter_client.post(
            f"/relaunch/{team_id}/message/00000000-0000-0000-0000-000000000000",
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

        v1_adapter_client.delete(f"/process/{team_id}/archive")
        time.sleep(0.5)

    def test_patch_state_not_running(self, v1_adapter_client: TestClient) -> None:
        """AC #5: PATCH /state/{id}/of/{agent} returns 404 for stopped team."""
        team_id = _create_v1_team(v1_adapter_client)

        # Stop the team first
        v1_adapter_client.delete(f"/process/{team_id}/archive")
        time.sleep(0.5)

        resp = v1_adapter_client.patch(
            f"/state/{team_id}/of/@Manager",
            json={"content": "state update"},
        )
        assert resp.status_code == 404

    def test_get_config_team(self, v1_adapter_client: TestClient) -> None:
        """AC #5: GET /config/team returns catalog entries."""
        resp = v1_adapter_client.get("/config/team")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        entry = data[0]
        assert "id" in entry
        assert "type" in entry
        assert entry["type"] == "team"
        assert "data" in entry

    def test_get_config_agent(self, v1_adapter_client: TestClient) -> None:
        """AC #5: GET /config/agent returns catalog entries."""
        resp = v1_adapter_client.get("/config/agent")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_get_config_unknown_type(self, v1_adapter_client: TestClient) -> None:
        """AC #5: GET /config/unknown returns 400."""
        resp = v1_adapter_client.get("/config/unknown")
        assert resp.status_code == 400

    def test_get_team_configs(self, v1_adapter_client: TestClient) -> None:
        """AC #5: GET /team-configs returns team catalog entries as dict."""
        resp = v1_adapter_client.get("/team-configs/")
        assert resp.status_code == 200
        data = resp.json()
        # ADR-004 (8.2) changed response from list to dict keyed by team name
        assert isinstance(data, dict)
        assert len(data) >= 1
        for _name, entry in data.items():
            assert "module" in entry
            assert "setup" in entry

    def test_get_feedback_empty(self, v1_adapter_client: TestClient) -> None:
        """AC #5: GET /get-feedback returns empty list (stub)."""
        resp = v1_adapter_client.get("/get-feedback")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_set_feedback_ok(self, v1_adapter_client: TestClient) -> None:
        """AC #5: POST /set-feedback returns ok (stub)."""
        resp = v1_adapter_client.post(
            "/set-feedback",
            json={"id": "fb-1", "content": "Great!", "rating": 5},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_put_config_create_and_update(self, v1_adapter_client: TestClient) -> None:
        """AC #5: PUT /config/{config_type} creates a new entry and updates an existing one."""
        # Get an existing entry to know the data shape
        resp = v1_adapter_client.get("/config/team")
        assert resp.status_code == 200
        existing = resp.json()[0]

        # Create a new entry via PUT /config/{config_type} (ADR-004 path shape)
        put_body = {
            "id": "v1-put-test",
            "name": "V1 PUT Test",
            "config": {**existing["data"], "id": "v1-put-test", "name": "V1 PUT Test"},
            "dry_run": False,
        }
        resp = v1_adapter_client.put("/config/team", json=put_body)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Update the entry via PUT
        put_body["config"]["name"] = "V1 PUT Updated"
        resp = v1_adapter_client.put("/config/team", json=put_body)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Clean up via DELETE /config/{config_type}/{config_id} (ADR-004 path shape)
        resp = v1_adapter_client.delete("/config/team/v1-put-test")
        assert resp.status_code == 200

    def test_delete_config_not_found(self, v1_adapter_client: TestClient) -> None:
        """AC #5: DELETE /config/{config_type}/{config_id} for nonexistent returns 404."""
        # ADR-004 path shape: DELETE /config/{config_type}/{config_id}
        resp = v1_adapter_client.delete("/config/team/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# AC #5 — llm_context WS envelope type
# ---------------------------------------------------------------------------


class TestLlmContextEnvelope:
    """Verify the V1 WS handler emits llm_context envelope type."""

    def test_classify_envelope_type_for_context_changed(self) -> None:
        """AC #5: _classify_envelope_type returns 'llm_context' for ContextChangedMessage."""
        from akgentic.infra.server.routes.frontend_adapter.angular_v1.ws import (
            _classify_envelope_type,
        )

        # _classify_envelope_type checks type(event).__name__ == "ContextChangedMessage"
        class ContextChangedMessage:  # noqa: N801
            """Fake message whose class name matches the real ContextChangedMessage."""

            def __init__(self) -> None:
                self.id = "test-id"
                self.sender = None

        result = _classify_envelope_type(ContextChangedMessage())  # type: ignore[arg-type]
        assert result == "llm_context"

    def test_classify_envelope_returns_message_for_user_message(self) -> None:
        """AC #5: _classify_envelope_type returns 'message' for UserMessage."""
        from akgentic.core.messages.message import UserMessage

        from akgentic.infra.server.routes.frontend_adapter.angular_v1.ws import (
            _classify_envelope_type,
        )

        msg = UserMessage(content="hello")
        result = _classify_envelope_type(msg)
        assert result == "message"
