"""Integration tests -- ADR-004 V1 Frontend Adapter Compatibility.

Validates all V1 adapter changes from Stories 8.1 and 8.2:
- /processes plural route alias
- V1ActorAddress orchestrator shape in V1ProcessContext
- params with workspace/knowledge_graph keys
- PUT /config/{config_type} new URL path shape
- DELETE /config/{config_type}/{config_id} URL params
- Human input with message dict body
- Auth stubs (/auth/me, /auth/ws-ticket, /auth/logout)
- GET /team-configs/ dict response
- GET /llm_context/{id} grouped response
- GET /states/{id} grouped response
- WebSocket error envelope (verified via existing unit tests)
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

CATALOG_ENTRY_ID = "test-team"
POLL_INTERVAL_S = 1.0
POLL_TIMEOUT_S = 60.0


def _create_v1_team(client: TestClient) -> str:
    """POST /process/{type} (V1) and return the team_id."""
    resp = client.post(f"/process/{CATALOG_ENTRY_ID}")
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert "status" in data
    assert "params" in data
    team_id: str = data["id"]
    return team_id


def _wait_for_v1_messages(
    client: TestClient,
    team_id: str,
    timeout: float = POLL_TIMEOUT_S,
) -> list[dict[str, object]]:
    """Poll GET /messages/{id} until an LLM response from @Manager appears."""
    deadline = time.monotonic() + timeout
    messages: list[dict[str, object]] = []
    while time.monotonic() < deadline:
        resp = client.get(f"/messages/{team_id}")
        assert resp.status_code == 200
        messages = resp.json()
        if _has_v1_llm_content(messages):
            return messages
        time.sleep(POLL_INTERVAL_S)
    pytest.fail(
        f"Timed out after {timeout}s waiting for V1 LLM response "
        f"(got {len(messages)} messages, none with LLM content)"
    )


def _has_v1_llm_content(messages: list[dict[str, object]]) -> bool:
    """Check if any V1 message entry contains LLM content from @Manager."""
    for msg in messages:
        sender = msg.get("sender")
        content = msg.get("content")
        if (
            isinstance(sender, str)
            and "@Manager" in sender
            and isinstance(content, str)
            and len(content) > 0
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# AC #1 -- Process endpoint integration (FR1, FR2, FR3, FR13)
# ---------------------------------------------------------------------------


class TestProcessEndpoint:
    """Test GET /processes returns flat list with V1ActorAddress orchestrator."""

    def test_get_processes_returns_flat_list(
        self, v1_adapter_client: TestClient,
    ) -> None:
        """AC1: GET /processes returns 200 with flat list response."""
        team_id = _create_v1_team(v1_adapter_client)

        resp = v1_adapter_client.get("/processes/")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

        v1_adapter_client.delete(f"/process/{team_id}/archive")
        time.sleep(0.5)

    def test_orchestrator_v1_actor_address_fields(
        self, v1_adapter_client: TestClient,
    ) -> None:
        """AC1: orchestrator is V1ActorAddress with name, role, and string defaults."""
        team_id = _create_v1_team(v1_adapter_client)

        resp = v1_adapter_client.get("/processes/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

        process = data[0]
        orch = process["orchestrator"]
        assert isinstance(orch, dict)
        assert "name" in orch
        assert "role" in orch
        assert "__actor_address__" in orch
        assert "address" in orch
        assert "agent_id" in orch
        assert "squad_id" in orch

        # All string-type fields
        for field in ("name", "role", "__actor_address__", "address", "agent_id", "squad_id"):
            assert isinstance(orch[field], str)

        v1_adapter_client.delete(f"/process/{team_id}/archive")
        time.sleep(0.5)

    def test_params_workspace_and_knowledge_graph(
        self, v1_adapter_client: TestClient,
    ) -> None:
        """AC1: params contains workspace and knowledge_graph keys."""
        team_id = _create_v1_team(v1_adapter_client)

        resp = v1_adapter_client.get("/processes/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

        process = data[0]
        params = process["params"]
        assert isinstance(params, dict)
        assert "workspace" in params
        assert "knowledge_graph" in params

        v1_adapter_client.delete(f"/process/{team_id}/archive")
        time.sleep(0.5)

    def test_processes_alias_matches_process_list(
        self, v1_adapter_client: TestClient,
    ) -> None:
        """AC1: GET /process/ returns same data as GET /processes."""
        team_id = _create_v1_team(v1_adapter_client)

        resp_processes = v1_adapter_client.get("/processes/")
        resp_process = v1_adapter_client.get("/process/")
        assert resp_processes.status_code == 200
        assert resp_process.status_code == 200

        data_processes = resp_processes.json()
        data_process = resp_process.json()
        assert len(data_processes) == len(data_process)

        v1_adapter_client.delete(f"/process/{team_id}/archive")
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# AC #2 -- Config endpoint integration (FR4, FR5)
# ---------------------------------------------------------------------------


class TestConfigEndpoint:
    """Test PUT /config/{config_type} and DELETE /config/{config_type}/{config_id}."""

    def test_put_config_new_url_shape(
        self, v1_adapter_client: TestClient,
    ) -> None:
        """AC2: PUT /config/team with V1ConfigPutBody succeeds."""
        # Get existing entry to build a valid config
        resp = v1_adapter_client.get("/config/team")
        assert resp.status_code == 200
        existing = resp.json()[0]

        body = {
            "id": "adr004-put-test",
            "name": "ADR-004 PUT Test",
            "config": {**existing["data"], "id": "adr004-put-test", "name": "ADR-004 PUT Test"},
            "dry_run": False,
        }
        resp = v1_adapter_client.put("/config/team", json=body)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Cleanup
        v1_adapter_client.delete("/config/team/adr004-put-test")

    def test_delete_config_new_url_shape(
        self, v1_adapter_client: TestClient,
    ) -> None:
        """AC2: DELETE /config/team/{config_id} with URL params succeeds."""
        # Create an entry first
        resp = v1_adapter_client.get("/config/team")
        assert resp.status_code == 200
        existing = resp.json()[0]

        body = {
            "id": "adr004-del-test",
            "name": "ADR-004 DEL Test",
            "config": {**existing["data"], "id": "adr004-del-test", "name": "ADR-004 DEL Test"},
            "dry_run": False,
        }
        v1_adapter_client.put("/config/team", json=body)

        # Delete via new URL shape
        resp = v1_adapter_client.delete("/config/team/adr004-del-test")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# AC #3 -- Human input endpoint integration (FR6)
# ---------------------------------------------------------------------------


class TestHumanInput:
    """Test POST /process_human_input/{id}/human/{proxy} with message dict."""

    def test_human_input_with_message_body(
        self, v1_adapter_client: TestClient,
    ) -> None:
        """AC3: human input with message dict containing id field succeeds."""
        team_id = _create_v1_team(v1_adapter_client)

        # Send initial message and wait for LLM response
        v1_adapter_client.patch(
            f"/process/{team_id}",
            json={"content": "Say hello in one word."},
        )
        messages = _wait_for_v1_messages(v1_adapter_client, team_id)

        # Find a message ID
        assert len(messages) >= 1
        msg_id = messages[0]["id"]

        # Send human input with the new message body shape
        resp = v1_adapter_client.post(
            f"/process_human_input/{team_id}/human/@Human",
            json={
                "content": "reply",
                "message": {"id": str(msg_id), "content": "original"},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Cleanup
        v1_adapter_client.delete(f"/process/{team_id}/archive")
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# AC #4 -- Auth stubs integration (FR7, FR8, FR9)
# ---------------------------------------------------------------------------


class TestAuthStubs:
    """Test auth stub endpoints for Community tier."""

    def test_auth_me(self, v1_adapter_client: TestClient) -> None:
        """AC4: GET /auth/me returns anonymous user dict."""
        resp = v1_adapter_client.get("/auth/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"user_id": "anonymous", "email": "", "name": "Anonymous"}

    def test_auth_ws_ticket(self, v1_adapter_client: TestClient) -> None:
        """AC4: GET /auth/ws-ticket returns noauth ticket."""
        resp = v1_adapter_client.get("/auth/ws-ticket")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"ticket": "noauth"}

    def test_auth_logout(self, v1_adapter_client: TestClient) -> None:
        """AC4: GET /auth/logout returns none auth_type."""
        resp = v1_adapter_client.get("/auth/logout")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"auth_type": "none"}


# ---------------------------------------------------------------------------
# AC #5 -- Team configs dict response integration (FR10)
# ---------------------------------------------------------------------------


class TestTeamConfigs:
    """Test GET /team-configs/ returns dict keyed by team name."""

    def test_team_configs_returns_dict(
        self, v1_adapter_client: TestClient,
    ) -> None:
        """AC5: GET /team-configs/ returns a dict (not a list)."""
        resp = v1_adapter_client.get("/team-configs/")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_team_configs_has_seeded_entry(
        self, v1_adapter_client: TestClient,
    ) -> None:
        """AC5: at least one entry exists from seeded catalog."""
        resp = v1_adapter_client.get("/team-configs/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

    def test_team_configs_entry_shape(
        self, v1_adapter_client: TestClient,
    ) -> None:
        """AC5: each value has module (str) and setup (str, parseable as JSON) keys."""
        resp = v1_adapter_client.get("/team-configs/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

        for _name, entry in data.items():
            assert "module" in entry
            assert "setup" in entry
            assert isinstance(entry["module"], str)
            assert isinstance(entry["setup"], str)
            # setup must be parseable as JSON
            parsed = json.loads(entry["setup"])
            assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# AC #6 -- LLM context grouped response integration (FR11)
# ---------------------------------------------------------------------------


class TestLlmContextGrouped:
    """Test GET /llm_context/{id} returns dict keyed by agent ID."""

    def test_llm_context_grouped_response(
        self, v1_adapter_client: TestClient,
    ) -> None:
        """AC6: GET /llm_context/{id} returns dict with agent keys and context lists."""
        team_id = _create_v1_team(v1_adapter_client)

        # Send message and wait for LLM response
        v1_adapter_client.patch(
            f"/process/{team_id}",
            json={"content": "Say hello in one word."},
        )
        _wait_for_v1_messages(v1_adapter_client, team_id)

        resp = v1_adapter_client.get(f"/llm_context/{team_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

        # At least one agent key should exist
        assert len(data) >= 1
        for _agent_id, value in data.items():
            assert "context" in value
            assert isinstance(value["context"], list)

        # Cleanup
        v1_adapter_client.delete(f"/process/{team_id}/archive")
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# AC #7 -- States grouped response integration (FR12)
# ---------------------------------------------------------------------------


class TestStatesGrouped:
    """Test GET /states/{id} returns dict keyed by agent ID."""

    def test_states_grouped_response(
        self, v1_adapter_client: TestClient,
    ) -> None:
        """AC7: GET /states/{id} returns dict."""
        team_id = _create_v1_team(v1_adapter_client)

        # Agent startup may or may not trigger StateChangedMessage events
        # depending on agent implementation. Give a moment for startup.
        time.sleep(1.0)

        resp = v1_adapter_client.get(f"/states/{team_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

        # If entries present, verify shape
        for _agent_id, value in data.items():
            assert "schema" in value
            assert "state" in value
            assert isinstance(value["schema"], dict)
            assert isinstance(value["state"], dict)

        # Cleanup
        v1_adapter_client.delete(f"/process/{team_id}/archive")
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# AC #8 -- WebSocket error envelope integration (FR14)
# ---------------------------------------------------------------------------


class TestWebSocketErrorEnvelope:
    """Verify WebSocket error envelope -- existing unit tests in test_angular_v1_ws.py.

    The WebSocket error envelope (ErrorMessage -> type: "error") is thoroughly
    tested at the unit level in tests/test_angular_v1_ws.py (TestWrapErrorMessage
    class). An end-to-end WebSocket error integration test would require triggering
    a real error during LLM processing, which is fragile and non-deterministic.

    This test class confirms the unit-level coverage by exercising the same
    functions in the integration context.
    """

    def test_classify_error_message_returns_error(self) -> None:
        """AC8: _classify_envelope_type(ErrorMessage) returns 'error'."""
        from akgentic.core.messages.orchestrator import ErrorMessage

        from akgentic.infra.server.routes.frontend_adapter.angular_v1.ws import (
            _classify_envelope_type,
        )

        msg = ErrorMessage(exception_value="test error", exception_type="ValueError")
        result = _classify_envelope_type(msg)
        assert result == "error"

    def test_wrap_event_error_produces_error_payload(self) -> None:
        """AC8: wrap_event with ErrorMessage produces ErrorPayload with type=='error'."""
        import uuid
        from datetime import UTC, datetime

        from akgentic.core.messages.orchestrator import ErrorMessage
        from akgentic.team.models import PersistedEvent

        from akgentic.infra.server.routes.frontend_adapter import ErrorPayload
        from akgentic.infra.server.routes.frontend_adapter.angular_v1.ws import (
            wrap_event,
        )

        msg = ErrorMessage(exception_value="something broke", exception_type="RuntimeError")
        ev = PersistedEvent(
            team_id=uuid.uuid4(),
            sequence=1,
            event=msg,
            timestamp=datetime.now(tz=UTC),
        )
        wrapped = wrap_event(ev)
        assert isinstance(wrapped.payload, ErrorPayload)
        assert wrapped.payload.type == "error"
        assert wrapped.payload.message == "something broke"
