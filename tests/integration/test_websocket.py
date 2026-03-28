"""Integration tests — WebSocket event streaming with real actors and real LLM."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATALOG_ENTRY_ID = "test-team"
POLL_INTERVAL_S = 1.0
POLL_TIMEOUT_S = 60.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_team(client: TestClient) -> str:
    """POST /teams and return the team_id."""
    resp = client.post("/teams/", json={"catalog_entry_id": CATALOG_ENTRY_ID})
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "running"
    return data["team_id"]


def _wait_for_llm_response(
    client: TestClient,
    team_id: str,
    timeout: float = POLL_TIMEOUT_S,
) -> list[dict[str, object]]:
    """Poll GET /teams/{team_id}/events until @Manager responds."""
    deadline = time.monotonic() + timeout
    events: list[dict[str, object]] = []
    while time.monotonic() < deadline:
        resp = client.get(f"/teams/{team_id}/events")
        assert resp.status_code == 200
        events = resp.json()["events"]
        if _has_llm_content(events):
            return events
        time.sleep(POLL_INTERVAL_S)
    pytest.fail(
        f"Timed out after {timeout}s waiting for LLM response "
        f"(got {len(events)} events, none with LLM content)"
    )


def _has_llm_content(events: list[dict[str, object]]) -> bool:
    """Check if any event contains LLM-generated content from @Manager."""
    for ev_wrapper in events:
        ev = ev_wrapper["event"]
        if not isinstance(ev, dict):
            continue
        msg = ev.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        sender = ev.get("sender")
        if not isinstance(sender, dict):
            continue
        if (
            isinstance(content, str)
            and len(content) > 0
            and sender.get("name") == "@Manager"
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWebSocketIntegration:
    """Integration tests for WebSocket event streaming (AC #1)."""

    def test_ws_create_team_send_message_receive_events(
        self, integration_client: TestClient,
    ) -> None:
        """AC #1: Create team, open WS, send message, verify events delivered via WS."""
        team_id = _create_team(integration_client)

        with integration_client.websocket_connect(f"/ws/{team_id}") as ws:
            # Send a message to trigger events through the actor system
            time.sleep(0.3)
            integration_client.post(
                f"/teams/{team_id}/message",
                json={"content": "Say hello"},
            )

            data = ws.receive_json(mode="text")
            assert isinstance(data, dict)
            assert "__model__" in data

        # Stop team before fixture teardown to avoid LLM-in-flight hang
        integration_client.post(f"/teams/{team_id}/stop")
        time.sleep(0.5)

    def test_ws_full_round_trip_with_llm(
        self, integration_client: TestClient,
    ) -> None:
        """AC #1: Send message via HTTP, verify LLM response arrives, open WS for new events."""
        team_id = _create_team(integration_client)

        # Send message and wait for LLM response via REST polling
        integration_client.post(
            f"/teams/{team_id}/message",
            json={"content": "Reply with exactly one word."},
        )
        events = _wait_for_llm_response(integration_client, team_id)
        assert _has_llm_content(events)

        # Now open WS and send another message — verify WS delivers events
        with integration_client.websocket_connect(f"/ws/{team_id}") as ws:
            time.sleep(0.3)
            integration_client.post(
                f"/teams/{team_id}/message",
                json={"content": "Say yes."},
            )
            data = ws.receive_json(mode="text")
            assert isinstance(data, dict)
            assert "__model__" in data

            # Verify at least one event has SentMessage or ReceivedMessage
            model = str(data.get("__model__", ""))
            assert "SentMessage" in model or "ReceivedMessage" in model, (
                f"Expected SentMessage or ReceivedMessage, got: {model}"
            )

        # Stop team before fixture teardown to avoid LLM-in-flight hang
        integration_client.post(f"/teams/{team_id}/stop")
        time.sleep(0.5)

    def test_ws_restore_receives_events(
        self, integration_client: TestClient,
    ) -> None:
        """AC #1: Connect WS to stopped team, restore, verify events flow after restore."""
        team_id = _create_team(integration_client)

        # Stop the team
        resp = integration_client.post(f"/teams/{team_id}/stop")
        assert resp.status_code == 204

        with integration_client.websocket_connect(f"/ws/{team_id}") as ws:
            # Restore the team — idle WS should start receiving events
            integration_client.post(f"/teams/{team_id}/restore")
            time.sleep(0.5)

            # Send a message to generate events
            integration_client.post(
                f"/teams/{team_id}/message",
                json={"content": "hello after restore"},
            )

            data = ws.receive_json(mode="text")
            assert isinstance(data, dict)
            assert "__model__" in data

        # Stop team before fixture teardown to avoid LLM-in-flight hang
        integration_client.post(f"/teams/{team_id}/stop")
        time.sleep(0.5)
