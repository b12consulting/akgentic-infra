"""Integration tests — WebSocket event streaming with real actors and real LLM."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from ._helpers import create_team, has_llm_content, wait_for_llm_response

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWebSocketIntegration:
    """Integration tests for WebSocket event streaming (AC #1)."""

    def test_ws_create_team_send_message_receive_events(
        self, integration_client: TestClient,
    ) -> None:
        """AC #1: Create team, open WS, send message, verify events delivered via WS."""
        team_id = create_team(integration_client)

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
            model = str(data.get("__model__", ""))
            assert "SentMessage" in model or "ReceivedMessage" in model, (
                f"Expected SentMessage or ReceivedMessage, got: {model}"
            )

        # Stop team before fixture teardown to avoid LLM-in-flight hang
        integration_client.post(f"/teams/{team_id}/stop")
        time.sleep(0.5)

    def test_ws_full_round_trip_with_llm(
        self, integration_client: TestClient,
    ) -> None:
        """AC #1: Send message via HTTP, verify LLM response arrives, open WS for new events."""
        team_id = create_team(integration_client)

        # Send message and wait for LLM response via REST polling
        integration_client.post(
            f"/teams/{team_id}/message",
            json={"content": "Reply with exactly one word."},
        )
        events = wait_for_llm_response(integration_client, team_id)
        assert has_llm_content(events)

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
        team_id = create_team(integration_client)

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
