"""Integration tests — event persistence via REST after team lifecycle operations."""

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
# Tests (AC #2)
# ---------------------------------------------------------------------------


class TestEventPersistence:
    """Integration tests for event persistence after team lifecycle operations."""

    def test_events_persisted_after_stop(
        self, integration_client: TestClient,
    ) -> None:
        """AC #2: Create team, send message, wait for LLM, stop, verify events via REST."""
        team_id = _create_team(integration_client)

        # Send message and wait for LLM response
        integration_client.post(
            f"/teams/{team_id}/message",
            json={"content": "Reply with exactly one word."},
        )
        events = _wait_for_llm_response(integration_client, team_id)
        assert _has_llm_content(events)

        # Stop the team
        resp = integration_client.post(f"/teams/{team_id}/stop")
        assert resp.status_code == 204

        # Events should still be accessible via REST after stop
        resp = integration_client.get(f"/teams/{team_id}/events")
        assert resp.status_code == 200
        stopped_events = resp.json()["events"]

        # Verify SentMessage and ReceivedMessage are present
        models = [
            ev["event"].get("__model__", "")
            for ev in stopped_events
            if isinstance(ev.get("event"), dict)
        ]
        has_sent = any("SentMessage" in str(m) for m in models)
        has_received = any("ReceivedMessage" in str(m) for m in models)
        assert has_sent or has_received, (
            f"Expected SentMessage or ReceivedMessage in persisted events, got: {models}"
        )

    def test_event_count_after_send_and_respond(
        self, integration_client: TestClient,
    ) -> None:
        """AC #2: Verify event count >= 3 after a full send-and-respond cycle."""
        team_id = _create_team(integration_client)

        integration_client.post(
            f"/teams/{team_id}/message",
            json={"content": "What is 1 + 1? Answer with the number."},
        )
        events = _wait_for_llm_response(integration_client, team_id)
        assert len(events) >= 3, (
            f"Expected >= 3 events after send-and-respond cycle, got {len(events)}"
        )

    def test_events_survive_stop_restore_cycle(
        self, integration_client: TestClient,
    ) -> None:
        """AC #2: Events from before stop are still accessible after restore."""
        team_id = _create_team(integration_client)

        # Send message and wait for LLM response
        integration_client.post(
            f"/teams/{team_id}/message",
            json={"content": "Respond with one word."},
        )
        events_before = _wait_for_llm_response(integration_client, team_id)
        count_before = len(events_before)
        assert count_before >= 3

        # Stop
        resp = integration_client.post(f"/teams/{team_id}/stop")
        assert resp.status_code == 204

        # Restore
        resp = integration_client.post(f"/teams/{team_id}/restore")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

        # Events from before stop should still be accessible
        resp = integration_client.get(f"/teams/{team_id}/events")
        assert resp.status_code == 200
        events_after = resp.json()["events"]
        assert len(events_after) >= count_before, (
            f"Events lost after stop/restore: had {count_before}, now {len(events_after)}"
        )
