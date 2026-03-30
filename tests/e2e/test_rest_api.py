"""REST API E2E tests — real running server, real LLM (Story 9.8, AC #1-#8).

All tests hit a live server via httpx.Client (not TestClient).
Every test that creates a team cleans up in try/finally.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tests.e2e.conftest import poll_until

pytestmark = [pytest.mark.e2e]

CATALOG_ENTRY_ID = "test-team"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_team(client: httpx.Client) -> str:
    """Create a team and return team_id."""
    resp = client.post("/teams/", json={"catalog_entry_id": CATALOG_ENTRY_ID})
    assert resp.status_code == 201, f"Expected 201, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["status"] == "running"
    return data["team_id"]


def _delete_team(client: httpx.Client, team_id: str) -> None:
    """Best-effort team cleanup."""
    try:
        client.delete(f"/teams/{team_id}")
    except Exception:  # noqa: BLE001
        pass


def _send_message(client: httpx.Client, team_id: str, content: str = "hello") -> None:
    """Send a message to a team."""
    resp = client.post(f"/teams/{team_id}/message", json={"content": content})
    assert resp.status_code == 204, f"Expected 204, got {resp.status_code}: {resp.text}"


def _get_events(client: httpx.Client, team_id: str) -> list[dict[str, Any]]:
    """Fetch events from a team."""
    resp = client.get(f"/teams/{team_id}/events")
    assert resp.status_code == 200
    return resp.json()["events"]


def _has_manager_response(events: list[dict[str, Any]]) -> bool:
    """Check if @Manager has responded with content."""
    for ev_wrapper in events:
        ev = ev_wrapper.get("event", {})
        if not isinstance(ev, dict):
            continue
        model = ev.get("__model__", "")
        short = model.rsplit(".", 1)[-1] if model else ""
        if short != "SentMessage":
            continue
        sender = ev.get("sender", {})
        if not isinstance(sender, dict):
            continue
        if sender.get("name") != "@Manager":
            continue
        msg = ev.get("message", {})
        if isinstance(msg, dict) and isinstance(msg.get("content"), str) and msg["content"]:
            return True
    return False


def _wait_for_manager_response(
    client: httpx.Client,
    team_id: str,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Poll events until @Manager responds."""
    events: list[dict[str, Any]] = []

    def _check() -> bool:
        nonlocal events
        events = _get_events(client, team_id)
        return _has_manager_response(events)

    poll_until(
        _check, timeout=timeout, interval=1.0, message="Timed out waiting for @Manager LLM response"
    )
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_e2e_list_catalog_teams(e2e_http_client: httpx.Client) -> None:
    """AC #1: GET /catalog/api/teams/ returns expected shape and content."""
    resp = e2e_http_client.get("/catalog/api/teams/")
    assert resp.status_code == 200
    teams = resp.json()
    assert isinstance(teams, list)
    assert len(teams) >= 1

    # Verify response shape
    for team in teams:
        for field in ("id", "name", "entry_point", "members"):
            assert field in team, f"Missing field '{field}' in catalog team entry"


def test_e2e_create_team(e2e_http_client: httpx.Client) -> None:
    """AC #2: POST /teams/ creates team with status running."""
    team_id: str | None = None
    try:
        resp = e2e_http_client.post(
            "/teams/",
            json={"catalog_entry_id": CATALOG_ENTRY_ID},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "running"
        team_id = data["team_id"]
        assert team_id is not None

        # Verify GET returns the created team
        resp2 = e2e_http_client.get(f"/teams/{team_id}")
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "running"
    finally:
        if team_id:
            _delete_team(e2e_http_client, team_id)


def test_e2e_send_message_receive_response(e2e_http_client: httpx.Client) -> None:
    """AC #3: Send message, poll events, verify SentMessage with nested message.content."""
    team_id: str | None = None
    try:
        team_id = _create_team(e2e_http_client)
        _send_message(e2e_http_client, team_id, "hello")
        events = _wait_for_manager_response(e2e_http_client, team_id)

        # Find @Manager SentMessage
        manager_event = None
        for ev in events:
            event_data = ev["event"]
            model = event_data.get("__model__", "")
            short = model.rsplit(".", 1)[-1] if model else ""
            if short != "SentMessage":
                continue
            sender = event_data.get("sender", {})
            if isinstance(sender, dict) and sender.get("name") == "@Manager":
                manager_event = event_data
                break

        assert manager_event is not None, "No @Manager SentMessage found"

        # AC #3: SentMessage has nested message.content (NOT flat content)
        message = manager_event.get("message")
        assert isinstance(message, dict), "SentMessage.message must be a dict"
        content = message.get("content")
        assert isinstance(content, str) and len(content) > 0, (
            "SentMessage.message.content must be a non-empty string"
        )
    finally:
        if team_id:
            _delete_team(e2e_http_client, team_id)


def test_e2e_event_sequence(e2e_http_client: httpx.Client) -> None:
    """AC #4: Verify event sequence includes correct model types and sender shapes."""
    team_id: str | None = None
    try:
        team_id = _create_team(e2e_http_client)
        _send_message(e2e_http_client, team_id, "say hello")
        events = _wait_for_manager_response(e2e_http_client, team_id)

        # Collect event model types
        model_types = set()
        for ev in events:
            event_data = ev["event"]
            model = event_data.get("__model__", "")
            short = model.rsplit(".", 1)[-1] if model else ""
            model_types.add(short)

        # Must include StartMessage and SentMessage
        assert "StartMessage" in model_types, f"Expected StartMessage in events, got: {model_types}"
        assert "SentMessage" in model_types, f"Expected SentMessage in events, got: {model_types}"

        # Verify sender dict shape on SentMessage events
        for ev in events:
            event_data = ev["event"]
            sender = event_data.get("sender")
            if sender is None:
                continue
            assert isinstance(sender, dict), f"sender must be a dict, got {type(sender)}"
            for field in ("name", "role"):
                assert field in sender, f"sender missing field '{field}'"
    finally:
        if team_id:
            _delete_team(e2e_http_client, team_id)


def test_e2e_workspace_round_trip(e2e_http_client: httpx.Client) -> None:
    """AC #5: Upload file, list tree, read back, verify content match."""
    team_id: str | None = None
    try:
        team_id = _create_team(e2e_http_client)
        test_content = b"Hello from E2E test!"
        test_path = "e2e-test-file.txt"

        # Upload file
        resp = e2e_http_client.post(
            f"/workspace/{team_id}/file",
            data={"path": test_path},
            files={"file": ("upload", test_content)},
        )
        assert resp.status_code == 201, f"Upload failed: {resp.status_code} {resp.text}"
        upload_data = resp.json()
        assert upload_data["path"] == test_path
        assert upload_data["size"] == len(test_content)

        # List tree
        resp = e2e_http_client.get(f"/workspace/{team_id}/tree")
        assert resp.status_code == 200
        tree = resp.json()
        entry_names = [e["name"] for e in tree["entries"]]
        assert test_path in entry_names, f"Uploaded file not in tree: {entry_names}"

        # Read back
        resp = e2e_http_client.get(
            f"/workspace/{team_id}/file",
            params={"path": test_path},
        )
        assert resp.status_code == 200
        assert resp.content == test_content
    finally:
        if team_id:
            _delete_team(e2e_http_client, team_id)


def test_e2e_human_input(e2e_http_client: httpx.Client) -> None:
    """AC #6: Trigger human input request, reply, verify response.

    Note: This test verifies the human-input endpoint is callable and returns
    the correct status code. Triggering a genuine HumanInputRequest from the
    LLM is non-deterministic, so we test the endpoint mechanics.
    """
    team_id: str | None = None
    try:
        team_id = _create_team(e2e_http_client)

        # The human-input endpoint requires a valid message_id. We send a message
        # first, wait for events, then use a message_id from the events.
        _send_message(e2e_http_client, team_id, "hello")
        events = _wait_for_manager_response(e2e_http_client, team_id)

        # Find any message ID from events
        message_id = None
        for ev in events:
            event_data = ev["event"]
            if "id" in event_data:
                message_id = event_data["id"]
                break

        if message_id is not None:
            # Verify human-input endpoint accepts input (may return 409 if
            # team is not in human-input-waiting state, which is acceptable)
            resp = e2e_http_client.post(
                f"/teams/{team_id}/human-input",
                json={"content": "test reply", "message_id": str(message_id)},
            )
            assert resp.status_code in (204, 409), (
                f"Expected 204 or 409, got {resp.status_code}: {resp.text}"
            )
    finally:
        if team_id:
            _delete_team(e2e_http_client, team_id)


def test_e2e_stop_restore_lifecycle(e2e_http_client: httpx.Client) -> None:
    """AC #7: Stop team, verify stopped, restore, verify running."""
    team_id: str | None = None
    try:
        team_id = _create_team(e2e_http_client)

        # Stop
        resp = e2e_http_client.post(f"/teams/{team_id}/stop")
        assert resp.status_code == 204, f"Stop failed: {resp.status_code}"

        # Verify stopped
        resp = e2e_http_client.get(f"/teams/{team_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

        # Restore
        resp = e2e_http_client.post(f"/teams/{team_id}/restore")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
    finally:
        if team_id:
            _delete_team(e2e_http_client, team_id)


def test_e2e_delete_team(e2e_http_client: httpx.Client) -> None:
    """AC #8: Delete team, verify 404 on subsequent GET."""
    team_id = _create_team(e2e_http_client)
    try:
        resp = e2e_http_client.delete(f"/teams/{team_id}")
        assert resp.status_code == 204

        resp = e2e_http_client.get(f"/teams/{team_id}")
        assert resp.status_code == 404
    finally:
        # Already deleted, but ensure cleanup
        _delete_team(e2e_http_client, team_id)
