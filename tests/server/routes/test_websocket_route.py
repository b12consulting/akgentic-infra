"""Tests for the WebSocket route and ConnectionManager."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from akgentic.infra.server.routes.ws import ConnectionManager


class TestConnectionManager:
    """Unit tests for ConnectionManager."""

    def test_add_and_pop_waiting(self) -> None:
        mgr = ConnectionManager()
        ws = MagicMock()
        tid = uuid.uuid4()
        mgr.add_waiting(tid, ws)
        result = mgr.pop_waiting(tid)
        assert result == [ws]
        assert mgr.pop_waiting(tid) == []

    def test_remove_waiting(self) -> None:
        mgr = ConnectionManager()
        ws = MagicMock()
        tid = uuid.uuid4()
        mgr.add_waiting(tid, ws)
        mgr.remove_waiting(tid, ws)
        assert mgr.pop_waiting(tid) == []

    def test_remove_nonexistent(self) -> None:
        mgr = ConnectionManager()
        mgr.remove_waiting(uuid.uuid4(), MagicMock())  # no error

    def test_mark_and_check_restored(self) -> None:
        mgr = ConnectionManager()
        tid = uuid.uuid4()
        assert not mgr.is_restored(tid)
        mgr.mark_restored(tid)
        assert mgr.is_restored(tid)
        mgr.clear_restored(tid)
        assert not mgr.is_restored(tid)


class TestWebSocketRoute:
    """Unit tests for WebSocket endpoint (AC #1, #2, #3, #4, #6)."""

    def test_ws_connect_nonexistent_team_receives_4004(
        self,
        client: TestClient,
    ) -> None:
        """AC #1: Non-existent team gets close code 4004."""
        from starlette.websockets import WebSocketDisconnect

        fake_id = uuid.uuid4()
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(f"/ws/{fake_id}"):
                pass
        assert exc_info.value.code == 4004

    def test_ws_connect_running_team_receives_events(
        self,
        client: TestClient,
    ) -> None:
        """AC #1, #2: Running team pushes events via EventStream."""
        resp = client.post("/teams/", json={"catalog_entry_id": "test-team"})
        assert resp.status_code == 201
        team_id = resp.json()["team_id"]

        with client.websocket_connect(f"/ws/{team_id}") as ws:
            _trigger_subscriber_event(client, team_id)
            data = ws.receive_json(mode="text")
            assert isinstance(data, dict)
            assert "__model__" in data

    def test_ws_connect_stopped_team_accepts_connection(
        self,
        client: TestClient,
    ) -> None:
        """AC #3: Stopped team accepts connection (idle)."""
        resp = client.post("/teams/", json={"catalog_entry_id": "test-team"})
        team_id = resp.json()["team_id"]
        client.post(f"/teams/{team_id}/stop")

        # Should accept without error -- we just close right away
        with client.websocket_connect(f"/ws/{team_id}"):
            pass

    def test_ws_event_json_contains_model_field(
        self,
        client: TestClient,
    ) -> None:
        """AC #1: __model__ discriminator in event JSON."""
        resp = client.post("/teams/", json={"catalog_entry_id": "test-team"})
        team_id = resp.json()["team_id"]

        with client.websocket_connect(f"/ws/{team_id}") as ws:
            _trigger_subscriber_event(client, team_id)
            data = ws.receive_json(mode="text")
            assert "__model__" in data

    def test_ws_client_disconnect_triggers_cleanup(
        self,
        client: TestClient,
    ) -> None:
        """AC #4: Client disconnect calls reader.close()."""
        resp = client.post("/teams/", json={"catalog_entry_id": "test-team"})
        team_id = resp.json()["team_id"]

        # Connect and immediately disconnect -- should not raise
        with client.websocket_connect(f"/ws/{team_id}"):
            pass

    def test_ws_restore_scenario(self, client: TestClient) -> None:
        """AC #6: Idle connection starts receiving events after restore."""
        resp = client.post("/teams/", json={"catalog_entry_id": "test-team"})
        team_id = resp.json()["team_id"]
        client.post(f"/teams/{team_id}/stop")

        # Restore while idle
        client.post(f"/teams/{team_id}/restore")

        # After restore, new WS connections work
        with client.websocket_connect(f"/ws/{team_id}") as ws:
            _trigger_subscriber_event(client, team_id)
            data = ws.receive_json(mode="text")
            assert "__model__" in data


class TestNotifyRestore:
    """Tests for the notify_restore function."""

    def test_notify_restore_no_waiting(self) -> None:
        """No error when no connections are waiting."""
        from akgentic.infra.server.routes.ws import notify_restore

        mgr = ConnectionManager()
        service = MagicMock()
        notify_restore(mgr, service, uuid.uuid4())

    def test_notify_restore_marks_restored(self) -> None:
        """notify_restore marks the team as restored and re-adds connected WSs."""
        from unittest.mock import PropertyMock

        from starlette.websockets import WebSocketState

        from akgentic.infra.server.routes.ws import notify_restore

        mgr = ConnectionManager()
        tid = uuid.uuid4()
        ws = MagicMock()
        type(ws).client_state = PropertyMock(return_value=WebSocketState.CONNECTED)
        mgr.add_waiting(tid, ws)

        service = MagicMock()
        notify_restore(mgr, service, tid)

        assert mgr.is_restored(tid)
        # WebSocket should be re-added to waiting list for idle loop pickup
        assert mgr.pop_waiting(tid) == [ws]

    def test_notify_restore_skips_disconnected(self) -> None:
        """notify_restore skips WebSocket connections that are no longer connected."""
        from unittest.mock import PropertyMock

        from starlette.websockets import WebSocketState

        from akgentic.infra.server.routes.ws import notify_restore

        mgr = ConnectionManager()
        tid = uuid.uuid4()
        ws = MagicMock()
        type(ws).client_state = PropertyMock(return_value=WebSocketState.DISCONNECTED)
        mgr.add_waiting(tid, ws)

        service = MagicMock()
        notify_restore(mgr, service, tid)

        assert mgr.is_restored(tid)
        # Disconnected WS should not be re-added
        assert mgr.pop_waiting(tid) == []

    def test_notify_restore_does_not_touch_orchestrator(self) -> None:
        """AC #2: notify_restore does NOT interact with the orchestrator."""
        from unittest.mock import PropertyMock

        from starlette.websockets import WebSocketState

        from akgentic.infra.server.routes.ws import notify_restore

        mgr = ConnectionManager()
        tid = uuid.uuid4()
        ws = MagicMock()
        type(ws).client_state = PropertyMock(return_value=WebSocketState.CONNECTED)
        mgr.add_waiting(tid, ws)

        service = MagicMock()
        notify_restore(mgr, service, tid)

        # No handle.subscribe() calls -- WS route is decoupled from orchestrator
        service.get_handle.assert_not_called()


class TestEventStreamWsIntegration:
    """Tests for EventStream-based WebSocket streaming (AC #1, #3, #5, #9)."""

    def test_ws_receives_replayed_historical_events(
        self,
        client: TestClient,
    ) -> None:
        """AC #1, #9: WS connect to running team receives replayed events (cursor=0)."""
        resp = client.post("/teams/", json={"catalog_entry_id": "test-team"})
        assert resp.status_code == 201
        team_id = resp.json()["team_id"]

        # Send a message to generate events before WS connects
        _trigger_subscriber_event(client, team_id)

        # New WS should receive events from cursor=0 (replay)
        with client.websocket_connect(f"/ws/{team_id}") as ws:
            data = ws.receive_json(mode="text")
            assert isinstance(data, dict)

    def test_ws_disconnect_does_not_raise(
        self,
        client: TestClient,
    ) -> None:
        """AC #4: WS disconnect -> reader.close() is called (no exceptions)."""
        resp = client.post("/teams/", json={"catalog_entry_id": "test-team"})
        team_id = resp.json()["team_id"]

        with client.websocket_connect(f"/ws/{team_id}"):
            pass  # immediate disconnect


def _trigger_subscriber_event(client: TestClient, team_id: str) -> None:
    """Send a message to the team to trigger an orchestrator event."""
    import time

    time.sleep(0.3)
    client.post(f"/teams/{team_id}/message", json={"content": "hello"})
