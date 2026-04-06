"""Integration tests — graceful shutdown sequence (Story 14.5, AC #1-3, #5).

Exercises the real wired shutdown stack (LocalWorkerHandle, ConnectionManager,
lifespan) with TestModel LLM injection.  No mocks except targeted patching in
``test_stop_all_skips_failures_integration``.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.routes.ws import ConnectionManager

from ._helpers import create_team

pytestmark = [pytest.mark.smoke]


# ---------------------------------------------------------------------------
# AC #1 — stop_all stops running teams and calls ActorSystem.shutdown()
# ---------------------------------------------------------------------------


def test_stop_all_stops_running_teams_integration(
    smoke_services: CommunityServices,
    smoke_client: TestClient,
) -> None:
    """Create a team, verify running, call stop_all(), verify stopped."""
    team_id = create_team(smoke_client)

    # Verify team is running
    resp = smoke_client.get(f"/teams/{team_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"

    # stop_all should stop team and shut down actor system
    smoke_services.worker_handle.stop_all()

    # After stop_all, team should be stopped (or absent from runtimes)
    resp = smoke_client.get(f"/teams/{team_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"


# ---------------------------------------------------------------------------
# AC #2 — stop_all skips failures without blocking remaining teams
# ---------------------------------------------------------------------------


def test_stop_all_skips_failures_integration(
    smoke_services: CommunityServices,
    smoke_client: TestClient,
) -> None:
    """Create two teams, make first stop fail, verify second still stopped."""
    team_id_1 = create_team(smoke_client)
    team_id_2 = create_team(smoke_client)

    tid_1 = uuid.UUID(team_id_1)

    original_stop = smoke_services.worker_handle.stop_team

    def _failing_stop(team_id: uuid.UUID) -> None:
        if team_id == tid_1:
            raise RuntimeError("Simulated stop failure")
        original_stop(team_id)

    with patch.object(
        smoke_services.worker_handle, "stop_team", side_effect=_failing_stop
    ):
        smoke_services.worker_handle.stop_all()

    # First team: stop failed, but stop_all should have continued
    # Second team: should be stopped
    resp2 = smoke_client.get(f"/teams/{team_id_2}")
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "stopped"


# ---------------------------------------------------------------------------
# AC #3 — lifespan shutdown sequence: draining, disconnect_all, stop_all
# ---------------------------------------------------------------------------


def test_lifespan_shutdown_sequence(
    smoke_app: FastAPI,
    smoke_services: CommunityServices,
) -> None:
    """Verify lifespan sets draining=True and calls disconnect_all then stop_all."""
    call_order: list[str] = []

    original_disconnect_all = smoke_app.state.connection_manager.disconnect_all
    original_stop_all = smoke_services.worker_handle.stop_all

    async def tracked_disconnect_all() -> None:
        call_order.append("disconnect_all")
        await original_disconnect_all()

    def tracked_stop_all() -> None:
        call_order.append("stop_all")
        original_stop_all()

    smoke_app.state.connection_manager.disconnect_all = tracked_disconnect_all
    smoke_services.worker_handle.stop_all = tracked_stop_all

    # Use TestClient as context manager to trigger lifespan startup + shutdown
    with TestClient(smoke_app):
        assert smoke_app.state.draining is False

    # After exiting the context manager, shutdown has run
    assert smoke_app.state.draining is True
    assert "disconnect_all" in call_order
    assert "stop_all" in call_order
    assert call_order.index("disconnect_all") < call_order.index("stop_all")


# ---------------------------------------------------------------------------
# AC #5 — WebSocket receives close code 1001 on shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_websocket_receives_1001_on_shutdown() -> None:
    """Verify disconnect_all sends close code 1001 to tracked WebSocket connections.

    Uses a real ConnectionManager with an AsyncMock WebSocket to verify the
    close code without the threading complexity of TestClient WebSocket.
    This validates the same code path that the lifespan shutdown invokes.
    """
    from unittest.mock import AsyncMock

    conn_mgr = ConnectionManager()
    ws = AsyncMock()
    conn_mgr.track(ws)

    assert ws in conn_mgr._active

    await conn_mgr.disconnect_all()

    ws.close.assert_awaited_once_with(code=1001, reason="Server shutting down")
    assert len(conn_mgr._active) == 0


def test_websocket_tracked_during_connection(
    smoke_app: FastAPI,
    smoke_client: TestClient,
) -> None:
    """Verify that a WebSocket connection to a real team is tracked by ConnectionManager."""
    team_id = create_team(smoke_client)
    conn_mgr: ConnectionManager = smoke_app.state.connection_manager

    # Before connecting, no active WebSockets
    initial_count = len(conn_mgr._active)

    with smoke_client.websocket_connect(f"/ws/{team_id}"):
        # During connection, the WebSocket should be tracked
        assert len(conn_mgr._active) > initial_count

    # After disconnect, the WebSocket is untracked
    assert len(conn_mgr._active) == initial_count
