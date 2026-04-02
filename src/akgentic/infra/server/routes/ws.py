"""WebSocket route for real-time event streaming via EventStream."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import cast

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from akgentic.infra.protocols.event_stream import StreamClosed
from akgentic.infra.server.routes.frontend_adapter import FrontendAdapter
from akgentic.infra.server.services.team_service import TeamService
from akgentic.team.models import TeamStatus

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Tracks idle WebSocket connections waiting for team restore.

    Stored on ``app.state.connection_manager`` so the restore endpoint
    can notify waiting connections. Uses a ``_restored`` set to signal
    idle loops that a team has been restored.
    """

    def __init__(self) -> None:
        self._waiting: dict[uuid.UUID, list[WebSocket]] = {}
        self._restored: set[uuid.UUID] = set()

    def add_waiting(self, team_id: uuid.UUID, ws: WebSocket) -> None:
        """Register an idle WebSocket waiting for a team to be restored."""
        self._waiting.setdefault(team_id, []).append(ws)

    def remove_waiting(self, team_id: uuid.UUID, ws: WebSocket) -> None:
        """Remove a WebSocket from the waiting list."""
        conns = self._waiting.get(team_id)
        if conns is not None:
            try:
                conns.remove(ws)
            except ValueError:
                pass
            if not conns:
                del self._waiting[team_id]

    def pop_waiting(self, team_id: uuid.UUID) -> list[WebSocket]:
        """Pop and return all waiting connections for a team."""
        return self._waiting.pop(team_id, [])

    def is_restored(self, team_id: uuid.UUID) -> bool:
        """Check if a team has been restored."""
        return team_id in self._restored

    def mark_restored(self, team_id: uuid.UUID) -> None:
        """Mark a team as restored so idle loops can detect it."""
        self._restored.add(team_id)

    def clear_restored(self, team_id: uuid.UUID) -> None:
        """Clear the restored flag for a team."""
        self._restored.discard(team_id)


def _get_team_service(ws: WebSocket) -> TeamService:
    """Extract TeamService from app.state."""
    return cast(TeamService, ws.app.state.team_service)


def _get_connection_manager(ws: WebSocket) -> ConnectionManager:
    """Extract ConnectionManager from app.state."""
    return cast(ConnectionManager, ws.app.state.connection_manager)


@router.websocket("/ws/{team_id}")
async def websocket_events(websocket: WebSocket, team_id: uuid.UUID) -> None:
    """Stream real-time orchestrator events over WebSocket.

    - Running team: subscribes and pushes events in real-time.
    - Stopped team: accepts connection and idles until restored.
    - Non-existent team: rejects with close code 4004.
    """
    service = _get_team_service(websocket)
    conn_mgr = _get_connection_manager(websocket)

    process = service.get_team(team_id)
    if process is None:
        logger.info("WebSocket rejected: team_id=%s (not found)", team_id)
        await websocket.close(code=4004, reason="Team not found")
        return

    await websocket.accept()
    logger.info("WebSocket connected: team_id=%s", team_id)

    adapter: FrontendAdapter | None = getattr(websocket.app.state, "frontend_adapter", None)

    if process.status == TeamStatus.RUNNING:
        await _run_streaming_loop(websocket, service, team_id, conn_mgr, adapter)
    else:
        await _run_idle_loop(websocket, team_id, conn_mgr, service, adapter)


async def _run_streaming_loop(
    websocket: WebSocket,
    service: TeamService,
    team_id: uuid.UUID,
    conn_mgr: ConnectionManager,
    adapter: FrontendAdapter | None = None,
) -> None:
    """Subscribe to EventStream and forward events over WebSocket."""
    logger.debug("Streaming loop started: team_id=%s", team_id)
    event_stream = service.get_event_stream()
    try:
        reader = event_stream.subscribe(team_id, cursor=0)
    except StreamClosed:
        logger.debug("Stream already closed for team %s, entering idle loop", team_id)
        await _run_idle_loop(websocket, team_id, conn_mgr, service, adapter)
        return

    loop = asyncio.get_running_loop()
    try:
        while True:
            event = await loop.run_in_executor(None, reader.read_next, 0.5)
            if event is None:
                if websocket.client_state != WebSocketState.CONNECTED:
                    break
                continue

            try:
                if adapter is not None:
                    wrapped = adapter.wrap_ws_event(event)
                    await websocket.send_text(wrapped.model_dump_json())
                else:
                    await websocket.send_text(event.model_dump_json())
                logger.debug("Event forwarded to client: team_id=%s", team_id)
            except Exception:  # noqa: BLE001
                logger.debug("Skipping unserializable event: team_id=%s", team_id)
    except StreamClosed:
        logger.debug("StreamClosed for team %s, transitioning to idle loop", team_id)
        await _run_idle_loop(websocket, team_id, conn_mgr, service, adapter)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: team_id=%s", team_id)
    finally:
        reader.close()


async def _run_idle_loop(
    websocket: WebSocket,
    team_id: uuid.UUID,
    conn_mgr: ConnectionManager,
    service: TeamService,
    adapter: FrontendAdapter | None = None,
) -> None:
    """Wait for team restore or client disconnect.

    The ``timeout=1.0`` polling loop checks ``conn_mgr.is_restored()``
    on each timeout iteration. ``notify_restore()`` marks the team as
    restored, and this loop detects it on the next timeout iteration.
    """
    logger.debug("WebSocket idle, waiting for restore: team_id=%s", team_id)
    conn_mgr.add_waiting(team_id, websocket)
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except TimeoutError:
                if conn_mgr.is_restored(team_id):
                    conn_mgr.clear_restored(team_id)
                    await _run_streaming_loop(websocket, service, team_id, conn_mgr, adapter)
                    return
            except WebSocketDisconnect:
                return
    finally:
        conn_mgr.remove_waiting(team_id, websocket)


def notify_restore(
    conn_mgr: ConnectionManager,
    service: TeamService,
    team_id: uuid.UUID,
) -> None:
    """Called when a team is restored -- signal all waiting WebSocket connections.

    Marks the team as restored in ConnectionManager. Each idle loop detects
    this on its next timeout iteration and starts streaming via EventStream.
    """
    waiting = conn_mgr.pop_waiting(team_id)
    if not waiting:
        return

    logger.info("Notifying %d waiting WebSocket(s) for team %s", len(waiting), team_id)

    conn_mgr.mark_restored(team_id)

    # Re-add all connected WebSockets so they can pick up the restored signal
    for ws in waiting:
        if ws.client_state != WebSocketState.CONNECTED:
            continue
        conn_mgr.add_waiting(team_id, ws)
