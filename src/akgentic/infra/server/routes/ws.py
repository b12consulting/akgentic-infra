"""WebSocket route for real-time event streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import uuid
from datetime import UTC, datetime
from queue import Empty
from typing import cast

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from akgentic.infra.adapters.websocket_subscriber import WebSocketEventSubscriber
from akgentic.infra.server.routes.frontend_adapter import FrontendAdapter
from akgentic.infra.server.services.team_service import TeamService
from akgentic.team.models import PersistedEvent, TeamStatus

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Tracks idle WebSocket connections waiting for team restore.

    Stored on ``app.state.connection_manager`` so the restore endpoint
    can notify waiting connections. Also maps restored WebSockets to
    their subscribers (avoiding monkey-patching attributes on WebSocket).
    """

    def __init__(self) -> None:
        self._waiting: dict[uuid.UUID, list[WebSocket]] = {}
        self._subscribers: dict[int, WebSocketEventSubscriber] = {}

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
        self._subscribers.pop(id(ws), None)

    def pop_waiting(self, team_id: uuid.UUID) -> list[WebSocket]:
        """Pop and return all waiting connections for a team."""
        return self._waiting.pop(team_id, [])

    def set_subscriber(self, ws: WebSocket, sub: WebSocketEventSubscriber) -> None:
        """Associate a subscriber with a restored WebSocket connection."""
        self._subscribers[id(ws)] = sub

    def get_subscriber(self, ws: WebSocket) -> WebSocketEventSubscriber | None:
        """Look up the subscriber assigned to a WebSocket, if any."""
        return self._subscribers.pop(id(ws), None)


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
        await websocket.close(code=4004, reason="Team not found")
        return

    await websocket.accept()

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
    """Subscribe to orchestrator events and forward them over WebSocket."""
    handle = service.get_handle(team_id)
    if handle is None:
        await websocket.close(code=1011, reason="Runtime not available")
        return

    subscriber = WebSocketEventSubscriber()
    handle.subscribe(subscriber)

    try:
        await _send_loop(websocket, subscriber, team_id, adapter)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            handle.unsubscribe(subscriber)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to unsubscribe (orchestrator may be stopped)")


async def _send_loop(
    websocket: WebSocket,
    subscriber: WebSocketEventSubscriber,
    team_id: uuid.UUID | None = None,
    adapter: FrontendAdapter | None = None,
) -> None:
    """Read from subscriber queue and send over WebSocket."""
    loop = asyncio.get_running_loop()
    q = subscriber.get_queue()
    # Local sequence counter for frontend adapter framing, NOT the global
    # persisted event sequence. Resets to 0 on each WebSocket (re)connect.
    # This is intentional — the frontend uses it for frame ordering within
    # a single connection, not for durable event ordering.
    seq = 0
    while True:
        try:
            item: str | None = await loop.run_in_executor(None, _queue_get, q)
        except _QueueTimeoutError:
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            continue

        if item is None:
            await websocket.close(code=1000)
            break

        if adapter is not None and team_id is not None:
            try:
                msg_data = json.loads(item)
            except json.JSONDecodeError:
                logger.warning("Malformed JSON in event queue, sending raw text")
                await websocket.send_text(item)
                continue
            event = PersistedEvent.model_validate({
                "team_id": str(team_id),
                "sequence": seq,
                "event": msg_data,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            })
            seq += 1
            wrapped = adapter.wrap_ws_event(event)
            await websocket.send_text(wrapped.model_dump_json())
        else:
            await websocket.send_text(item)


async def _run_idle_loop(
    websocket: WebSocket,
    team_id: uuid.UUID,
    conn_mgr: ConnectionManager,
    service: TeamService,
    adapter: FrontendAdapter | None = None,
) -> None:
    """Wait for team restore or client disconnect.

    The ``timeout=1.0`` polling loop is a signaling mechanism:
    ``notify_restore()`` sets a subscriber on idle WebSockets via the
    ConnectionManager, and this loop detects it on the next timeout
    iteration. Direct signaling is not used because there is no safe
    way to interrupt a blocking ``receive_text()`` from another thread.
    """
    conn_mgr.add_waiting(team_id, websocket)
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except TimeoutError:
                sub = conn_mgr.get_subscriber(websocket)
                if sub is not None:
                    await _send_loop(websocket, sub, team_id, adapter)
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
    """Called when a team is restored — activate all waiting WebSocket connections.

    Creates a WebSocketEventSubscriber per waiting connection, subscribes each
    to the restored team's orchestrator, and signals them to start streaming.
    """
    waiting = conn_mgr.pop_waiting(team_id)
    if not waiting:
        return

    handle = service.get_handle(team_id)
    if handle is None:
        return

    for ws in waiting:
        if ws.client_state != WebSocketState.CONNECTED:
            continue
        subscriber = WebSocketEventSubscriber()
        handle.subscribe(subscriber)
        conn_mgr.set_subscriber(ws, subscriber)


class _QueueTimeoutError(Exception):
    """Raised when queue.get times out."""


def _queue_get(q: queue.Queue[str | None]) -> str | None:  # noqa: F821
    """Blocking queue get with timeout for use in run_in_executor."""

    try:
        return q.get(timeout=0.5)
    except Empty:
        raise _QueueTimeoutError from None
