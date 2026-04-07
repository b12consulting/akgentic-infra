"""Tests for WebSocket streaming loop CancelledError handling (ADR-015, AC4).

Unit tests that exercise ``_run_streaming_loop`` when the executor future is
cancelled (simulating Uvicorn forced task cancellation after
``timeout_graceful_shutdown`` expires).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from akgentic.infra.server.routes.ws import ConnectionManager, _run_streaming_loop


def _make_mocks() -> tuple[AsyncMock, MagicMock, uuid.UUID, ConnectionManager]:
    """Build minimal mocks for _run_streaming_loop."""
    websocket = AsyncMock()
    websocket.client_state = MagicMock()

    service = MagicMock()
    event_stream = MagicMock()
    reader = MagicMock()
    event_stream.subscribe.return_value = reader
    service.get_event_stream.return_value = event_stream

    team_id = uuid.uuid4()
    conn_mgr = ConnectionManager()

    return websocket, service, team_id, conn_mgr


async def test_cancelled_error_closes_reader() -> None:
    """CancelledError triggers reader.close() -- no resource leak (AC4)."""
    websocket, service, team_id, conn_mgr = _make_mocks()
    reader = service.get_event_stream().subscribe.return_value

    # Make run_in_executor raise CancelledError immediately
    with patch("asyncio.get_running_loop") as mock_loop:
        loop_inst = MagicMock()
        mock_loop.return_value = loop_inst

        future: asyncio.Future[None] = asyncio.Future()
        future.cancel()
        loop_inst.run_in_executor.return_value = future

        await _run_streaming_loop(websocket, service, team_id, conn_mgr)

    reader.close.assert_called_once()


async def test_cancelled_error_logs_streaming_stopped(caplog: pytest.LogCaptureFixture) -> None:
    """CancelledError emits 'WebSocket streaming stopped' log message (AC4)."""
    websocket, service, team_id, conn_mgr = _make_mocks()

    with patch("asyncio.get_running_loop") as mock_loop:
        loop_inst = MagicMock()
        mock_loop.return_value = loop_inst

        future: asyncio.Future[None] = asyncio.Future()
        future.cancel()
        loop_inst.run_in_executor.return_value = future

        with caplog.at_level(logging.INFO, logger="akgentic.infra.server.routes.ws"):
            await _run_streaming_loop(websocket, service, team_id, conn_mgr)

    assert any("WebSocket streaming stopped" in msg for msg in caplog.messages)


async def test_cancelled_error_does_not_propagate() -> None:
    """CancelledError is caught -- no unhandled exception escapes (AC4)."""
    websocket, service, team_id, conn_mgr = _make_mocks()

    with patch("asyncio.get_running_loop") as mock_loop:
        loop_inst = MagicMock()
        mock_loop.return_value = loop_inst

        future: asyncio.Future[None] = asyncio.Future()
        future.cancel()
        loop_inst.run_in_executor.return_value = future

        # Should complete without raising
        await _run_streaming_loop(websocket, service, team_id, conn_mgr)
