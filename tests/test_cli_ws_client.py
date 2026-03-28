"""Tests for WsClient WebSocket wrapper."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets.exceptions

from akgentic.infra.cli.ws_client import WsClient


class TestUrlConversion:
    def test_http_to_ws(self) -> None:
        client = WsClient("http://localhost:8000", "team-1")
        assert client.url == "ws://localhost:8000/ws/team-1"

    def test_https_to_wss(self) -> None:
        client = WsClient("https://example.com", "team-2")
        assert client.url == "wss://example.com/ws/team-2"

    def test_preserves_port(self) -> None:
        client = WsClient("http://host:9090", "t3")
        assert client.url == "ws://host:9090/ws/t3"

    def test_api_key_stored(self) -> None:
        client = WsClient("http://localhost:8000", "t1", api_key="secret")
        assert ("Authorization", "Bearer secret") in client._headers

    def test_no_api_key(self) -> None:
        client = WsClient("http://localhost:8000", "t1")
        assert client._headers == []


class TestConnect:
    async def test_connect_success(self) -> None:
        mock_ws = AsyncMock()
        with patch(
            "akgentic.infra.cli.ws_client.websockets.asyncio.client.connect",
            new_callable=AsyncMock,
            return_value=mock_ws,
        ):
            client = WsClient("http://localhost:8000", "t1")
            result = await client.connect()
            assert result is client
            assert client._ws is mock_ws

    async def test_connect_refused_exits(self) -> None:
        with patch(
            "akgentic.infra.cli.ws_client.websockets.asyncio.client.connect",
            new_callable=AsyncMock,
            side_effect=ConnectionRefusedError("refused"),
        ):
            client = WsClient("http://localhost:8000", "t1")
            with pytest.raises(SystemExit) as exc_info:
                await client.connect()
            assert exc_info.value.code == 1

    async def test_connect_os_error_exits(self) -> None:
        with patch(
            "akgentic.infra.cli.ws_client.websockets.asyncio.client.connect",
            new_callable=AsyncMock,
            side_effect=OSError("network error"),
        ):
            client = WsClient("http://localhost:8000", "t1")
            with pytest.raises(SystemExit) as exc_info:
                await client.connect()
            assert exc_info.value.code == 1

    async def test_connect_invalid_status_404_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 404
        exc = websockets.exceptions.InvalidStatus(mock_response)
        with patch(
            "akgentic.infra.cli.ws_client.websockets.asyncio.client.connect",
            new_callable=AsyncMock,
            side_effect=exc,
        ):
            client = WsClient("http://localhost:8000", "t1")
            with pytest.raises(SystemExit) as exc_info:
                await client.connect()
            assert exc_info.value.code == 1
        assert "team not found" in capsys.readouterr().err

    async def test_connect_invalid_handshake_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exc = websockets.exceptions.InvalidHandshake("bad handshake")
        with patch(
            "akgentic.infra.cli.ws_client.websockets.asyncio.client.connect",
            new_callable=AsyncMock,
            side_effect=exc,
        ):
            client = WsClient("http://localhost:8000", "t1")
            with pytest.raises(SystemExit) as exc_info:
                await client.connect()
            assert exc_info.value.code == 1
        assert "handshake failed" in capsys.readouterr().err


class TestReceiveEvent:
    async def test_receive_json(self) -> None:
        event = {"__model__": "SentMessage", "content": "hello"}
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps(event))
        client = WsClient("http://localhost:8000", "t1")
        client._ws = mock_ws
        result = await client.receive_event()
        assert result == event

    async def test_receive_bytes(self) -> None:
        event = {"__model__": "SentMessage", "content": "hi"}
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps(event).encode("utf-8"))
        client = WsClient("http://localhost:8000", "t1")
        client._ws = mock_ws
        result = await client.receive_event()
        assert result == event

    async def test_receive_not_connected_raises(self) -> None:
        client = WsClient("http://localhost:8000", "t1")
        with pytest.raises(RuntimeError, match="Not connected"):
            await client.receive_event()


class TestClose:
    async def test_close(self) -> None:
        mock_ws = AsyncMock()
        client = WsClient("http://localhost:8000", "t1")
        client._ws = mock_ws
        await client.close()
        mock_ws.close.assert_awaited_once()
        assert client._ws is None

    async def test_close_when_not_connected(self) -> None:
        client = WsClient("http://localhost:8000", "t1")
        await client.close()  # Should not raise


class TestContextManager:
    async def test_async_with(self) -> None:
        mock_ws = AsyncMock()
        with patch(
            "akgentic.infra.cli.ws_client.websockets.asyncio.client.connect",
            new_callable=AsyncMock,
            return_value=mock_ws,
        ):
            client = WsClient("http://localhost:8000", "t1")
            async with client as ws:
                assert ws is client
                assert client._ws is mock_ws
            mock_ws.close.assert_awaited_once()
