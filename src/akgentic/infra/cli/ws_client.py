"""WebSocket client wrapper for real-time event streaming."""

from __future__ import annotations

import json
import sys
from typing import Any

import websockets.asyncio.client
import websockets.exceptions


class WsClient:
    """Async WebSocket client for streaming team events."""

    def __init__(self, base_url: str, team_id: str, api_key: str | None = None) -> None:
        ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
        self._url = f"{ws_url}/ws/{team_id}"
        self._headers: list[tuple[str, str]] = []
        if api_key:
            self._headers.append(("Authorization", f"Bearer {api_key}"))
        self._ws: websockets.asyncio.client.ClientConnection | None = None

    @property
    def url(self) -> str:
        """Return the resolved WebSocket URL."""
        return self._url

    async def connect(self) -> WsClient:
        """Open WebSocket connection."""
        try:
            self._ws = await websockets.asyncio.client.connect(
                self._url,
                additional_headers=self._headers,
            )
        except (ConnectionRefusedError, OSError) as exc:
            print(f"Connection error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        except websockets.exceptions.InvalidStatus as exc:
            status = exc.response.status_code
            if status == 404 or status == 403:
                print("Error: team not found", file=sys.stderr)
            else:
                print(f"WebSocket rejected: HTTP {status}", file=sys.stderr)
            raise SystemExit(1) from exc
        except websockets.exceptions.InvalidHandshake as exc:
            print(f"WebSocket handshake failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        return self

    async def receive_event(self) -> dict[str, Any]:
        """Read next JSON message from WebSocket."""
        if self._ws is None:
            raise RuntimeError("Not connected")
        raw = await self._ws.recv()
        text = raw if isinstance(raw, str) else raw.decode("utf-8")
        return json.loads(text)  # type: ignore[no-any-return]

    async def close(self) -> None:
        """Close WebSocket connection."""
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def __aenter__(self) -> WsClient:
        return await self.connect()

    async def __aexit__(self, *_args: object) -> None:
        await self.close()
