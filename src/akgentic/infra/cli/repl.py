"""Interactive chat REPL for team communication."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from akgentic.infra.cli.client import ApiClient
from akgentic.infra.cli.formatters import OutputFormat
from akgentic.infra.cli.ws_client import WsClient

# Event types to display vs skip
_DISPLAY_EVENTS = {"SentMessage", "ErrorMessage"}


class ChatSession:
    """REPL session combining REST message sending with WebSocket event streaming."""

    def __init__(
        self,
        client: ApiClient,
        ws_client: WsClient,
        team_id: str,
        fmt: OutputFormat,
    ) -> None:
        self._client = client
        self._ws = ws_client
        self._team_id = team_id
        self._fmt = fmt
        self._running = True

    async def run(self) -> None:
        """Main REPL loop: connect WS, replay history, read input, stream events."""
        async with self._ws:
            self._replay_history()
            print(f"Connected to team {self._team_id}. Type /quit or Ctrl+C to exit.")

            receive_task = asyncio.create_task(self._receive_loop())
            try:
                await self._input_loop()
            except KeyboardInterrupt:
                pass
            finally:
                self._running = False
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
                print("Session closed.")

    async def _input_loop(self) -> None:
        """Read user input and send messages."""
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                line = await loop.run_in_executor(None, _read_input, "You: ")
            except (EOFError, KeyboardInterrupt):
                break

            if not line.strip():
                continue

            if line.strip() == "/quit":
                break

            # Send message via REST API (run in executor to avoid blocking)
            try:
                await loop.run_in_executor(None, self._client.send_message, self._team_id, line)
            except SystemExit:
                print("Error sending message.", file=sys.stderr)

    async def _receive_loop(self) -> None:
        """Background coroutine: read WebSocket events and print them."""
        while self._running:
            try:
                event = await self._ws.receive_event()
                _print_event(event)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                if self._running:
                    break

    def _replay_history(self) -> None:
        """Fetch and display past events before starting the REPL."""
        try:
            events = self._client.get_events(self._team_id)
        except SystemExit:
            return
        displayed = False
        for evt in events:
            inner = evt.get("event", {})
            if _print_event(evt if "__model__" in evt else {"event": inner}):
                displayed = True
        if displayed:
            print("--- history ---")


def _read_input(prompt: str) -> str:
    """Read a line from stdin (used via run_in_executor)."""
    return input(prompt)


def _print_event(data: dict[str, Any]) -> bool:
    """Format and print a single event. Returns True if something was printed."""
    event = data.get("event", data)
    if isinstance(event, str):
        try:
            event = json.loads(event)
        except (json.JSONDecodeError, TypeError):
            return False

    model = event.get("__model__", "")
    if model not in _DISPLAY_EVENTS:
        return False

    if model == "ErrorMessage":
        content = event.get("content", event.get("error", ""))
        print(f"[error] {content}")
        return True

    # SentMessage — agent response
    sender = event.get("sender", "agent")
    content = event.get("content", "")
    if content:
        print(f"[{sender}] {content}")
        return True
    return False
