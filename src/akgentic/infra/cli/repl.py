"""Interactive chat REPL for team communication."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import websockets.exceptions

from akgentic.infra.cli.client import ApiClient
from akgentic.infra.cli.commands import CommandRegistry, build_default_registry
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
        *,
        server_url: str = "http://localhost:8000",
        api_key: str | None = None,
    ) -> None:
        self.client = client
        self.ws_client = ws_client
        self.team_id = team_id
        self.fmt = fmt
        self.server_url = server_url
        self.api_key = api_key
        self.command_registry: CommandRegistry = build_default_registry()
        self._running = True
        self._receive_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        """Main REPL loop: connect WS, replay history, read input, stream events."""
        async with self.ws_client:
            self._replay_history()
            print(f"Connected to team {self.team_id}. Type /quit or Ctrl+C to exit.")

            self._receive_task = asyncio.create_task(self._receive_loop())
            try:
                await self._input_loop()
            except KeyboardInterrupt:
                pass
            finally:
                self._running = False
                if self._receive_task is not None:
                    self._receive_task.cancel()
                    try:
                        await self._receive_task
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

            # Try dispatching as a slash command
            if await self.command_registry.dispatch(line, self):
                continue

            # Send message via REST API (run in executor to avoid blocking)
            try:
                await loop.run_in_executor(None, self.client.send_message, self.team_id, line)
            except SystemExit:
                print("Error sending message.", file=sys.stderr)

    async def _receive_loop(self) -> None:
        """Background coroutine: read WebSocket events and print them."""
        while self._running:
            try:
                event = await self.ws_client.receive_event()
                _print_event(event)
            except asyncio.CancelledError:
                raise
            except websockets.exceptions.ConnectionClosed as exc:
                if exc.rcvd is not None and exc.rcvd.code == 4004:
                    print("Error: team not found", file=sys.stderr)
                elif exc.rcvd is not None and exc.rcvd.code not in (1000, 1001):
                    print(f"Connection closed: {exc.rcvd.reason or exc.rcvd.code}", file=sys.stderr)
                break
            except Exception:  # noqa: BLE001
                if self._running:
                    break

    def _replay_history(self) -> None:
        """Fetch and display past events before starting the REPL."""
        try:
            events = self.client.get_events(self.team_id)
        except SystemExit:
            return
        self._display_events(events)

    async def replay_history_async(self) -> None:
        """Async version of _replay_history — uses run_in_executor to avoid blocking."""
        loop = asyncio.get_running_loop()
        try:
            events = await loop.run_in_executor(
                None, self.client.get_events, self.team_id
            )
        except SystemExit:
            return
        self._display_events(events)

    def _display_events(self, events: list[dict[str, Any]]) -> None:
        """Print a list of events, adding a history separator if any were displayed."""
        displayed = False
        for evt in events:
            if _print_event(evt):
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
