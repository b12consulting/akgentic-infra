"""Interactive chat REPL for team communication."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import websockets.exceptions

from akgentic.infra.cli.client import ApiClient
from akgentic.infra.cli.commands import CommandRegistry, build_default_registry
from akgentic.infra.cli.formatters import OutputFormat
from akgentic.infra.cli.renderers import RichRenderer
from akgentic.infra.cli.ws_client import WsClient

# Event types to display vs skip
_DISPLAY_EVENTS = {"SentMessage", "ErrorMessage", "EventMessage"}


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
        renderer: RichRenderer | None = None,
    ) -> None:
        self.client = client
        self.ws_client = ws_client
        self.team_id = team_id
        self.fmt = fmt
        self.server_url = server_url
        self.api_key = api_key
        self.renderer = renderer or RichRenderer()
        self.command_registry: CommandRegistry = build_default_registry()
        self._running = True
        self._receive_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        """Main REPL loop: connect WS, replay history, read input, stream events."""
        async with self.ws_client:
            self._replay_history()
            self.renderer.render_system_message(
                f"Connected to team {self.team_id}. Type /quit or Ctrl+C to exit."
            )

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
                self.renderer.render_system_message("Session closed.")

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
                self.renderer.render_error("Error sending message.")

    async def _receive_loop(self) -> None:
        """Background coroutine: read WebSocket events and render them."""
        while self._running:
            try:
                event = await self.ws_client.receive_event()
                self._render_event(event)
            except asyncio.CancelledError:
                raise
            except websockets.exceptions.ConnectionClosed as exc:
                if exc.rcvd is not None and exc.rcvd.code == 4004:
                    self.renderer.render_error("team not found")
                elif exc.rcvd is not None and exc.rcvd.code not in (1000, 1001):
                    self.renderer.render_system_message(
                        f"Connection closed: {exc.rcvd.reason or exc.rcvd.code}"
                    )
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
            events = await loop.run_in_executor(None, self.client.get_events, self.team_id)
        except SystemExit:
            return
        self._display_events(events)

    def _display_events(self, events: list[dict[str, Any]]) -> None:
        """Render a list of events, adding a history separator if any were displayed."""
        displayed = False
        for evt in events:
            if self._render_event(evt):
                displayed = True
        if displayed:
            self.renderer.render_history_separator()

    def _render_event(self, data: dict[str, Any]) -> bool:
        """Format and render a single event. Returns True if something was rendered."""
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
            self.renderer.render_error(str(content))
            return True

        if model == "EventMessage":
            return self._render_event_message(event)

        # SentMessage — agent response
        sender = event.get("sender", "agent")
        content = event.get("content", "")
        if content:
            self.renderer.render_agent_message(str(sender), str(content))
            return True
        return False

    def _render_event_message(self, event: dict[str, Any]) -> bool:
        """Handle EventMessage — inspect nested event for tool calls / human input."""
        nested = event.get("event", {})
        if isinstance(nested, str):
            try:
                nested = json.loads(nested)
            except (json.JSONDecodeError, TypeError):
                return False

        if not isinstance(nested, dict):
            return False

        nested_model = nested.get("__model__", "")

        # Tool call events
        if "tool_name" in nested:
            tool_name = str(nested["tool_name"])
            tool_input = str(nested.get("args", nested.get("input", "")))
            tool_output = nested.get("result")
            self.renderer.render_tool_call(
                tool_name,
                tool_input,
                str(tool_output) if tool_output is not None else None,
            )
            return True

        # Human input request
        if "HumanInput" in nested_model or "RequestInput" in nested_model:
            prompt_text = str(nested.get("prompt", nested.get("content", "Input requested")))
            self.renderer.render_human_input_request(prompt_text)
            return True

        return False


def _read_input(prompt: str) -> str:
    """Read a line from stdin (used via run_in_executor)."""
    return input(prompt)


def _print_event(data: dict[str, Any]) -> bool:
    """Backward-compatible module-level event printer.

    Delegates to a default RichRenderer for use by external callers.
    """
    return _default_renderer_session._render_event(data)


class _DefaultRendererSession:
    """Minimal wrapper providing _render_event with a default RichRenderer."""

    def __init__(self) -> None:
        self.renderer = RichRenderer()

    def _render_event(self, data: dict[str, Any]) -> bool:
        """Format and render a single event using the default renderer."""
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
            self.renderer.render_error(str(content))
            return True

        if model == "SentMessage":
            sender = event.get("sender", "agent")
            content = event.get("content", "")
            if content:
                self.renderer.render_agent_message(str(sender), str(content))
                return True
            return False

        return False


_default_renderer_session = _DefaultRendererSession()
