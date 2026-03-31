"""Interactive chat REPL for team communication."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import websockets.exceptions
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory

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
        self._pending_reply_id: str | None = None
        self._pending_agent_name: str | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._prompt_session: PromptSession[str] = PromptSession(
            history=InMemoryHistory(),
            completer=_SlashCompleter(self.command_registry),
        )

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

    def _get_prompt(self) -> str:
        """Return the current input prompt, reflecting pending reply state."""
        if self._pending_reply_id:
            name = self._pending_agent_name or "Agent"
            return f"Reply to {name}: "
        return "You: "

    async def _input_loop(self) -> None:
        """Read user input and send messages."""
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                line = await loop.run_in_executor(
                    None, self._prompt_session.prompt, self._get_prompt()
                )
            except (EOFError, KeyboardInterrupt):
                break

            if not line.strip():
                continue

            if line.strip() == "/quit":
                break

            # Try dispatching as a slash command
            if await self.command_registry.dispatch(line, self):
                continue

            # Route as human-input reply if pending
            if self._pending_reply_id:
                try:
                    reply_id = self._pending_reply_id
                    self._pending_reply_id = None
                    self._pending_agent_name = None
                    await loop.run_in_executor(
                        None, self.client.human_input, self.team_id, line, reply_id
                    )
                except SystemExit:
                    self.renderer.render_error("Error sending reply.")
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
        self._display_events([e.model_dump() for e in events])

    async def replay_history_async(self) -> None:
        """Async version of _replay_history — uses run_in_executor to avoid blocking."""
        loop = asyncio.get_running_loop()
        try:
            events = await loop.run_in_executor(None, self.client.get_events, self.team_id)
        except SystemExit:
            return
        self._display_events([e.model_dump() for e in events])

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

        def _set_pending(message_id: str, agent_name: str) -> None:
            self._pending_reply_id = message_id
            self._pending_agent_name = agent_name

        return _render_event_impl(data, self.renderer, on_human_input=_set_pending)


class _SlashCompleter(Completer):
    """Auto-complete slash commands from the command registry."""

    def __init__(self, registry: CommandRegistry) -> None:
        self._registry = registry

    def get_completions(
        self, document: Any, complete_event: Any
    ) -> Any:  # noqa: ANN401
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        partial = text[1:]  # strip leading /
        for cmd in self._registry.commands.values():
            if cmd.name.startswith(partial):
                yield Completion(
                    f"/{cmd.name}",
                    start_position=-len(text),
                    display=f"/{cmd.name}",
                    display_meta=cmd.help_text,
                )


def _render_event_impl(
    data: dict[str, Any],
    renderer: RichRenderer,
    on_human_input: Callable[[str, str], None] | None = None,
) -> bool:
    """Shared event rendering logic used by both ChatSession and _print_event."""
    event = data.get("event", data)
    if isinstance(event, str):
        try:
            event = json.loads(event)
        except (json.JSONDecodeError, TypeError):
            return False

    model = event.get("__model__", "")
    # Match short suffix (e.g. "SentMessage") from fully qualified model names
    short_model = model.rsplit(".", 1)[-1] if model else ""
    if short_model not in _DISPLAY_EVENTS:
        return False

    if short_model == "ErrorMessage":
        content = event.get("exception_value", event.get("content", event.get("error", "")))
        renderer.render_error(str(content))
        return True

    if short_model == "EventMessage":
        return _render_event_message_impl(
            event, renderer, on_human_input=on_human_input, outer_data=data
        )

    # SentMessage — agent response; content lives inside nested "message"
    raw_sender = event.get("sender", "agent")
    sender = raw_sender.get("name", "agent") if isinstance(raw_sender, dict) else str(raw_sender)
    message = event.get("message", {})
    content = message.get("content", "") if isinstance(message, dict) else ""
    if content:
        renderer.render_agent_message(sender, str(content))
        return True
    return False


def _render_tool_call(nested: dict[str, Any], renderer: RichRenderer) -> None:
    """Extract and render a tool call event."""
    tool_name = str(nested["tool_name"])
    raw_input = nested.get("arguments", nested.get("args", nested.get("input", "")))
    if isinstance(raw_input, dict | list):
        tool_input = json.dumps(raw_input, indent=2)
    else:
        tool_input = str(raw_input)
    raw_output = nested.get("result")
    if raw_output is None:
        tool_output = None
    elif isinstance(raw_output, dict | list):
        tool_output = json.dumps(raw_output, indent=2)
    else:
        tool_output = str(raw_output)
    renderer.render_tool_call(tool_name, tool_input, tool_output)


def _notify_human_input(
    outer_data: dict[str, Any],
    on_human_input: Callable[[str, str], None],
) -> None:
    """Extract message_id and agent_name from outer event data and invoke callback."""
    raw_id = outer_data.get("id")
    if not raw_id:
        return
    message_id = str(raw_id)
    raw_sender = outer_data.get("sender", "Agent")
    if isinstance(raw_sender, dict):
        agent_name = str(raw_sender.get("name", "Agent"))
    else:
        agent_name = str(raw_sender)
    on_human_input(message_id, agent_name)


def _render_event_message_impl(
    event: dict[str, Any],
    renderer: RichRenderer,
    on_human_input: Callable[[str, str], None] | None = None,
    outer_data: dict[str, Any] | None = None,
) -> bool:
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

    if "tool_name" in nested:
        _render_tool_call(nested, renderer)
        return True

    if "HumanInput" in nested_model or "RequestInput" in nested_model:
        prompt_text = str(nested.get("prompt", nested.get("content", "Input requested")))
        renderer.render_human_input_request(prompt_text)
        if on_human_input is not None and outer_data is not None:
            _notify_human_input(outer_data, on_human_input)
        return True

    return False


def _print_event(data: dict[str, Any]) -> bool:
    """Backward-compatible module-level event printer.

    Delegates to the shared rendering logic with a default RichRenderer.
    """
    return _render_event_impl(data, _default_renderer)


_default_renderer = RichRenderer()
