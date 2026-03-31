"""Interactive chat REPL for team communication."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory

from akgentic.infra.cli.client import ApiClient, ApiError, TeamInfo
from akgentic.infra.cli.commands import CommandRegistry, build_default_registry
from akgentic.infra.cli.connection import ConnectionManager
from akgentic.infra.cli.formatters import OutputFormat
from akgentic.infra.cli.renderers import RichRenderer
from akgentic.infra.cli.ws_client import WsConnectionError

# Event types to display vs skip
_DISPLAY_EVENTS = {"SentMessage", "ErrorMessage", "EventMessage"}

_STOPPED_PAGE_SIZE = 5


def _short_id(team_id: str) -> str:
    """Truncate a UUID to the first 13 characters."""
    return team_id[:13] if len(team_id) > 13 else team_id


class TeamSelector:
    """Interactive team selection / creation flow before entering chat."""

    def __init__(self, client: ApiClient, renderer: RichRenderer) -> None:
        self._client = client
        self._renderer = renderer

    def run(self) -> str | None:
        """Show the startup menu and return the selected/created team_id, or None to quit."""
        while True:
            teams = self._fetch_teams()
            running = [t for t in teams if t.status == "running"]
            stopped = [t for t in teams if t.status == "stopped"]

            self._render_menu(running)
            choice = input("\n  > ").strip()

            result = self._handle_choice(choice, running, stopped)
            if result is not None:
                return result if result != "" else None

    def _render_menu(self, running: list[TeamInfo]) -> None:
        """Display the welcome screen with running teams and catalog."""
        catalog = self._fetch_catalog()
        self._renderer.render_border()
        self._renderer.render_welcome_header()
        if running:
            numbered = [
                (i + 1, t.name, _short_id(t.team_id), t.status)
                for i, t in enumerate(running)
            ]
            self._renderer.render_team_list(numbered, title="Running teams:")
        if catalog:
            self._renderer.render_catalog_list(catalog)
        self._renderer.render_border()
        self._renderer.render_startup_hints(len(running), has_stopped=True)

    def _handle_choice(
        self,
        choice: str,
        running: list[TeamInfo],
        stopped: list[TeamInfo],
    ) -> str | None:
        """Process a user choice. Returns team_id, empty string for quit, or None to loop."""
        if not choice or choice in ("/quit", "q"):
            return ""  # signal quit

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(running):
                return running[idx].team_id
            self._renderer.render_error(f"Invalid selection: {choice}")
            return None

        if choice.startswith("c "):
            return self._handle_create(choice[2:].strip())

        if choice == "s":
            return self._browse_stopped(stopped)

        self._renderer.render_error(f"Unknown choice: {choice}")
        return None

    def _handle_create(self, entry_id: str) -> str | None:
        """Create a team from a catalog entry. Returns team_id or None on failure."""
        if not entry_id:
            self._renderer.render_error("Usage: c <catalog_entry>")
            return None
        try:
            team = self._client.create_team(entry_id)
            return team.team_id
        except ApiError:
            self._renderer.render_error(f"Failed to create team from '{entry_id}'")
            return None

    def _fetch_teams(self) -> list[TeamInfo]:
        """Fetch all teams from the server."""
        try:
            return self._client.list_teams()
        except ApiError:
            return []

    def _fetch_catalog(self) -> list[tuple[str, str]]:
        """Fetch catalog entries as (id, description) tuples."""
        try:
            entries = self._client.list_catalog_teams()
            return [(e.id, e.description) for e in entries]
        except (ApiError, Exception):  # noqa: BLE001
            return []

    def _browse_stopped(self, stopped: list[TeamInfo]) -> str | None:
        """Paginated browser for stopped teams. Returns team_id or None to go back."""
        if not stopped:
            self._renderer.render_system_message("No stopped teams.")
            return None

        page = 0
        while True:
            start = page * _STOPPED_PAGE_SIZE
            page_teams = stopped[start : start + _STOPPED_PAGE_SIZE]
            if not page_teams:
                page = 0
                continue

            has_next = start + _STOPPED_PAGE_SIZE < len(stopped)

            self._renderer.render_border()
            numbered = [
                (i + 1, t.name, _short_id(t.team_id), t.status)
                for i, t in enumerate(page_teams)
            ]
            title = f"Stopped teams ({len(stopped)} total):"
            self._renderer.render_team_list(numbered, title=title)
            self._renderer.render_border()
            self._renderer.render_pagination_hints(has_next)

            choice = input("\n  > ").strip()

            if not choice or choice == "b":
                return None

            if choice == "n" and has_next:
                page += 1
                continue

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(page_teams):
                    team = page_teams[idx]
                    try:
                        self._client.restore_team(team.team_id)
                        return team.team_id
                    except ApiError:
                        self._renderer.render_error(f"Failed to restore team {team.name}")
                        continue
                self._renderer.render_error(f"Invalid selection: {choice}")
                continue

            self._renderer.render_error(f"Unknown choice: {choice}")


class ChatSession:
    """REPL session combining REST message sending with WebSocket event streaming."""

    def __init__(
        self,
        client: ApiClient,
        conn: ConnectionManager,
        team_id: str,
        fmt: OutputFormat,
        *,
        server_url: str = "http://localhost:8000",
        api_key: str | None = None,
        renderer: RichRenderer | None = None,
    ) -> None:
        self.client = client
        self.conn = conn
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
        self._team_name: str = ""
        self._team_status: str = ""
        self._prompt_session: PromptSession[str] = PromptSession(
            history=InMemoryHistory(),
            completer=_SlashCompleter(self.command_registry),
        )

    def _fetch_team_info(self) -> None:
        """Fetch and cache team name and status for the status bar."""
        try:
            team = self.client.get_team(self.team_id)
            self._team_name = team.name
            self._team_status = team.status
        except ApiError:
            self._team_name = "(unknown)"
            self._team_status = "?"

    async def run(self) -> None:
        """Main REPL loop: connect WS, replay history, read input, stream events."""
        async with self.conn:
            self._fetch_team_info()
            self.renderer.render_border()
            self._replay_history()

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
                self.renderer.render_border()
                self.renderer.render_system_message("Session closed.")

    def _read_input(self) -> str:
        """Render bottom border, prompt for input, then status bar below."""
        self.renderer.render_border()
        line = self._prompt_session.prompt("> ")
        self.renderer.render_border()
        self.renderer.render_status_bar(
            self._team_name or "(unknown)",
            self.team_id,
            self._team_status or "?",
        )
        return line

    async def _input_loop(self) -> None:
        """Read user input and send messages."""
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                line = await loop.run_in_executor(None, self._read_input)
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
                    await loop.run_in_executor(
                        None, self.client.human_input, self.team_id, line, reply_id
                    )
                    # Clear only after successful send
                    self._pending_reply_id = None
                    self._pending_agent_name = None
                except ApiError:
                    self.renderer.render_error("Error sending reply. Try again.")
                continue

            # Send message via REST API (run in executor to avoid blocking)
            try:
                await loop.run_in_executor(None, self.client.send_message, self.team_id, line)
            except ApiError:
                self.renderer.render_error("Error sending message.")

    async def _receive_loop(self) -> None:
        """Background coroutine: read WebSocket events and render them."""
        while self._running:
            try:
                event = await self.conn.receive_event()
                self._render_event(event)
            except asyncio.CancelledError:
                raise
            except WsConnectionError as exc:
                self.renderer.render_error(f"Connection lost: {exc.reason}")
                break

    def _replay_history(self) -> None:
        """Fetch and display past events before starting the REPL."""
        try:
            events = self.client.get_events(self.team_id)
        except ApiError:
            return
        self._display_events([e.model_dump() for e in events])

    async def replay_history_async(self) -> None:
        """Async version of _replay_history — uses run_in_executor to avoid blocking."""
        loop = asyncio.get_running_loop()
        try:
            events = await loop.run_in_executor(None, self.client.get_events, self.team_id)
        except ApiError:
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
