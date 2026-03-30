"""Slash command registry and dispatcher for in-session REPL commands."""

from __future__ import annotations

import asyncio
import json as json_mod
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from akgentic.infra.cli.ws_client import WsClient

if TYPE_CHECKING:
    from akgentic.infra.cli.repl import ChatSession

# Type alias for command handlers: async def handler(args: str, session: ChatSession) -> None
CommandHandler = Callable[[str, "ChatSession"], Awaitable[None]]


@dataclass
class SlashCommand:
    """A registered slash command."""

    name: str
    handler: CommandHandler
    help_text: str
    usage: str


class CommandRegistry:
    """Registry and dispatcher for slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(
        self,
        name: str,
        handler: CommandHandler,
        help_text: str,
        usage: str,
    ) -> None:
        """Register a slash command."""
        self._commands[name] = SlashCommand(
            name=name, handler=handler, help_text=help_text, usage=usage
        )

    async def dispatch(self, line: str, session: ChatSession) -> bool:
        """Parse and dispatch a slash command.

        Returns True if a command was handled (or an unknown /command was caught),
        False if the line is not a command (no / prefix).
        """
        stripped = line.strip()
        if not stripped.startswith("/"):
            return False

        parts = stripped.split(maxsplit=1)
        cmd_name = parts[0][1:]  # strip leading /
        args = parts[1] if len(parts) > 1 else ""

        if cmd_name in self._commands:
            await self._commands[cmd_name].handler(args, session)
            return True

        print(f"Unknown command: /{cmd_name}. Type /help for available commands.")
        return True

    @property
    def commands(self) -> dict[str, SlashCommand]:
        """Return registered commands."""
        return self._commands


# -- Built-in commands --


async def _help_handler(args: str, session: ChatSession) -> None:
    """List all registered commands."""
    registry = session.command_registry
    print("Available commands:")
    for cmd in registry.commands.values():
        print(f"  /{cmd.name:<12s} {cmd.help_text}")
        if cmd.usage:
            print(f"  {'':12s}   Usage: {cmd.usage}")


# -- Team inspection commands --


async def _status_handler(args: str, session: ChatSession) -> None:
    """Show team status and agent states."""
    loop = asyncio.get_running_loop()
    try:
        team = await loop.run_in_executor(None, session.client.get_team, session.team_id)
    except SystemExit:
        print("Error fetching team status.", file=sys.stderr)
        return

    print(f"Team: {team.name}")
    print(f"Status: {team.status}")


async def _agents_handler(args: str, session: ChatSession) -> None:
    """List team members with roles and state."""
    loop = asyncio.get_running_loop()
    try:
        events = await loop.run_in_executor(None, session.client.get_events, session.team_id)
    except SystemExit:
        print("Error fetching agents.", file=sys.stderr)
        return

    seen: dict[str, str] = {}
    for e in events:
        evt = e.model_dump().get("event", {})
        if evt.get("__model__", "").endswith("StartMessage"):
            sender = evt.get("sender", {})
            name = sender.get("name", "")
            role = sender.get("role", "")
            if name and name != "orchestrator":
                seen[name] = role

    if not seen:
        print("No agents found.")
        return

    print("Team agents:")
    for name, role in seen.items():
        print(f"  {name:<16s} ({role})")


# -- History command --


async def _history_handler(args: str, session: ChatSession) -> None:
    """Show recent messages."""
    limit = 20
    if args.strip():
        try:
            limit = int(args.strip())
        except ValueError:
            print("Usage: /history [N]  — N must be a positive integer.")
            return
        if limit <= 0:
            print("Usage: /history [N]  — N must be a positive integer.")
            return

    loop = asyncio.get_running_loop()
    try:
        events = await loop.run_in_executor(None, session.client.get_events, session.team_id)
    except SystemExit:
        print("Error fetching history.", file=sys.stderr)
        return

    # Convert EventInfo models to dicts for the rendering pipeline
    event_dicts = [e.model_dump() for e in events]
    displayable = [e for e in event_dicts if _is_displayable(e)]
    for evt in displayable[-limit:]:
        session._render_event(evt)


def _is_displayable(data: dict[str, Any]) -> bool:
    """Check if an event is displayable (SentMessage or ErrorMessage)."""
    event = data.get("event", data)
    if isinstance(event, str):
        try:
            event = json_mod.loads(event)
        except (json_mod.JSONDecodeError, TypeError):
            return False
    model = event.get("__model__", "")
    # Keep in sync with repl._DISPLAY_EVENTS
    return model in {"SentMessage", "ErrorMessage", "EventMessage"}


# -- Workspace commands --


async def _files_handler(args: str, session: ChatSession) -> None:
    """Show workspace file tree."""
    loop = asyncio.get_running_loop()
    try:
        tree = await loop.run_in_executor(None, session.client.workspace_tree, session.team_id)
    except SystemExit:
        print("Error fetching file tree.", file=sys.stderr)
        return

    if not tree.entries:
        print("(empty workspace)")
        return
    for entry in tree.entries:
        prefix = "D " if entry.is_dir else "  "
        suffix = f"  ({entry.size} bytes)" if not entry.is_dir else ""
        print(f"{prefix}{entry.name}{suffix}")


async def _read_handler(args: str, session: ChatSession) -> None:
    """Read a file from the workspace."""
    path = args.strip()
    if not path:
        print("Usage: /read <path>")
        return

    loop = asyncio.get_running_loop()
    try:
        data = await loop.run_in_executor(
            None, session.client.workspace_read, session.team_id, path
        )
    except SystemExit:
        print(f"Error reading file: {path}", file=sys.stderr)
        return

    try:
        print(data.decode("utf-8"), end="")
    except UnicodeDecodeError:
        print(f"(binary file, {len(data)} bytes)")


async def _upload_handler(args: str, session: ChatSession) -> None:
    """Upload a local file to the workspace."""
    local_path = args.strip()
    if not local_path:
        print("Usage: /upload <local_path>")
        return

    loop = asyncio.get_running_loop()
    p = Path(local_path)

    is_file = await loop.run_in_executor(None, p.is_file)
    if not is_file:
        print(f"Error: {local_path} is not a file or does not exist.")
        return

    file_data = await loop.run_in_executor(None, p.read_bytes)
    try:
        result = await loop.run_in_executor(
            None, session.client.workspace_upload, session.team_id, p.name, file_data
        )
    except SystemExit:
        print(f"Error uploading file: {local_path}", file=sys.stderr)
        return

    print(f"Uploaded {result.path} ({result.size} bytes)")


# -- Lifecycle commands --


async def _stop_handler(args: str, session: ChatSession) -> None:
    """Stop the team."""
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, session.client.stop_team, session.team_id)
    except SystemExit:
        print("Error stopping team.", file=sys.stderr)
        return

    print(f"Team {session.team_id} stopped.")


async def _restore_handler(args: str, session: ChatSession) -> None:
    """Restore a stopped team."""
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, session.client.restore_team, session.team_id)
    except SystemExit:
        print("Error restoring team.", file=sys.stderr)
        return

    print(f"Team {session.team_id} restored. Live events resumed.")


# -- Switch command --


async def _switch_handler(args: str, session: ChatSession) -> None:
    """Switch to a different team."""
    new_team_id = args.strip()
    if not new_team_id:
        print("Usage: /switch <team_id>")
        return

    old_team_id = session.team_id
    old_ws = session.ws_client

    # Close current WebSocket
    await old_ws.close()

    # Create new WsClient using stored server_url and api_key
    new_ws = WsClient(
        base_url=session.server_url,
        team_id=new_team_id,
        api_key=session.api_key,
    )

    try:
        await new_ws.connect()
    except (SystemExit, Exception):  # noqa: BLE001
        print(f"Team {new_team_id} not found.", file=sys.stderr)
        # Restore previous connection
        restored = WsClient(
            base_url=session.server_url,
            team_id=old_team_id,
            api_key=session.api_key,
        )
        try:
            await restored.connect()
            session.ws_client = restored
        except Exception:  # noqa: BLE001
            print("Failed to restore previous connection.", file=sys.stderr)
        return

    # Update session state
    session.team_id = new_team_id
    session.ws_client = new_ws

    # Cancel old receive loop before replaying history
    if session._receive_task is not None:
        session._receive_task.cancel()
        try:
            await session._receive_task
        except asyncio.CancelledError:
            pass

    # Replay history for the new team (non-blocking)
    await session.replay_history_async()

    session._receive_task = asyncio.create_task(session._receive_loop())

    print(f"Switched to team {new_team_id}.")


def build_default_registry() -> CommandRegistry:
    """Create a command registry with all built-in commands."""
    registry = CommandRegistry()
    registry.register("help", _help_handler, "Show available commands", "/help")
    registry.register("status", _status_handler, "Show team status", "/status")
    registry.register("agents", _agents_handler, "List team agents", "/agents")
    registry.register("history", _history_handler, "Show recent messages", "/history [N]")
    registry.register("files", _files_handler, "Show workspace files", "/files")
    registry.register("read", _read_handler, "Read a workspace file", "/read <path>")
    registry.register("upload", _upload_handler, "Upload a file to workspace", "/upload <path>")
    registry.register("stop", _stop_handler, "Stop the team", "/stop")
    registry.register("restore", _restore_handler, "Restore a stopped team", "/restore")
    registry.register("switch", _switch_handler, "Switch to another team", "/switch <team_id>")
    return registry
