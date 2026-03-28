"""Typer CLI application for akgentic-infra."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer

from akgentic.infra.cli.client import ApiClient
from akgentic.infra.cli.formatters import OutputFormat, format_output
from akgentic.infra.cli.repl import ChatSession
from akgentic.infra.cli.ws_client import WsClient

app = typer.Typer(name="ak-infra", help="Akgentic Infrastructure CLI")
team_app = typer.Typer(name="team", help="Manage agent teams")
workspace_app = typer.Typer(name="workspace", help="Manage team workspace files")

app.add_typer(team_app, name="team")
app.add_typer(workspace_app, name="workspace")


# -- shared state --


class _State:
    """Holds resolved global options for commands."""

    client: ApiClient
    fmt: OutputFormat
    server: str
    api_key: str | None


_state = _State()


def _print(data: object, columns: list[str] | None = None) -> None:
    """Print formatted output using the current output format."""
    typer.echo(format_output(data, _state.fmt, columns))


# -- global callback --


@app.callback()
def main(
    server: Annotated[
        str, typer.Option("--server", help="Server base URL")
    ] = "http://localhost:8000",
    api_key: Annotated[
        str | None, typer.Option("--api-key", help="API key for authentication")
    ] = None,
    fmt: Annotated[
        OutputFormat, typer.Option("--format", help="Output format")
    ] = OutputFormat.table,
) -> None:
    """Akgentic Infrastructure CLI — manage teams, messaging, and workspace."""
    _state.client = ApiClient(base_url=server, api_key=api_key)
    _state.fmt = fmt
    _state.server = server
    _state.api_key = api_key


# -- team commands --

_team_columns = ["team_id", "name", "status", "created_at"]
_team_detail_columns = ["team_id", "name", "status", "user_id", "created_at", "updated_at"]


@team_app.command("list")
def team_list() -> None:
    """List all teams."""
    teams = _state.client.list_teams()
    _print(teams, _team_columns)


@team_app.command("get")
def team_get(team_id: str) -> None:
    """Show team detail."""
    team = _state.client.get_team(team_id)
    _print(team, _team_detail_columns)


@team_app.command("create")
def team_create(catalog_entry: str) -> None:
    """Create a team from a catalog entry."""
    team = _state.client.create_team(catalog_entry)
    _print(team, _team_detail_columns)


@team_app.command("delete")
def team_delete(team_id: str) -> None:
    """Delete a team."""
    _state.client.delete_team(team_id)
    if _state.fmt == OutputFormat.table:
        typer.echo(f"Team {team_id} deleted.")
    else:
        _print({"team_id": team_id, "status": "deleted"})


@team_app.command("restore")
def team_restore(team_id: str) -> None:
    """Restore a stopped team."""
    team = _state.client.restore_team(team_id)
    _print(team, _team_detail_columns)


@team_app.command("events")
def team_events(team_id: str) -> None:
    """Show team events."""
    events = _state.client.get_events(team_id)
    _print(events, ["sequence", "timestamp", "event"])


# -- top-level message / reply --


@app.command("message")
def message(team_id: str, content: str) -> None:
    """Send a message to a team (non-interactive)."""
    _state.client.send_message(team_id, content)
    if _state.fmt == OutputFormat.table:
        typer.echo("Message sent.")
    else:
        _print({"team_id": team_id, "status": "sent"})


@app.command("reply")
def reply(
    team_id: str,
    content: str,
    message_id: Annotated[str, typer.Option("--message-id", help="Original message ID")],
) -> None:
    """Reply with human input to an agent request."""
    _state.client.human_input(team_id, content, message_id)
    if _state.fmt == OutputFormat.table:
        typer.echo("Reply sent.")
    else:
        _print({"team_id": team_id, "message_id": message_id, "status": "sent"})


# -- chat --


@app.command("chat")
def chat(
    team_id: Annotated[str, typer.Argument(help="Team ID to chat with")],
    create: Annotated[
        str | None, typer.Option("--create", help="Create team from catalog entry first")
    ] = None,
) -> None:
    """Interactive chat REPL — connect to a team via WebSocket."""
    if create is not None:
        team = _state.client.create_team(create)
        team_id = str(team["team_id"])
        typer.echo(f"Created team {team_id}")

    ws = WsClient(
        base_url=_state.server,
        team_id=team_id,
        api_key=_state.api_key,
    )
    session = ChatSession(_state.client, ws, team_id, _state.fmt)
    asyncio.run(session.run())


# -- workspace commands --


@workspace_app.command("tree")
def workspace_tree(team_id: str) -> None:
    """Show workspace file tree."""
    tree: dict[str, Any] = _state.client.workspace_tree(team_id)
    if _state.fmt != OutputFormat.table:
        _print(tree)
        return
    entries: list[dict[str, Any]] = tree.get("entries", [])
    if not entries:
        typer.echo("(empty workspace)")
        return
    for entry in entries:
        prefix = "📁 " if entry.get("is_dir") else "   "
        name = entry.get("name", "")
        size = entry.get("size", 0)
        suffix = f"  ({size} bytes)" if not entry.get("is_dir") else ""
        typer.echo(f"{prefix}{name}{suffix}")


@workspace_app.command("read")
def workspace_read(team_id: str, path: str) -> None:
    """Read a file from the workspace."""
    data = _state.client.workspace_read(team_id, path)
    try:
        typer.echo(data.decode("utf-8"), nl=False)
    except UnicodeDecodeError:
        typer.echo(f"(binary file, {len(data)} bytes)")


@workspace_app.command("upload")
def workspace_upload(team_id: str, local_path: str) -> None:
    """Upload a local file to the workspace."""
    p = Path(local_path)
    if not p.is_file():
        typer.echo(f"Error: {local_path} is not a file", err=True)
        raise typer.Exit(code=1)
    file_data = p.read_bytes()
    result: dict[str, Any] = _state.client.workspace_upload(team_id, p.name, file_data)
    path = result.get("path", p.name)
    size = result.get("size", len(file_data))
    typer.echo(f"Uploaded {path} ({size} bytes)")
