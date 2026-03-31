"""Tests for in-session slash command registry, handlers, and ChatSession integration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from akgentic.infra.cli.client import (
    EventInfo,
    TeamInfo,
    WorkspaceEntry,
    WorkspaceTreeInfo,
    WorkspaceUploadInfo,
)
from akgentic.infra.cli.commands import (
    CommandRegistry,
    _agents_handler,
    _create_handler,
    _delete_handler,
    _events_handler,
    _files_handler,
    _help_handler,
    _history_handler,
    _info_handler,
    _read_handler,
    _restore_handler,
    _stop_handler,
    _switch_handler,
    _teams_handler,
    _upload_handler,
    build_default_registry,
)
from akgentic.infra.cli.formatters import OutputFormat
from akgentic.infra.cli.repl import ChatSession
from tests.fixtures.events import _make_proxy, make_sent_message, make_start_message

from .conftest import captured_renderer as _captured_renderer
from .conftest import make_session as _make_session
from .conftest import mock_ws as _mock_ws

_PROMPT_PATH = "prompt_toolkit.PromptSession.prompt"

# -- Helpers --

def _mock_client(**overrides: Any) -> MagicMock:
    """Build a mock ApiClient with in-session-specific defaults."""
    mock = MagicMock()
    mock.get_events.return_value = []
    mock.send_message.return_value = None
    mock.get_team.return_value = TeamInfo(
        team_id="t1",
        name="Test Team",
        status="running",
        user_id="user-1",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )
    mock.workspace_tree.return_value = WorkspaceTreeInfo(
        team_id="t1",
        path="/",
        entries=[
            WorkspaceEntry(name="docs", is_dir=True, size=0),
            WorkspaceEntry(name="readme.md", is_dir=False, size=42),
        ],
    )
    mock.workspace_read.return_value = b"file content here"
    mock.workspace_upload.return_value = WorkspaceUploadInfo(path="test.txt", size=5)
    mock.stop_team.return_value = None
    mock.delete_team.return_value = None
    mock.restore_team.return_value = TeamInfo(
        team_id="t1",
        name="Test Team",
        status="running",
        user_id="user-1",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )
    for k, v in overrides.items():
        setattr(mock, k, v)
    return mock


# =============================================================================
# Task 8.1: Test CommandRegistry
# =============================================================================


class TestCommandRegistryDispatch:
    """Test dispatch returns True for registered commands, False for non-commands."""

    async def test_dispatch_returns_true_for_registered_command(self) -> None:
        registry = CommandRegistry()
        handler = AsyncMock()
        registry.register("test", handler, "Test command", "/test")
        session = _make_session()

        result = await registry.dispatch("/test", session)

        assert result is True
        handler.assert_called_once_with("", session)

    async def test_dispatch_returns_false_for_non_command(self) -> None:
        registry = CommandRegistry()
        session = _make_session()

        result = await registry.dispatch("hello world", session)

        assert result is False

    async def test_dispatch_passes_args_to_handler(self) -> None:
        registry = CommandRegistry()
        handler = AsyncMock()
        registry.register("test", handler, "Test", "/test <arg>")
        session = _make_session()

        await registry.dispatch("/test some args here", session)

        handler.assert_called_once_with("some args here", session)

    async def test_unknown_command_prints_error_returns_true(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        registry = CommandRegistry()
        session = _make_session()

        result = await registry.dispatch("/xxx", session)

        assert result is True
        out = capsys.readouterr().out
        assert "Unknown command: /xxx" in out
        assert "/help" in out

    async def test_dispatch_strips_whitespace(self) -> None:
        registry = CommandRegistry()
        handler = AsyncMock()
        registry.register("test", handler, "Test", "/test")
        session = _make_session()

        result = await registry.dispatch("  /test  ", session)

        assert result is True
        handler.assert_called_once()


class TestHelpCommand:
    """Test /help lists all registered commands."""

    async def test_help_lists_all_commands(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = _make_session()
        registry = build_default_registry()
        session.command_registry = registry

        await _help_handler("", session)

        out = capsys.readouterr().out
        assert "/teams" in out
        assert "/create" in out
        assert "/delete" in out
        assert "/info" in out
        assert "/events" in out
        assert "/agents" in out
        assert "/history" in out
        assert "/files" in out
        assert "/read" in out
        assert "/upload" in out
        assert "/stop" in out
        assert "/restore" in out
        assert "/switch" in out
        assert "/help" in out
        assert "/status" not in out


# =============================================================================
# Task 8.2: Test each command handler in isolation
# =============================================================================


class TestTeamsHandler:
    async def test_lists_teams(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.list_teams.return_value = [
            TeamInfo(
                team_id="t1", name="Team A", status="running",
                user_id="u1", created_at="2026-01-01", updated_at="2026-01-01",
            ),
            TeamInfo(
                team_id="t2", name="Team B", status="stopped",
                user_id="u1", created_at="2026-01-01", updated_at="2026-01-01",
            ),
        ]
        session = _make_session(client=client)

        await _teams_handler("", session)

        out = capsys.readouterr().out
        assert "Team A" in out
        assert "Team B" in out
        assert "running" in out
        assert "stopped" in out

    async def test_highlights_current_team(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.list_teams.return_value = [
            TeamInfo(
                team_id="t1", name="Team A", status="running",
                user_id="u1", created_at="2026-01-01", updated_at="2026-01-01",
            ),
            TeamInfo(
                team_id="t2", name="Team B", status="stopped",
                user_id="u1", created_at="2026-01-01", updated_at="2026-01-01",
            ),
        ]
        session = _make_session(client=client)

        await _teams_handler("", session)

        out = capsys.readouterr().out
        lines = out.strip().split("\n")
        # The line with t1 should have (current) marker
        t1_line = [line for line in lines if "t1" in line][0]
        assert "(current)" in t1_line
        t2_line = [line for line in lines if "t2" in line][0]
        assert "(current)" not in t2_line

    async def test_empty_teams_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.list_teams.return_value = []
        session = _make_session(client=client)

        await _teams_handler("", session)

        out = capsys.readouterr().out
        assert "No teams found" in out

    async def test_handles_api_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.list_teams.side_effect = SystemExit(1)
        session = _make_session(client=client)

        await _teams_handler("", session)

        err = capsys.readouterr().err
        assert "Error" in err


class TestCreateHandler:
    async def test_creates_and_switches(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.create_team.return_value = TeamInfo(
            team_id="new-t", name="New Team", status="running",
            user_id="u1", created_at="2026-01-01", updated_at="2026-01-01",
        )
        session = _make_session(client=client)
        session._receive_task = asyncio.create_task(asyncio.sleep(100))

        new_ws_mock = _mock_ws()
        with patch("akgentic.infra.cli.commands.WsClient", return_value=new_ws_mock):
            await _create_handler("my-catalog-entry", session)

        out = capsys.readouterr().out
        assert "Created team: New Team" in out
        assert "new-t" in out
        client.create_team.assert_called_once_with("my-catalog-entry")
        # Should have auto-switched
        assert session.team_id == "new-t"

    async def test_missing_arg_shows_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = _make_session()

        await _create_handler("", session)

        out = capsys.readouterr().out
        assert "Usage" in out

    async def test_handles_api_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.create_team.side_effect = SystemExit(1)
        session = _make_session(client=client)

        await _create_handler("my-entry", session)

        err = capsys.readouterr().err
        assert "Error" in err


class TestDeleteHandler:
    async def test_delete_confirms_and_proceeds(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = _mock_client()
        session = _make_session(client=client)

        with patch("akgentic.infra.cli.commands.builtins.input", return_value="y"):
            await _delete_handler("", session)

        out = capsys.readouterr().out
        assert "deleted" in out.lower()
        client.delete_team.assert_called_once_with("t1")

    async def test_delete_aborts_on_no(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        session = _make_session(client=client)

        with patch("akgentic.infra.cli.commands.builtins.input", return_value="n"):
            await _delete_handler("", session)

        out = capsys.readouterr().out
        assert "Aborted" in out
        client.delete_team.assert_not_called()

    async def test_defaults_to_current_team(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = _mock_client()
        session = _make_session(client=client)

        with patch("akgentic.infra.cli.commands.builtins.input", return_value="y"):
            await _delete_handler("", session)

        client.get_team.assert_called_once_with("t1")
        client.delete_team.assert_called_once_with("t1")

    async def test_delete_current_team_shows_switch_msg(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = _mock_client()
        session = _make_session(client=client)

        with patch("akgentic.infra.cli.commands.builtins.input", return_value="y"):
            await _delete_handler("", session)

        out = capsys.readouterr().out
        assert "/switch" in out or "/teams" in out

    async def test_handles_api_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.get_team.side_effect = SystemExit(1)
        session = _make_session(client=client)

        await _delete_handler("some-id", session)

        err = capsys.readouterr().err
        assert "Error" in err


class TestInfoHandler:
    async def test_info_current_team(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        session = _make_session(client=client)

        await _info_handler("", session)

        out = capsys.readouterr().out
        assert "Test Team" in out
        assert "running" in out
        assert "user-1" in out
        client.get_team.assert_called_once_with("t1")

    async def test_info_specific_team(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        session = _make_session(client=client)

        await _info_handler("t2", session)

        client.get_team.assert_called_once_with("t2")

    async def test_handles_api_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.get_team.side_effect = SystemExit(1)
        session = _make_session(client=client)

        await _info_handler("", session)

        err = capsys.readouterr().err
        assert "Error" in err


class TestEventsHandler:
    async def test_default_limit(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        events = [
            EventInfo(
                team_id="t1", sequence=i,
                event={"type": f"event-{i}"}, timestamp="2026-01-01T00:00:00",
            )
            for i in range(25)
        ]
        client.get_events.return_value = events
        session = _make_session(client=client)

        await _events_handler("", session)

        out = capsys.readouterr().out
        # Default limit 20, so first 5 should be missing
        assert "event-0" not in out
        assert "event-4" not in out
        assert "event-5" in out
        assert "event-24" in out

    async def test_custom_limit(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        events = [
            EventInfo(
                team_id="t1", sequence=i,
                event={"type": f"event-{i}"}, timestamp="2026-01-01T00:00:00",
            )
            for i in range(25)
        ]
        client.get_events.return_value = events
        session = _make_session(client=client)

        await _events_handler("10", session)

        out = capsys.readouterr().out
        assert "event-14" not in out
        assert "event-15" in out
        assert "event-24" in out

    async def test_invalid_limit(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = _make_session()

        await _events_handler("abc", session)

        out = capsys.readouterr().out
        assert "Usage" in out

    async def test_zero_limit(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = _make_session()

        await _events_handler("0", session)

        out = capsys.readouterr().out
        assert "Usage" in out

    async def test_negative_limit(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = _make_session()

        await _events_handler("-3", session)

        out = capsys.readouterr().out
        assert "Usage" in out

    async def test_no_events(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.get_events.return_value = []
        session = _make_session(client=client)

        await _events_handler("", session)

        out = capsys.readouterr().out
        # No events = no output (no crash)
        assert out == ""

    async def test_handles_api_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.get_events.side_effect = SystemExit(1)
        session = _make_session(client=client)

        await _events_handler("", session)

        err = capsys.readouterr().err
        assert "Error" in err


class TestAgentsHandler:
    async def test_shows_agents_from_events(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.get_events.return_value = [
            EventInfo(
                team_id="t1",
                sequence=1,
                timestamp="2026-01-01T00:00:00",
                event=make_start_message(
                    sender=_make_proxy(name="@Manager", role="Manager"),
                ),
            ),
        ]
        session = _make_session(client=client)

        await _agents_handler("", session)

        out = capsys.readouterr().out
        assert "@Manager" in out
        assert "Manager" in out

    async def test_no_agents_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        session = _make_session(client=client)

        await _agents_handler("", session)

        out = capsys.readouterr().out
        assert "No agents found" in out

    async def test_handles_api_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.get_events.side_effect = SystemExit(1)
        session = _make_session(client=client)

        await _agents_handler("", session)

        err = capsys.readouterr().err
        assert "Error" in err


class TestHistoryHandler:
    async def test_shows_history_default_limit(self) -> None:
        events = [
            EventInfo(
                team_id="t1",
                sequence=i,
                event=make_sent_message(content=f"msg-{i}"),
                timestamp="2026-01-01T00:00:00",
            )
            for i in range(25)
        ]
        client = _mock_client()
        client.get_events.return_value = events
        renderer, buf = _captured_renderer()
        session = _make_session(client=client, renderer=renderer)

        await _history_handler("", session)

        out = buf.getvalue()
        # Default limit 20, so first 5 should be missing
        assert "msg-0" not in out
        assert "msg-4" not in out
        assert "msg-5" in out
        assert "msg-24" in out

    async def test_custom_limit(self) -> None:
        events = [
            EventInfo(
                team_id="t1",
                sequence=i,
                event=make_sent_message(content=f"msg-{i}"),
                timestamp="2026-01-01T00:00:00",
            )
            for i in range(25)
        ]
        client = _mock_client()
        client.get_events.return_value = events
        renderer, buf = _captured_renderer()
        session = _make_session(client=client, renderer=renderer)

        await _history_handler("10", session)

        out = buf.getvalue()
        assert "msg-14" not in out
        assert "msg-15" in out
        assert "msg-24" in out

    async def test_invalid_limit(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = _make_session()

        await _history_handler("abc", session)

        out = capsys.readouterr().out
        assert "Usage" in out

    async def test_zero_limit(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = _make_session()

        await _history_handler("0", session)

        out = capsys.readouterr().out
        assert "Usage" in out

    async def test_negative_limit(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = _make_session()

        await _history_handler("-5", session)

        out = capsys.readouterr().out
        assert "Usage" in out

    async def test_filters_non_displayable(self) -> None:
        events = [
            EventInfo(
                team_id="t1",
                sequence=1,
                # StateChangedMessage has no factory -- acceptable per story notes
                event={"__model__": "StateChangedMessage", "state": "running"},
                timestamp="2026-01-01T00:00:00",
            ),
            EventInfo(
                team_id="t1",
                sequence=2,
                event=make_sent_message(content="visible"),
                timestamp="2026-01-01T00:00:00",
            ),
        ]
        client = _mock_client()
        client.get_events.return_value = events
        renderer, buf = _captured_renderer()
        session = _make_session(client=client, renderer=renderer)

        await _history_handler("", session)

        out = buf.getvalue()
        assert "visible" in out
        assert "StateChanged" not in out

    async def test_handles_api_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.get_events.side_effect = SystemExit(1)
        session = _make_session(client=client)

        await _history_handler("", session)

        err = capsys.readouterr().err
        assert "Error" in err


class TestFilesHandler:
    async def test_shows_file_tree(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        session = _make_session(client=client)

        await _files_handler("", session)

        out = capsys.readouterr().out
        assert "docs" in out
        assert "readme.md" in out
        assert "42 bytes" in out

    async def test_empty_workspace(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.workspace_tree.return_value = WorkspaceTreeInfo(team_id="t1", path="/", entries=[])
        session = _make_session(client=client)

        await _files_handler("", session)

        out = capsys.readouterr().out
        assert "empty" in out.lower()

    async def test_handles_api_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.workspace_tree.side_effect = SystemExit(1)
        session = _make_session(client=client)

        await _files_handler("", session)

        err = capsys.readouterr().err
        assert "Error" in err


class TestReadHandler:
    async def test_reads_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        session = _make_session(client=client)

        await _read_handler("readme.md", session)

        out = capsys.readouterr().out
        assert "file content here" in out

    async def test_no_arg_shows_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = _make_session()

        await _read_handler("", session)

        out = capsys.readouterr().out
        assert "Usage" in out

    async def test_binary_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.workspace_read.return_value = bytes(range(256))
        session = _make_session(client=client)

        await _read_handler("binary.bin", session)

        out = capsys.readouterr().out
        assert "binary" in out.lower() or "bytes" in out.lower()

    async def test_handles_api_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.workspace_read.side_effect = SystemExit(1)
        session = _make_session(client=client)

        await _read_handler("test.txt", session)

        err = capsys.readouterr().err
        assert "Error" in err


class TestUploadHandler:
    async def test_uploads_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello")
        client = _mock_client()
        session = _make_session(client=client)

        await _upload_handler(str(f), session)

        out = capsys.readouterr().out
        assert "Uploaded" in out
        client.workspace_upload.assert_called_once()

    async def test_no_arg_shows_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = _make_session()

        await _upload_handler("", session)

        out = capsys.readouterr().out
        assert "Usage" in out

    async def test_missing_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = _make_session()

        await _upload_handler("/nonexistent/file.txt", session)

        out = capsys.readouterr().out
        assert "Error" in out or "not a file" in out

    async def test_handles_api_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello")
        client = _mock_client()
        client.workspace_upload.side_effect = SystemExit(1)
        session = _make_session(client=client)

        await _upload_handler(str(f), session)

        err = capsys.readouterr().err
        assert "Error" in err


class TestStopHandler:
    async def test_stops_team(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        session = _make_session(client=client)

        await _stop_handler("", session)

        out = capsys.readouterr().out
        assert "Team t1 stopped." in out
        client.stop_team.assert_called_once_with("t1")
        client.delete_team.assert_not_called()

    async def test_handles_api_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.stop_team.side_effect = SystemExit(1)
        session = _make_session(client=client)

        await _stop_handler("", session)

        err = capsys.readouterr().err
        assert "Error" in err


class TestRestoreHandler:
    async def test_restores_current_team_no_arg(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = _mock_client()
        session = _make_session(client=client)

        await _restore_handler("", session)

        out = capsys.readouterr().out
        assert "Team t1 restored. Live events resumed." in out
        client.restore_team.assert_called_once_with("t1")
        # No switch should happen when restoring current team
        assert session.team_id == "t1"

    async def test_restores_different_team_and_switches(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = _mock_client()
        session = _make_session(client=client)
        session._receive_task = asyncio.create_task(asyncio.sleep(100))

        new_ws_mock = _mock_ws()
        with patch("akgentic.infra.cli.commands.WsClient", return_value=new_ws_mock):
            await _restore_handler("t2", session)

        out = capsys.readouterr().out
        assert "Team t2 restored. Live events resumed." in out
        client.restore_team.assert_called_once_with("t2")
        # Should have auto-switched
        assert session.team_id == "t2"

    async def test_restores_same_team_id_no_switch(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = _mock_client()
        session = _make_session(client=client)

        await _restore_handler("t1", session)

        out = capsys.readouterr().out
        assert "Team t1 restored. Live events resumed." in out
        client.restore_team.assert_called_once_with("t1")

    async def test_handles_api_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.restore_team.side_effect = SystemExit(1)
        session = _make_session(client=client)

        await _restore_handler("", session)

        err = capsys.readouterr().err
        assert "Error" in err


class TestSwitchHandler:
    async def test_switches_team(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        old_ws = _mock_ws()
        session = _make_session(client=client, ws=old_ws)
        session._receive_task = asyncio.create_task(asyncio.sleep(100))

        new_ws_mock = _mock_ws()
        with patch("akgentic.infra.cli.commands.WsClient", return_value=new_ws_mock) as ws_cls:
            await _switch_handler("t2", session)

        old_ws.close.assert_called_once()
        assert session.team_id == "t2"
        assert session.ws_client is new_ws_mock
        # Verify server_url and api_key are passed to new WsClient
        ws_cls.assert_called_once_with(
            base_url="http://localhost:8000",
            team_id="t2",
            api_key="test-key",
        )
        out = capsys.readouterr().out
        assert "Switched to team t2" in out

    async def test_no_arg_shows_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = _make_session()

        await _switch_handler("", session)

        out = capsys.readouterr().out
        assert "Usage" in out

    async def test_switch_fails_restores_old(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        old_ws = _mock_ws()
        session = _make_session(client=client, ws=old_ws)

        # First call creates the new WsClient (connect will fail),
        # second call creates the restored WsClient for the old team
        new_ws_mock = _mock_ws()
        new_ws_mock.connect = AsyncMock(side_effect=SystemExit(1))
        restored_ws_mock = _mock_ws()
        with patch(
            "akgentic.infra.cli.commands.WsClient",
            side_effect=[new_ws_mock, restored_ws_mock],
        ):
            await _switch_handler("bad-team", session)

        err = capsys.readouterr().err
        assert "not found" in err
        # Session ws should be the restored connection
        assert session.ws_client is restored_ws_mock
        assert session.team_id == "t1"  # unchanged


# =============================================================================
# Task 8.3: Test integration with ChatSession._input_loop
# =============================================================================


class TestChatSessionCommandIntegration:
    async def test_info_dispatched_not_sent(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        ws = _mock_ws()
        session = ChatSession(client, ws, "t1", OutputFormat.table)

        with patch(
            _PROMPT_PATH,
            side_effect=["/info", "/quit"],
        ):
            await session.run()

        # /info should have triggered get_team, not send_message
        client.get_team.assert_called_once_with("t1")
        client.send_message.assert_not_called()

    async def test_regular_text_sent_as_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        ws = _mock_ws()
        session = ChatSession(client, ws, "t1", OutputFormat.table)

        with patch(
            _PROMPT_PATH,
            side_effect=["hello there", "/quit"],
        ):
            await session.run()

        client.send_message.assert_called_once_with("t1", "hello there")

    async def test_quit_still_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        ws = _mock_ws()
        session = ChatSession(client, ws, "t1", OutputFormat.table)

        with patch(
            _PROMPT_PATH,
            side_effect=["/quit"],
        ):
            await session.run()

        out = capsys.readouterr().out
        assert "Session closed." in out

    async def test_unknown_slash_command_not_sent(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        ws = _mock_ws()
        session = ChatSession(client, ws, "t1", OutputFormat.table)

        with patch(
            _PROMPT_PATH,
            side_effect=["/unknown", "/quit"],
        ):
            await session.run()

        out = capsys.readouterr().out
        assert "Unknown command: /unknown" in out
        client.send_message.assert_not_called()


# =============================================================================
# Task 8.4: Verify build_default_registry has all commands
# =============================================================================


class TestBuildDefaultRegistry:
    def test_all_commands_registered(self) -> None:
        registry = build_default_registry()
        expected = {
            "help",
            "teams",
            "create",
            "delete",
            "info",
            "events",
            "agents",
            "history",
            "files",
            "read",
            "upload",
            "stop",
            "restore",
            "switch",
        }
        assert set(registry.commands.keys()) == expected

    def test_status_not_registered(self) -> None:
        registry = build_default_registry()
        assert "status" not in registry.commands

    def test_restore_usage_updated(self) -> None:
        registry = build_default_registry()
        assert "[team_id]" in registry.commands["restore"].usage
