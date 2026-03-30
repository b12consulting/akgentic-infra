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
    _files_handler,
    _help_handler,
    _history_handler,
    _read_handler,
    _restore_handler,
    _status_handler,
    _stop_handler,
    _switch_handler,
    _upload_handler,
    build_default_registry,
)
from akgentic.infra.cli.formatters import OutputFormat
from akgentic.infra.cli.renderers import RichRenderer
from akgentic.infra.cli.repl import ChatSession

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
        assert "/status" in out
        assert "/agents" in out
        assert "/history" in out
        assert "/files" in out
        assert "/read" in out
        assert "/upload" in out
        assert "/stop" in out
        assert "/restore" in out
        assert "/switch" in out
        assert "/help" in out


# =============================================================================
# Task 8.2: Test each command handler in isolation
# =============================================================================


class TestStatusHandler:
    async def test_shows_team_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        session = _make_session(client=client)

        await _status_handler("", session)

        out = capsys.readouterr().out
        assert "Test Team" in out
        assert "running" in out

    async def test_handles_api_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.get_team.side_effect = SystemExit(1)
        session = _make_session(client=client)

        await _status_handler("", session)

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
                event={
                    "__model__": "StartMessage",
                    "sender": {"name": "@Manager", "role": "Manager"},
                },
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
                event={"__model__": "SentMessage", "sender": "bot", "message": {"content": f"msg-{i}"}},
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
                event={"__model__": "SentMessage", "sender": "bot", "message": {"content": f"msg-{i}"}},
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
                event={"__model__": "StateChangedMessage", "state": "running"},
                timestamp="2026-01-01T00:00:00",
            ),
            EventInfo(
                team_id="t1",
                sequence=2,
                event={"__model__": "SentMessage", "sender": "bot", "message": {"content": "visible"}},
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
    async def test_restores_team(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        session = _make_session(client=client)

        await _restore_handler("", session)

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
    async def test_status_dispatched_not_sent(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        ws = _mock_ws()
        session = ChatSession(client, ws, "t1", OutputFormat.table)

        with patch(
            _PROMPT_PATH,
            side_effect=["/status", "/quit"],
        ):
            await session.run()

        # /status should have triggered get_team, not send_message
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
            "status",
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
