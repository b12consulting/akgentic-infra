"""Tests for TeamSelector -- team selection and creation flow."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from akgentic.infra.cli.client import ApiError, TeamInfo
from akgentic.infra.cli.team_selector import TeamSelector, _short_id

from .conftest import captured_renderer as _captured_renderer
from .conftest import mock_client as _shared_mock_client


def _mock_client(**overrides: Any) -> MagicMock:
    """Build a mock ApiClient with defaults for team selector tests."""
    return _shared_mock_client(**overrides)


def _make_running_teams(n: int = 2) -> list[TeamInfo]:
    """Create a list of running TeamInfo instances."""
    return [
        TeamInfo(
            team_id=f"team-{i}",
            name=f"Team {i}",
            status="running",
            user_id="u1",
            created_at="2025-01-01",
            updated_at="2025-01-01",
        )
        for i in range(1, n + 1)
    ]


def _make_stopped_teams(n: int = 2) -> list[TeamInfo]:
    """Create a list of stopped TeamInfo instances."""
    return [
        TeamInfo(
            team_id=f"stopped-{i}",
            name=f"Stopped Team {i}",
            status="stopped",
            user_id="u1",
            created_at="2025-01-01",
            updated_at="2025-01-01",
        )
        for i in range(1, n + 1)
    ]


class TestShortId:
    def test_truncates_long_uuid(self) -> None:
        assert _short_id("abcdefghijklmn") == "abcdefghijklm"

    def test_short_id_unchanged(self) -> None:
        assert _short_id("abc") == "abc"

    def test_exactly_13_chars(self) -> None:
        assert _short_id("1234567890123") == "1234567890123"


class TestTeamSelectorRun:
    def test_digit_selection_returns_team_id(self) -> None:
        renderer, buf = _captured_renderer()
        running = _make_running_teams(2)
        client = _mock_client(list_teams=MagicMock(return_value=running))
        selector = TeamSelector(client, renderer)

        with patch("builtins.input", return_value="1"):
            result = selector.run()

        assert result == "team-1"

    def test_quit_returns_none(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        selector = TeamSelector(client, renderer)

        with patch("builtins.input", return_value="q"):
            result = selector.run()

        assert result is None

    def test_empty_input_returns_none(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        selector = TeamSelector(client, renderer)

        with patch("builtins.input", return_value=""):
            result = selector.run()

        assert result is None

    def test_create_returns_team_id(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        selector = TeamSelector(client, renderer)

        with patch("builtins.input", return_value="c my-entry"):
            result = selector.run()

        assert result == "new"
        client.create_team.assert_called_once_with("my-entry")


class TestTeamSelectorHandleCreate:
    def test_empty_entry_renders_error(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        selector = TeamSelector(client, renderer)
        result = selector._handle_create("")
        assert result is None
        out = buf.getvalue()
        assert "Usage: c" in out
        assert "catalog_entry" in out

    def test_api_error_renders_error(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        client.create_team.side_effect = ApiError(500, "fail")
        selector = TeamSelector(client, renderer)
        result = selector._handle_create("my-entry")
        assert result is None
        out = buf.getvalue()
        assert "Failed to create team" in out


class TestTeamSelectorBrowseStopped:
    def test_no_stopped_teams_renders_message(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        selector = TeamSelector(client, renderer)
        result = selector._browse_stopped([])
        assert result is None
        out = buf.getvalue()
        assert "No stopped teams" in out

    def test_back_returns_none(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        selector = TeamSelector(client, renderer)
        stopped = _make_stopped_teams(2)

        with patch("builtins.input", return_value="b"):
            result = selector._browse_stopped(stopped)

        assert result is None

    def test_digit_restores_and_returns_team_id(self) -> None:
        renderer, buf = _captured_renderer()
        client = _mock_client()
        selector = TeamSelector(client, renderer)
        stopped = _make_stopped_teams(2)

        with patch("builtins.input", return_value="1"):
            result = selector._browse_stopped(stopped)

        assert result == "stopped-1"
        client.restore_team.assert_called_once_with("stopped-1")
