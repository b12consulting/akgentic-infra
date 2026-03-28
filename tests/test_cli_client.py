"""Tests for akgentic.infra.cli.client.ApiClient."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from akgentic.infra.cli.client import ApiClient


def _transport(
    status: int = 200,
    json_body: dict[str, Any] | list[dict[str, Any]] | None = None,
    content: bytes = b"",
) -> httpx.MockTransport:
    """Build a MockTransport returning a fixed response."""

    def handler(request: httpx.Request) -> httpx.Response:
        if json_body is not None:
            return httpx.Response(status, json=json_body)
        return httpx.Response(status, content=content)

    return httpx.MockTransport(handler)


def _client(
    transport: httpx.MockTransport | httpx.BaseTransport,
    api_key: str | None = None,
) -> ApiClient:
    """Build an ApiClient backed by a mock transport."""
    c = ApiClient(base_url="http://test", api_key=api_key)
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    c._client = httpx.Client(base_url="http://test", transport=transport, headers=headers)
    return c


# -- team endpoints --


class TestListTeams:
    def test_returns_team_list(self) -> None:
        teams = [{"team_id": "abc", "name": "t1", "status": "running"}]
        client = _client(_transport(json_body={"teams": teams}))
        result = client.list_teams()
        assert result == teams

    def test_empty_list(self) -> None:
        client = _client(_transport(json_body={"teams": []}))
        assert client.list_teams() == []


class TestGetTeam:
    def test_returns_team_dict(self) -> None:
        team = {"team_id": "abc", "name": "t1"}
        client = _client(_transport(json_body=team))
        assert client.get_team("abc") == team


class TestCreateTeam:
    def test_sends_catalog_entry(self) -> None:
        created = {"team_id": "new-id", "name": "new"}
        sent_bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent_bodies.append(json.loads(request.content))
            return httpx.Response(200, json=created)

        client = _client(httpx.MockTransport(handler))
        result = client.create_team("my-entry")
        assert result == created
        assert sent_bodies[0] == {"catalog_entry_id": "my-entry"}


class TestDeleteTeam:
    def test_no_return(self) -> None:
        client = _client(_transport(status=204, content=b""))
        # Should not raise
        client.delete_team("abc")


class TestRestoreTeam:
    def test_returns_restored(self) -> None:
        team = {"team_id": "abc", "status": "running"}
        client = _client(_transport(json_body=team))
        assert client.restore_team("abc") == team


class TestGetEvents:
    def test_returns_event_list(self) -> None:
        events = [{"sequence": 1, "event": {"type": "msg"}}]
        client = _client(_transport(json_body={"events": events}))
        assert client.get_events("abc") == events


# -- messaging --


class TestSendMessage:
    def test_sends_content(self) -> None:
        sent: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={})

        client = _client(httpx.MockTransport(handler))
        client.send_message("team-1", "hello")
        assert sent[0] == {"content": "hello"}


class TestHumanInput:
    def test_sends_content_and_message_id(self) -> None:
        sent: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={})

        client = _client(httpx.MockTransport(handler))
        client.human_input("team-1", "yes", "msg-42")
        assert sent[0] == {"content": "yes", "message_id": "msg-42"}


# -- workspace --


class TestWorkspaceTree:
    def test_returns_tree(self) -> None:
        tree = {"team_id": "t1", "path": "/", "entries": []}
        client = _client(_transport(json_body=tree))
        assert client.workspace_tree("t1") == tree


class TestWorkspaceRead:
    def test_returns_bytes(self) -> None:
        client = _client(_transport(content=b"file content here"))
        assert client.workspace_read("t1", "readme.md") == b"file content here"


class TestWorkspaceUpload:
    def test_sends_multipart(self) -> None:
        result = {"path": "readme.md", "size": 5}
        urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            urls.append(str(request.url))
            return httpx.Response(200, json=result)

        client = _client(httpx.MockTransport(handler))
        resp = client.workspace_upload("t1", "readme.md", b"hello")
        assert resp == result
        assert "/workspace/t1/file" in urls[0]


# -- error handling --


class TestErrorHandling:
    def test_404_raises_exit(self) -> None:
        client = _client(_transport(status=404, json_body={"detail": "Not found"}))
        with pytest.raises(RuntimeError):
            client.get_team("missing")

    def test_500_raises_exit(self) -> None:
        client = _client(_transport(status=500, content=b"Internal error"))
        with pytest.raises(RuntimeError):
            client.list_teams()

    def test_409_raises_exit(self) -> None:
        client = _client(_transport(status=409, json_body={"detail": "Conflict"}))
        with pytest.raises(RuntimeError):
            client.delete_team("abc")

    def test_404_prints_detail(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _client(_transport(status=404, json_body={"detail": "Not found"}))
        with pytest.raises(RuntimeError):
            client.get_team("missing")
        assert "Not found" in capsys.readouterr().err


class TestApiKey:
    def test_api_key_header(self) -> None:
        headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            headers.update(dict(request.headers))
            return httpx.Response(200, json={"teams": []})

        client = _client(httpx.MockTransport(handler), api_key="secret-key")
        client.list_teams()
        assert headers.get("authorization") == "Bearer secret-key"
