"""HTTP client wrapper for the akgentic-infra REST API."""

from __future__ import annotations

import sys
from typing import Any

import httpx
import typer


class ApiClient:
    """Thin HTTP client mapping CLI commands to server endpoints."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=base_url, headers=headers)

    # -- helpers --

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        files: dict[str, Any] | None = None,
        data: dict[str, str] | None = None,
    ) -> httpx.Response:
        resp = self._client.request(
            method,
            path,
            json=json,
            params=params,
            files=files,
            data=data,
        )
        if not resp.is_success:
            detail = ""
            try:
                body = resp.json()
                detail = body.get("detail", "")
            except Exception:  # noqa: BLE001
                pass
            msg = f"HTTP {resp.status_code}"
            if detail:
                msg += f": {detail}"
            print(msg, file=sys.stderr)
            raise typer.Exit(code=1)
        return resp

    # -- team endpoints --

    def list_teams(self) -> list[dict[str, Any]]:
        """GET /teams → list of team dicts."""
        resp = self._request("GET", "/teams")
        return resp.json()["teams"]  # type: ignore[no-any-return]

    def get_team(self, team_id: str) -> dict[str, Any]:
        """GET /teams/{team_id} → team dict."""
        return self._request("GET", f"/teams/{team_id}").json()  # type: ignore[no-any-return]

    def create_team(self, catalog_entry_id: str) -> dict[str, Any]:
        """POST /teams → created team dict."""
        resp = self._request("POST", "/teams", json={"catalog_entry_id": catalog_entry_id})
        return resp.json()  # type: ignore[no-any-return]

    def delete_team(self, team_id: str) -> None:
        """DELETE /teams/{team_id}."""
        self._request("DELETE", f"/teams/{team_id}")

    def restore_team(self, team_id: str) -> dict[str, Any]:
        """POST /teams/{team_id}/restore → restored team dict."""
        return self._request("POST", f"/teams/{team_id}/restore").json()  # type: ignore[no-any-return]

    def get_events(self, team_id: str) -> list[dict[str, Any]]:
        """GET /teams/{team_id}/events → list of event dicts."""
        resp = self._request("GET", f"/teams/{team_id}/events")
        return resp.json()["events"]  # type: ignore[no-any-return]

    # -- messaging --

    def send_message(self, team_id: str, content: str) -> None:
        """POST /teams/{team_id}/message."""
        self._request("POST", f"/teams/{team_id}/message", json={"content": content})

    def human_input(self, team_id: str, content: str, message_id: str) -> None:
        """POST /teams/{team_id}/human-input."""
        self._request(
            "POST",
            f"/teams/{team_id}/human-input",
            json={"content": content, "message_id": message_id},
        )

    # -- workspace --

    def workspace_tree(self, team_id: str) -> dict[str, Any]:
        """GET /workspace/{team_id}/tree → tree dict."""
        return self._request("GET", f"/workspace/{team_id}/tree").json()  # type: ignore[no-any-return]

    def workspace_read(self, team_id: str, path: str) -> bytes:
        """GET /workspace/{team_id}/file → raw file bytes."""
        resp = self._request("GET", f"/workspace/{team_id}/file", params={"path": path})
        return resp.content

    def workspace_upload(self, team_id: str, path: str, file_data: bytes) -> dict[str, Any]:
        """POST /workspace/{team_id}/file → upload response dict."""
        resp = self._request(
            "POST",
            f"/workspace/{team_id}/file",
            data={"path": path},
            files={"file": ("upload", file_data)},
        )
        return resp.json()  # type: ignore[no-any-return]
