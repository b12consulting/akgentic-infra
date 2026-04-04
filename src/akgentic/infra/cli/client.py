"""HTTP client wrapper for the akgentic-infra REST API."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel, model_validator

from akgentic.core.messages.message import Message
from akgentic.core.utils.deserializer import deserialize_object

_log = logging.getLogger(__name__)

# -- CLI-side response models (independent of server models) --


class TeamInfo(BaseModel):
    """Team response from GET/POST /teams endpoints."""

    team_id: str
    name: str
    status: str
    user_id: str
    created_at: str
    updated_at: str


class TeamListInfo(BaseModel):
    """Wrapper for GET /teams list response."""

    teams: list[TeamInfo]


class EventInfo(BaseModel):
    """Single event from GET /teams/{team_id}/events."""

    team_id: str
    sequence: int
    event: dict[str, object] | Message
    timestamp: str

    @model_validator(mode="after")
    def _deserialize_event(self) -> EventInfo:
        """Deserialize raw event dict into a typed Message if possible."""
        if isinstance(self.event, dict) and "__model__" in self.event:
            try:
                result = deserialize_object(dict(self.event))
                if isinstance(result, Message):
                    self.event = result
            except ValueError:
                _log.debug("EventInfo: failed to deserialize event dict", exc_info=True)
        return self


class EventListInfo(BaseModel):
    """Wrapper for GET /teams/{team_id}/events list response."""

    events: list[EventInfo]


class WorkspaceEntry(BaseModel):
    """Single entry in workspace tree."""

    name: str
    is_dir: bool
    size: int


class WorkspaceTreeInfo(BaseModel):
    """Response from GET /workspace/{team_id}/tree."""

    team_id: str
    path: str
    entries: list[WorkspaceEntry]


class WorkspaceUploadInfo(BaseModel):
    """Response from POST /workspace/{team_id}/file."""

    path: str
    size: int


class CatalogTeamInfo(BaseModel):
    """Catalog team entry for display (subset of server-side TeamEntry)."""

    id: str
    name: str
    description: str


class ApiError(Exception):
    """HTTP API call failed."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}" if detail else f"HTTP {status_code}")

    @property
    def retryable(self) -> bool:
        """True for transient server errors and network issues."""
        return self.status_code >= 500 or self.status_code == 429 or self.status_code == 0


class ApiClient:
    """Thin HTTP client mapping CLI commands to server endpoints."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=base_url,
            headers=headers,
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> ApiClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

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
        try:
            resp = self._client.request(
                method,
                path,
                json=json,
                params=params,
                files=files,
                data=data,
            )
        except httpx.TimeoutException as exc:
            raise ApiError(0, f"Request timed out: {method} {path}") from exc
        except httpx.ConnectError as exc:
            raise ApiError(0, f"Connection failed: {exc}") from exc

        if not resp.is_success:
            detail = ""
            try:
                body = resp.json()
                detail = body.get("detail", "")
            except Exception:  # noqa: BLE001
                pass
            raise ApiError(resp.status_code, detail)
        return resp

    # -- catalog endpoints --

    def list_catalog_teams(self) -> list[CatalogTeamInfo]:
        """GET /catalog/api/teams/ -> list of CatalogTeamInfo models."""
        resp = self._request("GET", "/catalog/api/teams/")
        return [CatalogTeamInfo.model_validate(entry) for entry in resp.json()]

    # -- team endpoints --

    def list_teams(self) -> list[TeamInfo]:
        """GET /teams → list of TeamInfo models."""
        resp = self._request("GET", "/teams")
        return TeamListInfo.model_validate(resp.json()).teams

    def get_team(self, team_id: str) -> TeamInfo:
        """GET /teams/{team_id} → TeamInfo model."""
        return TeamInfo.model_validate(self._request("GET", f"/teams/{team_id}").json())

    def create_team(self, catalog_entry_id: str) -> TeamInfo:
        """POST /teams → created TeamInfo model."""
        resp = self._request("POST", "/teams", json={"catalog_entry_id": catalog_entry_id})
        return TeamInfo.model_validate(resp.json())

    def stop_team(self, team_id: str) -> None:
        """POST /teams/{team_id}/stop — stop actors but preserve event store data."""
        self._request("POST", f"/teams/{team_id}/stop")

    def delete_team(self, team_id: str) -> None:
        """DELETE /teams/{team_id}."""
        self._request("DELETE", f"/teams/{team_id}")

    def restore_team(self, team_id: str) -> TeamInfo:
        """POST /teams/{team_id}/restore → restored TeamInfo model."""
        return TeamInfo.model_validate(self._request("POST", f"/teams/{team_id}/restore").json())

    def get_events(self, team_id: str) -> list[EventInfo]:
        """GET /teams/{team_id}/events → list of EventInfo models."""
        resp = self._request("GET", f"/teams/{team_id}/events")
        return EventListInfo.model_validate(resp.json()).events

    # -- messaging --

    def send_message(self, team_id: str, content: str) -> None:
        """POST /teams/{team_id}/message."""
        self._request("POST", f"/teams/{team_id}/message", json={"content": content})

    def send_message_to(self, team_id: str, agent_name: str, content: str) -> None:
        """POST /teams/{team_id}/message/{agent_name}."""
        self._request(
            "POST", f"/teams/{team_id}/message/{agent_name}", json={"content": content}
        )

    def human_input(self, team_id: str, content: str, message_id: str) -> None:
        """POST /teams/{team_id}/human-input."""
        self._request(
            "POST",
            f"/teams/{team_id}/human-input",
            json={"content": content, "message_id": message_id},
        )

    # -- workspace --

    def workspace_tree(self, team_id: str) -> WorkspaceTreeInfo:
        """GET /workspace/{team_id}/tree → WorkspaceTreeInfo model."""
        return WorkspaceTreeInfo.model_validate(
            self._request("GET", f"/workspace/{team_id}/tree").json()
        )

    def workspace_read(self, team_id: str, path: str) -> bytes:
        """GET /workspace/{team_id}/file → raw file bytes."""
        resp = self._request("GET", f"/workspace/{team_id}/file", params={"path": path})
        return resp.content

    def workspace_upload(self, team_id: str, path: str, file_data: bytes) -> WorkspaceUploadInfo:
        """POST /workspace/{team_id}/file → WorkspaceUploadInfo model."""
        resp = self._request(
            "POST",
            f"/workspace/{team_id}/file",
            data={"path": path},
            files={"file": ("upload", file_data)},
        )
        return WorkspaceUploadInfo.model_validate(resp.json())
