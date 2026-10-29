"""Over HTTP, on a community app whose store retires deleted teams.

A retired team answers exactly like a purged one — 404 on read, 404 on a second
delete, absent from the default listing — while ``?status=deleted`` still lists
it for audit.

The team is stopped before it is deleted, for the reason
``test_team_deletion_removes_the_bound_tree`` gives: the stop path's
subscribers can still be flushing store writes while the delete runs.
"""

from __future__ import annotations

import uuid

import pytest
from akgentic.team.models import DeleteMode
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.settings import CommunitySettings


@pytest.fixture()
def seeded_settings(seeded_settings: CommunitySettings) -> CommunitySettings:
    """The suite's seeded settings, with the record half of deletion set to ``soft``."""
    return seeded_settings.model_copy(update={"team_delete_mode": DeleteMode.SOFT})


def _team_ids(client: TestClient, query: str = "") -> set[str]:
    resp = client.get(f"/teams/{query}")
    assert resp.status_code == 200
    return {team["team_id"] for team in resp.json()["teams"]}


def test_a_retired_team_is_not_found_and_deleting_it_twice_is_a_404(
    client: TestClient, community_services: CommunityServices
) -> None:
    team_id = client.post("/teams/", json={"catalog_namespace": "test-team"}).json()["team_id"]
    assert client.post(f"/teams/{team_id}/stop").status_code == 204

    assert client.delete(f"/teams/{team_id}").status_code == 204

    assert client.get(f"/teams/{team_id}").status_code == 404
    assert client.delete(f"/teams/{team_id}").status_code == 404
    assert team_id not in _team_ids(client)
    assert team_id in _team_ids(client, "?status=deleted")
    # Retired, not purged: the store still holds the record.
    assert community_services.event_store.load_team(uuid.UUID(team_id)) is not None


def test_every_gated_route_and_the_websocket_do_not_find_a_retired_team(
    client: TestClient, community_services: CommunityServices
) -> None:
    """The retired record's events are still in the store, yet no read reaches them."""
    team_id = client.post("/teams/", json={"catalog_namespace": "test-team"}).json()["team_id"]
    assert client.post(f"/teams/{team_id}/stop").status_code == 204
    assert client.delete(f"/teams/{team_id}").status_code == 204
    assert community_services.event_store.load_events(uuid.UUID(team_id))

    assert client.get(f"/teams/{team_id}/events").status_code == 404
    assert client.get(f"/teams/{team_id}/agent-states").status_code == 404
    assert client.post(f"/teams/{team_id}/restore").status_code == 404
    assert client.post(f"/teams/{team_id}/message", json={"content": "hi"}).status_code == 404
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(f"/ws/{team_id}"):
            pass
    assert exc_info.value.code == 4004
