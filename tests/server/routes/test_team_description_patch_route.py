"""Route-level tests for ``PATCH /teams/{team_id}/description`` — Story 80.1.

Everything here goes through the real HTTP surface and the real community
wiring, so an asserted value has genuinely travelled route → request-model
normalisation → service → the store's conditional write, and back out through
``GET /teams``. A mock would happily record a call that wrote a value and left
``description_origin`` behind.

Values use ``acme`` / ``contoso`` placeholders (Golden Rule #9).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock

from akgentic.team.models import DescriptionOrigin, Process, TeamStatus
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from akgentic.infra.server.app import create_app
from akgentic.infra.server.auth import RequestUser, get_request_user
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community

DESCRIPTION = "Triage inbound acme cases"


def _create(client: TestClient) -> str:
    """Create a team from the seeded namespace and return its id."""
    resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert resp.status_code == 201
    return str(resp.json()["team_id"])


def _patch(client: TestClient, team_id: str, description: str | None) -> Response:
    """Send a description PATCH and return the raw response."""
    return client.patch(f"/teams/{team_id}/description", json={"description": description})


def _stored(services: CommunityServices, team_id: str) -> Process:
    """Read the persisted ``Process`` straight off the store."""
    process = services.worker_handle.get_team(uuid.UUID(team_id))
    assert process is not None
    return process


def _assert_persisted_answer(resp: Response, stored: Process, description: str | None) -> None:
    """The PATCH body is exactly what the store returned: value, owner and stamp.

    ``updated_at`` is the stamp the write set, carried so a client's local save
    can keep its cached team as fresh as the store's record. Compared against
    the persisted ``Process`` rather than against "now".
    """
    body = resp.json()
    assert set(body) == {"description", "origin", "updated_at"}
    assert body["description"] == description
    assert body["origin"] == "user"
    assert datetime.fromisoformat(body["updated_at"]) == stored.updated_at
    assert stored.team_description == description
    assert stored.description_origin is DescriptionOrigin.USER


def _as_alice(app: FastAPI) -> str:
    """Create a team owned by another user and hand back its id."""
    app.dependency_overrides[get_request_user] = lambda: RequestUser(
        user_id="alice", email="alice@example.com"
    )
    try:
        return _create(TestClient(app))
    finally:
        app.dependency_overrides.clear()


# --- AC2: the response carries what was persisted ---


def test_patch_round_trips_the_trimmed_description(
    client: TestClient, community_services: CommunityServices
) -> None:
    team_id = _create(client)

    resp = _patch(client, team_id, f"  {DESCRIPTION}  ")

    assert resp.status_code == 200
    _assert_persisted_answer(resp, _stored(community_services, team_id), DESCRIPTION)
    assert client.get(f"/teams/{team_id}").json()["description"] == DESCRIPTION


def test_patch_updated_at_is_the_stamp_the_write_set(
    client: TestClient, community_services: CommunityServices
) -> None:
    """The stamp moves with the write and the response carries the moved stamp."""
    team_id = _create(client)
    before = _stored(community_services, team_id).updated_at

    resp = _patch(client, team_id, DESCRIPTION)

    after = _stored(community_services, team_id).updated_at
    assert after > before
    assert datetime.fromisoformat(resp.json()["updated_at"]) == after
    assert datetime.fromisoformat(client.get(f"/teams/{team_id}").json()["updated_at"]) == after


# --- AC3: a team that never set one reads null ---


def test_fresh_team_reads_null_on_get_and_on_the_list(client: TestClient) -> None:
    team_id = _create(client)

    fetched = client.get(f"/teams/{team_id}").json()
    assert "description" in fetched
    assert fetched["description"] is None

    listed = client.get("/teams").json()["teams"]
    assert [team["description"] for team in listed] == [None]


# --- AC4: null clears, and the clear is owned ---


def test_null_clears_and_keeps_the_user_as_owner(
    client: TestClient, community_services: CommunityServices
) -> None:
    team_id = _create(client)
    assert _patch(client, team_id, DESCRIPTION).status_code == 200

    resp = _patch(client, team_id, None)

    assert resp.status_code == 200
    _assert_persisted_answer(resp, _stored(community_services, team_id), None)


# --- AC5: whitespace is trimmed; all-whitespace becomes null ---


def test_all_whitespace_is_a_clear(
    client: TestClient, community_services: CommunityServices
) -> None:
    team_id = _create(client)

    resp = _patch(client, team_id, "   \n\t ")

    assert resp.status_code == 200
    _assert_persisted_answer(resp, _stored(community_services, team_id), None)


# --- AC6: over-length is a 422 before any write ---


def test_over_length_is_422_and_writes_nothing(
    client: TestClient, community_services: CommunityServices
) -> None:
    """501 after trimming is refused and the record is byte-identical, stamp included."""
    team_id = _create(client)
    before = _stored(community_services, team_id)

    resp = _patch(client, team_id, " " + "x" * 501 + " ")

    assert resp.status_code == 422
    assert "500" in resp.text
    assert _stored(community_services, team_id) == before


def test_exactly_the_cap_after_trimming_is_accepted(client: TestClient) -> None:
    team_id = _create(client)

    resp = _patch(client, team_id, " " + "x" * 500 + " ")

    assert resp.status_code == 200
    assert resp.json()["description"] == "x" * 500


def test_validation_failure_is_422_not_409(client: TestClient) -> None:
    """The 422 must not travel through the ValueError → 404/409 string mapper."""
    team_id = _create(client)
    resp = _patch(client, team_id, "x" * 501)
    assert resp.status_code not in (404, 409)


# --- AC7: any non-deleted status is writable ---


def test_stopped_team_accepts_the_patch_and_persists(
    client: TestClient, community_services: CommunityServices
) -> None:
    team_id = _create(client)
    assert client.post(f"/teams/{team_id}/stop").status_code == 204

    resp = _patch(client, team_id, DESCRIPTION)

    assert resp.status_code == 200
    assert resp.json()["description"] == DESCRIPTION
    assert _stored(community_services, team_id).team_description == DESCRIPTION


# --- AC8: a DELETED record is a 409 ---


def test_deleted_record_is_409_and_unchanged(
    client: TestClient, community_services: CommunityServices
) -> None:
    """A soft-deleted record is a conflict, not a missing team, and stays as it was.

    The community tier purges on delete, so the ``DELETED`` record is written by
    hand: stop, read, re-save with only the status flipped. The access gate
    still passes (the owner matches), so the answer comes from the route body.
    """
    team_id = _create(client)
    assert client.post(f"/teams/{team_id}/stop").status_code == 204
    deleted = _stored(community_services, team_id).model_copy(update={"status": TeamStatus.DELETED})
    community_services.event_store.save_team(deleted)

    resp = _patch(client, team_id, DESCRIPTION)

    assert resp.status_code == 409
    assert "deleted" in resp.json()["detail"]
    assert _stored(community_services, team_id) == deleted


# --- AC9: unknown and foreign teams are indistinguishable 404s ---


def test_foreign_and_missing_teams_answer_exactly_like_the_metadata_endpoint(
    app: FastAPI,
) -> None:
    """Three 404s with one body: a caller cannot enumerate teams they may not see."""
    alice_team = _as_alice(app)
    client = TestClient(app)
    missing_id = str(uuid.uuid4())

    foreign = _patch(client, alice_team, DESCRIPTION)
    missing = _patch(client, missing_id, DESCRIPTION)
    metadata = client.patch(f"/teams/{missing_id}/metadata", json={"metadata": {}})

    assert foreign.status_code == missing.status_code == metadata.status_code == 404
    assert foreign.json() == missing.json() == metadata.json()


def test_a_team_that_vanishes_between_the_read_and_the_write_is_the_same_404(
    client: TestClient, community_services: CommunityServices
) -> None:
    """The gate passed, then the store's conditional write matched nothing.

    A delete landing between the lifecycle read and the write is the one way
    the route's own not-found arm is reached: ``require_team_access`` answers
    every other 404 before the body runs. Exercised by stubbing the store's
    answer rather than racing a real delete, and the body must equal the
    gate's so the two cases stay indistinguishable to the caller.
    """
    team_id = _create(client)
    service = community_services.team_service
    assert service is not None
    spy = MagicMock(wraps=service._services.event_store)  # noqa: SLF001 — the seam under test
    spy.update_team_description.return_value = None
    service._services.event_store = spy  # type: ignore[assignment]  # noqa: SLF001

    resp = _patch(client, team_id, DESCRIPTION)

    assert resp.status_code == 404
    assert resp.json() == _patch(client, str(uuid.uuid4()), DESCRIPTION).json()


def test_a_user_id_in_the_body_is_inert(
    app: FastAPI, community_services: CommunityServices
) -> None:
    """Ownership comes from the identity seam; naming the owner in the body changes nothing."""
    alice_team = _as_alice(app)

    resp = TestClient(app).patch(
        f"/teams/{alice_team}/description",
        json={"description": DESCRIPTION, "user_id": "alice"},
    )

    assert resp.status_code == 404
    assert _stored(community_services, alice_team).team_description is None


# --- AC14: statelessness ---


def test_a_second_replica_reads_the_patched_description(
    client: TestClient, seeded_settings: CommunitySettings
) -> None:
    """A separately wired app over the same store sees the update."""
    team_id = _create(client)
    assert _patch(client, team_id, DESCRIPTION).status_code == 200

    replica_services = wire_community(seeded_settings)
    try:
        replica = TestClient(create_app(replica_services, seeded_settings))
        fetched = replica.get(f"/teams/{team_id}")
        assert fetched.status_code == 200
        assert fetched.json()["description"] == DESCRIPTION
    finally:
        replica_services.actor_system.shutdown()


# --- the pre-existing surface is untouched ---


def test_patch_does_not_disturb_the_other_team_routes(client: TestClient) -> None:
    team_id = _create(client)
    assert _patch(client, team_id, DESCRIPTION).status_code == 200

    fetched = client.get(f"/teams/{team_id}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "running"

    listed = client.get("/teams")
    assert listed.status_code == 200
    assert listed.json()["total_count"] == 1
