"""Tests for team CRUD endpoints using FastAPI TestClient."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from akgentic.infra.errors import TeamStateConflictError
from akgentic.infra.server.app import create_app
from akgentic.infra.server.auth import RequestUser, get_request_user
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.routes._team_access import get_team_service
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community


def test_create_team_success(client: TestClient) -> None:
    """POST /teams with valid catalog entry returns 201."""
    resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert resp.status_code == 201
    data = resp.json()
    assert "team_id" in data
    assert data["status"] == "running"
    assert data["name"] == "Test Team"


def test_create_team_invalid_entry(client: TestClient) -> None:
    """POST /teams with unknown catalog entry returns 404."""
    resp = client.post("/teams/", json={"catalog_namespace": "nonexistent"})
    assert resp.status_code == 404


# --- POST /teams failure modes: absent, teamless, invalid (Epic 59 Part A) ---


def test_create_team_absent_namespace_is_404_naming_the_namespace(client: TestClient) -> None:
    """A namespace that holds nothing at all keeps today's 404 and its message."""
    resp = client.post("/teams/", json={"catalog_namespace": "nonexistent"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Catalog namespace not found"


def test_create_team_namespace_without_team_entry_is_404_saying_so(client: TestClient) -> None:
    """A namespace that exists but holds no team entry is a 404 that says which.

    "Not found" about a namespace the operator can see listed is the same lie
    this epic removes from the invalid case, so the two 404s carry different
    messages.
    """
    resp = client.post("/teams/", json={"catalog_namespace": "teamless"})
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert "has no team entry" in detail
    assert "teamless" in detail
    assert detail != "Catalog namespace not found"


def test_create_team_invalid_namespace_is_409_carrying_the_catalog_message(
    client: TestClient,
) -> None:
    """A namespace that exists and fails validation is a 409 with the diagnosis.

    The status alone is not the fix — the body text is. Asserting only the code
    would stay green if the message were replaced by "Catalog namespace not
    found" again.
    """
    resp = client.post("/teams/", json={"catalog_namespace": "broken-team"})
    assert resp.status_code == 409
    body = resp.json()
    assert "ref marker" in body["detail"]
    assert "'params'" in body["detail"]
    assert isinstance(body["errors"], list)
    assert body["errors"]
    assert any("ref marker" in err for err in body["errors"])


def test_create_team_409_body_matches_the_resolve_endpoint(client: TestClient) -> None:
    """``POST /teams`` and ``/admin/catalog/team/{ns}/resolve`` answer identically.

    Same catalog state, same diagnosis, same body — that equivalence is the
    reason 409 was chosen over a fresh status, and nothing else pins it: the
    moment the re-raise is "tidied" into a local ``HTTPException(409)``, the
    ``errors`` list vanishes and only this assertion notices.
    """
    created = client.post("/teams/", json={"catalog_namespace": "broken-team"})
    resolved = client.get("/admin/catalog/team/broken-team/resolve")

    assert created.status_code == resolved.status_code == 409
    assert created.json().keys() == resolved.json().keys()
    assert created.json() == resolved.json()


def test_create_team_valid_namespace_still_201(client: TestClient) -> None:
    """The fourth outcome: a namespace that resolves is unchanged by the split."""
    resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert resp.status_code == 201


def test_list_teams_empty(client: TestClient) -> None:
    """GET /teams returns an empty list and total_count == 0 when no teams exist."""
    resp = client.get("/teams/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["teams"] == []
    assert body["total_count"] == 0


def test_list_teams_after_create(client: TestClient) -> None:
    """GET /teams returns created teams and total_count == the owned count."""
    client.post("/teams/", json={"catalog_namespace": "test-team"})
    resp = client.get("/teams/")
    assert resp.status_code == 200
    body = resp.json()
    teams = body["teams"]
    assert len(teams) == 1
    assert teams[0]["name"] == "Test Team"
    assert body["total_count"] == 1


def test_get_team_success(client: TestClient) -> None:
    """GET /teams/{id} returns team detail."""
    create_resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    team_id = create_resp.json()["team_id"]
    resp = client.get(f"/teams/{team_id}")
    assert resp.status_code == 200
    assert resp.json()["team_id"] == team_id


def test_get_team_not_found(client: TestClient) -> None:
    """GET /teams/{id} returns 404 for unknown team."""
    resp = client.get(f"/teams/{uuid.uuid4()}")
    assert resp.status_code == 404


@pytest.mark.skip(
    reason="Flaky: race in TeamManager.delete_team — on_stop subscribers still "
    "flushing event_store writes while rmtree runs; pre-existing, not introduced by Epic 22."
)
def test_delete_team_success(client: TestClient) -> None:
    """DELETE /teams/{id} returns 204 and removes team."""
    create_resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    team_id = create_resp.json()["team_id"]
    resp = client.delete(f"/teams/{team_id}")
    assert resp.status_code == 204
    # Verify team is actually gone
    get_resp = client.get(f"/teams/{team_id}")
    assert get_resp.status_code == 404


def test_delete_team_not_found(client: TestClient) -> None:
    """DELETE /teams/{id} returns 404 for unknown team."""
    resp = client.delete(f"/teams/{uuid.uuid4()}")
    assert resp.status_code == 404


# --- Team-state errors are classified by type, not by message (Epic 59 Part B) ---


def test_absent_team_is_404_on_every_previously_flattening_route(client: TestClient) -> None:
    """The three routes that flattened every ValueError still answer 404 when absent.

    The flattening was only ever right for this case; keeping it pinned is what
    stops the fix from over-correcting into 409 for a genuinely missing team.
    """
    absent = uuid.uuid4()
    assert client.delete(f"/teams/{absent}").status_code == 404
    assert client.get(f"/teams/{absent}/events").status_code == 404
    assert client.get(f"/teams/{absent}/agent-states").status_code == 404


def test_acting_on_a_running_team_is_409_naming_the_condition(client: TestClient) -> None:
    """A real, live team refuses an operation as a conflict — never as "not found".

    Driven end to end against a team the app actually started: the response must
    name the state that caused the refusal, because "Team not found" about a
    running team sends the reader looking for the wrong problem entirely.
    """
    team_id = client.post("/teams/", json={"catalog_namespace": "test-team"}).json()["team_id"]

    resp = client.post(f"/teams/{team_id}/restore")

    assert resp.status_code == 409
    assert "already running" in resp.json()["detail"]
    assert client.get(f"/teams/{team_id}").status_code == 200  # still there, still visible


def test_deleting_a_running_team_succeeds_rather_than_conflicting(client: TestClient) -> None:
    """A RUNNING team deletes cleanly, because the service stops it first.

    This is the reported defect's premise, pinned rather than reported: deleting
    a live team was said to answer 404, and it does not — ``delete_team`` stops a
    RUNNING team before handing off, so the team package's "currently running"
    refusal is never reached from this route. Nothing else holds that ordering
    in place; reorder the stop and the route silently answers 404 for a team
    that exists and is running, which is exactly the lie this epic removes.
    """
    team_id = client.post("/teams/", json={"catalog_namespace": "test-team"}).json()["team_id"]

    assert client.delete(f"/teams/{team_id}").status_code == 204


def test_state_conflict_from_the_service_is_409_on_the_flattening_routes(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conflict surfacing on those three routes answers 409, not 404.

    The team, the app and the service are the real wired ones; only the single
    call under test is forced to report a conflict. On the community tier no
    conflict reaches these three routes today (``delete_team`` stops a running
    team before deleting, and the two reads only ever fail as "not found"), so
    without this the type-based mapping would ship unexercised and the next
    condition to arrive would land back on 404.
    """
    team_id = client.post("/teams/", json={"catalog_namespace": "test-team"}).json()["team_id"]
    service = client.app.state.services.team_service

    def _conflict(_team_id: uuid.UUID, **_kwargs: object) -> None:
        raise TeamStateConflictError(f"Team {_team_id} is currently running")

    monkeypatch.setattr(service, "get_agent_states", _conflict)
    monkeypatch.setattr(service, "delete_team", _conflict)
    monkeypatch.setattr(service, "get_events", _conflict)

    states = client.get(f"/teams/{team_id}/agent-states")
    assert states.status_code == 409
    assert "currently running" in states.json()["detail"]

    deleted = client.delete(f"/teams/{team_id}")
    assert deleted.status_code == 409
    assert "currently running" in deleted.json()["detail"]

    events = client.get(f"/teams/{team_id}/events")
    assert events.status_code == 409
    assert "currently running" in events.json()["detail"]


def test_unclassified_value_error_keeps_its_404_on_the_flattening_routes(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare ValueError the service did not classify answers exactly as before.

    Those arrive from the team package, whose conditions this layer cannot name.
    Preserving 404 keeps the change additive — the fallback exists so an
    unclassified error cannot become a 500.
    """
    team_id = client.post("/teams/", json={"catalog_namespace": "test-team"}).json()["team_id"]
    service = client.app.state.services.team_service

    def _bare(_team_id: uuid.UUID) -> None:
        msg = "something the service never classified"
        raise ValueError(msg)

    monkeypatch.setattr(service, "get_agent_states", _bare)

    resp = client.get(f"/teams/{team_id}/agent-states")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Team not found"


def test_routes_using_the_message_matcher_keep_todays_statuses(client: TestClient) -> None:
    """The routes left on ``_raise_action_error`` answer exactly as before.

    The typed errors subclass ``ValueError`` and carry unchanged messages, so
    the helper keeps producing today's answers — this pins that the migration of
    the three routes changed nothing for the others.
    """
    team_id = client.post("/teams/", json={"catalog_namespace": "test-team"}).json()["team_id"]
    absent = uuid.uuid4()

    assert client.post(f"/teams/{team_id}/restore").status_code == 409  # already running
    assert client.post(f"/teams/{team_id}/stop").status_code == 204
    assert client.post(f"/teams/{team_id}/stop").status_code == 409  # already stopped
    assert client.post(f"/teams/{team_id}/message", json={"content": "hi"}).status_code == 409
    assert client.post(f"/teams/{absent}/stop").status_code == 404
    assert client.post(f"/teams/{absent}/restore").status_code == 404


# --- RequestUser identity seam (ADR-023 Story 26.1) ---


def test_create_team_default_identity_anonymous(client: TestClient) -> None:
    """With no override, POST /teams persists user_id == 'anonymous' (AC #5)."""
    resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert resp.status_code == 201
    assert resp.json()["user_id"] == "anonymous"


@pytest.fixture()
def overridden_user_client(app: FastAPI) -> Generator[TestClient, None, None]:
    """TestClient with get_request_user overridden to a fixed identity."""
    app.dependency_overrides[get_request_user] = lambda: RequestUser(
        user_id="alice", email="alice@example.com"
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_create_team_uses_overridden_identity(
    overridden_user_client: TestClient,
) -> None:
    """When get_request_user is overridden, POST /teams persists that user_id (AC #6)."""
    resp = overridden_user_client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert resp.status_code == 201
    assert resp.json()["user_id"] == "alice"


def test_list_teams_filters_by_overridden_identity(
    overridden_user_client: TestClient,
) -> None:
    """Under the override, GET /teams returns only the overridden user's teams (AC #6)."""
    overridden_user_client.post("/teams/", json={"catalog_namespace": "test-team"})
    resp = overridden_user_client.get("/teams/")
    assert resp.status_code == 200
    body = resp.json()
    teams = body["teams"]
    assert len(teams) == 1
    assert teams[0]["user_id"] == "alice"
    assert body["total_count"] == 1


# --- require_team_access community no-break (ADR-034 AC5/AC6) ---


def test_team_owner_access_passes_for_anonymous(client: TestClient) -> None:
    """Community (no middleware): the anonymous owner reaches its own team routes.

    GET /teams/{id} and GET /teams/{id}/events pass (200, NOT 401/404) because
    require_team_access allows when ``process.user_id == "anonymous"``.
    """
    team_id = client.post("/teams/", json={"catalog_namespace": "test-team"}).json()["team_id"]
    assert client.get(f"/teams/{team_id}").status_code == 200
    assert client.get(f"/teams/{team_id}/events").status_code == 200


def test_team_access_denies_non_owner_with_404(
    app: FastAPI, overridden_user_client: TestClient
) -> None:
    """A non-owner non-admin gets 404 (no existence leak) on another user's team.

    ``alice`` (the override) creates and owns the team; a different identity
    (the default anonymous principal, the same ``app``) must NOT see it — the
    per-team routes return 404, not 200, and not 403.
    """
    # alice creates and owns the team.
    team_id = overridden_user_client.post(
        "/teams/", json={"catalog_namespace": "test-team"}
    ).json()["team_id"]
    # alice (owner) can read it.
    assert overridden_user_client.get(f"/teams/{team_id}").status_code == 200
    # A different identity must NOT see it — 404 over 403. Drop alice's identity
    # override so the seam resolves the default anonymous principal.
    app.dependency_overrides.clear()
    anonymous_client = TestClient(app)
    assert anonymous_client.get(f"/teams/{team_id}").status_code == 404
    assert anonymous_client.get(f"/teams/{team_id}/events").status_code == 404


# --- Classic offset+total pagination (Story 37.1, ADR-032 §Decision 1-2) ---


def _create_teams(client: TestClient, n: int) -> None:
    """Create ``n`` teams for the default (anonymous) identity."""
    for _ in range(n):
        resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
        assert resp.status_code == 201


def test_list_teams_default_page_returns_total_count(client: TestClient) -> None:
    """No query params: total_count == full owned count; teams capped at the default."""
    _create_teams(client, 3)
    resp = client.get("/teams/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_count"] == 3
    assert len(body["teams"]) == 3  # well under the 250 default


def test_list_teams_page_size_slices_and_orders(client: TestClient) -> None:
    """?page=&size= slice the owned set; total_count stays full; pages don't overlap."""
    _create_teams(client, 5)
    resp1 = client.get("/teams/", params={"page": 1, "size": 2})
    resp2 = client.get("/teams/", params={"page": 2, "size": 2})
    resp3 = client.get("/teams/", params={"page": 3, "size": 2})
    assert resp1.status_code == resp2.status_code == resp3.status_code == 200
    body1, body2, body3 = resp1.json(), resp2.json(), resp3.json()

    # total_count is the full owned count on every page.
    assert body1["total_count"] == body2["total_count"] == body3["total_count"] == 5
    assert len(body1["teams"]) == 2
    assert len(body2["teams"]) == 2
    assert len(body3["teams"]) == 1  # last partial page

    ids = [t["team_id"] for t in body1["teams"] + body2["teams"] + body3["teams"]]
    assert len(set(ids)) == 5  # contiguous, no overlap, no gap

    # Ordering is created_at DESC, team_id DESC — newest first across all pages.
    created = [
        (t["created_at"], t["team_id"]) for t in body1["teams"] + body2["teams"] + body3["teams"]
    ]
    assert created == sorted(created, reverse=True)


def test_list_teams_out_of_range_page_empty_with_total(client: TestClient) -> None:
    """An out-of-range page returns an empty list with the correct total_count."""
    _create_teams(client, 3)
    resp = client.get("/teams/", params={"page": 99, "size": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["teams"] == []
    assert body["total_count"] == 3


def test_list_teams_size_clamped_no_500(client: TestClient) -> None:
    """size <= 0 and size > 500 are clamped to [1, 500]; the request never 500s."""
    _create_teams(client, 2)
    resp_low = client.get("/teams/", params={"size": 0})
    assert resp_low.status_code == 200
    assert len(resp_low.json()["teams"]) == 1  # clamped to 1

    resp_high = client.get("/teams/", params={"size": 99999})
    assert resp_high.status_code == 200
    assert len(resp_high.json()["teams"]) <= 500


# --- Statelessness (Story 37.1 AC #7): a page is a pure function of args + store ---


@pytest.fixture()
def second_replica_client(
    seeded_settings: CommunitySettings,
) -> Generator[TestClient, None, None]:
    """A second, independently-wired app over the SAME on-disk event store.

    Simulates a different worker/replica: it shares no in-process state with the
    primary ``client`` — only the persisted store contents (same event_store_path).
    """
    services = wire_community(seeded_settings)
    application = create_app(services, seeded_settings)
    yield TestClient(application)
    services.actor_system.shutdown()


def test_pages_consistent_across_independent_replicas(
    client: TestClient,
    second_replica_client: TestClient,
) -> None:
    """AC #7: page 1 from one replica and page 2 from another serving the same
    store form a contiguous, non-overlapping walk — no reliance on per-request state.
    """
    _create_teams(client, 5)

    resp_a = client.get("/teams/", params={"page": 1, "size": 2})
    assert resp_a.status_code == 200
    body_a = resp_a.json()
    assert body_a["total_count"] == 5
    page1 = {t["team_id"] for t in body_a["teams"]}

    # Replica B (separate app/services) serves page 2 over the shared store.
    resp_b = second_replica_client.get("/teams/", params={"page": 2, "size": 2})
    assert resp_b.status_code == 200
    body_b = resp_b.json()
    assert body_b["total_count"] == 5
    page2 = {t["team_id"] for t in body_b["teams"]}

    assert page1.isdisjoint(page2)  # B did not re-show A's page
    assert len(page1 | page2) == 4  # contiguous walk across the replica boundary


# --- Optional ?status= lifecycle filter (Epic 49) ---
#
# The filtered requests use "/teams" with no trailing slash: that is the
# registered path, so the query string never depends on Starlette's
# slash-redirect preserving it.


def test_list_teams_status_running_excludes_stopped(client: TestClient) -> None:
    """?status=running returns only the running team; omitting it returns both."""
    running = client.post("/teams/", json={"catalog_namespace": "test-team"})
    stopped = client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert running.status_code == 201
    assert stopped.status_code == 201
    running_id = running.json()["team_id"]
    stopped_id = stopped.json()["team_id"]
    assert client.post(f"/teams/{stopped_id}/stop").status_code == 204

    filtered = client.get("/teams", params={"status": "running"})
    assert filtered.status_code == 200
    assert [t["team_id"] for t in filtered.json()["teams"]] == [running_id]

    unfiltered = client.get("/teams")
    assert unfiltered.status_code == 200
    assert {t["team_id"] for t in unfiltered.json()["teams"]} == {running_id, stopped_id}


def test_list_teams_status_total_count_is_filtered(client: TestClient) -> None:
    """total_count counts the STATUS-FILTERED set, not the whole owned set.

    A total carried over from the unfiltered query would tell a paginating
    client there are more pages than the filter can ever produce. Both teams
    are owned by the same identity, so only the status filter can move the
    number.
    """
    _create_teams(client, 3)
    unfiltered = client.get("/teams")
    assert unfiltered.status_code == 200
    stopped_id = unfiltered.json()["teams"][0]["team_id"]
    assert client.post(f"/teams/{stopped_id}/stop").status_code == 204

    assert unfiltered.json()["total_count"] == 3

    filtered = client.get("/teams", params={"status": "running"})
    assert filtered.status_code == 200
    body = filtered.json()
    assert body["total_count"] == 2
    assert len(body["teams"]) == 2

    stopped_only = client.get("/teams", params={"status": "stopped"})
    assert stopped_only.status_code == 200
    assert stopped_only.json()["total_count"] == 1


def test_list_teams_status_paginates_within_the_filtered_set(client: TestClient) -> None:
    """?status= composes with ?page=/?size=: the page slices the filtered set."""
    _create_teams(client, 4)
    stopped_id = client.get("/teams").json()["teams"][0]["team_id"]
    assert client.post(f"/teams/{stopped_id}/stop").status_code == 204

    page1 = client.get("/teams", params={"status": "running", "page": 1, "size": 2})
    page2 = client.get("/teams", params={"status": "running", "page": 2, "size": 2})
    assert page1.status_code == page2.status_code == 200

    # Three running teams remain, so the total is 3 on every page of the filter.
    assert page1.json()["total_count"] == page2.json()["total_count"] == 3
    ids = [t["team_id"] for t in page1.json()["teams"] + page2.json()["teams"]]
    assert len(ids) == 3
    assert stopped_id not in ids  # the filter holds across the page boundary


def test_list_teams_unknown_status_returns_422(client: TestClient) -> None:
    """An unknown status is rejected by FastAPI's own enum validation."""
    resp = client.get("/teams", params={"status": "bogus"})
    assert resp.status_code == 422


def test_list_teams_status_does_not_reach_across_users(app: FastAPI) -> None:
    """?status=running narrows within the caller's teams — it never widens past them.

    Another user's *running* team is the case that would surface a bypassed
    owner filter, so one is created under a second identity first. The total
    must stay owner-scoped too, or the count alone leaks the other user's team.
    """
    app.dependency_overrides[get_request_user] = lambda: RequestUser(
        user_id="alice", email="alice@example.com"
    )
    alice_resp = TestClient(app).post("/teams/", json={"catalog_namespace": "test-team"})
    assert alice_resp.status_code == 201
    app.dependency_overrides.clear()

    default_client = TestClient(app)
    own_resp = default_client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert own_resp.status_code == 201

    resp = default_client.get("/teams", params={"status": "running"})
    assert resp.status_code == 200
    body = resp.json()
    assert [t["team_id"] for t in body["teams"]] == [own_resp.json()["team_id"]]
    assert body["total_count"] == 1


# --- Repeated ?meta.<key>=<term> filter — parsing and no-filter behaviour (Epics 53, 65) ---
#
# The teams seeded here carry no metadata (the ``test-team`` card declares no
# metadata_type), so this file covers the halves that need none: how the query
# string is parsed into the filter delegated downstream, rejection of the one
# parameter that is still malformed, and the guarantee that a request without
# any ``meta.`` parameter is unchanged. The matching behaviour itself lives
# beside the metadata-carrying catalog fixture, in test_team_metadata_routes.py.
#
# A repeated key is no longer a 422 — it is the ordered term list this surface
# exists to carry (Epic 65). The spec that pinned that rejection, and its
# "equality-only" message, is deleted rather than weakened: the behaviour it
# guarded is now a feature.


def test_list_teams_empty_meta_key_is_422(client: TestClient) -> None:
    """``?meta.=acme`` names no key, so there is nothing to filter on."""
    resp = client.get("/teams", params={"meta.": "acme"})
    assert resp.status_code == 422
    assert "meta." in resp.json()["detail"]


def test_list_teams_without_meta_params_is_unchanged_field_by_field(client: TestClient) -> None:
    """No ``meta.`` parameter: every pre-existing field keeps its name, type and value.

    Asserted field by field rather than against a frozen response dict — each
    entry legitimately gained a ``metadata`` key in Story 53-1, and a whole-dict
    comparison would pin the absence of the very field the epic adds.
    """
    created = client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert created.status_code == 201
    created_body = created.json()

    resp = client.get("/teams")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_count"] == 1
    entry = body["teams"][0]

    assert entry["team_id"] == created_body["team_id"]
    assert entry["name"] == "Test Team"
    assert entry["status"] == "running"
    assert entry["user_id"] == "anonymous"
    assert entry["created_at"] == created_body["created_at"]
    assert entry["updated_at"] == created_body["updated_at"]
    # The one additive change of Story 53.1, null for a team carrying no metadata.
    assert entry["metadata"] is None
    # ...and the one additive change of Story 65.1. The key set is asserted as a
    # whole so "the body is what it was plus one key" is pinned in both
    # directions: a field quietly dropped fails here too.
    assert entry["catalog_namespace"] == "test-team"
    assert set(entry) == {
        "team_id",
        "name",
        "status",
        "user_id",
        "created_at",
        "updated_at",
        "metadata",
        "catalog_namespace",
    }


def test_list_teams_unknown_non_meta_params_are_still_ignored(client: TestClient) -> None:
    """A parameter that is not ``meta.``-prefixed is ignored, and page/size still apply."""
    _create_teams(client, 3)
    resp = client.get("/teams", params={"page": 2, "size": 2, "metaphor": "not-a-filter"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_count"] == 3
    assert len(body["teams"]) == 1  # second page of a 3-team set at size 2


# --- What GET /teams delegates downstream (Epic 65) ---
#
# Two seams, deliberately probed separately.
#
# ``_delegated_to_service`` spies on the boundary the route owns: how a query
# string becomes the ``metadata`` mapping and the ``catalog_namespace`` term.
# ``_delegated_to_store`` goes one layer further and records what reaches
# ``EventStore.list_teams``, which is where the verbatim guarantee has to hold —
# a term is escaped for a dialect only *after* this point, inside akgentic-team,
# so a term pre-escaped anywhere above the store is escaped twice and matches
# nothing. That defect is invisible to a result-level assertion in this
# repository: the community store matches literally in Python, while the
# escaping it would compose with lives in the Mongo and Postgres dialects this
# suite never exercises.


def _delegated_to_service(
    app: FastAPI,
    community_services: CommunityServices,
    params: Any,
) -> dict[str, Any]:
    """Return the kwargs ``GET /teams`` delegated to ``TeamService.list_teams``.

    The spy *wraps* the wired service rather than replacing it, so the request
    is served for real and the recorded call is the one that actually ran.
    """
    spy = MagicMock(wraps=community_services.team_service)
    app.dependency_overrides[get_team_service] = lambda: spy
    try:
        resp = TestClient(app).get("/teams", params=params)
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert spy.list_teams.call_count == 1
    assert spy.list_teams.call_args.args == ()
    return dict(spy.list_teams.call_args.kwargs)


def _delegated_to_store(
    client: TestClient,
    community_services: CommunityServices,
    params: Any,
) -> dict[str, Any]:
    """Return the kwargs that reached ``EventStore.list_teams`` for a ``GET /teams``.

    Asserts the delegated kwarg SET on every call, so no filter the store does
    not declare can appear on any path: ``catalog_namespace`` is applied in the
    service, and pushing it down would be an akgentic-team Protocol change.
    """
    mock_store = MagicMock()
    mock_store.list_teams.return_value = []
    community_services.event_store = mock_store  # type: ignore[assignment]

    resp = client.get("/teams", params=params)

    assert resp.status_code == 200
    assert mock_store.list_teams.call_count == 1
    assert mock_store.list_teams.call_args.args == ()
    kwargs = dict(mock_store.list_teams.call_args.kwargs)
    assert set(kwargs) == {"user_id", "status", "metadata"}
    return kwargs


def test_repeated_meta_key_delegates_an_ordered_term_list(
    app: FastAPI, community_services: CommunityServices
) -> None:
    """A repeated key is a term list, in arrival order — not a resolved single value.

    The request MUST use a genuinely repeated query key rather than a dict, or
    it cannot tell correct behaviour from last-wins: reading
    ``request.query_params`` instead of ``multi_items()`` would deliver
    ``{"tenant": ["me"]}`` and look right in every single-term test.
    """
    kwargs = _delegated_to_service(
        app, community_services, [("meta.tenant", "ac"), ("meta.tenant", "me")]
    )
    assert kwargs["metadata"] == {"tenant": ["ac", "me"]}


def test_repeated_meta_key_is_no_longer_rejected(client: TestClient) -> None:
    """The repeated-key 422 is gone: two terms on one key is what this surface carries.

    They OR-combine in the store — ordinary faceted search — so the pair that
    could never both hold under equality matching is now the ordinary request.
    """
    resp = client.get("/teams", params=[("meta.tenant", "acme"), ("meta.tenant", "contoso")])
    assert resp.status_code == 200


def test_distinct_meta_keys_each_carry_their_own_list(
    app: FastAPI, community_services: CommunityServices
) -> None:
    """Two different keys are two entries, each a one-term list."""
    kwargs = _delegated_to_service(
        app, community_services, {"meta.tenant": "acme", "meta.case": "C-1"}
    )
    assert kwargs["metadata"] == {"tenant": ["acme"], "case": ["C-1"]}


def test_a_single_blank_term_delegates_no_filter_at_all(
    app: FastAPI, community_services: CommunityServices
) -> None:
    """``?meta.tenant=`` reduces to nothing, and nothing is ``None`` — never ``{}``.

    Neither ``{}`` nor ``{"tenant": [""]}``: an empty conjunction is a query
    some backends still translate, and an empty term would prefix-match every
    entry under its key, which under a disjunction *widens* the answer to
    everything rather than failing to narrow it.
    """
    kwargs = _delegated_to_service(app, community_services, {"meta.tenant": ""})
    assert kwargs["metadata"] is None


def test_a_blank_meta_term_answers_exactly_what_no_filter_answers(client: TestClient) -> None:
    """AC 6, behaviourally: a filter that reduces to nothing changes no answer."""
    assert client.post("/teams/", json={"catalog_namespace": "test-team"}).status_code == 201

    unfiltered = client.get("/teams")
    blank = client.get("/teams", params={"meta.tenant": ""})
    assert unfiltered.status_code == blank.status_code == 200
    assert blank.json() == unfiltered.json()


def test_an_all_blank_key_is_dropped_while_its_siblings_survive(
    app: FastAPI, community_services: CommunityServices
) -> None:
    """A key whose terms are all blank contributes no entry; the other key stands.

    Asserted as an exact mapping rather than by membership: a surviving
    ``"tenant": []`` entry is precisely the empty disjunction the store's
    contract forbids, and ``"tenant" in kwargs["metadata"]`` is how it would
    slip through.
    """
    kwargs = _delegated_to_service(
        app, community_services, [("meta.tenant", ""), ("meta.case", "C-1")]
    )
    assert kwargs["metadata"] == {"case": ["C-1"]}


def test_a_key_keeps_the_terms_that_are_not_blank(
    app: FastAPI, community_services: CommunityServices
) -> None:
    """A blank term among real ones is dropped without taking its key with it."""
    kwargs = _delegated_to_service(
        app,
        community_services,
        [("meta.tenant", "acme"), ("meta.tenant", ""), ("meta.tenant", "contoso")],
    )
    assert kwargs["metadata"] == {"tenant": ["acme", "contoso"]}


_METACHARACTER_TERMS = ["a.b", "50%", "a_b", "a*b", "ac|me", "back\\slash"]
"""Terms carrying regex, ``LIKE`` and index-separator metacharacters.

``.`` and ``*`` are regex (Mongo), ``%`` and ``_`` are ``LIKE`` wildcards
(Postgres), ``|`` separates key from value inside an index entry, and ``\\`` is
the escape character of all three. Each store escapes for its own dialect, once,
after this point.
"""


def test_metacharacter_terms_reach_the_store_byte_identical(
    client: TestClient, community_services: CommunityServices
) -> None:
    """Terms travel verbatim: nothing here escapes, renders, trims or case-folds.

    This is the load-bearing assertion of the whole seam, and it has to be on
    the delegated call rather than on the rows that come back. A pre-escape
    looks *correct* in this repository's suite — the term was escaped, as
    intended — and correct in akgentic-team's, where a mangled term arrived and
    was faithfully escaped again. It is wrong only in composition, which neither
    suite exercises.
    """
    params = [("meta.tenant", term) for term in _METACHARACTER_TERMS]

    kwargs = _delegated_to_store(client, community_services, params)

    assert kwargs["metadata"] == {"tenant": _METACHARACTER_TERMS}


def test_mixed_case_terms_are_not_folded_at_this_seam(
    client: TestClient, community_services: CommunityServices
) -> None:
    """Case-insensitivity is the store's, from casefolding the index on both sides.

    Folding here would be invisible end to end — the answers match — while
    quietly making this layer a second place that decides matching semantics.
    """
    kwargs = _delegated_to_store(client, community_services, {"meta.tenant": "AcMe"})
    assert kwargs["metadata"] == {"tenant": ["AcMe"]}


# --- ?catalog_namespace= : the filter the store does not carry (Epic 65) ---


def test_catalog_namespace_is_never_passed_to_the_store(
    client: TestClient, community_services: CommunityServices
) -> None:
    """``EventStore.list_teams`` has no such parameter, so nothing may push it down.

    Adding one would be an akgentic-team Protocol change. The delegated kwargs
    stay exactly ``{user_id, status, metadata}`` — and ``user_id`` stays on the
    call, so the namespace path cannot reach past the caller's own teams.
    """
    kwargs = _delegated_to_store(
        client, community_services, {"catalog_namespace": "test-team", "meta.tenant": "acme"}
    )
    assert kwargs["user_id"] == "anonymous"
    assert kwargs["metadata"] == {"tenant": ["acme"]}


def test_catalog_namespace_is_forwarded_to_the_service(
    app: FastAPI, community_services: CommunityServices
) -> None:
    """The route hands the raw term to the service, which is where it is applied."""
    kwargs = _delegated_to_service(app, community_services, {"catalog_namespace": "test-team"})
    assert kwargs["catalog_namespace"] == "test-team"


def test_catalog_namespace_narrows_the_page_and_the_count(client: TestClient) -> None:
    """Only teams created from the namespace are returned, and counted."""
    created = client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert created.status_code == 201

    resp = client.get("/teams", params={"catalog_namespace": "test-team"})
    assert resp.status_code == 200
    body = resp.json()
    assert [t["team_id"] for t in body["teams"]] == [created.json()["team_id"]]
    assert body["total_count"] == 1

    missing = client.get("/teams", params={"catalog_namespace": "no-such-namespace"})
    assert missing.status_code == 200
    assert missing.json()["teams"] == []
    assert missing.json()["total_count"] == 0


def test_blank_catalog_namespace_answers_exactly_what_omitting_it_answers(
    client: TestClient,
) -> None:
    """A blank is an empty form field, not a filter on the literal empty string.

    Every seeded team here has a namespace, so a filter on ``""`` would answer
    an empty page — which is how the difference shows.
    """
    assert client.post("/teams/", json={"catalog_namespace": "test-team"}).status_code == 201

    omitted = client.get("/teams")
    blank = client.get("/teams", params={"catalog_namespace": ""})
    assert omitted.status_code == blank.status_code == 200
    assert blank.json() == omitted.json()
    assert blank.json()["total_count"] == 1


# --- catalog_namespace on the wire, from the server's producer (Epic 65) ---


def test_catalog_namespace_is_reported_by_every_server_route(client: TestClient) -> None:
    """Create, get and list all report the namespace the team was created from."""
    created = client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert created.status_code == 201
    assert created.json()["catalog_namespace"] == "test-team"

    team_id = created.json()["team_id"]
    fetched = client.get(f"/teams/{team_id}")
    assert fetched.status_code == 200
    assert fetched.json()["catalog_namespace"] == "test-team"

    listed = client.get("/teams")
    assert listed.status_code == 200
    assert [t["catalog_namespace"] for t in listed.json()["teams"]] == ["test-team"]
