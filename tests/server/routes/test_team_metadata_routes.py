"""Route-level tests for the team-metadata HTTP surface — Stories 53.1 and 53.2.

Two catalog namespaces are seeded: ``acme-cases`` whose card declares a
``metadata_type``, and ``acme-plain`` whose card declares none. Everything here
goes through the real HTTP surface and the real community wiring, so a value
asserted on a response has genuinely travelled catalog → validation → placement
→ event store → conversion point — and, for the ``?meta.`` filter, back out
through the store's own index matching rather than a mock's recorded call.

The ``?meta.`` cases that need no metadata at all — the 422s and the
no-filter backward-compatibility check — live in test_team_routes.py beside the
plain catalog fixture.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from akgentic.infra.server.app import create_app
from akgentic.infra.server.auth import RequestUser, get_request_user
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.fixtures.team_metadata import (
    AcmeCaseMetadata,
    make_metadata_body,
    seed_metadata_namespace,
)

TYPED_NS = "acme-cases"
"""Namespace whose TeamCard declares a metadata_type."""

UNTYPED_NS = "acme-plain"
"""Namespace whose TeamCard declares no metadata contract."""

# A real, importable, harmless class. A nonexistent path would let the rejection
# tests pass on an ImportError even if the server had honoured the tag — the
# false green that would hide exactly the vulnerability this rule exists for.
REAL_IMPORTABLE_CLASS = "akgentic.infra.server.models.TeamResponse"


@pytest.fixture()
def metadata_settings(tmp_path: Path) -> CommunitySettings:
    """Community settings with both metadata namespaces seeded."""
    settings = CommunitySettings(
        workspaces_root=tmp_path / "workspaces",
        event_store_path=tmp_path / "event_store",
        catalog_path=tmp_path / "catalog",
    )
    seed_metadata_namespace(settings.catalog_path, TYPED_NS, with_type=True)
    seed_metadata_namespace(settings.catalog_path, UNTYPED_NS, with_type=False)
    return settings


@pytest.fixture()
def metadata_services(
    metadata_settings: CommunitySettings,
) -> Generator[CommunityServices, None, None]:
    """Wired community services over the metadata catalog."""
    services = wire_community(metadata_settings)
    yield services
    services.actor_system.shutdown()


@pytest.fixture()
def metadata_app(
    metadata_services: CommunityServices,
    metadata_settings: CommunitySettings,
) -> FastAPI:
    """The metadata-aware app itself, for tests that override a dependency on it."""
    return create_app(metadata_services, metadata_settings)


@pytest.fixture()
def metadata_client(metadata_app: FastAPI) -> TestClient:
    """HTTP client over the metadata-aware app."""
    return TestClient(metadata_app)


def _team_count(client: TestClient) -> int:
    """Number of teams the store holds for the caller."""
    resp = client.get("/teams")
    assert resp.status_code == 200
    return int(resp.json()["total_count"])


def _cached_handle_count(services: CommunityServices) -> int:
    """Number of live handles in the runtime cache."""
    return len(services.runtime_cache._handles)  # noqa: SLF001 — asserting the AC directly


def _assert_nothing_created(
    client: TestClient,
    services: CommunityServices,
    before_teams: int,
    before_handles: int,
) -> None:
    """AC #5: a rejected body leaves neither a Process nor a cached handle behind."""
    assert _team_count(client) == before_teams
    assert _cached_handle_count(services) == before_handles


# --- AC #1 / #12: omitting metadata is unchanged behaviour, plus one new key ---


def test_create_without_metadata_is_unchanged_and_returns_null(
    metadata_client: TestClient,
) -> None:
    """A create that omits metadata behaves exactly as before, field by field.

    Asserted field by field rather than against a frozen response dict: the
    response legitimately gains a ``metadata`` key, and a whole-dict comparison
    would pin the absence of the very field this story adds (FR10).
    """
    resp = metadata_client.post("/teams/", json={"catalog_namespace": TYPED_NS})
    assert resp.status_code == 201
    body = resp.json()

    assert body["name"] == "Acme Case Team"
    assert body["status"] == "running"
    assert body["user_id"] == "anonymous"
    assert isinstance(body["team_id"], str)
    assert isinstance(body["created_at"], str)
    assert isinstance(body["updated_at"], str)
    # The one additive change: present, and null.
    assert body["metadata"] is None


def test_create_without_metadata_persists_no_metadata(
    metadata_client: TestClient,
    metadata_services: CommunityServices,
) -> None:
    """Omitting metadata leaves ``Process.metadata`` unset and its index empty."""
    resp = metadata_client.post("/teams/", json={"catalog_namespace": TYPED_NS})
    assert resp.status_code == 201
    process = metadata_services.worker_handle.get_team(resp.json()["team_id"])
    assert process is not None
    assert process.metadata is None
    assert process.metadata_indexes == []


# --- AC #2 / #12: a valid body is validated, persisted, and read back ---


def test_valid_metadata_is_persisted_as_the_declared_type(
    metadata_client: TestClient,
    metadata_services: CommunityServices,
) -> None:
    """The persisted Process carries an instance of the card's declared type."""
    resp = metadata_client.post(
        "/teams/",
        json={"catalog_namespace": TYPED_NS, "metadata": {"tenant": "acme", "case": "C-1234"}},
    )
    assert resp.status_code == 201

    process = metadata_services.worker_handle.get_team(resp.json()["team_id"])
    assert process is not None
    assert isinstance(process.metadata, AcmeCaseMetadata)
    assert process.metadata.tenant == "acme"
    assert process.metadata.case == "C-1234"
    # The index is derived once, inside akgentic-team — never re-derived here.
    assert set(process.metadata.index_entries()) <= set(process.metadata_indexes)


def test_valid_metadata_is_returned_in_the_create_response(
    metadata_client: TestClient,
) -> None:
    """AC #12: the 201 body carries the metadata, not just the store."""
    body = make_metadata_body(note="escalated")
    resp = metadata_client.post(
        "/teams/", json={"catalog_namespace": TYPED_NS, "metadata": body}
    )
    assert resp.status_code == 201
    returned = resp.json()["metadata"]
    assert returned["tenant"] == "acme"
    assert returned["case"] == "C-1234"
    assert returned["note"] == "escalated"


def test_get_team_returns_metadata_from_the_conversion_point(
    metadata_client: TestClient,
) -> None:
    """AC #12: a later GET carries it too — the conversion point populates it.

    A route that merely echoed the request body back would pass the create test
    and fail this one, which is why both exist.
    """
    created = metadata_client.post(
        "/teams/",
        json={"catalog_namespace": TYPED_NS, "metadata": make_metadata_body()},
    )
    assert created.status_code == 201
    team_id = created.json()["team_id"]

    fetched = metadata_client.get(f"/teams/{team_id}")
    assert fetched.status_code == 200
    assert fetched.json()["metadata"] == created.json()["metadata"]


def test_list_teams_inherits_the_metadata_field(metadata_client: TestClient) -> None:
    """GET /teams inherits the field through TeamListResponse's TeamResponse."""
    assert (
        metadata_client.post(
            "/teams/",
            json={"catalog_namespace": TYPED_NS, "metadata": make_metadata_body()},
        ).status_code
        == 201
    )
    listed = metadata_client.get("/teams")
    assert listed.status_code == 200
    assert listed.json()["teams"][0]["metadata"]["case"] == "C-1234"


# --- AC #3: a card declaring no contract ---


def test_metadata_for_a_card_declaring_none_is_422(
    metadata_client: TestClient,
    metadata_services: CommunityServices,
) -> None:
    """A non-empty body against an untyped card is refused, and creates nothing."""
    before_teams = _team_count(metadata_client)
    before_handles = _cached_handle_count(metadata_services)

    resp = metadata_client.post(
        "/teams/",
        json={"catalog_namespace": UNTYPED_NS, "metadata": {"tenant": "acme"}},
    )
    assert resp.status_code == 422
    assert "no metadata contract" in resp.json()["detail"]
    _assert_nothing_created(metadata_client, metadata_services, before_teams, before_handles)


@pytest.mark.parametrize("empty", [{}, None])
def test_empty_metadata_against_an_untyped_card_creates_the_team(
    metadata_client: TestClient,
    empty: dict[str, Any] | None,
) -> None:
    """AC #3: the guard is on truthiness — an empty body is not an error."""
    resp = metadata_client.post(
        "/teams/", json={"catalog_namespace": UNTYPED_NS, "metadata": empty}
    )
    assert resp.status_code == 201
    assert resp.json()["metadata"] is None


# --- AC #4: schema failures name the offending field ---


def test_missing_required_field_is_422_naming_the_field(
    metadata_client: TestClient,
    metadata_services: CommunityServices,
) -> None:
    """A missing required field is named, not reported as a bare 'invalid metadata'."""
    before_teams = _team_count(metadata_client)
    before_handles = _cached_handle_count(metadata_services)

    resp = metadata_client.post(
        "/teams/", json={"catalog_namespace": TYPED_NS, "metadata": {"tenant": "acme"}}
    )
    assert resp.status_code == 422
    assert "case" in resp.json()["detail"]
    _assert_nothing_created(metadata_client, metadata_services, before_teams, before_handles)


def test_wrong_scalar_type_is_422_naming_the_field(metadata_client: TestClient) -> None:
    """A wrong scalar type is named too."""
    resp = metadata_client.post(
        "/teams/",
        json={
            "catalog_namespace": TYPED_NS,
            "metadata": {"tenant": "acme", "case": {"not": "a string"}},
        },
    )
    assert resp.status_code == 422
    assert "case" in resp.json()["detail"]


def test_validation_failure_is_422_not_409(metadata_client: TestClient) -> None:
    """The 422 must not travel through the ValueError → 404/409 string mapper.

    That helper maps anything without "not found" / "deleted" to 409, so a
    regression that routed this through it would surface as a conflict.
    """
    resp = metadata_client.post(
        "/teams/", json={"catalog_namespace": TYPED_NS, "metadata": {"tenant": "acme"}}
    )
    assert resp.status_code == 422
    assert resp.status_code not in (404, 409)


# --- AC #6 / #7 / #8: the __model__ rejection ---


def test_top_level_model_tag_is_422(
    metadata_client: TestClient,
    metadata_services: CommunityServices,
) -> None:
    """A ``__model__`` key at the top level is refused and names itself."""
    before_teams = _team_count(metadata_client)
    before_handles = _cached_handle_count(metadata_services)

    resp = metadata_client.post(
        "/teams/",
        json={
            "catalog_namespace": TYPED_NS,
            "metadata": {"__model__": REAL_IMPORTABLE_CLASS, "tenant": "acme", "case": "C-1"},
        },
    )
    assert resp.status_code == 422
    assert "__model__" in resp.json()["detail"]
    _assert_nothing_created(metadata_client, metadata_services, before_teams, before_handles)


def test_nested_model_tag_is_422(metadata_client: TestClient) -> None:
    """A ``__model__`` one level down is refused — the scan recurses."""
    resp = metadata_client.post(
        "/teams/",
        json={
            "catalog_namespace": TYPED_NS,
            "metadata": {
                "tenant": "acme",
                "case": "C-1",
                "owner": {"__model__": REAL_IMPORTABLE_CLASS, "email": "ops@contoso.example"},
            },
        },
    )
    assert resp.status_code == 422
    assert "__model__" in resp.json()["detail"]


def test_model_tag_with_a_real_importable_class_pins_the_policy(
    metadata_client: TestClient,
) -> None:
    """AC #7: the rejection is the policy, not a failed import.

    ``REAL_IMPORTABLE_CLASS`` genuinely imports, so this test cannot pass by
    accident on an ImportError. If a future refactor routed the body through
    ``deserialize_object``, the tag would resolve and this test would fail.
    """
    resp = metadata_client.post(
        "/teams/",
        json={"catalog_namespace": TYPED_NS, "metadata": {"__model__": REAL_IMPORTABLE_CLASS}},
    )
    assert resp.status_code == 422
    assert "__model__" in resp.json()["detail"]


def test_model_tag_beats_the_no_contract_message(metadata_client: TestClient) -> None:
    """AC #8: the scan runs unconditionally, before the metadata_type check.

    Against a card declaring no contract, the caller must still be told about
    ``__model__`` — the type-naming attempt is the security-relevant condition,
    and "this team takes no metadata" would hide that it was noticed at all.
    """
    resp = metadata_client.post(
        "/teams/",
        json={"catalog_namespace": UNTYPED_NS, "metadata": {"__model__": REAL_IMPORTABLE_CLASS}},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "__model__" in detail
    assert "no metadata contract" not in detail


# --- AC #13: the response carries plain fields, tag stripped, and round-trips ---


def test_no_model_tag_anywhere_in_the_response_metadata(metadata_client: TestClient) -> None:
    """The response metadata carries no ``__model__`` key, at any depth.

    This is what stops a later refactor from "simplifying" the conversion into a
    bare ``model_dump()``, which would emit the tag for the model and every
    nested sub-model.
    """
    resp = metadata_client.post(
        "/teams/",
        json={
            "catalog_namespace": TYPED_NS,
            "metadata": make_metadata_body(
                owner={"email": "ops@contoso.example"},
                watchers=[{"email": "watcher@contoso.example"}],
            ),
        },
    )
    assert resp.status_code == 201
    metadata = resp.json()["metadata"]
    assert "__model__" not in metadata
    assert metadata["owner"] is not None
    assert "__model__" not in metadata["owner"]
    assert metadata["owner"]["email"] == "ops@contoso.example"
    # ...and inside a list element, which a dict-only recursion would miss.
    assert all("__model__" not in watcher for watcher in metadata["watchers"])
    assert metadata["watchers"][0]["email"] == "watcher@contoso.example"


def test_response_metadata_is_accepted_verbatim_on_a_second_create(
    metadata_client: TestClient,
) -> None:
    """AC #13 round-trip: the server accepts its own output unchanged.

    GET → modify → PATCH is the most ordinary client pattern there is; a
    response carrying a tag the API then rejects would 422 on the server's own
    output. Pinned behaviourally rather than by inspecting a key.
    """
    first = metadata_client.post(
        "/teams/",
        json={
            "catalog_namespace": TYPED_NS,
            "metadata": make_metadata_body(
                owner={"email": "ops@contoso.example"},
                watchers=[{"email": "watcher@contoso.example"}],
            ),
        },
    )
    assert first.status_code == 201
    echoed = first.json()["metadata"]

    second = metadata_client.post(
        "/teams/", json={"catalog_namespace": TYPED_NS, "metadata": echoed}
    )
    assert second.status_code == 201
    assert second.json()["metadata"] == echoed


# --- AC #10: statelessness ---


def test_a_second_replica_reads_the_same_metadata(
    metadata_client: TestClient,
    metadata_settings: CommunitySettings,
) -> None:
    """A separately wired app over the same store returns the same metadata.

    Nothing about the resolved metadata_type or the validated value is cached in
    app state or a module global, so any replica serves any request identically.
    """
    created = metadata_client.post(
        "/teams/",
        json={"catalog_namespace": TYPED_NS, "metadata": make_metadata_body()},
    )
    assert created.status_code == 201
    team_id = created.json()["team_id"]

    replica_services = wire_community(metadata_settings)
    try:
        replica = TestClient(create_app(replica_services, metadata_settings))
        fetched = replica.get(f"/teams/{team_id}")
        assert fetched.status_code == 200
        assert fetched.json()["metadata"] == created.json()["metadata"]
    finally:
        replica_services.actor_system.shutdown()


# ---------------------------------------------------------------------------
# Story 53.2 — GET /teams ?meta.<key>=<value>
#
# These run against the real YAML event store, so the filter genuinely travels
# route → service → store and is matched against the index the store derived on
# create. Nothing here re-derives an index entry or escapes a value: that is
# akgentic-team's job and happens exactly once, there.
# ---------------------------------------------------------------------------


def _create_team_with(client: TestClient, **fields: Any) -> str:
    """Create a typed-namespace team carrying ``fields`` and return its id."""
    resp = client.post(
        "/teams/",
        json={"catalog_namespace": TYPED_NS, "metadata": make_metadata_body(**fields)},
    )
    assert resp.status_code == 201
    return str(resp.json()["team_id"])


def _has_model_tag(value: Any) -> bool:
    """Report whether ``__model__`` appears anywhere in *value*.

    Deliberately a local walk rather than the server's own scanner: checking the
    strip with the code it ships beside would let a shared blind spot — a
    container neither of them recurses into — pass as a green test.
    """
    if isinstance(value, dict):
        return "__model__" in value or any(_has_model_tag(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_model_tag(item) for item in value)
    return False


def test_single_meta_filter_narrows_the_page(metadata_client: TestClient) -> None:
    """AC #1: ``?meta.tenant=acme`` returns only the teams carrying that value."""
    acme_id = _create_team_with(metadata_client, tenant="acme", case="C-1")
    _create_team_with(metadata_client, tenant="contoso", case="C-2")

    resp = metadata_client.get("/teams", params={"meta.tenant": "acme"})
    assert resp.status_code == 200
    body = resp.json()
    assert [t["team_id"] for t in body["teams"]] == [acme_id]
    assert body["total_count"] == 1


def test_distinct_meta_keys_and_combine(metadata_client: TestClient) -> None:
    """AC #2: two keys are a conjunction — matching one of them is not enough."""
    both_id = _create_team_with(metadata_client, tenant="acme", case="C-1")
    _create_team_with(metadata_client, tenant="acme", case="C-2")
    _create_team_with(metadata_client, tenant="contoso", case="C-1")

    resp = metadata_client.get("/teams", params={"meta.tenant": "acme", "meta.case": "C-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert [t["team_id"] for t in body["teams"]] == [both_id]
    assert body["total_count"] == 1


def test_meta_filter_matching_nothing_is_empty_with_zero_total(
    metadata_client: TestClient,
) -> None:
    """A filter no team satisfies is an empty page and a zero total, not a 404."""
    _create_team_with(metadata_client, tenant="acme", case="C-1")

    resp = metadata_client.get("/teams", params={"meta.tenant": "nobody"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["teams"] == []
    assert body["total_count"] == 0


def test_meta_filter_total_count_is_the_filtered_count_across_pages(
    metadata_client: TestClient,
) -> None:
    """AC #4: the total is the FILTERED count, and it holds on every page.

    Seven teams are owned and three match. A total carried over from the
    unfiltered set would promise a paginating client four more pages than the
    filter can ever produce — the failure mode of filtering the slice instead of
    pushing the filter down, which also yields short pages for no visible reason.
    """
    matching = {_create_team_with(metadata_client, tenant="acme", case=f"C-{i}") for i in range(3)}
    for i in range(4):
        _create_team_with(metadata_client, tenant="contoso", case=f"C-{i}")

    params = {"meta.tenant": "acme", "size": 2}
    page1 = metadata_client.get("/teams", params={**params, "page": 1})
    page2 = metadata_client.get("/teams", params={**params, "page": 2})
    page3 = metadata_client.get("/teams", params={**params, "page": 3})
    assert page1.status_code == page2.status_code == page3.status_code == 200

    totals = [page1.json()["total_count"], page2.json()["total_count"], page3.json()["total_count"]]
    assert totals == [3, 3, 3]
    assert 7 not in totals  # the unfiltered count never surfaces in a filtered answer

    assert len(page1.json()["teams"]) == 2
    assert len(page2.json()["teams"]) == 1
    assert page3.json()["teams"] == []

    walked = [t["team_id"] for t in page1.json()["teams"] + page2.json()["teams"]]
    assert len(walked) == len(set(walked))  # contiguous, no overlap across the boundary
    assert set(walked) == matching


def test_meta_filter_does_not_reach_across_users(metadata_app: FastAPI) -> None:
    """AC #5: another user's identically-tagged team is neither returned NOR counted.

    Metadata is caller-supplied and non-secret, so a filter evaluated without the
    owner scope would turn this route into a cross-tenant enumeration primitive:
    guess a tenant, read the total, learn whether another customer exists. The
    count is asserted for exactly that reason — a leak through it alone is a leak.
    """
    metadata_app.dependency_overrides[get_request_user] = lambda: RequestUser(
        user_id="alice", email="alice@example.com"
    )
    _create_team_with(TestClient(metadata_app), tenant="acme", case="C-1")
    metadata_app.dependency_overrides.clear()

    default_client = TestClient(metadata_app)
    own_id = _create_team_with(default_client, tenant="acme", case="C-1")

    resp = default_client.get("/teams", params={"meta.tenant": "acme", "meta.case": "C-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert [t["team_id"] for t in body["teams"]] == [own_id]
    assert body["total_count"] == 1


def test_meta_filter_composes_with_the_status_filter(metadata_client: TestClient) -> None:
    """AC #6: the metadata and status filters narrow together, and the total follows."""
    running_id = _create_team_with(metadata_client, tenant="acme", case="C-1")
    stopped_id = _create_team_with(metadata_client, tenant="acme", case="C-2")
    _create_team_with(metadata_client, tenant="contoso", case="C-3")
    assert metadata_client.post(f"/teams/{stopped_id}/stop").status_code == 204

    resp = metadata_client.get("/teams", params={"meta.tenant": "acme", "status": "running"})
    assert resp.status_code == 200
    body = resp.json()
    assert [t["team_id"] for t in body["teams"]] == [running_id]
    assert body["total_count"] == 1


def test_listed_entries_carry_no_model_tag_at_any_depth(metadata_client: TestClient) -> None:
    """AC #11: no ``__model__`` survives to the list wire, nested or in a list element.

    Inherited from the shared conversion point rather than implemented here, but
    asserted here because GET /teams is the highest-volume path and the likeliest
    place a later refactor grows its own converter and quietly re-emits the tag.
    """
    _create_team_with(
        metadata_client,
        tenant="acme",
        case="C-1",
        owner={"email": "ops@contoso.example", "squad": "support"},
        watchers=[{"email": "watcher@contoso.example"}],
    )

    resp = metadata_client.get("/teams", params={"meta.tenant": "acme"})
    assert resp.status_code == 200
    entry = resp.json()["teams"][0]
    metadata = entry["metadata"]

    # The nested shapes are actually present, so the walk below has something to find.
    assert metadata["owner"]["email"] == "ops@contoso.example"
    assert metadata["watchers"][0]["email"] == "watcher@contoso.example"
    assert not _has_model_tag(entry)


def test_filtered_pages_are_consistent_across_independent_replicas(
    metadata_client: TestClient,
    metadata_settings: CommunitySettings,
) -> None:
    """AC #9: a separately wired app serves the same filtered walk over the same store.

    Nothing about the parsed filter or the page is cached in app state, a module
    global or a service attribute, so which replica answers cannot change the
    answer.
    """
    for i in range(4):
        _create_team_with(metadata_client, tenant="acme", case=f"C-{i}")

    first = metadata_client.get("/teams", params={"meta.tenant": "acme", "page": 1, "size": 2})
    assert first.status_code == 200
    assert first.json()["total_count"] == 4
    page1 = {t["team_id"] for t in first.json()["teams"]}

    replica_services = wire_community(metadata_settings)
    try:
        replica = TestClient(create_app(replica_services, metadata_settings))
        second = replica.get("/teams", params={"meta.tenant": "acme", "page": 2, "size": 2})
        assert second.status_code == 200
        assert second.json()["total_count"] == 4
        page2 = {t["team_id"] for t in second.json()["teams"]}
    finally:
        replica_services.actor_system.shutdown()

    assert page1.isdisjoint(page2)
    assert len(page1 | page2) == 4
