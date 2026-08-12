"""Route-level tests for ``PATCH /teams/{team_id}/metadata`` — Story 53.3.

A distinct file from ``test_team_metadata_routes.py`` so sibling stories sharing
the epic branch do not collide in one file.

Everything here goes through the real HTTP surface and the real community
wiring, so an asserted value has genuinely travelled route → validation →
akgentic-team's ordered write path → event store, and — for the ``?meta.``
assertions — back out through the store's own index matching. That is what makes
the re-index test meaningful: a mock would happily record a call that wrote a
value and left a stale index behind.

Field names and values use ``acme`` / ``contoso`` placeholders (Golden Rule #9).
"""

from __future__ import annotations

import uuid
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
# test pass on an ImportError even if the server had honoured the tag — the
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


def _create(client: TestClient, namespace: str = TYPED_NS, **fields: Any) -> str:
    """Create a team carrying ``fields`` as metadata and return its id."""
    payload: dict[str, Any] = {"catalog_namespace": namespace}
    if fields:
        payload["metadata"] = make_metadata_body(**fields)
    resp = client.post("/teams/", json=payload)
    assert resp.status_code == 201
    return str(resp.json()["team_id"])


def _patch(client: TestClient, team_id: str, metadata: dict[str, Any]) -> Any:
    """Send a metadata PATCH and return the raw response."""
    return client.patch(f"/teams/{team_id}/metadata", json={"metadata": metadata})


def _stored_metadata(services: CommunityServices, team_id: str) -> Any:
    """Read the metadata straight off the persisted Process."""
    process = services.worker_handle.get_team(uuid.UUID(team_id))
    assert process is not None
    return process.metadata


def _filter(client: TestClient, key: str, value: str) -> tuple[list[str], int]:
    """Return the team ids and total_count for a ``?meta.<key>=<value>`` query."""
    resp = client.get("/teams", params={f"meta.{key}": value})
    assert resp.status_code == 200
    body = resp.json()
    return [str(t["team_id"]) for t in body["teams"]], int(body["total_count"])


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


# --- AC1: the route returns the updated document ---


def test_patch_returns_updated_metadata(metadata_client: TestClient) -> None:
    """A valid complete body is a 200 carrying the new metadata as plain JSON."""
    team_id = _create(metadata_client, tenant="acme", case="C-1234")

    resp = _patch(metadata_client, team_id, make_metadata_body(tenant="contoso", case="C-9999"))
    assert resp.status_code == 200
    metadata = resp.json()["metadata"]
    assert metadata["tenant"] == "contoso"
    assert metadata["case"] == "C-9999"


def test_patch_persists_the_declared_type(
    metadata_client: TestClient,
    metadata_services: CommunityServices,
) -> None:
    """The persisted Process carries an instance of the card's declared type."""
    team_id = _create(metadata_client, tenant="acme", case="C-1234")
    resp = _patch(metadata_client, team_id, {"tenant": "contoso", "case": "C-9999"})
    assert resp.status_code == 200

    stored = _stored_metadata(metadata_services, team_id)
    assert isinstance(stored, AcmeCaseMetadata)
    assert stored.tenant == "contoso"
    assert stored.case == "C-9999"


def test_patch_response_carries_no_model_tag_and_round_trips(
    metadata_client: TestClient,
) -> None:
    """The response is tag-free at any depth and is accepted verbatim back.

    GET → modify → PATCH is the most ordinary client pattern there is; a
    response carrying a tag the route then rejects would 422 on the server's own
    output.
    """
    team_id = _create(metadata_client, tenant="acme", case="C-1")

    first = _patch(
        metadata_client,
        team_id,
        make_metadata_body(
            tenant="contoso",
            case="C-2",
            owner={"email": "ops@contoso.example"},
            watchers=[{"email": "watcher@contoso.example"}],
        ),
    )
    assert first.status_code == 200
    echoed = first.json()["metadata"]
    assert echoed["owner"]["email"] == "ops@contoso.example"
    assert echoed["watchers"][0]["email"] == "watcher@contoso.example"
    assert not _has_model_tag(echoed)

    second = _patch(metadata_client, team_id, echoed)
    assert second.status_code == 200
    assert second.json()["metadata"] == echoed


# --- AC2: replace, never merge ---


def test_patch_replaces_document_omitted_field_is_gone(
    metadata_client: TestClient,
    metadata_services: CommunityServices,
) -> None:
    """A field present before and omitted now is gone — from value and index alike.

    ``note`` stands in for the story's optional field (``case`` is required on
    the fixture model) and ``case`` carries the indexed half of the assertion.
    A merge would keep the old ``note`` and keep matching the old ``case``.
    """
    team_id = _create(metadata_client, tenant="acme", case="C-1234", note="escalated")

    resp = _patch(metadata_client, team_id, {"tenant": "contoso", "case": "C-9999"})
    assert resp.status_code == 200
    assert resp.json()["metadata"]["note"] is None

    stored = _stored_metadata(metadata_services, team_id)
    assert isinstance(stored, AcmeCaseMetadata)
    assert stored.note is None
    assert stored.tenant == "contoso"

    # Both halves: the old entry is gone AND the new one is there. Asserting only
    # the empty result would also pass against a filter that matches nothing at all.
    assert _filter(metadata_client, "case", "C-9999") == ([team_id], 1)
    assert _filter(metadata_client, "case", "C-1234") == ([], 0)


def test_patch_drops_the_index_entry_of_an_omitted_indexed_field(
    metadata_client: TestClient,
    metadata_services: CommunityServices,
) -> None:
    """An *indexed* field that goes absent takes its index entry with it.

    The distinct half of replace-vs-merge, and the one the required indexed
    fields cannot show: ``tenant`` and ``case`` always have a value, so every
    other assertion here only proves an entry was **overwritten**. A write path
    that rewrote the entries it was given and left the rest of the previous
    index in place passes all of them and fails only this one — the index has to
    *shrink*, not merely change.
    """
    team_id = _create(metadata_client, tenant="acme", case="C-1", channel="email")
    bystander = _create(metadata_client, tenant="acme", case="C-2", channel="email")

    matched, total = _filter(metadata_client, "channel", "email")
    assert set(matched) == {team_id, bystander}
    assert total == 2

    resp = _patch(metadata_client, team_id, {"tenant": "acme", "case": "C-1"})
    assert resp.status_code == 200
    assert resp.json()["metadata"]["channel"] is None

    stored = _stored_metadata(metadata_services, team_id)
    assert isinstance(stored, AcmeCaseMetadata)
    assert stored.channel is None

    # The entry is gone for this team only; the bystander still carries its own.
    assert _filter(metadata_client, "channel", "email") == ([bystander], 1)
    assert _filter(metadata_client, "case", "C-1") == ([team_id], 1)


def test_empty_document_clears_metadata_and_its_whole_index(
    metadata_client: TestClient,
    metadata_services: CommunityServices,
) -> None:
    """An empty document is the limit case of replace: value and index both empty.

    It is the only path on which infra hands ``None`` down to the write seam,
    and the only one that produces a null ``metadata`` in the response — the
    reason ``TeamMetadataResponse.metadata`` is nullable at all. Every indexed
    entry must go, not just the ones a new document would have overwritten.
    """
    team_id = _create(metadata_client, tenant="acme", case="C-1", channel="email")
    bystander = _create(metadata_client, tenant="acme", case="C-2", channel="email")

    resp = _patch(metadata_client, team_id, {})
    assert resp.status_code == 200
    assert resp.json()["metadata"] is None

    assert _stored_metadata(metadata_services, team_id) is None

    assert _filter(metadata_client, "tenant", "acme") == ([bystander], 1)
    assert _filter(metadata_client, "case", "C-1") == ([], 0)
    assert _filter(metadata_client, "channel", "email") == ([bystander], 1)


# --- AC3: a __model__ key is a 422, even for a real class ---


def test_model_key_is_422_even_for_a_real_importable_class(
    metadata_client: TestClient,
    metadata_services: CommunityServices,
) -> None:
    """The rejection is the policy, not a failed import — and nothing is written.

    ``REAL_IMPORTABLE_CLASS`` genuinely imports, so this cannot pass by accident
    on an ImportError. If a future refactor routed the body through
    ``deserialize_object``, the tag would resolve and this test would fail.
    """
    team_id = _create(metadata_client, tenant="acme", case="C-1234")
    before = _stored_metadata(metadata_services, team_id)

    resp = _patch(
        metadata_client,
        team_id,
        {"__model__": REAL_IMPORTABLE_CLASS, "tenant": "contoso", "case": "C-9999"},
    )
    assert resp.status_code == 422
    assert "__model__" in resp.json()["detail"]
    assert _stored_metadata(metadata_services, team_id) == before


def test_nested_model_key_is_422(
    metadata_client: TestClient,
    metadata_services: CommunityServices,
) -> None:
    """A ``__model__`` one level down is refused too — the scan recurses."""
    team_id = _create(metadata_client, tenant="acme", case="C-1234")
    before = _stored_metadata(metadata_services, team_id)

    resp = _patch(
        metadata_client,
        team_id,
        {
            "tenant": "contoso",
            "case": "C-9999",
            "owner": {"__model__": REAL_IMPORTABLE_CLASS, "email": "ops@contoso.example"},
        },
    )
    assert resp.status_code == 422
    assert "__model__" in resp.json()["detail"]
    assert _stored_metadata(metadata_services, team_id) == before


# --- AC4: a card declaring no metadata contract ---


def test_team_without_metadata_type_is_422(
    metadata_client: TestClient,
    metadata_services: CommunityServices,
) -> None:
    """Patching a team whose card declares no contract is refused, and writes nothing."""
    team_id = _create(metadata_client, namespace=UNTYPED_NS)

    resp = _patch(metadata_client, team_id, {"tenant": "acme"})
    assert resp.status_code == 422
    assert "no metadata contract" in resp.json()["detail"]
    assert _stored_metadata(metadata_services, team_id) is None


# --- AC5: validation failure ---


def test_invalid_body_is_422_and_writes_nothing(
    metadata_client: TestClient,
    metadata_services: CommunityServices,
) -> None:
    """A body failing the declared schema names the field and leaves the value alone."""
    team_id = _create(metadata_client, tenant="acme", case="C-1234")
    before = _stored_metadata(metadata_services, team_id)

    resp = _patch(metadata_client, team_id, {"tenant": "contoso"})
    assert resp.status_code == 422
    assert "case" in resp.json()["detail"]
    assert _stored_metadata(metadata_services, team_id) == before


def test_validation_failure_is_422_not_409(metadata_client: TestClient) -> None:
    """The 422 must not travel through the ValueError → 404/409 string mapper.

    That helper maps anything without "not found" / "deleted" to 409, so a
    regression routing this through it would surface as a conflict.
    """
    team_id = _create(metadata_client, tenant="acme", case="C-1234")
    resp = _patch(metadata_client, team_id, {"tenant": "contoso"})
    assert resp.status_code not in (404, 409)


# --- AC6: unknown team ---


def test_unknown_team_id_is_404(metadata_client: TestClient) -> None:
    """A team_id naming no team is a 404."""
    resp = _patch(metadata_client, str(uuid.uuid4()), {"tenant": "acme", "case": "C-1"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Team not found"


# --- AC7: the derived index is re-computed, not just the value overwritten ---


def test_patch_reindexes_so_new_value_matches_and_old_does_not(
    metadata_client: TestClient,
) -> None:
    """The load-bearing test: the filter index follows the value.

    A change that writes ``Process.metadata`` and leaves ``metadata_indexes``
    stale passes every read-back assertion above and fails only here — which is
    why both queries are asserted, and why ``total_count`` is asserted alongside
    the rows. A stale entry that no longer returns the team can still count it.

    A second team keeps its old tenant throughout, so the "old value matches
    nothing" half cannot pass merely because the store went empty.
    """
    team_id = _create(metadata_client, tenant="acme", case="C-1")
    bystander = _create(metadata_client, tenant="northwind", case="C-2")

    assert _filter(metadata_client, "tenant", "acme") == ([team_id], 1)

    assert _patch(metadata_client, team_id, {"tenant": "contoso", "case": "C-1"}).status_code == 200

    assert _filter(metadata_client, "tenant", "contoso") == ([team_id], 1)
    assert _filter(metadata_client, "tenant", "acme") == ([], 0)
    # The untouched team is still indexed under its own value.
    assert _filter(metadata_client, "tenant", "northwind") == ([bystander], 1)


# --- AC8: another user's team is indistinguishable from a missing one ---


def test_other_users_team_is_indistinguishable_from_missing(metadata_app: FastAPI) -> None:
    """A foreign team answers exactly as an unknown one: same status, same body.

    Asserted as an equality between the two responses rather than against the
    literal 404, so a future change that alters the not-found answer has to
    alter both together or fail here. A distinguishable answer would let a
    caller enumerate teams they may not see.
    """
    metadata_app.dependency_overrides[get_request_user] = lambda: RequestUser(
        user_id="alice", email="alice@example.com"
    )
    alice_team = _create(TestClient(metadata_app), tenant="acme", case="C-1")
    metadata_app.dependency_overrides.clear()

    client = TestClient(metadata_app)
    foreign = _patch(client, alice_team, {"tenant": "contoso", "case": "C-2"})
    missing = _patch(client, str(uuid.uuid4()), {"tenant": "contoso", "case": "C-2"})

    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()


def test_foreign_user_id_in_the_body_changes_nothing(
    metadata_app: FastAPI,
    metadata_services: CommunityServices,
) -> None:
    """Ownership comes from the identity seam; a user_id in the body is inert.

    The body names alice, who owns the team. If ownership were read from the
    payload this would succeed — so the 404, and the unchanged stored value,
    are what pin the seam.
    """
    metadata_app.dependency_overrides[get_request_user] = lambda: RequestUser(
        user_id="alice", email="alice@example.com"
    )
    alice_team = _create(TestClient(metadata_app), tenant="acme", case="C-1")
    metadata_app.dependency_overrides.clear()

    client = TestClient(metadata_app)
    resp = client.patch(
        f"/teams/{alice_team}/metadata",
        json={"metadata": {"tenant": "contoso", "case": "C-2"}, "user_id": "alice"},
    )
    assert resp.status_code == 404

    stored = _stored_metadata(metadata_services, alice_team)
    assert isinstance(stored, AcmeCaseMetadata)
    assert stored.tenant == "acme"


# --- AC9: a stopped team can be patched ---


def test_stopped_team_can_be_patched_and_persists(
    metadata_client: TestClient,
    metadata_services: CommunityServices,
) -> None:
    """A stopped team accepts the write; no orchestrator to push to is not an error.

    The filter assertion matters as much as the read-back: the persisted value
    and its index must both move for a team with no live actor at all.
    """
    team_id = _create(metadata_client, tenant="acme", case="C-1")
    assert metadata_client.post(f"/teams/{team_id}/stop").status_code == 204

    resp = _patch(metadata_client, team_id, {"tenant": "contoso", "case": "C-1"})
    assert resp.status_code == 200
    assert resp.json()["metadata"]["tenant"] == "contoso"

    stored = _stored_metadata(metadata_services, team_id)
    assert isinstance(stored, AcmeCaseMetadata)
    assert stored.tenant == "contoso"
    assert _filter(metadata_client, "tenant", "contoso") == ([team_id], 1)
    assert _filter(metadata_client, "tenant", "acme") == ([], 0)


# --- AC11: statelessness ---


def test_a_second_replica_reads_the_patched_metadata(
    metadata_client: TestClient,
    metadata_settings: CommunitySettings,
) -> None:
    """A separately wired app over the same store sees the update.

    Nothing about the resolved metadata_type or the written value is cached in
    app state or a module global, so any replica serves any request identically.
    """
    team_id = _create(metadata_client, tenant="acme", case="C-1")
    assert _patch(metadata_client, team_id, {"tenant": "contoso", "case": "C-1"}).status_code == 200

    replica_services = wire_community(metadata_settings)
    try:
        replica = TestClient(create_app(replica_services, metadata_settings))
        fetched = replica.get(f"/teams/{team_id}")
        assert fetched.status_code == 200
        assert fetched.json()["metadata"]["tenant"] == "contoso"
    finally:
        replica_services.actor_system.shutdown()


# --- AC13: the pre-existing surface is untouched ---


def test_patch_does_not_disturb_the_other_team_routes(metadata_client: TestClient) -> None:
    """Create, get and list still answer as before around a metadata update."""
    team_id = _create(metadata_client, tenant="acme", case="C-1")
    assert _patch(metadata_client, team_id, {"tenant": "contoso", "case": "C-1"}).status_code == 200

    fetched = metadata_client.get(f"/teams/{team_id}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "running"

    listed = metadata_client.get("/teams")
    assert listed.status_code == 200
    assert listed.json()["total_count"] == 1
