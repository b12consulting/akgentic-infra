"""End-to-end tests for the single ServerError -> HTTP handler via POST /teams.

A stub ``PlacementStrategy`` raising a placement error is swapped into the
community services before building the app, so ``POST /teams`` drives the
status/code/Retry-After mapping infra owns. Community's ``LocalPlacement``
never naturally exhausts capacity, so the stub is the only way to exercise
these branches (ADR-031 §Decision 4 / §Validation).
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock

import pytest
from akgentic.infra.protocols.placement import (
    DeclaredWorkspaces,
    NoCapacityError,
    WorkerRejectedError,
)
from akgentic.infra.server.app import create_app
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community
from akgentic.tool.workspace import METADATA_SCOPE, WorkspaceTool
from fastapi.testclient import TestClient

from tests.fixtures.team_metadata import seed_metadata_namespace


class _RaisingPlacement:
    """Stub PlacementStrategy whose create_team raises a fixed placement error."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self._instance_id = uuid.uuid4()

    @property
    def instance_id(self) -> uuid.UUID:
        return self._instance_id

    def create_team(
        self,
        team_card: object,
        user_id: str,
        user_email: str = "",
        team_id: uuid.UUID | None = None,
        catalog_namespace: str | None = None,
        metadata: object | None = None,
        workspaces: DeclaredWorkspaces | None = None,
    ) -> object:
        raise self._error


def _client_with_placement(
    services: CommunityServices,
    settings: CommunitySettings,
    error: Exception,
) -> TestClient:
    """Swap a raising placement stub into the services, then build the app."""
    services.placement = _RaisingPlacement(error)  # type: ignore[assignment]
    return TestClient(create_app(services, settings))


@pytest.fixture()
def no_capacity_client(
    community_services: CommunityServices,
    seeded_settings: CommunitySettings,
) -> Generator[TestClient, None, None]:
    """Client whose placement raises NoCapacityError."""
    yield _client_with_placement(
        community_services,
        seeded_settings,
        NoCapacityError("No worker available with capacity to place team"),
    )


@pytest.fixture()
def worker_rejected_client(
    community_services: CommunityServices,
    seeded_settings: CommunitySettings,
) -> Generator[TestClient, None, None]:
    """Client whose placement raises WorkerRejectedError."""
    yield _client_with_placement(
        community_services,
        seeded_settings,
        WorkerRejectedError("Worker returned 500 from create"),
    )


def test_no_capacity_maps_to_503(no_capacity_client: TestClient) -> None:
    """AC #13: NoCapacityError -> 503 + Retry-After + code=no_worker_capacity."""
    resp = no_capacity_client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert resp.status_code == 503
    assert "retry-after" in {k.lower() for k in resp.headers}
    body = resp.json()
    assert body["code"] == "no_worker_capacity"
    assert isinstance(body["detail"], str) and body["detail"]


def test_worker_rejected_maps_to_502(worker_rejected_client: TestClient) -> None:
    """AC #14: WorkerRejectedError -> 502 + code=worker_rejected."""
    resp = worker_rejected_client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert resp.status_code == 502
    body = resp.json()
    assert body["code"] == "worker_rejected"
    assert isinstance(body["detail"], str) and body["detail"]


# ---------------------------------------------------------------------------
# Story 68.2, AC #7 — a tier that routes gets the key and the refusal through
# the seam, unchanged, over HTTP.
# ---------------------------------------------------------------------------

ONE_META_NS = "acme-one-meta"
TWO_META_NS = "acme-two-meta"
_ACME_CASE = {"tenant": "acme", "case": "C1234"}


class _KeyRecordingPlacement:
    """A transcription check of the seam, not a router.

    Does what a multi-worker tier's placement must do at the seam — call
    ``workspaces.routing_key()`` and let its refusal propagate unchanged — and
    records what it saw before delegating to an inner ``MagicMock`` placement.
    It has no worker list and no instance table on purpose: it proves the key
    and the refusal reach the tier that asks for them, and nothing about where
    a second team would land.
    """

    def __init__(self, inner: MagicMock) -> None:
        self.inner = inner
        self.received: list[DeclaredWorkspaces | None] = []
        self.keys: list[PurePosixPath | None] = []
        self._instance_id = uuid.uuid4()

    @property
    def instance_id(self) -> uuid.UUID:
        return self._instance_id

    def create_team(
        self,
        team_card: object,
        user_id: str,
        user_email: str = "",
        team_id: uuid.UUID | None = None,
        catalog_namespace: str | None = None,
        metadata: object | None = None,
        workspaces: DeclaredWorkspaces | None = None,
    ) -> object:
        self.received.append(workspaces)
        if workspaces is None:
            msg = "TeamService never sends None"
            raise AssertionError(msg)
        # Raises UnroutableWorkspacesError for two _meta/ trees; propagated as is.
        self.keys.append(workspaces.routing_key())
        return self.inner.create_team(
            team_card,
            user_id,
            user_email=user_email,
            team_id=team_id,
            catalog_namespace=catalog_namespace,
            metadata=metadata,
            workspaces=workspaces,
        )


@pytest.fixture()
def routing_client(
    tmp_path: Path,
) -> Generator[tuple[TestClient, _KeyRecordingPlacement, CommunityServices], None, None]:
    """A wired app whose placement is the routing double, swapped in after the build.

    ``TeamService`` reads ``self._services.placement`` at call time, so the swap
    works on the container the app already holds. The catalog carries one team
    declaring one ``_meta/`` tree and one declaring two.
    """
    settings = CommunitySettings(
        workspaces_root=tmp_path / "workspaces",
        event_store_path=tmp_path / "event_store",
        catalog_path=tmp_path / "catalog",
    )
    seed_metadata_namespace(
        settings.catalog_path,
        ONE_META_NS,
        with_type=True,
        tools=[WorkspaceTool(workspace_metadata_keys=["tenant"])],
    )
    seed_metadata_namespace(
        settings.catalog_path,
        TWO_META_NS,
        with_type=True,
        tools=[
            WorkspaceTool(workspace_metadata_keys=["tenant"]),
            WorkspaceTool(workspace_metadata_keys=["case"]),
        ],
    )
    services = wire_community(settings)
    try:
        app = create_app(services, settings)
        double = _KeyRecordingPlacement(MagicMock())
        services.placement = double  # type: ignore[assignment]
        yield TestClient(app), double, services
    finally:
        services.actor_system.shutdown()


def test_one_metadata_tree_reaches_the_routing_tier_as_its_key(
    routing_client: tuple[TestClient, _KeyRecordingPlacement, CommunityServices],
) -> None:
    """AC #7: the key the double recorded is the tree, and placement was called with the value.

    The 502 ``placement_consistency`` is the expected exit — the inner mock
    persists no ``Process`` — and it is asserted rather than ignored, so a
    regression that failed earlier for another reason cannot hide behind it.
    """
    client, double, _ = routing_client

    resp = client.post("/teams/", json={"catalog_namespace": ONE_META_NS, "metadata": _ACME_CASE})

    assert resp.status_code == 502
    assert resp.json()["code"] == "placement_consistency"
    tree = PurePosixPath(METADATA_SCOPE) / "tenant-acme"
    assert double.keys == [tree]
    assert double.received == [DeclaredWorkspaces(shared={tree})]
    double.inner.create_team.assert_called_once()
    assert double.inner.create_team.call_args.kwargs["workspaces"] == DeclaredWorkspaces(
        shared={tree}
    )


def test_two_metadata_trees_are_refused_with_409_before_any_worker_is_asked(
    routing_client: tuple[TestClient, _KeyRecordingPlacement, CommunityServices],
) -> None:
    """AC #7: rule 3 over HTTP — 409, the typed code, both trees named, nothing created."""
    client, double, services = routing_client

    resp = client.post("/teams/", json={"catalog_namespace": TWO_META_NS, "metadata": _ACME_CASE})

    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "workspace_affinity_unsatisfiable"
    assert f"{METADATA_SCOPE}/tenant-acme" in body["detail"]
    assert f"{METADATA_SCOPE}/case-C1234" in body["detail"]
    assert "retry-after" not in {k.lower() for k in resp.headers}
    # The seam was reached with the value; the worker behind it never was.
    assert len(double.received) == 1
    assert double.keys == []
    double.inner.create_team.assert_not_called()
    _, total = services.team_service.list_teams(user_id="anonymous")
    assert total == 0


def test_an_unsatisfiable_workspace_card_is_422_before_the_seam(
    routing_client: tuple[TestClient, _KeyRecordingPlacement, CommunityServices],
) -> None:
    """AC #7: ``workspace_unresolvable`` is answered by the single ``ServerError`` handler.

    The body carries ``code``, which the route's own ``MetadataValidationError``
    branch never sets — so the key proves the handler path, with no route
    change. The double was never called at all: the refusal is pre-dispatch.
    """
    client, double, services = routing_client

    resp = client.post("/teams/", json={"catalog_namespace": ONE_META_NS})

    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "workspace_unresolvable"
    assert "carries no metadata" in body["detail"]
    assert double.received == []
    double.inner.create_team.assert_not_called()
    _, total = services.team_service.list_teams(user_id="anonymous")
    assert total == 0
