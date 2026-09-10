"""Service-level tests for metadata on ``TeamService.create_team`` — Story 53.1.

Complements the route tests: these pin the *delegated call shape* (that the
validated model reaches placement, and that placement is not reached at all when
validation fails), which no response-body assertion can show.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock

import pytest
from akgentic.infra.errors import (
    MetadataValidationError,
    PlacementConsistencyError,
    WorkspaceDeclarationError,
)
from akgentic.infra.protocols.placement import DeclaredWorkspaces, UnroutableWorkspacesError
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.services._workspace_paths import declared_workspaces
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community
from akgentic.tool.workspace import METADATA_SCOPE, WorkspaceTool

from tests.fixtures.team_metadata import AcmeCaseMetadata, seed_metadata_namespace

TYPED_NS = "acme-cases"
UNTYPED_NS = "acme-plain"
# Story 68.2: typed teams whose Manager declares a metadata-keyed workspace —
# one card on two keys, and two cards on two key sets (the unsatisfiable shape).
META_WORKSPACE_NS = "acme-meta-workspace"
TWO_META_NS = "acme-two-meta"


@pytest.fixture()
def metadata_service(tmp_path: Path) -> Generator[TeamService, None, None]:
    """TeamService over a catalog holding one typed and one untyped team card."""
    settings = CommunitySettings(
        workspaces_root=tmp_path / "workspaces",
        event_store_path=tmp_path / "event_store",
        catalog_path=tmp_path / "catalog",
    )
    seed_metadata_namespace(settings.catalog_path, TYPED_NS, with_type=True)
    seed_metadata_namespace(settings.catalog_path, UNTYPED_NS, with_type=False)
    seed_metadata_namespace(
        settings.catalog_path,
        META_WORKSPACE_NS,
        with_type=True,
        tools=[WorkspaceTool(workspace_metadata_keys=["tenant", "case"])],
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
    services: CommunityServices = wire_community(settings)
    yield TeamService(services=services, workspaces_root=settings.workspaces_root)
    services.actor_system.shutdown()


def _mock_placement(service: TeamService) -> MagicMock:
    """Swap a MagicMock placement in and return it for call inspection."""
    placement = MagicMock()
    service._services.placement = placement  # type: ignore[assignment]
    return placement


def test_validated_model_is_forwarded_to_placement(metadata_service: TeamService) -> None:
    """The *validated model* reaches placement — not the raw request dict.

    Asserted on the call rather than only on the persisted Process: a stubbed
    placement accepts any kwargs silently, so a pass-through that dropped or
    forwarded the wrong value would otherwise go green.
    """
    placement = _mock_placement(metadata_service)

    # The specific type, not a bare Exception: the mock placement persists no
    # Process, so the consistency guard is the *expected* failure. A bare
    # Exception would also swallow a regression that failed earlier, for an
    # unrelated reason, on its way to the same assertions.
    with pytest.raises(PlacementConsistencyError):
        metadata_service.create_team(
            catalog_namespace=TYPED_NS,
            user_id="alice",
            metadata={"tenant": "acme", "case": "C-1234"},
        )

    forwarded = placement.create_team.call_args.kwargs["metadata"]
    assert isinstance(forwarded, AcmeCaseMetadata)
    assert forwarded.tenant == "acme"
    assert forwarded.case == "C-1234"


def test_placement_is_never_reached_when_validation_fails(
    metadata_service: TeamService,
) -> None:
    """AC #5 at the service seam: a rejected body creates nothing downstream.

    The route tests assert the team count is unchanged; this asserts the
    stronger, more direct fact — placement was never invoked at all.
    """
    placement = _mock_placement(metadata_service)

    with pytest.raises(MetadataValidationError):
        metadata_service.create_team(
            catalog_namespace=TYPED_NS, user_id="alice", metadata={"tenant": "acme"}
        )

    placement.create_team.assert_not_called()


def test_model_tag_is_rejected_at_the_service_layer(metadata_service: TeamService) -> None:
    """The ``__model__`` refusal is in the service, not only in the route."""
    placement = _mock_placement(metadata_service)

    with pytest.raises(MetadataValidationError, match="__model__"):
        metadata_service.create_team(
            catalog_namespace=TYPED_NS,
            user_id="alice",
            metadata={"__model__": "akgentic.infra.server.models.TeamResponse"},
        )

    placement.create_team.assert_not_called()


def test_metadata_for_an_untyped_card_is_rejected(metadata_service: TeamService) -> None:
    """A card declaring no ``metadata_type`` refuses a non-empty body."""
    placement = _mock_placement(metadata_service)

    with pytest.raises(MetadataValidationError, match="no metadata contract"):
        metadata_service.create_team(
            catalog_namespace=UNTYPED_NS, user_id="alice", metadata={"tenant": "acme"}
        )

    placement.create_team.assert_not_called()


def test_omitted_metadata_forwards_none(metadata_service: TeamService) -> None:
    """A caller that supplies no metadata forwards ``None``, unchanged behaviour."""
    placement = _mock_placement(metadata_service)

    with pytest.raises(PlacementConsistencyError):
        metadata_service.create_team(catalog_namespace=TYPED_NS, user_id="alice")

    assert placement.create_team.call_args.kwargs["metadata"] is None


def test_valid_metadata_lands_on_the_persisted_process(metadata_service: TeamService) -> None:
    """End to end through the real placement: the value reaches the store."""
    process = metadata_service.create_team(
        catalog_namespace=TYPED_NS,
        user_id="alice",
        metadata={"tenant": "acme", "case": "C-1234", "note": "escalated"},
    )
    assert isinstance(process.metadata, AcmeCaseMetadata)
    assert process.metadata.tenant == "acme"
    assert process.metadata.note == "escalated"
    # Index derivation happens once, inside akgentic-team — never re-derived here.
    assert "tenant|acme" in process.metadata_indexes


# ---------------------------------------------------------------------------
# Story 68.2 — the resolved workspaces reach placement, or nothing does
# ---------------------------------------------------------------------------


def test_the_resolved_workspaces_reach_placement_unconditionally(
    metadata_service: TeamService,
) -> None:
    """AC #5: ``workspaces=`` carries the value the pre-dispatch resolver produces.

    Asserted on the value — equal to what ``declared_workspaces`` returns for the
    resolved card, the request's owner and its validated metadata, and equal to
    the literal tree — never on ``isinstance`` alone, which a value built from
    the wrong inputs would also satisfy.
    """
    placement = _mock_placement(metadata_service)

    with pytest.raises(PlacementConsistencyError):
        metadata_service.create_team(
            catalog_namespace=META_WORKSPACE_NS,
            user_id="alice",
            metadata={"tenant": "acme", "case": "C1234"},
        )

    forwarded = placement.create_team.call_args.kwargs["workspaces"]
    assert isinstance(forwarded, DeclaredWorkspaces)
    card = metadata_service._services.catalog.load_team(META_WORKSPACE_NS)
    expected = declared_workspaces(
        card,
        user_id="alice",
        team_id=None,
        metadata=AcmeCaseMetadata(tenant="acme", case="C1234"),
    )
    assert forwarded == expected
    assert forwarded.shared == {PurePosixPath(METADATA_SCOPE) / "tenant-acme__case-C1234"}
    # No default card on this team and no id supplied: nothing to put in ``own``.
    assert forwarded.own is None


def test_an_unsatisfiable_workspace_card_is_refused_before_placement(
    metadata_service: TeamService,
) -> None:
    """AC #5: a metadata card on a request carrying no metadata is a 422, and nothing is created.

    Same shape as ``test_placement_is_never_reached_when_validation_fails``: the
    strong fact is that placement was never invoked, not that a count is
    unchanged. The detail carries the tool resolver's own message, naming the
    keys, so the admin is told which card could not be satisfied.
    """
    placement = _mock_placement(metadata_service)

    with pytest.raises(WorkspaceDeclarationError) as excinfo:
        metadata_service.create_team(catalog_namespace=META_WORKSPACE_NS, user_id="alice")

    err = excinfo.value
    assert err.status_code == 422
    assert err.code == "workspace_unresolvable"
    assert "carries no metadata" in err.detail
    assert "tenant" in err.detail and "case" in err.detail
    placement.create_team.assert_not_called()


def test_a_team_declaring_two_metadata_trees_reaches_placement_on_the_community_tier(
    metadata_service: TeamService,
) -> None:
    """AC #6, service layer: the community tier refuses nothing.

    ``TeamService`` computes the value and passes it through without consulting
    it; the only exit is the consistency error the mock's missing ``Process``
    causes, exactly as in ``test_validated_model_is_forwarded_to_placement``.
    The last assertion is what keeps this from being vacuous: the value that
    reached placement is one a tier that routes **would** refuse.
    """
    placement = _mock_placement(metadata_service)

    with pytest.raises(PlacementConsistencyError):
        metadata_service.create_team(
            catalog_namespace=TWO_META_NS,
            user_id="alice",
            metadata={"tenant": "acme", "case": "C1234"},
        )

    forwarded = placement.create_team.call_args.kwargs["workspaces"]
    assert forwarded.shared == {
        PurePosixPath(METADATA_SCOPE) / "tenant-acme",
        PurePosixPath(METADATA_SCOPE) / "case-C1234",
    }
    with pytest.raises(UnroutableWorkspacesError):
        forwarded.routing_key()


def test_create_team_holds_no_metadata_state(metadata_service: TeamService) -> None:
    """AC #10: creating with metadata adds no attribute that survives the call."""
    before = dict(vars(metadata_service))
    metadata_service.create_team(
        catalog_namespace=TYPED_NS,
        user_id="alice",
        metadata={"tenant": "acme", "case": "C-1234"},
    )
    after = dict(vars(metadata_service))
    assert before.keys() == after.keys()
    assert all(before[key] is after[key] for key in before)
