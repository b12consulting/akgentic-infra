"""Service-level tests for metadata on ``TeamService.create_team`` — Story 53.1.

Complements the route tests: these pin the *delegated call shape* (that the
validated model reaches placement, and that placement is not reached at all when
validation fails), which no response-body assertion can show.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from akgentic.infra.errors import MetadataValidationError, PlacementConsistencyError
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community

from tests.fixtures.team_metadata import AcmeCaseMetadata, seed_metadata_namespace

TYPED_NS = "acme-cases"
UNTYPED_NS = "acme-plain"


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
