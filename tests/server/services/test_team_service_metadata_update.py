"""Service-level tests for ``TeamService.update_team_metadata`` — Story 53.3.

Complements the route tests: these pin what no response body can show — that the
update reaches the worker-handle seam exactly once carrying the *validated
model*, that infra performs no write of its own alongside it, and that two
service instances over one store answer identically.

Field names and values use ``acme`` / ``contoso`` placeholders (Golden Rule #9).
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from akgentic.infra.errors import MetadataValidationError
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community

from tests.fixtures.team_metadata import AcmeCaseMetadata, seed_metadata_namespace

TYPED_NS = "acme-cases"
UNTYPED_NS = "acme-plain"


@pytest.fixture()
def metadata_settings(tmp_path: Path) -> CommunitySettings:
    """Community settings with one typed and one untyped team namespace."""
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
def metadata_service(
    metadata_services: CommunityServices,
    metadata_settings: CommunitySettings,
) -> TeamService:
    """TeamService over the real community wiring."""
    return TeamService(
        services=metadata_services, workspaces_root=metadata_settings.workspaces_root
    )


def _create(service: TeamService, namespace: str = TYPED_NS, **fields: Any) -> uuid.UUID:
    """Create a team through the real path and return its id."""
    process = service.create_team(
        catalog_namespace=namespace, user_id="alice", metadata=dict(fields) or None
    )
    return process.team_id


def _spy_worker_handle(service: TeamService, team_id: uuid.UUID) -> MagicMock:
    """Wrap the real worker handle in a spy that records the update call.

    ``get_team`` keeps delegating to the real handle so the ``Process`` — and
    therefore the declared ``metadata_type`` — is genuine; only the update is
    intercepted, and it returns the real result so the caller still reads a
    persisted value.
    """
    real = service._services.worker_handle  # noqa: SLF001 — swapping the seam under test
    spy = MagicMock(wraps=real)
    spy.get_team.side_effect = real.get_team
    spy.update_team_metadata.side_effect = real.update_team_metadata
    service._services.worker_handle = spy  # type: ignore[assignment]
    assert real.get_team(team_id) is not None
    return spy


def test_update_delegates_to_the_seam_once_with_the_validated_model(
    metadata_service: TeamService,
) -> None:
    """The seam receives the validated model, not the raw request dict.

    Asserted on the call rather than only on the persisted value: the store
    would accept either, so a pass-through that handed the dict down would only
    fail further inside akgentic-team, or not at all.
    """
    team_id = _create(metadata_service, tenant="acme", case="C-1")
    spy = _spy_worker_handle(metadata_service, team_id)

    metadata_service.update_team_metadata(team_id, {"tenant": "contoso", "case": "C-2"})

    spy.update_team_metadata.assert_called_once()
    called_id, forwarded = spy.update_team_metadata.call_args.args
    assert called_id == team_id
    assert isinstance(forwarded, AcmeCaseMetadata)
    assert forwarded.tenant == "contoso"
    assert forwarded.case == "C-2"


def test_update_returns_what_the_seam_persisted(metadata_service: TeamService) -> None:
    """The return value comes off the returned Process, not off the request body.

    A service that echoed its argument back would look identical on the happy
    path and hide a write that never landed.
    """
    team_id = _create(metadata_service, tenant="acme", case="C-1")

    returned = metadata_service.update_team_metadata(team_id, {"tenant": "contoso", "case": "C-2"})

    assert isinstance(returned, AcmeCaseMetadata)
    process = metadata_service.get_team(team_id)
    assert process is not None
    assert returned == process.metadata


def test_update_performs_no_push_or_cache_or_stream_write(
    metadata_service: TeamService,
) -> None:
    """Infra adds nothing beside the ordered write path it delegates to.

    The runtime cache, the team handle it hands out and the event stream are all
    left untouched: index derivation and the best-effort orchestrator push
    belong to akgentic-team, and a second write here would be a second,
    unordered path next to the one the decision defines.
    """
    team_id = _create(metadata_service, tenant="acme", case="C-1")
    handle = metadata_service.get_handle(team_id)
    assert handle is not None

    cache = MagicMock(wraps=metadata_service._services.runtime_cache)  # noqa: SLF001
    stream = MagicMock(wraps=metadata_service._services.event_stream)  # noqa: SLF001
    metadata_service._services.runtime_cache = cache  # type: ignore[assignment]
    metadata_service._services.event_stream = stream  # type: ignore[assignment]
    metadata_service._cache = cache  # noqa: SLF001 — the service caches the reference

    spied_handle = MagicMock(wraps=handle)
    cache.get.return_value = spied_handle

    metadata_service.update_team_metadata(team_id, {"tenant": "contoso", "case": "C-2"})

    cache.store.assert_not_called()
    cache.remove.assert_not_called()
    stream.append.assert_not_called()
    stream.remove.assert_not_called()
    assert spied_handle.mock_calls == []


def test_unknown_team_raises_value_error_carrying_the_404_substring(
    metadata_service: TeamService,
) -> None:
    """The message must contain ``not found`` — the router maps 404 by that substring.

    A rephrased message would silently become a 409 conflict at the HTTP edge.
    """
    with pytest.raises(ValueError, match="not found"):
        metadata_service.update_team_metadata(uuid.uuid4(), {"tenant": "acme", "case": "C-1"})


def test_validation_failure_never_reaches_the_seam(metadata_service: TeamService) -> None:
    """A rejected body stops above the write path — nothing downstream is called."""
    team_id = _create(metadata_service, tenant="acme", case="C-1")
    spy = _spy_worker_handle(metadata_service, team_id)

    with pytest.raises(MetadataValidationError):
        metadata_service.update_team_metadata(team_id, {"tenant": "contoso"})

    spy.update_team_metadata.assert_not_called()


def test_model_tag_never_reaches_the_seam(metadata_service: TeamService) -> None:
    """The ``__model__`` refusal lives in the service, not only in the route."""
    team_id = _create(metadata_service, tenant="acme", case="C-1")
    spy = _spy_worker_handle(metadata_service, team_id)

    with pytest.raises(MetadataValidationError, match="__model__"):
        metadata_service.update_team_metadata(
            team_id, {"__model__": "akgentic.infra.server.models.TeamResponse"}
        )

    spy.update_team_metadata.assert_not_called()


def test_untyped_card_is_refused_at_the_service_layer(metadata_service: TeamService) -> None:
    """A team whose card declares no contract refuses a non-empty document."""
    team_id = _create(metadata_service, namespace=UNTYPED_NS)
    spy = _spy_worker_handle(metadata_service, team_id)

    with pytest.raises(MetadataValidationError, match="no metadata contract"):
        metadata_service.update_team_metadata(team_id, {"tenant": "acme"})

    spy.update_team_metadata.assert_not_called()


def test_update_holds_no_state_between_calls(metadata_service: TeamService) -> None:
    """The call adds no attribute to the service that survives it."""
    team_id = _create(metadata_service, tenant="acme", case="C-1")

    before = dict(vars(metadata_service))
    metadata_service.update_team_metadata(team_id, {"tenant": "contoso", "case": "C-2"})
    after = dict(vars(metadata_service))

    assert before.keys() == after.keys()
    assert all(before[key] is after[key] for key in before)


def test_two_service_instances_over_one_store_behave_identically(
    metadata_services: CommunityServices,
    metadata_settings: CommunitySettings,
) -> None:
    """Nothing is remembered between calls, or between service instances.

    Two successive patches issued through two independently constructed services
    produce the same result as two through one — which is what "any replica
    serves any request" means at this layer.
    """
    first = TeamService(
        services=metadata_services, workspaces_root=metadata_settings.workspaces_root
    )
    second = TeamService(
        services=metadata_services, workspaces_root=metadata_settings.workspaces_root
    )
    team_id = _create(first, tenant="acme", case="C-1")

    first.update_team_metadata(team_id, {"tenant": "contoso", "case": "C-2"})
    returned = second.update_team_metadata(team_id, {"tenant": "northwind", "case": "C-3"})

    assert isinstance(returned, AcmeCaseMetadata)
    assert returned.tenant == "northwind"
    process = first.get_team(team_id)
    assert process is not None
    assert process.metadata == returned
