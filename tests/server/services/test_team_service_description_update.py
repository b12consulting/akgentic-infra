"""Service-level tests for ``TeamService.update_team_description`` — Story 80.1.

Complements the route tests: these pin what no response body can show — that
the write reaches the server's own event store exactly once, as a ``USER``
write, that the worker handle is consulted for the lifecycle answer and nothing
else, that infra performs no write of its own alongside it, and that two
service instances over one store answer identically.

Values use ``acme`` / ``contoso`` placeholders (Golden Rule #9).
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from akgentic.team.models import DescriptionOrigin, Process, TeamStatus

from akgentic.infra.errors import TeamNotFoundError, TeamStateConflictError
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import CommunitySettings

DESCRIPTION = "Triage inbound acme cases"


def _create(service: TeamService) -> uuid.UUID:
    """Create a team through the real path and return its id."""
    return service.create_team(catalog_namespace="test-team", user_id="alice").team_id


def _spy_event_store(service: TeamService) -> MagicMock:
    """Wrap the real event store in a spy that still performs the write."""
    real = service._services.event_store  # noqa: SLF001 — swapping the seam under test
    spy = MagicMock(wraps=real)
    service._services.event_store = spy  # type: ignore[assignment]
    return spy


def _spy_worker_handle(service: TeamService) -> MagicMock:
    """Wrap the real worker handle in a spy that still answers ``get_team``."""
    real = service._services.worker_handle  # noqa: SLF001 — swapping the seam under test
    spy = MagicMock(wraps=real)
    service._services.worker_handle = spy  # type: ignore[assignment]
    return spy


def _save_as_deleted(service: TeamService, team_id: uuid.UUID) -> Process:
    """Hold a ``DELETED`` record in the store, the only way the community tier can.

    ``TeamManager.delete_team`` purges the document, so a soft-deleted record
    has to be written by hand: stop the team, then re-save its ``Process`` with
    the status flipped and nothing else touched.
    """
    service.stop_team(team_id)
    process = service.get_team(team_id)
    assert process is not None
    deleted = process.model_copy(update={"status": TeamStatus.DELETED})
    service._services.event_store.save_team(deleted)  # noqa: SLF001 — seeding the store
    return deleted


def test_update_delegates_to_the_store_once_as_a_user_write(team_service: TeamService) -> None:
    """One conditional write, origin ``USER``, carrying the value as handed in.

    Asserted on the call rather than only on the persisted value: the endpoint
    is the only ``USER`` writer, and a service that passed ``AUTO`` would still
    land on a fresh team and only fail against a description the user already
    owns.
    """
    team_id = _create(team_service)
    store = _spy_event_store(team_service)

    team_service.update_team_description(team_id, DESCRIPTION)

    store.update_team_description.assert_called_once_with(
        team_id, DESCRIPTION, DescriptionOrigin.USER
    )


def test_update_returns_the_process_the_store_returned(team_service: TeamService) -> None:
    """The return value is the store's own answer — not a re-read, not the input."""
    team_id = _create(team_service)
    real_store = team_service._services.event_store  # noqa: SLF001 — captured before the spy
    store = _spy_event_store(team_service)
    returned_by_store: list[Process | None] = []

    def _recording(*args: object, **kwargs: object) -> Process | None:
        result: Process | None = real_store.update_team_description(*args, **kwargs)  # type: ignore[arg-type]
        returned_by_store.append(result)
        return result

    store.update_team_description.side_effect = _recording

    returned = team_service.update_team_description(team_id, DESCRIPTION)

    assert returned is returned_by_store[0]
    assert returned.team_description == DESCRIPTION
    assert returned.description_origin is DescriptionOrigin.USER


def test_the_worker_handle_is_consulted_for_the_lifecycle_answer_only(
    team_service: TeamService,
) -> None:
    """``get_team`` resolves the 404 / 409; no verb travels to the worker.

    The write goes to the server's store directly, so the only thing the
    handle may be asked is what the record says.
    """
    team_id = _create(team_service)
    handle = _spy_worker_handle(team_service)

    team_service.update_team_description(team_id, DESCRIPTION)

    assert handle.method_calls, "the lifecycle read must go through the handle"
    assert {call[0] for call in handle.method_calls} == {"get_team"}


def test_update_performs_no_cache_or_stream_or_handle_write(team_service: TeamService) -> None:
    """Infra adds nothing beside the one store write.

    The runtime cache, the live team handle and the event stream are all left
    untouched: a second write here would be a second, unordered path next to
    the conditional one the store owns.
    """
    team_id = _create(team_service)
    live_handle = team_service.get_handle(team_id)
    assert live_handle is not None

    cache = MagicMock(wraps=team_service._services.runtime_cache)  # noqa: SLF001
    stream = MagicMock(wraps=team_service._services.event_stream)  # noqa: SLF001
    team_service._services.runtime_cache = cache  # type: ignore[assignment]
    team_service._services.event_stream = stream  # type: ignore[assignment]
    team_service._cache = cache  # noqa: SLF001 — the service caches the reference
    spied_handle = MagicMock(wraps=live_handle)
    cache.get.return_value = spied_handle

    team_service.update_team_description(team_id, DESCRIPTION)

    assert cache.mock_calls == []
    assert stream.mock_calls == []
    assert spied_handle.mock_calls == []


def test_unknown_team_raises_not_found(team_service: TeamService) -> None:
    with pytest.raises(TeamNotFoundError, match="not found"):
        team_service.update_team_description(uuid.uuid4(), DESCRIPTION)


def test_a_team_that_vanishes_between_the_read_and_the_write_raises_not_found(
    team_service: TeamService,
) -> None:
    """The store answering ``None`` after the lifecycle read passed is a typed 404.

    The read and the write are two store round trips, and the port returns
    ``None`` when the conditional write matches no record — a delete landing in
    between. That ``None`` must become the same error an unknown team raises,
    never an ``AttributeError`` on the way to building the response.
    """
    team_id = _create(team_service)
    store = _spy_event_store(team_service)
    store.update_team_description.return_value = None

    with pytest.raises(TeamNotFoundError, match="not found"):
        team_service.update_team_description(team_id, DESCRIPTION)

    store.update_team_description.assert_called_once_with(
        team_id, DESCRIPTION, DescriptionOrigin.USER
    )


def test_deleted_record_raises_a_state_conflict_and_writes_nothing(
    team_service: TeamService,
) -> None:
    """AC8 at the service: a typed conflict, raised before the store is reached."""
    team_id = _create(team_service)
    deleted = _save_as_deleted(team_service, team_id)
    store = _spy_event_store(team_service)

    with pytest.raises(TeamStateConflictError, match="deleted"):
        team_service.update_team_description(team_id, DESCRIPTION)

    store.update_team_description.assert_not_called()
    assert team_service.get_team(team_id) == deleted


def test_clear_is_a_user_write_of_none(team_service: TeamService) -> None:
    """``None`` travels to the store as a ``USER`` write, so the latch holds."""
    team_id = _create(team_service)
    team_service.update_team_description(team_id, DESCRIPTION)

    cleared = team_service.update_team_description(team_id, None)

    assert cleared.team_description is None
    assert cleared.description_origin is DescriptionOrigin.USER


def test_update_holds_no_state_between_calls(team_service: TeamService) -> None:
    """The call adds no attribute to the service that survives it."""
    team_id = _create(team_service)

    before = dict(vars(team_service))
    team_service.update_team_description(team_id, DESCRIPTION)
    after = dict(vars(team_service))

    assert before.keys() == after.keys()
    assert all(before[key] is after[key] for key in before)


def test_two_service_instances_over_one_store_behave_identically(
    community_services: CommunityServices,
    seeded_settings: CommunitySettings,
) -> None:
    """Nothing is remembered between calls, or between service instances."""
    first = TeamService(
        services=community_services, workspaces_root=seeded_settings.workspaces_root
    )
    second = TeamService(
        services=community_services, workspaces_root=seeded_settings.workspaces_root
    )
    team_id = _create(first)

    first.update_team_description(team_id, "Triage acme cases")
    returned = second.update_team_description(team_id, "Triage contoso cases")

    assert returned.team_description == "Triage contoso cases"
    process = first.get_team(team_id)
    assert process is not None
    assert process.team_description == "Triage contoso cases"
    assert process.description_origin is DescriptionOrigin.USER
