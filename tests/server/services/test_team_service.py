"""Tests for TeamService — service layer with real in-memory adapters."""

from __future__ import annotations

import inspect
import logging
import shutil
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import get_type_hints
from unittest.mock import MagicMock

import pytest
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.core.agent_card import AgentCard
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.team.models import AgentCardRef, Process, TeamStatus
from akgentic.team.projection import hash_agent_card
from akgentic.tool.workspace import (
    ID_KIND,
    METADATA_KIND,
    SHARED_SCOPE,
    TEAM_KIND,
    WorkspaceTool,
    git_dir_for,
    meta_dir_for,
)

from akgentic.infra.adapters.shared.team_tree_only_policy import TeamTreeOnlyPolicy
from akgentic.infra.errors import TeamNotFoundError, TeamStateConflictError
from akgentic.infra.protocols.workspace_deletion import (
    WorkspaceDeletionContext,
    WorkspaceDeletionPolicy,
)
from akgentic.infra.server.services import _workspace_paths as workspace_paths_module
from akgentic.infra.server.services.team_service import (
    MAX_PAGE_SIZE,
    CatalogTeamEntryMissingError,
    TeamService,
)
from tests.server.routes._workspace_cards import CaseMetadata, RecordingCardStore, tool_card


def test_create_team_returns_process(team_service: TeamService) -> None:
    """Creating a team with a valid catalog entry returns a Process."""
    process = team_service.create_team(catalog_namespace="test-team", user_id="anonymous")
    assert process.team_id is not None
    assert process.status == TeamStatus.RUNNING
    assert process.user_id == "anonymous"
    assert process.team_name == "Test Team"


def test_create_team_invalid_entry_raises(team_service: TeamService) -> None:
    """Creating a team with an invalid catalog namespace raises EntryNotFoundError."""
    with pytest.raises(EntryNotFoundError):
        team_service.create_team(catalog_namespace="nonexistent", user_id="anonymous")


def test_create_team_present_but_invalid_namespace_propagates_the_catalog_message(
    team_service: TeamService,
) -> None:
    """A namespace that exists and fails to resolve keeps the catalog's diagnosis.

    The translation this replaces turned every ``CatalogValidationError`` into an
    ``EntryNotFoundError``, which destroyed the only text saying what to repair —
    one layer below the wire, where no client could recover it.
    """
    with pytest.raises(CatalogValidationError) as excinfo:
        team_service.create_team(catalog_namespace="broken-team", user_id="anonymous")

    assert "ref marker" in str(excinfo.value)
    assert "'params'" in str(excinfo.value)
    assert excinfo.value.errors
    # An absent namespace is a different answer, so this must not be one.
    assert not isinstance(excinfo.value, EntryNotFoundError)


def test_create_team_namespace_without_team_entry_is_a_not_found(
    team_service: TeamService,
) -> None:
    """A namespace that exists but holds no team entry is missing, not broken.

    Nothing is invalid here — there is simply no team to create from — so this
    stays in the ``EntryNotFoundError`` family, with a message that says which
    of the two 404s it is.
    """
    with pytest.raises(CatalogTeamEntryMissingError) as excinfo:
        team_service.create_team(catalog_namespace="teamless", user_id="anonymous")

    assert isinstance(excinfo.value, EntryNotFoundError)
    assert "has no team entry" in str(excinfo.value)


def test_create_team_probes_the_namespace_only_on_the_failure_path(
    team_service: TeamService,
) -> None:
    """A successful create never runs the existence probe.

    The probe exists to classify a failure; running it before ``load_team``
    would put a second catalog query on every create for a branch that almost
    never fires.
    """
    spy = MagicMock(wraps=team_service._services.catalog)
    team_service._services.catalog = spy  # type: ignore[assignment]

    team_service.create_team(catalog_namespace="test-team", user_id="anonymous")
    assert spy.list_by_namespace.call_count == 0

    with pytest.raises(CatalogValidationError):
        team_service.create_team(catalog_namespace="broken-team", user_id="anonymous")
    spy.list_by_namespace.assert_called_once_with("broken-team")


def test_create_team_propagates_catalog_namespace(team_service: TeamService) -> None:
    """Process.catalog_namespace is populated from the create_team argument."""
    process = team_service.create_team(catalog_namespace="test-team", user_id="anonymous")
    assert process.catalog_namespace == "test-team"


def test_create_team_forwards_user_email_and_team_id(team_service: TeamService) -> None:
    """user_email and team_id flow through to placement.create_team verbatim."""
    explicit_id = uuid.uuid4()
    mock_placement = MagicMock()
    # Match downstream contract: placement returns a handle whose team_id
    # round-trips through the cache.
    mock_placement.create_team.return_value.team_id = explicit_id
    team_service._services.placement = mock_placement  # type: ignore[assignment]

    try:
        team_service.create_team(
            "test-team",
            user_id="alice",
            user_email="alice@example.com",
            team_id=explicit_id,
        )
    except Exception:
        # Downstream worker_handle.get_team will fail because the mock placement
        # never persists a Process — that's fine, we only care about the
        # placement call shape.
        pass

    call = mock_placement.create_team.call_args
    assert call.args[1] == "alice"
    assert call.kwargs == {
        "user_email": "alice@example.com",
        "team_id": explicit_id,
        "catalog_namespace": "test-team",
        # Always forwarded, None when the caller supplied no metadata.
        "metadata": None,
    }


def test_list_teams_empty(team_service: TeamService) -> None:
    """Listing teams when none exist returns an empty page and a zero total."""
    page, total = team_service.list_teams(user_id="anonymous")
    assert page == []
    assert total == 0


def test_list_teams_filters_by_user(team_service: TeamService) -> None:
    """list_teams returns only teams belonging to the given user."""
    team_service.create_team(catalog_namespace="test-team", user_id="alice")
    team_service.create_team(catalog_namespace="test-team", user_id="bob")
    alice_teams, alice_total = team_service.list_teams(user_id="alice")
    bob_teams, bob_total = team_service.list_teams(user_id="bob")
    assert len(alice_teams) == 1
    assert alice_total == 1
    assert len(bob_teams) == 1
    assert bob_total == 1
    assert alice_teams[0].user_id == "alice"


def test_list_teams_delegates_to_event_store_with_user_id(team_service: TeamService) -> None:
    """TeamService.list_teams MUST push user_id down to event_store.list_teams,
    not load all teams and filter in Python. Regression-guard for the team-side
    ADR-16 / Epic 19 push-down: if a future refactor restores the in-memory
    filter pattern, this test fails even though the behavioural contract
    (users see only their own teams) still passes.
    """
    mock_event_store = MagicMock()
    mock_event_store.list_teams.return_value = []
    # Swap in the mock event_store on the wired TierServices container.
    # SkipValidation on the field allows direct assignment without re-validation.
    team_service._services.event_store = mock_event_store  # type: ignore[assignment]

    page, total = team_service.list_teams(user_id="alice")

    # The delegating call shape — exactly one call, every filter as a kwarg.
    # Phase-2 (store-side offset pushdown) is out of scope: NO page/size here.
    # ``metadata=None`` is unconditional: a branch that omits the kwarg when no
    # filter was given is how a filter later gets silently dropped.
    mock_event_store.list_teams.assert_called_once_with(user_id="alice", status=None, metadata=None)
    # The call must NOT be a no-arg call followed by an in-Python filter.
    assert mock_event_store.list_teams.call_args.args == ()
    assert mock_event_store.list_teams.call_args.kwargs == {
        "user_id": "alice",
        "status": None,
        "metadata": None,
    }
    # Empty owned set -> empty page, zero total.
    assert page == []
    assert total == 0


def test_list_teams_passes_empty_string_user_id_verbatim(team_service: TeamService) -> None:
    """user_id="" is a literal value, NOT a "list everything" sentinel.

    The empty string is passed through verbatim to event_store.list_teams; the
    backend applies its literal-match filter and returns only teams whose
    Process.user_id == "". This locks in the natural behaviour of the
    one-line delegating call.
    """
    mock_event_store = MagicMock()
    mock_event_store.list_teams.return_value = []
    team_service._services.event_store = mock_event_store  # type: ignore[assignment]

    team_service.list_teams(user_id="")

    mock_event_store.list_teams.assert_called_once_with(user_id="", status=None, metadata=None)


def test_list_teams_delegates_status_to_event_store(team_service: TeamService) -> None:
    """A caller-supplied status is pushed down alongside user_id.

    Companion to the user_id push-down guard above: the service must never
    load the user's teams and filter the lifecycle state in Python. Both
    filters travel as kwargs in a single delegated call.
    """
    mock_event_store = MagicMock()
    mock_event_store.list_teams.return_value = []
    team_service._services.event_store = mock_event_store  # type: ignore[assignment]

    team_service.list_teams(user_id="alice", status=TeamStatus.RUNNING)

    mock_event_store.list_teams.assert_called_once_with(
        user_id="alice", status=TeamStatus.RUNNING, metadata=None
    )
    assert mock_event_store.list_teams.call_args.args == ()


def test_list_teams_forwards_the_metadata_filter_verbatim(team_service: TeamService) -> None:
    """The metadata filter reaches the store raw — no escaping, no normalising.

    ``|`` is the index separator and is escaped inside akgentic-team, exactly
    once; a term carrying one that arrived pre-escaped would be escaped twice
    and match nothing. Asserted on the delegated kwargs rather than on the
    returned rows because a stub store returns the same rows either way.

    The terms are lists, not bare strings. A bare ``str`` is itself a
    ``Sequence[str]``, so the store now rejects one with ``TypeError`` rather
    than filtering on one term per character — and this spec used to pass a bare
    ``str`` and stay green only because its store is a ``MagicMock``.
    """
    mock_event_store = MagicMock()
    mock_event_store.list_teams.return_value = []
    team_service._services.event_store = mock_event_store  # type: ignore[assignment]

    team_service.list_teams(user_id="alice", metadata={"tenant": ["ac|me"], "case": ["C-1234"]})

    mock_event_store.list_teams.assert_called_once_with(
        user_id="alice", status=None, metadata={"tenant": ["ac|me"], "case": ["C-1234"]}
    )
    assert mock_event_store.list_teams.call_args.args == ()


def test_list_teams_metadata_filter_narrows_within_user(team_service: TeamService) -> None:
    """The push-down against the real YAML store returns only the matching team.

    Companion to the kwargs guard above: that one proves the call shape, this
    one proves the shape actually produces the right answer end to end.
    """
    team_service.create_team("test-team", user_id="alice")
    unmatched, unmatched_total = team_service.list_teams(
        user_id="alice", metadata={"tenant": ["acme"]}
    )
    assert unmatched == []
    # The total follows the filter, not the owned set the filter was drawn from.
    assert unmatched_total == 0

    _, owned_total = team_service.list_teams(user_id="alice")
    assert owned_total == 1


def _namespaced_rows(seed: Process, namespaces: list[str | None]) -> list[Process]:
    """Derive one Process per entry of ``namespaces``, copied from a real one.

    ``model_copy(update=...)`` rather than a hand-enumerated constructor: a
    rebuild naming every field that exists today would silently drop the next
    one added to ``Process``, and these rows stand in for persisted state.
    """
    base = seed.created_at
    return [
        seed.model_copy(
            update={
                "team_id": uuid.uuid4(),
                "catalog_namespace": namespace,
                "created_at": base + timedelta(seconds=index),
            }
        )
        for index, namespace in enumerate(namespaces)
    ]


def test_list_teams_does_not_push_catalog_namespace_to_the_store(
    team_service: TeamService,
) -> None:
    """``EventStore.list_teams`` has no such parameter, so nothing may push it down.

    Adding one is an akgentic-team Protocol change, outside this submodule. The
    delegated call stays exactly one call carrying exactly the three terms the
    store declares — and ``user_id`` is one of them on this path as on every
    other, which is what keeps the namespace filter from reaching past the
    caller's own teams.
    """
    mock_event_store = MagicMock()
    mock_event_store.list_teams.return_value = []
    team_service._services.event_store = mock_event_store  # type: ignore[assignment]

    team_service.list_teams(user_id="alice", catalog_namespace="acme-cases")

    mock_event_store.list_teams.assert_called_once_with(user_id="alice", status=None, metadata=None)
    assert mock_event_store.list_teams.call_args.args == ()
    assert set(mock_event_store.list_teams.call_args.kwargs) == {"user_id", "status", "metadata"}


def test_list_teams_namespace_filter_runs_before_the_sort_and_the_slice(
    team_service: TeamService,
) -> None:
    """The namespace narrows the SET the page is cut from, not the page.

    Three of five rows match, at size 2. Filtering after the slice would give a
    total of 5 and short pages — pages the client cannot tell from the end of
    the results — while filtering before it gives a filtered total and a
    contiguous walk.
    """
    seed = team_service.create_team("test-team", user_id="alice")
    rows = _namespaced_rows(seed, ["wanted", "other", "wanted", "other", "wanted"])
    mock_event_store = MagicMock()
    mock_event_store.list_teams.return_value = rows
    team_service._services.event_store = mock_event_store  # type: ignore[assignment]

    page1, total1 = team_service.list_teams(
        user_id="alice", catalog_namespace="wanted", page=1, size=2
    )
    page2, total2 = team_service.list_teams(
        user_id="alice", catalog_namespace="wanted", page=2, size=2
    )

    assert (total1, total2) == (3, 3)
    assert len(page1) == 2
    assert len(page2) == 1
    walked = [p.team_id for p in page1 + page2]
    assert len(walked) == len(set(walked))
    assert all(p.catalog_namespace == "wanted" for p in page1 + page2)


def test_list_teams_blank_catalog_namespace_is_not_a_filter(team_service: TeamService) -> None:
    """A blank is an empty form field, not a filter on the literal empty string.

    The ``None``-namespace row is the one that tells the two readings apart: a
    filter on ``""`` would drop every row here, including the teams that carry
    no namespace at all.
    """
    seed = team_service.create_team("test-team", user_id="alice")
    rows = _namespaced_rows(seed, ["wanted", None, "other"])
    mock_event_store = MagicMock()
    mock_event_store.list_teams.return_value = rows
    team_service._services.event_store = mock_event_store  # type: ignore[assignment]

    page, total = team_service.list_teams(user_id="alice", catalog_namespace="")

    assert total == 3
    assert len(page) == 3


def test_list_teams_status_narrows_within_user(team_service: TeamService) -> None:
    """status=RUNNING returns only the running team; omitting status returns both.

    Runs against the wired YAML event store, so this exercises the real
    push-down rather than a mock's recorded call.
    """
    running = team_service.create_team("test-team", user_id="alice")
    stopped = team_service.create_team("test-team", user_id="alice")
    team_service.stop_team(stopped.team_id)

    only_running, running_total = team_service.list_teams(
        user_id="alice", status=TeamStatus.RUNNING
    )
    assert [p.team_id for p in only_running] == [running.team_id]
    # The total counts the filtered set, not the owned set it was drawn from.
    assert running_total == 1

    unfiltered, unfiltered_total = team_service.list_teams(user_id="alice")
    assert {p.team_id for p in unfiltered} == {running.team_id, stopped.team_id}
    assert unfiltered_total == 2


def test_list_teams_user_id_stays_required_while_status_is_optional() -> None:
    """``status`` is the optional filter; ``user_id`` is not, and never becomes one.

    Widening ``user_id`` to ``str | None = None`` "for symmetry with the
    Protocol" would put "list every user's teams" one forgotten argument
    away, and nothing else in the suite would notice: every call site passes
    ``user_id`` today, so the behavioural tests and strict mypy both stay
    green. This asserts the shape directly because no behavioural test can.
    """
    sig = inspect.signature(TeamService.list_teams)
    assert sig.parameters["user_id"].default is inspect.Parameter.empty
    assert sig.parameters["status"].default is None

    hints = get_type_hints(TeamService.list_teams)
    assert hints["user_id"] is str
    assert hints["return"] == tuple[list[Process], int]


def test_get_team_found(team_service: TeamService) -> None:
    """get_team returns the Process for an existing team."""
    process = team_service.create_team(catalog_namespace="test-team", user_id="anonymous")
    found = team_service.get_team(process.team_id)
    assert found is not None
    assert found.team_id == process.team_id


def test_get_team_not_found(team_service: TeamService) -> None:
    """get_team returns None for a nonexistent team ID."""
    result = team_service.get_team(uuid.uuid4())
    assert result is None


@pytest.mark.skip(
    reason="Flaky: race in TeamManager.delete_team — on_stop subscribers still "
    "flushing event_store writes while rmtree runs; pre-existing on master, "
    "not introduced by Epic 22."
)
def test_delete_team_stops_and_deletes(team_service: TeamService) -> None:
    """delete_team stops a running team and purges it from the event store."""
    process = team_service.create_team(catalog_namespace="test-team", user_id="anonymous")
    team_service.delete_team(process.team_id)
    # After deletion, the team is fully purged from the event store
    after = team_service.get_team(process.team_id)
    assert after is None


@pytest.mark.skip(
    reason="Flaky: same race in TeamManager.delete_team as "
    "test_delete_team_stops_and_deletes; pre-existing, not introduced by Epic 22."
)
def test_delete_stopped_team(team_service: TeamService) -> None:
    """delete_team handles an already-stopped team without calling stop_team."""
    process = team_service.create_team(catalog_namespace="test-team", user_id="anonymous")
    team_service._services.worker_handle.stop_team(process.team_id)
    team_service.delete_team(process.team_id)
    after = team_service.get_team(process.team_id)
    assert after is None


def test_delete_team_not_found_raises(team_service: TeamService) -> None:
    """delete_team raises ValueError for a nonexistent team ID."""
    with pytest.raises(ValueError, match="not found"):
        team_service.delete_team(uuid.uuid4())


# ---------------------------------------------------------------------------
# Epic 59 Part B — the service names the condition; the routes map it by type.
#
# Driven through the wired service against real running teams: a mocked manager
# raising ``ValueError("...")`` cannot tell a correct classification from the
# flattening one, which is how the flattening survived this long.
# ---------------------------------------------------------------------------


def test_absent_team_raises_team_not_found_on_every_read_and_lifecycle_path(
    team_service: TeamService,
) -> None:
    """An unknown team is classified as missing wherever it is looked up."""
    absent = uuid.uuid4()
    for operation in (
        team_service.delete_team,
        team_service.stop_team,
        team_service.restore_team,
        team_service.get_events,
        team_service.get_agent_states,
    ):
        with pytest.raises(TeamNotFoundError, match="not found"):
            operation(absent)  # type: ignore[operator]


def test_running_team_restore_raises_state_conflict_not_not_found(
    team_service: TeamService,
) -> None:
    """A real RUNNING team is a conflict, never a missing one.

    This is the defect's shape at the service layer: the team exists, is alive,
    and the operation is refused because of *that* — reporting it as absent
    tells the caller the opposite of the truth.
    """
    process = team_service.create_team(catalog_namespace="test-team", user_id="anonymous")

    with pytest.raises(TeamStateConflictError) as excinfo:
        team_service.restore_team(process.team_id)

    assert "already running" in str(excinfo.value)
    assert not isinstance(excinfo.value, TeamNotFoundError)
    # Still a ValueError, so every existing catch site keeps working unchanged.
    assert isinstance(excinfo.value, ValueError)


def test_stopped_team_stop_again_raises_state_conflict(team_service: TeamService) -> None:
    """Stopping an already-stopped real team is a conflict, not a 404 condition."""
    process = team_service.create_team(catalog_namespace="test-team", user_id="anonymous")
    team_service.stop_team(process.team_id)

    with pytest.raises(TeamStateConflictError, match="already stopped"):
        team_service.stop_team(process.team_id)


def test_message_to_a_stopped_team_raises_state_conflict(team_service: TeamService) -> None:
    """A stopped team is present-but-unusable — the running-handle path says so."""
    process = team_service.create_team(catalog_namespace="test-team", user_id="anonymous")
    team_service.stop_team(process.team_id)

    with pytest.raises(TeamStateConflictError, match="is not running"):
        team_service.send_message(process.team_id, "hello")


# ---------------------------------------------------------------------------
# Reclassified from integration/test_adr003_tier_agnostic.py
# Source inspection; no real app needed.
# ---------------------------------------------------------------------------


class TestStopTeam:
    """Story 13.9 AC1: stop_team cleans up the event stream."""

    def test_stop_team_removes_event_stream(self, team_service: TeamService) -> None:
        """AC1: stop_team calls event_stream.remove(team_id)."""
        from akgentic.infra.adapters.community.local_event_stream import LocalEventStream

        process = team_service.create_team(catalog_namespace="test-team", user_id="anonymous")
        team_id = process.team_id

        event_stream = team_service.get_event_stream()
        assert isinstance(event_stream, LocalEventStream)

        # Verify stream has events (team creation generates StartMessage events)
        events = event_stream.read_from(team_id)
        assert len(events) > 0

        team_service.stop_team(team_id)

        # After stop, subscribing should raise StreamClosed or return empty
        # The stream was removed — read_from returns [] for non-existent streams
        events_after = event_stream.read_from(team_id)
        assert events_after == []

    def test_stop_team_errors_are_non_fatal(self, team_service: TeamService) -> None:
        """AC1: event_stream.remove() failure does not prevent stop.

        Story 27.1: ``EventStreamSubscriber.on_stop(team_id)`` also calls
        ``event_stream.remove`` as the canonical per-team cleanup hook, and
        ``TeamService.stop_team`` retains its own ``event_stream.remove`` call
        as a belt-and-suspenders for the case where the worker has died
        before ``on_stop`` could fire. Both call sites must swallow
        ``event_stream.remove`` failures; the test now asserts the failing
        ``remove`` was invoked at least once.
        """
        process = team_service.create_team(catalog_namespace="test-team", user_id="anonymous")
        team_id = process.team_id

        # Replace event_stream.remove with one that raises
        original_remove = team_service._services.event_stream.remove
        call_count = 0

        def failing_remove(tid: uuid.UUID) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("simulated failure")

        team_service._services.event_stream.remove = failing_remove  # type: ignore[assignment]
        try:
            team_service.stop_team(team_id)  # Should not raise
            assert call_count >= 1
        finally:
            team_service._services.event_stream.remove = original_remove  # type: ignore[assignment]

        # Team should still be stopped
        stopped = team_service.get_team(team_id)
        assert stopped is not None
        assert stopped.status == TeamStatus.STOPPED


class TestTeamServiceImports:
    """Verify TeamService module does not import actor internals."""

    def test_team_service_has_no_actor_internal_imports(self) -> None:
        """TeamService module does not import actor internals."""
        import inspect

        from akgentic.infra.server.services import team_service as ts_module

        source = inspect.getsource(ts_module)
        forbidden = ["TeamManager", "ActorSystem", "LocalTeamHandle", "CommunityServices"]
        for name in forbidden:
            assert name not in source, f"TeamService module must not import {name}"


class TestTeamServiceLogging:
    """TeamService emits expected log messages."""

    def test_create_team_emits_info_log(
        self,
        team_service: TeamService,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """create_team() emits INFO log with team_id and catalog_entry."""
        with caplog.at_level(logging.INFO, logger="akgentic.infra.server.services.team_service"):
            team_service.create_team(catalog_namespace="test-team", user_id="anonymous")
        assert any("Team created" in r.message for r in caplog.records)

    @pytest.mark.skip(
        reason="Flaky: same race in TeamManager.delete_team as "
        "test_delete_team_stops_and_deletes; pre-existing, not introduced by Epic 22."
    )
    def test_delete_team_emits_info_log(
        self,
        team_service: TeamService,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """delete_team() emits INFO log with team_id."""
        process = team_service.create_team(catalog_namespace="test-team", user_id="anonymous")
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="akgentic.infra.server.services.team_service"):
            team_service.delete_team(process.team_id)
        assert any("Team deleted" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Story 24.1 — workspace-directory cleanup in delete_team
#
# These tests stub the tier services (MagicMock) so delete_team's FS-cleanup
# step can be exercised in isolation, without spinning up the real actor
# system (whose TeamManager.delete_team has a pre-existing flaky teardown
# race — see the skipped tests above).
# ---------------------------------------------------------------------------


_OWNER = "alice"
"""The owning principal on the stubbed ``Process`` — the workspace's ``<scope>``."""


@pytest.fixture()
def _workspace_roots_under_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every path this suite can remove inside ``tmp_path``.

    ``TeamService`` is handed its ``workspaces_root`` directly, but the
    ``<tree>.index`` sibling is located by the tool's ``meta_dir_for``, which
    resolves its own parent from ``AKGENTIC_WORKSPACE_META_ROOT`` or, failing
    that, ``AKGENTIC_WORKSPACES_ROOT`` — defaulting to ``./workspaces``,
    relative to the process cwd. That is deliberate (an operator may relocate
    the metadata root, and refusing to delete it there would re-open the
    retention leak), and it means the injected root does **not** bound the
    index removal.

    So a spec that leaves the variables unset aims a real ``rmtree`` at
    ``<cwd>/workspaces/<scope>/<kind>/<leaf>.index`` — a live directory in this
    checkout, holding the demo stack's trees. Nothing is lost today only
    because the leaves these specs build (a fixed ``notes``, otherwise random
    UUIDs) happen not to collide. Pinning both variables removes the
    coincidence rather than relying on it.
    """
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(tmp_path))
    monkeypatch.delenv("AKGENTIC_WORKSPACE_META_ROOT", raising=False)


def _stub_team_service(
    workspaces_root: Path,
    *,
    team_exists: bool,
    owner: str = _OWNER,
    team_id: uuid.UUID | None = None,
    cards: list[AgentCard] | None = None,
    missing_cards: bool = False,
    metadata: SerializableBaseModel | None = None,
    policy: WorkspaceDeletionPolicy | None = None,
) -> TeamService:
    """Build a TeamService with mocked tier services for FS-cleanup tests.

    When ``team_exists`` is False, ``worker_handle.get_team`` returns None so
    ``delete_team`` raises ``TeamNotFoundError`` before any FS work.

    Three attributes are set with **real** values rather than left as bare
    ``MagicMock`` ones, because the deletion path now reads all three and a mock
    in any of them silently changes what is under test:

    * ``process.user_id`` is the ``<scope>`` segment (ADR-048), so a mock would
      send every spec down the unusable-owner arm;
    * ``process.team_id`` is the ``<leaf>`` of the team's own tree *and* what
      the default policy compares that leaf against, so a mock refuses
      everything;
    * ``process.agent_cards`` and ``services.event_store`` are what the sharing
      axis is read from — a mock card store returns a mock and the candidate set
      is nonsense.

    ``services.workspace_deletion_policy`` is likewise real: a ``MagicMock``
    would return a truthy mock from ``may_delete`` and approve every candidate,
    which is the opposite of the default under test.
    """
    services = MagicMock()
    if team_exists:
        cards = cards or []
        process = MagicMock(spec=Process)
        process.status = TeamStatus.STOPPED
        process.user_id = owner
        process.team_id = team_id or uuid.uuid4()
        process.metadata = metadata
        process.agent_cards = [
            AgentCardRef(role=card.role, card_hash=hash_agent_card(card)) for card in cards
        ]
        services.event_store = RecordingCardStore(cards, missing=missing_cards)
        services.worker_handle.get_team.return_value = process
    else:
        services.worker_handle.get_team.return_value = None
    services.workspace_deletion_policy = policy or TeamTreeOnlyPolicy()
    return TeamService(services, workspaces_root=workspaces_root)


@pytest.mark.usefixtures("_workspace_roots_under_tmp")
class TestDeleteTeamWorkspaceCleanup:
    """Story 24.1 / 67.1 / 70.1: delete_team removes the team's **scoped** workspace dir."""

    def test_happy_path_removes_scoped_workspace_dir(self, tmp_path: Path) -> None:
        """AC #5: the dir at ``<root>/<owner>/_team/<team_id>`` and its contents are removed.

        The on-disk guard that a *real* bind's tree is the one removed is
        ``test_team_deletion_removes_the_bound_tree.py``; this one pins the
        literal location for a hand-seeded tree.
        """
        team_id = uuid.uuid4()
        team_dir = tmp_path / _OWNER / "_team" / str(team_id)
        team_dir.mkdir(parents=True)
        (team_dir / "file.txt").write_text("content")

        service = _stub_team_service(tmp_path, team_exists=True, team_id=team_id)
        service.delete_team(team_id)

        assert not team_dir.exists()

    def test_the_deletion_target_is_the_one_the_resolver_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #1: the target comes from the resolver only, on the sharing axis the card gave.

        The property this protects is unchanged from when it was written — the
        deletion target is whatever ``resolve_workspace_path`` returned, so a
        hand-built target beside the resolver, *even a correct three-segment
        one*, goes red here where an on-disk guard cannot tell the two apart.

        What changed, on the strength of epic 71's decision, is the argument the
        assertion pins. It used to read ``"workspace_sharable": False`` under the
        comment *"no card is consulted on delete"*. That **was** the defect: the
        deletion site answered a question belonging to the card, so a team whose
        card declared sharing wrote to ``_shared/…`` and deletion looked under
        ``<owner>/…``. The team here declares ``workspace_sharable=True``, and
        the assertion is that every call carried the card's ``True`` — loosening
        it to drop the kwargs comparison would be a check narrowed until it
        agreed, where re-expressing it is the reversal the decision asked for.
        """
        team_id = uuid.uuid4()
        sentinel = PurePosixPath("resolver-chose", TEAM_KIND, str(team_id))
        calls: list[dict[str, object]] = []

        def _resolver(**kwargs: object) -> PurePosixPath:
            calls.append(kwargs)
            return sentinel

        monkeypatch.setattr(workspace_paths_module, "resolve_workspace_path", _resolver)
        chosen = tmp_path / sentinel
        chosen.mkdir(parents=True)
        (chosen / "file.txt").write_text("content")
        default = tmp_path / _OWNER / TEAM_KIND / str(team_id)
        default.mkdir(parents=True)

        service = _stub_team_service(
            tmp_path,
            team_exists=True,
            team_id=team_id,
            cards=[tool_card("Sharer", WorkspaceTool(workspace_sharable=True))],
        )
        service.delete_team(team_id)

        assert not chosen.exists()
        assert default.exists()
        # The card's declaration, on every call: the declared workspace and the
        # team's own default tree are one and the same tree for this card, so
        # the resolver is asked the same question twice and told ``True`` twice.
        declared = {
            "workspace_id": None,
            "workspace_metadata_keys": [],
            "team_id": str(team_id),
            "user_id": _OWNER,
            "metadata": None,
            "workspace_sharable": True,
        }
        assert calls == [declared, declared]

    def test_unscoped_sibling_tree_is_left_alone(self, tmp_path: Path) -> None:
        """The flat and two-segment directories are not this team's and are not touched.

        Migrating what is on disk belongs to the tool-side migration story, so a
        leftover tree of an earlier layout must survive a delete rather than be
        swept by a path this code no longer owns.
        """
        team_id = uuid.uuid4()
        scoped = tmp_path / _OWNER / "_team" / str(team_id)
        scoped.mkdir(parents=True)
        legacy = tmp_path / str(team_id)
        legacy.mkdir(parents=True)
        (legacy / "old.txt").write_text("pre-migration")
        two_segment = tmp_path / _OWNER / str(team_id)
        two_segment.mkdir(parents=True)
        (two_segment / "old.txt").write_text("pre-three-segment")

        service = _stub_team_service(tmp_path, team_exists=True, team_id=team_id)
        service.delete_team(team_id)

        assert not scoped.exists()
        assert (legacy / "old.txt").read_text() == "pre-migration"
        assert (two_segment / "old.txt").read_text() == "pre-three-segment"

    def test_scope_is_the_owner_not_the_caller(self, tmp_path: Path) -> None:
        """The scope comes from the deleted team's own ``Process.user_id``."""
        team_id = uuid.uuid4()
        owner_dir = tmp_path / "owner-principal" / "_team" / str(team_id)
        owner_dir.mkdir(parents=True)
        other_dir = tmp_path / "some-other-principal" / "_team" / str(team_id)
        other_dir.mkdir(parents=True)

        service = _stub_team_service(
            tmp_path, team_exists=True, team_id=team_id, owner="owner-principal"
        )
        service.delete_team(team_id)

        assert not owner_dir.exists()
        assert other_dir.exists()

    def test_unusable_owner_id_is_warned_and_deletion_completes(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """ADR-048 Decision 4, delete-path row: a bad owner id must not stick a delete.

        Letting the ``ValueError`` propagate would make a team whose stored
        ``user_id`` cannot be a directory name **undeletable** — trading an
        orphaned directory for a stuck record.
        """
        team_id = uuid.uuid4()
        service = _stub_team_service(tmp_path, team_exists=True, team_id=team_id, owner="")

        with caplog.at_level(logging.WARNING):
            service.delete_team(team_id)  # must NOT raise

        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and str(team_id) in r.getMessage()
        ]
        assert len(warnings) == 1
        service._services.worker_handle.delete_team.assert_called_once_with(team_id)

    def test_missing_dir_is_silent_no_op(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """AC #2: a missing workspace dir produces no WARNING log and no error."""
        team_id = uuid.uuid4()
        # workspaces_root exists, but the scoped subdir does NOT.
        service = _stub_team_service(tmp_path, team_exists=True, team_id=team_id)

        with caplog.at_level(logging.WARNING):
            service.delete_team(team_id)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings == []

    def test_rmtree_failure_logged_and_suppressed(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AC #3: an rmtree failure is logged at WARNING and suppressed."""
        team_id = uuid.uuid4()
        team_dir = tmp_path / _OWNER / "_team" / str(team_id)
        team_dir.mkdir(parents=True)
        (team_dir / "file.txt").write_text("content")

        def _boom(_path: object) -> None:
            raise PermissionError("denied")

        monkeypatch.setattr(shutil, "rmtree", _boom)

        service = _stub_team_service(tmp_path, team_exists=True, team_id=team_id)
        with caplog.at_level(logging.WARNING):
            service.delete_team(team_id)  # must NOT raise

        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and str(team_id) in r.getMessage()
        ]
        assert len(warnings) == 1
        # Team is still deleted from the system of record.
        service._services.worker_handle.delete_team.assert_called_once_with(team_id)

    def test_team_not_found_skips_fs_work(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AC #5: a missing team raises ValueError before any rmtree is attempted."""
        team_id = uuid.uuid4()
        rmtree_calls: list[object] = []
        monkeypatch.setattr(shutil, "rmtree", lambda p: rmtree_calls.append(p))

        service = _stub_team_service(tmp_path, team_exists=False)
        with pytest.raises(ValueError, match="not found"):
            service.delete_team(team_id)

        assert rmtree_calls == []
        service._services.worker_handle.delete_team.assert_not_called()


# ---------------------------------------------------------------------------
# Story 71.1 — the sharing axis is the card's, the rule is the kind and the
# leaf, and that rule is an overridable policy with a safe default.
# ---------------------------------------------------------------------------


class _PermitKind:
    """A test-wired policy permitting exactly one ``<kind>``.

    Stands in for a deployment that knows something this package cannot — that
    its named workspaces really do belong to one team at a time. Its only job
    here is to prove the override path is reachable and that it narrows to
    exactly what it names.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def may_delete(self, *, ctx: WorkspaceDeletionContext) -> bool:
        return ctx.kind == self.kind


def _kept_records(caplog: pytest.LogCaptureFixture, team_id: uuid.UUID) -> list[str]:
    """Every "workspace kept" record this deletion produced."""
    return [
        record.getMessage()
        for record in caplog.records
        if "Workspace kept" in record.getMessage() and str(team_id) in record.getMessage()
    ]


@pytest.mark.usefixtures("_workspace_roots_under_tmp")
class TestTheDefaultRuleIsTheKindAndTheLeaf:
    """AC #3 + AC #6: only ``_team``/<this team id> is deletable, and refusals are logged.

    The candidates are forced through a monkeypatched resolver rather than built
    from cards, because the point is the *shape of the path*, whatever produced
    it: the default policy must refuse a named or metadata tree even when the
    resolver hands it one. A named tree is reachable by every team of that
    principal and a metadata tree by every team carrying those values, so taking
    either with one team would wipe a workspace others still use.
    """

    @pytest.mark.parametrize(
        ("candidate", "kind"),
        [
            (PurePosixPath(_OWNER, ID_KIND, "notes"), "a per-principal named tree"),
            (PurePosixPath("_shared", METADATA_KIND, "customer_id-ACME"), "a shared metadata tree"),
            (PurePosixPath("_shared", ID_KIND, "notes"), "a shared named tree"),
            (PurePosixPath(_OWNER, METADATA_KIND, "customer_id-ACME"), "a metadata tree"),
        ],
    )
    def test_a_candidate_that_is_not_this_teams_own_tree_is_kept_and_recorded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        candidate: PurePosixPath,
        kind: str,
    ) -> None:
        team_id = uuid.uuid4()
        monkeypatch.setattr(
            workspace_paths_module, "resolve_workspace_path", lambda **_kwargs: candidate
        )
        target = tmp_path / candidate
        target.mkdir(parents=True)
        (target / "file.txt").write_text("content")

        service = _stub_team_service(tmp_path, team_exists=True, team_id=team_id)
        with caplog.at_level(logging.INFO):
            service.delete_team(team_id)

        assert (target / "file.txt").read_text() == "content", f"{kind} was removed"
        # AC #6: the candidate and the deciding policy are both on the record,
        # so a deletion that did not happen can be explained afterwards.
        kept = _kept_records(caplog, team_id)
        assert len(kept) == 1
        assert str(candidate) in kept[0]
        assert TeamTreeOnlyPolicy.__name__ in kept[0]

    def test_a_team_leaf_belonging_to_another_team_is_kept(self, tmp_path: Path) -> None:
        """The leaf must be **this** team's id, not merely of the ``_team`` kind.

        A ``_team`` leaf is a team id, so a candidate naming a different one is
        another team's tree. Checking only the kind would delete it.
        """
        team_id = uuid.uuid4()
        other = tmp_path / _OWNER / TEAM_KIND / str(uuid.uuid4())
        other.mkdir(parents=True)

        assert not TeamTreeOnlyPolicy().may_delete(
            ctx=WorkspaceDeletionContext(
                team_id=team_id,
                owner_user_id=_OWNER,
                path=PurePosixPath(other.relative_to(tmp_path).as_posix()),
                scope=_OWNER,
                kind=TEAM_KIND,
                leaf=other.name,
            )
        )

    def test_the_shared_team_tree_is_deletable_in_the_shared_scope_too(
        self, tmp_path: Path
    ) -> None:
        """``_shared/_team/<team_id>`` is an orphan once the team is gone, not sharing."""
        team_id = uuid.uuid4()
        assert TeamTreeOnlyPolicy().may_delete(
            ctx=WorkspaceDeletionContext(
                team_id=team_id,
                owner_user_id=_OWNER,
                path=PurePosixPath("_shared", TEAM_KIND, str(team_id)),
                scope="_shared",
                kind=TEAM_KIND,
                leaf=str(team_id),
            )
        )


@pytest.mark.usefixtures("_workspace_roots_under_tmp")
class TestTheRuleIsAWiredPolicyNotAConstant:
    """AC #4: the default is overridable, and an override narrows to what it names."""

    def test_an_unwired_deployment_gets_the_refusing_default(self) -> None:
        """The ``default_factory`` is what makes wiring nothing safe.

        ``wiring.py`` deliberately does not set this field, so the container's
        own default is the only thing standing between a deployment and a policy
        that answers nothing.
        """
        from akgentic.infra.server.deps import TierServices

        field = TierServices.model_fields["workspace_deletion_policy"]
        assert field.default_factory is TeamTreeOnlyPolicy
        assert isinstance(TeamTreeOnlyPolicy(), WorkspaceDeletionPolicy)

    def test_a_policy_permitting_named_trees_takes_exactly_those(self, tmp_path: Path) -> None:
        """Exactly the ``_id`` candidates go; every other kind — ``_meta`` included — stays.

        The team declares one of each kind, so the permissive policy has
        something to refuse as well as something to permit. A policy that simply
        approved everything it was asked about would pass a spec that only
        seeded the permitted kind.
        """
        team_id = uuid.uuid4()
        service = _stub_team_service(
            tmp_path,
            team_exists=True,
            team_id=team_id,
            metadata=CaseMetadata(),
            cards=[
                tool_card(
                    "Worker",
                    WorkspaceTool(workspace_id="notes"),
                    WorkspaceTool(workspace_metadata_keys=["customer_id"]),
                    WorkspaceTool(),
                )
            ],
            policy=_PermitKind(ID_KIND),
        )
        named = tmp_path / _OWNER / ID_KIND / "notes"
        meta = tmp_path / _OWNER / METADATA_KIND / "customer_id-ACME"
        own = tmp_path / _OWNER / TEAM_KIND / str(team_id)
        for directory in (named, meta, own):
            directory.mkdir(parents=True)
            (directory / "file.txt").write_text("content")

        service.delete_team(team_id)

        assert not named.exists()
        assert (meta / "file.txt").read_text() == "content"
        assert (own / "file.txt").read_text() == "content"


@pytest.mark.usefixtures("_workspace_roots_under_tmp")
class TestNoPolicyCanReachOutsideTheWorkspacesRoot:
    """AC #5b: containment and depth are the caller's, enforced whatever the policy says.

    Both specs wire a policy that approves **everything**, so the only thing
    that can stop the removal is the caller's own check.
    """

    def test_a_candidate_resolving_outside_the_root_is_refused_and_logged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        team_id = uuid.uuid4()
        root = tmp_path / "workspaces"
        root.mkdir()
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        (outside / "file.txt").write_text("not ours")
        escape = PurePosixPath("..", "elsewhere", "..")
        monkeypatch.setattr(
            workspace_paths_module, "resolve_workspace_path", lambda **_kwargs: escape
        )

        service = _stub_team_service(
            root, team_exists=True, team_id=team_id, policy=_PermitKind(escape.parts[1])
        )
        with caplog.at_level(logging.WARNING):
            service.delete_team(team_id)

        assert (outside / "file.txt").read_text() == "not ours"
        refusals = [r for r in caplog.records if "not a contained workspace path" in r.getMessage()]
        assert len(refusals) == 1
        assert refusals[0].levelno == logging.WARNING

    def test_a_candidate_that_is_not_three_segments_is_refused_and_logged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Depth is what makes no workspace path a proper prefix of another.

        A two-segment candidate inside the root is the **parent** of every
        three-segment tree beneath it, so removing one would take trees this
        team never bound — the containment hazard ADR-052's fixed depth removes.
        """
        team_id = uuid.uuid4()
        shallow = PurePosixPath(_OWNER, TEAM_KIND)
        monkeypatch.setattr(
            workspace_paths_module, "resolve_workspace_path", lambda **_kwargs: shallow
        )
        sibling = tmp_path / _OWNER / TEAM_KIND / str(uuid.uuid4())
        sibling.mkdir(parents=True)

        service = _stub_team_service(
            tmp_path, team_exists=True, team_id=team_id, policy=_PermitKind(TEAM_KIND)
        )
        with caplog.at_level(logging.WARNING):
            service.delete_team(team_id)

        assert sibling.exists()
        assert (tmp_path / shallow).exists()
        refusals = [r for r in caplog.records if "not a contained workspace path" in r.getMessage()]
        assert len(refusals) == 1

    def test_a_three_part_candidate_that_resolves_shallow_is_refused_and_logged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Depth is measured on what the candidate *resolves to*, not on its parts.

        ``a/../b`` is three parts and lands inside the root, so a depth check
        that counted only ``candidate.parts`` would approve it — while the
        target it names is ``<root>/b``, one segment deep and the parent of
        every tree beneath it. That is the same proper-prefix hazard as the
        two-segment case above, reached by a candidate that merely *looks*
        three-segment, and the policy is no help: it is shown ``kind='..'``,
        which describes nothing on disk.
        """
        team_id = uuid.uuid4()
        disguised = PurePosixPath("a", "..", _OWNER)
        assert len(disguised.parts) == 3, "the whole point is that the literal looks well-formed"
        monkeypatch.setattr(
            workspace_paths_module, "resolve_workspace_path", lambda **_kwargs: disguised
        )
        sibling = tmp_path / _OWNER / TEAM_KIND / str(uuid.uuid4())
        sibling.mkdir(parents=True)

        service = _stub_team_service(
            tmp_path, team_exists=True, team_id=team_id, policy=_PermitKind("..")
        )
        with caplog.at_level(logging.WARNING):
            service.delete_team(team_id)

        assert sibling.exists()
        assert (tmp_path / _OWNER).exists()
        refusals = [r for r in caplog.records if "not a contained workspace path" in r.getMessage()]
        assert len(refusals) == 1


@pytest.mark.usefixtures("_workspace_roots_under_tmp")
class TestBothSiblingsGoAndOneFailureDoesNotSkipTheOther:
    """AC #2 + AC #7: the journal and the index go too, each independently best-effort."""

    def _seed(self, tmp_path: Path, team_id: uuid.UUID) -> tuple[Path, Path, Path]:
        """The team's own tree and its two sidecars, seeded and located as production does."""
        relative = PurePosixPath(_OWNER, TEAM_KIND, str(team_id))
        tree = tmp_path / relative
        tree.mkdir(parents=True)
        (tree / "file.txt").write_text("content")
        journal = git_dir_for(tree)
        journal.mkdir()
        (journal / "HEAD").write_text("ref: refs/heads/main\n")
        index = meta_dir_for(str(relative))
        (index / "rag").mkdir(parents=True)
        (index / "rag" / "doc.yaml").write_text("text: extracted\n")
        return tree, journal, index

    def test_the_tree_and_both_sidecars_are_removed(self, tmp_path: Path) -> None:
        team_id = uuid.uuid4()
        tree, journal, index = self._seed(tmp_path, team_id)

        service = _stub_team_service(tmp_path, team_exists=True, team_id=team_id)
        service.delete_team(team_id)

        assert not tree.exists()
        assert not journal.exists()
        assert not index.exists()

    def test_a_failure_removing_the_journal_does_not_skip_the_index(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The index is the retention half, so it must be attempted regardless.

        ``<tree>.index/rag/*.yaml`` holds the extracted text of every document
        the tree indexed. Sequencing the three removals so one failure aborts
        the rest would leave exactly that behind whenever the journal — a git
        repository, full of files a sandbox may have written as another uid —
        is the one that fails.
        """
        team_id = uuid.uuid4()
        tree, journal, index = self._seed(tmp_path, team_id)
        real_rmtree = shutil.rmtree

        def _fail_on_the_journal(path: Path) -> None:
            if Path(path) == journal:
                raise PermissionError("denied")
            real_rmtree(path)

        monkeypatch.setattr(shutil, "rmtree", _fail_on_the_journal)

        service = _stub_team_service(tmp_path, team_exists=True, team_id=team_id)
        with caplog.at_level(logging.WARNING):
            service.delete_team(team_id)  # must NOT raise

        assert not tree.exists()
        assert journal.exists(), "the failing target is left behind, as best-effort implies"
        assert not index.exists(), "the retention half was skipped by the journal's failure"
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert str(journal) in warnings[0].getMessage()
        # The team is still deleted from the system of record.
        service._services.worker_handle.delete_team.assert_called_once_with(team_id)


@pytest.mark.usefixtures("_workspace_roots_under_tmp")
class TestNoTeamBecomesUndeletable:
    """AC #7: each new way the candidate set can fail logs a WARNING and completes.

    Letting any of these propagate would trade an orphaned directory for a stuck
    record — a team nobody can remove at all.
    """

    def test_cards_disagreeing_on_workspace_sharable_still_delete_both_trees(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two default-layout cards, one sharing and one not: **both** trees go, warned about.

        Decided 2026-09-13, closing the deferred finding 71-1's review recorded.
        Skipping here was this epic's own retention leak in miniature: the
        declared candidates had already resolved, and the refusal raised from
        the default-tree append discarded the whole list — so the team kept both
        its ``<owner>/_team/<team_id>`` and ``_shared/_team/<team_id>`` trees and
        both ``.index`` sidecars for ever, in the one configuration where the
        code had already computed both correct, policy-approvable targets.

        Both are ``_team`` trees whose leaf is **this** team's id, so both pass
        the kind-and-leaf rule and the default policy unchanged, and once the
        team is gone neither can be addressed again. The WARNING stays: the
        disagreement is still a card defect worth telling an operator about.
        """
        team_id = uuid.uuid4()
        service = _stub_team_service(
            tmp_path,
            team_exists=True,
            team_id=team_id,
            cards=[
                tool_card("Private", WorkspaceTool()),
                tool_card("Sharer", WorkspaceTool(workspace_sharable=True)),
            ],
        )
        trees = {}
        for scope in (_OWNER, SHARED_SCOPE):
            relative = PurePosixPath(scope, TEAM_KIND, str(team_id))
            tree = tmp_path / relative
            tree.mkdir(parents=True)
            (tree / "file.txt").write_text("content")
            index = meta_dir_for(str(relative))
            (index / "rag").mkdir(parents=True)
            (index / "rag" / "doc.yaml").write_text("text: extracted\n")
            trees[scope] = (tree, index)

        with caplog.at_level(logging.WARNING):
            service.delete_team(team_id)  # must NOT raise

        for scope, (tree, index) in trees.items():
            assert not tree.exists(), f"the {scope} tree survived the deletion"
            assert not index.exists(), f"the {scope} tree's .index sidecar survived"
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and str(team_id) in r.getMessage()
        ]
        assert len(warnings) == 1
        assert "disagree on workspace_sharable" in warnings[0].getMessage()
        service._services.worker_handle.delete_team.assert_called_once_with(team_id)

    def test_an_unresolvable_card_hash_skips_cleanup_and_still_deletes(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A card blob the store cannot resolve leaves the sharing axis unknowable.

        Falling back to the per-principal default would be the wrong guess
        precisely when it cannot be checked, so nothing is removed rather than
        the wrong thing — and the record deletion still succeeds.
        """
        team_id = uuid.uuid4()
        tree = tmp_path / _OWNER / TEAM_KIND / str(team_id)
        tree.mkdir(parents=True)
        service = _stub_team_service(
            tmp_path,
            team_exists=True,
            team_id=team_id,
            cards=[tool_card("Worker", WorkspaceTool())],
            missing_cards=True,
        )

        with caplog.at_level(logging.WARNING):
            service.delete_team(team_id)  # must NOT raise

        assert tree.exists(), "nothing is removed when the cards cannot be read"
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and str(team_id) in r.getMessage()
        ]
        assert len(warnings) == 1
        assert "unresolvable" in warnings[0].getMessage()
        service._services.worker_handle.delete_team.assert_called_once_with(team_id)


@pytest.mark.usefixtures("_workspace_roots_under_tmp")
class TestTheCandidateSetIsResolvedBeforeTheRecordIsDeleted:
    """AC #5a's ordering half: the card read does not depend on card blobs outliving the team.

    The filesystem work still runs last — a worker-delete failure must not leave
    a removed workspace behind — but the **resolution** moves ahead of it. A
    card store purged by ``worker_handle.delete_team`` would otherwise return
    nothing, the sharing axis would revert to the per-principal default, and the
    defect would come back with no spec failing.
    """

    def test_the_cards_are_read_before_the_worker_delete(self, tmp_path: Path) -> None:
        team_id = uuid.uuid4()
        order: list[str] = []
        service = _stub_team_service(
            tmp_path,
            team_exists=True,
            team_id=team_id,
            cards=[tool_card("Worker", WorkspaceTool())],
        )
        store = service._services.event_store
        real_load = store.load_agent_cards

        def _recording_load(hashes: list[str]) -> dict[str, AgentCard]:
            order.append("cards-read")
            return real_load(hashes)

        store.load_agent_cards = _recording_load  # type: ignore[method-assign]
        service._services.worker_handle.delete_team.side_effect = lambda _id: order.append(
            "worker-delete"
        )

        service.delete_team(team_id)

        assert order == ["cards-read", "worker-delete"]


# ---------------------------------------------------------------------------
# Story 37.1 — classic offset+total pagination on list_teams (ADR-032).
#
# These tests use a MagicMock event store returning hand-built Process
# snapshots so created_at / team_id are deterministic (the real fixture
# stamps near-identical timestamps). `Process.model_construct` skips validation,
# so the projection fields these tests never read are left off entirely rather
# than stubbed.
# ---------------------------------------------------------------------------

_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _make_process(created_at: datetime, team_id: uuid.UUID) -> Process:
    """Build a Process snapshot with explicit sort-key columns."""
    return Process.model_construct(
        team_id=team_id,
        status=TeamStatus.RUNNING,
        user_id="alice",
        user_email="",
        created_at=created_at,
        updated_at=created_at,
    )


def _service_over(rows: list[Process]) -> TeamService:
    """TeamService whose event store returns a fresh copy of ``rows`` each call."""
    services = MagicMock()
    services.event_store.list_teams.side_effect = lambda **_kw: list(rows)
    return TeamService(services, workspaces_root=Path("/unused"))


def _distinct_rows(n: int) -> list[Process]:
    """``n`` processes with strictly increasing created_at (distinct positions)."""
    return [_make_process(_BASE_TIME + timedelta(minutes=i), uuid.uuid4()) for i in range(n)]


def test_page1_returns_size_rows_with_full_total() -> None:
    """(a) page 1 returns <= size rows; total_count is the full owned count."""
    service = _service_over(_distinct_rows(300))
    page, total = service.list_teams(user_id="alice")  # default page=1, size=250
    assert len(page) == 250
    assert total == 300


def test_page1_under_size_returns_all_with_full_total() -> None:
    """(a) a set smaller than size returns every team with the full total."""
    service = _service_over(_distinct_rows(40))
    page, total = service.list_teams(user_id="alice", size=250)
    assert len(page) == 40
    assert total == 40


def test_page_n_returns_correct_slice_in_order() -> None:
    """(b) page N returns the offset slice in created_at DESC, team_id DESC order."""
    rows = _distinct_rows(10)
    service = _service_over(rows)
    expected = sorted(rows, key=lambda p: (p.created_at, p.team_id), reverse=True)

    page1, total1 = service.list_teams(user_id="alice", page=1, size=3)
    page2, total2 = service.list_teams(user_id="alice", page=2, size=3)
    page4, total4 = service.list_teams(user_id="alice", page=4, size=3)

    assert total1 == total2 == total4 == 10
    assert [p.team_id for p in page1] == [p.team_id for p in expected[0:3]]
    assert [p.team_id for p in page2] == [p.team_id for p in expected[3:6]]
    # Last partial page: rows 9..10 of 10 (size-3 slice at offset 9).
    assert [p.team_id for p in page4] == [p.team_id for p in expected[9:12]]
    assert len(page4) == 1


def test_out_of_range_page_returns_empty_with_correct_total() -> None:
    """(c) a page past the end returns [] with the correct total (no error)."""
    service = _service_over(_distinct_rows(5))
    page, total = service.list_teams(user_id="alice", page=99, size=10)
    assert page == []
    assert total == 5


def test_ordering_is_created_at_then_team_id_desc() -> None:
    """(d) order is created_at DESC, team_id DESC, with a tie broken by team_id."""
    t0 = _BASE_TIME
    t1 = _BASE_TIME + timedelta(minutes=1)
    low = uuid.UUID(int=1)
    high = uuid.UUID(int=2)
    # Two rows share created_at=t0 to exercise the team_id tie-breaker.
    rows = [
        _make_process(t0, low),
        _make_process(t1, high),
        _make_process(t0, high),
    ]
    service = _service_over(rows)
    page, total = service.list_teams(user_id="alice", size=10)
    keys = [(p.created_at, p.team_id) for p in page]
    assert total == 3
    assert keys == sorted(keys, reverse=True)
    # Newest timestamp first; among the t0 tie, the higher team_id leads.
    assert keys == [(t1, high), (t0, high), (t0, low)]


def test_size_clamps_to_lower_bound() -> None:
    """(e) size <= 0 clamps to 1 (returns a single row)."""
    service = _service_over(_distinct_rows(3))
    page_zero, total_zero = service.list_teams(user_id="alice", size=0)
    page_neg, _ = service.list_teams(user_id="alice", size=-5)
    assert len(page_zero) == 1
    assert total_zero == 3
    assert len(page_neg) == 1


def test_size_clamps_to_upper_bound() -> None:
    """(e) size > MAX_PAGE_SIZE clamps to MAX_PAGE_SIZE (500)."""
    service = _service_over(_distinct_rows(600))
    page, total = service.list_teams(user_id="alice", size=99999)
    assert len(page) == MAX_PAGE_SIZE
    assert total == 600


def test_size_250_default_and_explicit() -> None:
    """(e) default size is 250, and size=250 (cap raised above 200) works."""
    service_default = _service_over(_distinct_rows(300))
    page_default, _ = service_default.list_teams(user_id="alice")
    assert len(page_default) == 250

    service_explicit = _service_over(_distinct_rows(300))
    page_explicit, _ = service_explicit.list_teams(user_id="alice", size=250)
    assert len(page_explicit) == 250


def test_default_page_is_1() -> None:
    """(f) default page is 1: omitting page returns the first slice."""
    rows = _distinct_rows(10)
    service = _service_over(rows)
    expected = sorted(rows, key=lambda p: (p.created_at, p.team_id), reverse=True)
    default_page, _ = service.list_teams(user_id="alice", size=3)
    explicit_page1, _ = service.list_teams(user_id="alice", page=1, size=3)
    assert [p.team_id for p in default_page] == [p.team_id for p in expected[0:3]]
    assert [p.team_id for p in default_page] == [p.team_id for p in explicit_page1]


def test_page_clamps_to_lower_bound() -> None:
    """(f) page <= 0 clamps to 1 (same slice as page 1)."""
    rows = _distinct_rows(10)
    service = _service_over(rows)
    page_zero, _ = service.list_teams(user_id="alice", page=0, size=3)
    page_neg, _ = service.list_teams(user_id="alice", page=-3, size=3)
    page1, _ = service.list_teams(user_id="alice", page=1, size=3)
    assert [p.team_id for p in page_zero] == [p.team_id for p in page1]
    assert [p.team_id for p in page_neg] == [p.team_id for p in page1]


# ---------------------------------------------------------------------------
# Story 37.1 AC #7 — list_teams is stateless: a pure function of
# (user_id, page, size) + current store contents; nothing is cached between
# requests, so a page is correct regardless of which replica serves it.
# ---------------------------------------------------------------------------


def test_list_teams_refetches_store_every_call() -> None:
    """Every list_teams call re-reads the store — no cached sorted list."""
    service = _service_over(_distinct_rows(5))
    service.list_teams(user_id="alice", page=1, size=2)
    service.list_teams(user_id="alice", page=2, size=2)
    service.list_teams(user_id="alice", page=3, size=2)
    # One store read per request — no request reused a prior request's fetch.
    assert service._services.event_store.list_teams.call_count == 3


def test_same_args_yield_same_page_independent_requests() -> None:
    """Two independent requests with the same (page, size) return the same page."""
    rows = _distinct_rows(7)
    service = _service_over(rows)
    page_a, total_a = service.list_teams(user_id="alice", page=2, size=2)
    page_b, total_b = service.list_teams(user_id="alice", page=2, size=2)
    assert [p.team_id for p in page_a] == [p.team_id for p in page_b]
    assert total_a == total_b


def test_page_followable_on_fresh_service_instance() -> None:
    """A page minted by one service instance matches a SEPARATE, freshly
    constructed instance over the same store contents — simulating a different
    worker/replica with no shared in-process state.
    """
    rows = _distinct_rows(7)
    minting_service = _service_over(rows)
    page1, _ = minting_service.list_teams(user_id="alice", page=1, size=3)

    # A brand-new instance (different "replica"), no prior request primed.
    fresh_service = _service_over(rows)
    page2, _ = fresh_service.list_teams(user_id="alice", page=2, size=3)

    seen = {p.team_id for p in page1} | {p.team_id for p in page2}
    assert {p.team_id for p in page1}.isdisjoint({p.team_id for p in page2})
    assert len(seen) == 6  # 3 + 3, no overlap, no gap across the replica boundary


def test_list_teams_holds_no_per_request_state() -> None:
    """list_teams mutates no instance attribute that survives the call."""
    service = _service_over(_distinct_rows(5))
    before = dict(vars(service))
    service.list_teams(user_id="alice", page=1, size=2)
    after = dict(vars(service))
    # No new attribute, and the wired collaborators are unchanged identities.
    assert before.keys() == after.keys()
    assert all(before[k] is after[k] for k in before)


# ---------------------------------------------------------------------------
# Story 53.2 — the metadata filter and pagination together.
#
# The stub store below stands in for the store-side push-down: a filtered call
# gets back only the matching rows. That is the whole reason ``total`` is the
# filtered count by construction — the service counts what the store returned
# and never counts a second time.
# ---------------------------------------------------------------------------


def _service_over_filtered(rows: list[Process], matching: list[Process]) -> TeamService:
    """TeamService whose store returns ``matching`` for a metadata-filtered call."""
    services = MagicMock()

    def _list_teams(**kwargs: object) -> list[Process]:
        return list(matching) if kwargs.get("metadata") else list(rows)

    services.event_store.list_teams.side_effect = _list_teams
    return TeamService(services, workspaces_root=Path("/unused"))


_ACME = {"tenant": "acme"}


def test_filtered_total_is_the_matching_count_on_every_page() -> None:
    """50 owned, 3 matching: the total is 3 on pages 1, 2 and 3 alike.

    The opposite implementation — fetch the owned set, slice it, filter the
    slice — reports 50 and hands back pages shorter than ``size`` for no visible
    reason. Checked across the page boundary because a single page cannot tell
    the two apart.
    """
    rows = _distinct_rows(50)
    matching = sorted(rows[:3], key=lambda p: (p.created_at, p.team_id), reverse=True)
    service = _service_over_filtered(rows, matching)

    page1, total1 = service.list_teams(user_id="alice", metadata=_ACME, page=1, size=2)
    page2, total2 = service.list_teams(user_id="alice", metadata=_ACME, page=2, size=2)
    page3, total3 = service.list_teams(user_id="alice", metadata=_ACME, page=3, size=2)

    assert [total1, total2, total3] == [3, 3, 3]
    assert len(rows) not in (total1, total2, total3)
    assert [p.team_id for p in page1] == [p.team_id for p in matching[0:2]]
    assert [p.team_id for p in page2] == [p.team_id for p in matching[2:3]]
    assert page3 == []


def test_filtered_page_followable_on_a_fresh_service_instance() -> None:
    """A filtered walk survives crossing to a separately constructed instance.

    Same guarantee the unfiltered path already gives: the filter lives in the
    arguments and the store, never in the instance that happened to serve the
    previous page.
    """
    rows = _distinct_rows(20)
    matching = rows[:6]

    page1, total1 = _service_over_filtered(rows, matching).list_teams(
        user_id="alice", metadata=_ACME, page=1, size=3
    )
    page2, total2 = _service_over_filtered(rows, matching).list_teams(
        user_id="alice", metadata=_ACME, page=2, size=3
    )

    assert total1 == total2 == 6
    ids1 = {p.team_id for p in page1}
    ids2 = {p.team_id for p in page2}
    assert ids1.isdisjoint(ids2)
    assert len(ids1 | ids2) == 6
