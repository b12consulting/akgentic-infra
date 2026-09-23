"""Tests for LocalPlacement adapter."""

from __future__ import annotations

import inspect
import logging
import threading
import time
import uuid
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from akgentic.infra.adapters.community import local_placement as local_placement_module
from akgentic.infra.adapters.community.local_placement import LocalPlacement
from akgentic.infra.adapters.community.local_team_handle import LocalTeamHandle
from akgentic.infra.protocols.placement import (
    PlacementError,
    PlacementStrategy,
)
from tests.fixtures.team_metadata import AcmeCaseMetadata


def _make_adapter() -> LocalPlacement:
    """Create a LocalPlacement with mock dependencies."""
    team_manager = MagicMock()
    service_registry = MagicMock()
    return LocalPlacement(team_manager, service_registry)


class TestLocalPlacementProtocolCompliance:
    """AC5: LocalPlacement implements PlacementStrategy protocol."""

    def test_satisfies_placement_strategy_protocol(self) -> None:
        """LocalPlacement structurally satisfies PlacementStrategy."""
        adapter = _make_adapter()
        assert isinstance(adapter, PlacementStrategy)

    def test_has_create_team_method(self) -> None:
        """LocalPlacement exposes create_team with correct signature."""
        adapter = _make_adapter()
        assert callable(adapter.create_team)

    def test_resume_team_signature(self) -> None:
        """resume_team has team_id parameter."""
        sig = inspect.signature(LocalPlacement.resume_team)
        assert "team_id" in sig.parameters

    def test_create_team_signature_matches_protocol(self) -> None:
        """create_team has all parameters matching PlacementStrategy."""
        sig = inspect.signature(LocalPlacement.create_team)
        assert "team_card" in sig.parameters
        assert "user_id" in sig.parameters
        assert "user_email" in sig.parameters
        assert "team_id" in sig.parameters
        assert "catalog_namespace" in sig.parameters
        assert sig.parameters["catalog_namespace"].default is None
        # Optional and defaulting to None, so pre-metadata callers are unaffected.
        assert "metadata" in sig.parameters
        assert sig.parameters["metadata"].default is None


class TestLocalPlacementBehavior:
    """AC5: LocalPlacement delegates to TeamManager and returns LocalTeamHandle."""

    def test_create_team_delegates_to_team_manager(self) -> None:
        """create_team calls TeamManager.create_team forwarding all args."""
        team_manager = MagicMock()
        service_registry = MagicMock()
        adapter = LocalPlacement(team_manager, service_registry)
        team_card = MagicMock()
        adapter.create_team(team_card, "user-1")
        team_manager.create_team.assert_called_once_with(
            team_card, "user-1", user_email="", team_id=None, catalog_namespace=None, metadata=None
        )

    def test_create_team_forwards_catalog_namespace(self) -> None:
        """create_team forwards catalog_namespace to TeamManager.create_team."""
        team_manager = MagicMock()
        service_registry = MagicMock()
        adapter = LocalPlacement(team_manager, service_registry)
        team_card = MagicMock()
        adapter.create_team(team_card, "user-1", catalog_namespace="ns-abc")
        team_manager.create_team.assert_called_once_with(
            team_card,
            "user-1",
            user_email="",
            team_id=None,
            catalog_namespace="ns-abc",
            metadata=None,
        )

    def test_create_team_forwards_user_email_and_team_id(self) -> None:
        """create_team forwards caller-supplied user_email and team_id verbatim."""
        team_manager = MagicMock()
        team_manager.get_team.return_value = None  # the key names no existing team
        service_registry = MagicMock()
        adapter = LocalPlacement(team_manager, service_registry)
        team_card = MagicMock()
        explicit_id = uuid.uuid4()
        adapter.create_team(team_card, "user-1", user_email="user@example.com", team_id=explicit_id)
        team_manager.create_team.assert_called_once_with(
            team_card,
            "user-1",
            user_email="user@example.com",
            team_id=explicit_id,
            catalog_namespace=None,
            metadata=None,
        )

    def test_create_team_forwards_validated_metadata(self) -> None:
        """A validated metadata model reaches TeamManager.create_team verbatim.

        Asserted on the *call*, not only on a persisted result: a MagicMock
        TeamManager accepts any kwargs silently, so a pass-through that dropped
        the value would otherwise go green.
        """
        team_manager = MagicMock()
        adapter = LocalPlacement(team_manager, MagicMock())
        team_card = MagicMock()
        metadata = AcmeCaseMetadata(tenant="acme", case="C-1234")

        adapter.create_team(team_card, "user-1", metadata=metadata)

        assert team_manager.create_team.call_args.kwargs["metadata"] is metadata

    def test_create_team_returns_local_team_handle(self) -> None:
        """create_team wraps TeamManager result in LocalTeamHandle."""
        team_manager = MagicMock()
        service_registry = MagicMock()
        adapter = LocalPlacement(team_manager, service_registry)
        result = adapter.create_team(MagicMock(), "user-1")
        assert isinstance(result, LocalTeamHandle)

    def test_resume_team_delegates_to_team_manager(self) -> None:
        """resume_team calls TeamManager.resume_team with correct team_id."""
        team_manager = MagicMock()
        adapter = LocalPlacement(team_manager, MagicMock())
        tid = uuid.uuid4()
        adapter.resume_team(tid)
        team_manager.resume_team.assert_called_once_with(tid)

    def test_resume_team_returns_local_team_handle(self) -> None:
        """resume_team wraps TeamManager result in LocalTeamHandle."""
        adapter = _make_adapter()
        result = adapter.resume_team(uuid.uuid4())
        assert isinstance(result, LocalTeamHandle)

    def test_instance_id_is_stable(self) -> None:
        """instance_id does not change between calls."""
        adapter = _make_adapter()
        assert adapter.instance_id == adapter.instance_id

    def test_instance_id_is_uuid(self) -> None:
        """instance_id is a uuid.UUID."""
        adapter = _make_adapter()
        assert isinstance(adapter.instance_id, uuid.UUID)

    def test_different_instances_have_different_ids(self) -> None:
        """Two LocalPlacement instances have different instance_ids."""
        a = _make_adapter()
        b = _make_adapter()
        assert a.instance_id != b.instance_id


class TestLocalPlacementCreateFailure:
    """AC12: a TeamManager.create_team failure surfaces as a PlacementError."""

    def test_create_team_failure_raises_placement_error(self) -> None:
        """A delegate exception is wrapped in PlacementError (a ServerError)."""
        team_manager = MagicMock()
        team_manager.create_team.side_effect = RuntimeError("boom")
        adapter = LocalPlacement(team_manager, MagicMock())
        with pytest.raises(PlacementError) as exc_info:
            adapter.create_team(MagicMock(), "user-1")
        # Wrapped, not re-raised verbatim: carries the placement HTTP mapping.
        assert exc_info.value.status_code == 503
        assert exc_info.value.__cause__ is not None

    def test_create_team_passes_through_placement_error(self) -> None:
        """An already-typed PlacementError propagates unchanged (not re-wrapped)."""
        original = PlacementError("already typed")
        team_manager = MagicMock()
        team_manager.create_team.side_effect = original
        adapter = LocalPlacement(team_manager, MagicMock())
        with pytest.raises(PlacementError) as exc_info:
            adapter.create_team(MagicMock(), "user-1")
        assert exc_info.value is original


class TestLocalPlacementResumeFailure:
    """The resume keeps its ValueError contract — it is not a PlacementError.

    ``TeamService.restore_team`` and the worker route classify a resume failure
    by string-matching its message ("not found" / "deleted" → 404, otherwise →
    409). A ``PlacementError`` is a ``ServerError``, so the single infra handler
    would take it instead and answer 503 to both. Every unit test of the resume
    would still pass; only these specs and the route specs notice.
    """

    def test_resume_failure_propagates_the_value_error_unchanged(self) -> None:
        """The exception reaches the caller as the same object, type and message."""
        original = ValueError("Team is currently running")
        team_manager = MagicMock()
        team_manager.resume_team.side_effect = original
        adapter = LocalPlacement(team_manager, MagicMock())

        with pytest.raises(ValueError) as exc_info:  # noqa: PT011 - identity asserted below
            adapter.resume_team(uuid.uuid4())

        assert exc_info.value is original
        assert not isinstance(exc_info.value, PlacementError)

    def test_a_failed_resume_releases_its_key(self) -> None:
        """A retry after a failed resume runs afresh instead of re-reading the failure."""
        team_manager = MagicMock()
        team_manager.resume_team.side_effect = [ValueError("boom"), MagicMock()]
        adapter = LocalPlacement(team_manager, MagicMock())
        team_id = uuid.uuid4()

        with pytest.raises(ValueError, match="boom"):
            adapter.resume_team(team_id)
        handle = adapter.resume_team(team_id)

        assert isinstance(handle, LocalTeamHandle)
        assert team_manager.resume_team.call_count == 2


# --- The creation key: team_id collapses concurrent creations, never addresses ---


class _GatedTeamManager:
    """A TeamManager whose claims block until released, to hold a key in flight.

    Serves creations and resumes alike, because the placement holds both in one
    table. ``get_team`` answers with an object carrying ``user_id`` rather than a
    bare sentinel: the create-side existing-team check only tests for None, but
    the resume reads the team's persisted owner off it.
    """

    def __init__(self, existing: set[uuid.UUID] | None = None, owner: str = "user-1") -> None:
        self.existing = existing or set()
        self.owner = owner
        self.release = threading.Event()
        self.creations = 0
        self.resumes = 0
        self.fail_with: Exception | None = None

    def get_team(self, team_id: uuid.UUID) -> object | None:
        if team_id not in self.existing:
            return None
        process = MagicMock()
        process.user_id = self.owner
        return process

    def create_team(self, team_card: object, user_id: str, **kwargs: object) -> MagicMock:
        self.creations += 1
        assert self.release.wait(5), "creation never released"
        if self.fail_with is not None:
            raise self.fail_with
        runtime = MagicMock()
        runtime.id = kwargs["team_id"]
        return runtime

    def resume_team(self, team_id: uuid.UUID) -> MagicMock:
        self.resumes += 1
        assert self.release.wait(5), "resume never released"
        if self.fail_with is not None:
            raise self.fail_with
        runtime = MagicMock()
        runtime.id = team_id
        return runtime


def _run(fn: Callable[[], LocalTeamHandle]) -> tuple[threading.Thread, dict[str, object]]:
    """Run ``fn`` on a thread, keeping its result or its exception."""
    out: dict[str, object] = {}

    def target() -> None:
        try:
            out["handle"] = fn()
        except Exception as exc:  # noqa: BLE001 - the spec inspects it
            out["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    return thread, out


def _await_log(caplog: pytest.LogCaptureFixture, text: str) -> None:
    """Wait until another thread has logged ``text`` — a deterministic rendezvous."""
    deadline = time.monotonic() + 5
    while not any(text in r.getMessage() for r in caplog.records):
        assert time.monotonic() < deadline, f"never logged: {text!r}"
        time.sleep(0.005)


class TestCreationKey:
    def test_a_duplicate_creation_parks_and_receives_the_same_team(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The second creation of an in-flight key waits and gets the very same handle."""
        caplog.set_level(logging.DEBUG, "akgentic.infra.adapters.community.local_placement")
        manager = _GatedTeamManager()
        placement = LocalPlacement(manager, MagicMock())  # type: ignore[arg-type]
        key = uuid.uuid4()

        first, first_out = _run(lambda: placement.create_team(MagicMock(), "user-1", team_id=key))
        second, second_out = _run(lambda: placement.create_team(MagicMock(), "user-1", team_id=key))
        _await_log(caplog, "Parking duplicate")
        manager.release.set()
        first.join(5)
        second.join(5)

        assert manager.creations == 1
        assert second_out["handle"] is first_out["handle"]

    def test_a_key_being_created_for_another_user_is_refused(self) -> None:
        """Collapsing into it would hand one user's team to another."""
        manager = _GatedTeamManager()
        placement = LocalPlacement(manager, MagicMock())  # type: ignore[arg-type]
        key = uuid.uuid4()
        first, _ = _run(lambda: placement.create_team(MagicMock(), "user-1", team_id=key))
        deadline = time.monotonic() + 5
        while manager.creations == 0:
            assert time.monotonic() < deadline, "first creation never started"
            time.sleep(0.005)

        with pytest.raises(PlacementError) as refused:
            placement.create_team(MagicMock(), "user-2", team_id=key)

        manager.release.set()
        first.join(5)
        assert refused.value.status_code == 409
        assert refused.value.code == "team_id_conflict"
        assert manager.creations == 1

    def test_a_key_naming_an_existing_team_is_refused_and_not_recreated(self) -> None:
        """A key is never an address: an existing team is neither returned nor overwritten."""
        key = uuid.uuid4()
        manager = _GatedTeamManager(existing={key})
        manager.release.set()
        placement = LocalPlacement(manager, MagicMock())  # type: ignore[arg-type]

        with pytest.raises(PlacementError) as refused:
            placement.create_team(MagicMock(), "user-1", team_id=key)

        assert refused.value.status_code == 409
        assert manager.creations == 0

    def test_a_parked_duplicate_fails_the_way_its_creation_failed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG, "akgentic.infra.adapters.community.local_placement")
        manager = _GatedTeamManager()
        manager.fail_with = RuntimeError("boom")
        placement = LocalPlacement(manager, MagicMock())  # type: ignore[arg-type]
        key = uuid.uuid4()

        first, first_out = _run(lambda: placement.create_team(MagicMock(), "user-1", team_id=key))
        second, second_out = _run(lambda: placement.create_team(MagicMock(), "user-1", team_id=key))
        _await_log(caplog, "Parking duplicate")
        manager.release.set()
        first.join(5)
        second.join(5)

        assert isinstance(first_out["error"], PlacementError)
        assert second_out["error"] is first_out["error"]

    def test_a_parked_duplicate_gives_up_on_a_stuck_creation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A creation that never finishes must not hold its duplicates' threads forever."""
        monkeypatch.setattr(local_placement_module, "_PARKED_CREATION_TIMEOUT_S", 0.05)
        manager = _GatedTeamManager()
        placement = LocalPlacement(manager, MagicMock())  # type: ignore[arg-type]
        key = uuid.uuid4()
        first, _ = _run(lambda: placement.create_team(MagicMock(), "user-1", team_id=key))
        deadline = time.monotonic() + 5
        while manager.creations == 0:
            assert time.monotonic() < deadline, "first creation never started"
            time.sleep(0.005)

        with pytest.raises(PlacementError, match="did not finish") as refused:
            placement.create_team(MagicMock(), "user-1", team_id=key)

        manager.release.set()
        first.join(5)
        assert refused.value.status_code == 503
        assert manager.creations == 1

    def test_a_failed_creation_releases_its_key(self) -> None:
        """A retry after a failure creates afresh instead of re-reading the old failure."""
        manager = _GatedTeamManager()
        manager.release.set()
        manager.fail_with = RuntimeError("boom")
        placement = LocalPlacement(manager, MagicMock())  # type: ignore[arg-type]
        key = uuid.uuid4()
        with pytest.raises(PlacementError):
            placement.create_team(MagicMock(), "user-1", team_id=key)

        manager.fail_with = None
        handle = placement.create_team(MagicMock(), "user-1", team_id=key)

        assert handle.team_id == key
        assert manager.creations == 2

    def test_no_key_means_nothing_is_collapsed(self) -> None:
        manager = _GatedTeamManager()
        manager.release.set()
        placement = LocalPlacement(manager, MagicMock())  # type: ignore[arg-type]

        placement.create_team(MagicMock(), "user-1")
        placement.create_team(MagicMock(), "user-1")

        assert manager.creations == 2


class TestResumeSharesTheCreationTable:
    """One table, one lock, one ``_park`` — resumes claim keys the way creations do."""

    def test_two_concurrent_resumes_start_one_runtime(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A second resume of an in-flight team waits and gets the very same handle.

        ``TeamManager.resume_team`` checks the status and proceeds, so two
        callers can both pass that check and both spawn actors. The placement is
        where that race is closed.
        """
        caplog.set_level(logging.DEBUG, "akgentic.infra.adapters.community.local_placement")
        team_id = uuid.uuid4()
        manager = _GatedTeamManager(existing={team_id})
        placement = LocalPlacement(manager, MagicMock())  # type: ignore[arg-type]

        first, first_out = _run(lambda: placement.resume_team(team_id))
        second, second_out = _run(lambda: placement.resume_team(team_id))
        _await_log(caplog, "Parking duplicate")
        manager.release.set()
        first.join(5)
        second.join(5)

        assert manager.resumes == 1
        assert second_out["handle"] is first_out["handle"]

    def test_a_parked_duplicate_resume_fails_the_way_its_resume_failed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The loser of a race against a not-stopped team sees the same ValueError.

        That is what keeps it mapping to 409 / 404 at the route rather than to
        a 503 the caller cannot act on.
        """
        caplog.set_level(logging.DEBUG, "akgentic.infra.adapters.community.local_placement")
        team_id = uuid.uuid4()
        manager = _GatedTeamManager(existing={team_id})
        manager.fail_with = ValueError(f"Team {team_id} is currently running")
        placement = LocalPlacement(manager, MagicMock())  # type: ignore[arg-type]

        first, first_out = _run(lambda: placement.resume_team(team_id))
        second, second_out = _run(lambda: placement.resume_team(team_id))
        _await_log(caplog, "Parking duplicate")
        manager.release.set()
        first.join(5)
        second.join(5)

        assert isinstance(first_out["error"], ValueError)
        assert not isinstance(first_out["error"], PlacementError)
        assert second_out["error"] is first_out["error"]

    def test_a_successful_resume_releases_its_key(self) -> None:
        """The key is popped on success too, not only on failure.

        ``test_a_failed_resume_releases_its_key`` covers only half of the
        ``finally``. Move the pop into the ``except`` branch — the shape a
        developer reaches for when the release reads like error handling — and
        every other spec stays green, because none resumes one team twice. What
        breaks in production is the next resume of that team: it parks on a
        future that resolved long ago and is handed a handle to a runtime that
        was stopped in between, with ``TeamManager.resume_team`` never called
        and nothing logged.
        """
        team_manager = MagicMock()
        team_manager.resume_team.side_effect = [MagicMock(), MagicMock()]
        adapter = LocalPlacement(team_manager, MagicMock())
        team_id = uuid.uuid4()

        first = adapter.resume_team(team_id)
        second = adapter.resume_team(team_id)

        assert team_manager.resume_team.call_count == 2
        assert second is not first

    def test_a_create_naming_a_team_whose_resume_is_in_flight_is_refused(self) -> None:
        """The create↔resume branch of the shared table is unreachable by construction.

        The create presents the resumed team's **own owner**, so the owner
        refusal inside ``_park`` cannot be what saves it. What refuses it is
        ``create_team``'s existing-team check: a key naming a team that already
        exists is never honoured, and a team being resumed exists by definition.
        Remove that check and the create parks on the resume and is handed the
        resumed team's handle — which is the whole reason this is a spec and not
        only a comment.
        """
        team_id = uuid.uuid4()
        manager = _GatedTeamManager(existing={team_id}, owner="user-1")
        placement = LocalPlacement(manager, MagicMock())  # type: ignore[arg-type]

        resumer, _ = _run(lambda: placement.resume_team(team_id))
        deadline = time.monotonic() + 5
        while manager.resumes == 0:
            assert time.monotonic() < deadline, "resume never started"
            time.sleep(0.005)

        creator, creator_out = _run(
            lambda: placement.create_team(MagicMock(), "user-1", team_id=team_id)
        )
        creator.join(5)
        parked = creator.is_alive()

        manager.release.set()
        resumer.join(5)
        creator.join(5)

        assert not parked, "the create parked on the in-flight resume instead of being refused"
        assert "handle" not in creator_out, "the create received the resumed team's handle"
        refused = creator_out["error"]
        assert isinstance(refused, PlacementError)
        assert refused.status_code == 409
        assert refused.code == "team_id_conflict"
        assert manager.creations == 0
