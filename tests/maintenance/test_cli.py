"""Backend discovery, rendering and exit codes for the sweep CLI."""

from __future__ import annotations

import uuid

import pytest
from akgentic.team.models import Process
from click.testing import Result
from typer.testing import CliRunner

from akgentic.infra.maintenance import cli
from akgentic.infra.maintenance.models import ReaperReport, ResourceKind, SweepReport
from tests.maintenance.conftest import FakeEventStore, FakeReaper, make_process, make_ref

# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------


class _StubVectorReaper:
    """Stands in for the real reaper so no cluster connection is attempted."""

    kind = ResourceKind.VECTOR

    def __init__(self, backend_name: str) -> None:
        self.backend_name = backend_name


def _stub_vector_reapers(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Report *names* as the configured vector backends, without a cluster."""
    monkeypatch.setattr(cli, "sweepable_backends", lambda: list(names))
    monkeypatch.setattr(cli, "VectorStoreReaper", _StubVectorReaper)


def test_no_vector_store_still_sweeps_the_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment with no cluster still has files to reclaim."""
    _stub_vector_reapers(monkeypatch)

    assert [reaper.kind for reaper in cli._build_reapers(set())] == [ResourceKind.WORKSPACE]


def test_a_qdrant_only_deployment_gets_a_vector_reaper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure this prevents: a Qdrant cluster reported clean by a sweep
    that only ever knew how to look at Weaviate."""
    _stub_vector_reapers(monkeypatch, "qdrant")

    reapers = cli._build_reapers(set())

    assert [reaper.kind for reaper in reapers] == [ResourceKind.VECTOR, ResourceKind.WORKSPACE]
    assert reapers[0].backend_name == "qdrant"


def test_one_reaper_is_built_per_configured_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two clusters leak independently, so two of them are swept independently."""
    _stub_vector_reapers(monkeypatch, "qdrant", "weaviate")

    reapers = cli._build_reapers(set())

    assert [getattr(r, "backend_name", None) for r in reapers] == ["qdrant", "weaviate", None]


def test_only_restricts_the_sweep_to_one_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--only workspace`` must not open a cluster connection at all."""
    _stub_vector_reapers(monkeypatch, "weaviate")

    reapers = cli._build_reapers({ResourceKind.WORKSPACE})

    assert [reaper.kind for reaper in reapers] == [ResourceKind.WORKSPACE]


def test_only_vector_covers_every_configured_vector_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One selector for the family is what an unattended schedule wants."""
    _stub_vector_reapers(monkeypatch, "qdrant", "weaviate")

    reapers = cli._build_reapers({ResourceKind.VECTOR})

    assert [reaper.kind for reaper in reapers] == [ResourceKind.VECTOR, ResourceKind.VECTOR]


def test_an_unconstructable_backend_is_reported_and_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing client extra degrades the sweep, never fails it."""

    def _explode(backend_name: str) -> _StubVectorReaper:
        msg = f"{backend_name}-client is not installed"
        raise ImportError(msg)

    _stub_vector_reapers(monkeypatch, "qdrant")
    monkeypatch.setattr(cli, "VectorStoreReaper", _explode)

    assert [reaper.kind for reaper in cli._build_reapers(set())] == [ResourceKind.WORKSPACE]


def test_one_unconstructable_backend_does_not_cost_the_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial install still reaps what it can."""

    def _selective(backend_name: str) -> _StubVectorReaper:
        if backend_name == "qdrant":
            msg = "qdrant-client is not installed"
            raise ImportError(msg)
        return _StubVectorReaper(backend_name)

    _stub_vector_reapers(monkeypatch, "qdrant", "weaviate")
    monkeypatch.setattr(cli, "VectorStoreReaper", _selective)

    reapers = cli._build_reapers(set())

    assert [getattr(r, "backend_name", None) for r in reapers] == ["weaviate", None]


def test_the_filesystem_store_is_read_when_mongo_is_unconfigured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """Community tier: no Mongo env, so the YAML event store answers."""
    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.delenv("MONGO_DB", raising=False)
    monkeypatch.setenv("AKGENTIC_EVENT_STORE_PATH", str(tmp_path))

    store = cli._build_event_store()

    assert store.list_teams() == []


def test_a_half_configured_mongo_does_not_silently_read_the_wrong_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """``MONGO_URI`` without ``MONGO_DB`` is a misconfiguration, not Mongo."""
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
    monkeypatch.delenv("MONGO_DB", raising=False)
    monkeypatch.setenv("AKGENTIC_EVENT_STORE_PATH", str(tmp_path))

    assert cli._build_event_store().list_teams() == []


# ---------------------------------------------------------------------------
# Rendering and exit codes
# ---------------------------------------------------------------------------


def _report(**overrides: object) -> SweepReport:
    """Build a one-reaper report, overriding the reaper entry's fields."""
    entry = ReaperReport(kind=ResourceKind.VECTOR, scanned=3, **overrides)  # type: ignore[arg-type]
    return SweepReport(applied=False, live_team_ids=2, reports=[entry])


def test_a_dry_run_says_so_in_the_first_line() -> None:
    """An operator must never mistake a plan for a completed deletion."""
    text = cli._render(_report(orphans=[make_ref(str(uuid.uuid4()))]))

    assert "DRY RUN" in text.splitlines()[0]
    assert "nothing deleted" in text.splitlines()[0]


def test_every_orphan_is_named_in_the_output() -> None:
    """The plan is only reviewable if it lists what it would delete."""
    dead = str(uuid.uuid4())
    text = cli._render(_report(orphans=[make_ref(dead, size=42)]))

    assert dead in text
    assert "42 rows" in text


def test_an_unavailable_backend_is_rendered_as_such() -> None:
    """ "scanned 0, orphaned 0" for a dead cluster would read as a clean sweep."""
    text = cli._render(_report(available=False, unavailable_reason="connection refused"))

    assert "UNAVAILABLE — connection refused" in text
    assert "orphaned 0" not in text


def test_a_clean_sweep_exits_zero() -> None:
    """Nothing to alert on."""
    assert cli._exit_code(_report()) == 0


def test_an_unreachable_backend_exits_nonzero() -> None:
    """A cron job that cannot reach its cluster must page, not report success."""
    assert cli._exit_code(_report(available=False, unavailable_reason="down")) == 1


def test_a_failed_purge_exits_nonzero() -> None:
    """A resource the backend refuses is a leak that will not clear itself."""
    assert cli._exit_code(_report(failures=["planning/team-a: shared collection"])) == 1


# ---------------------------------------------------------------------------
# End to end through the command
# ---------------------------------------------------------------------------


def _run(
    monkeypatch: pytest.MonkeyPatch,
    reaper: FakeReaper,
    processes: list[Process],
    args: list[str],
) -> Result:
    """Invoke the command with the store and reapers replaced by fakes."""
    monkeypatch.setattr(cli, "_build_event_store", lambda: FakeEventStore(processes))
    monkeypatch.setattr(cli, "_build_reapers", lambda _only: [reaper])
    return CliRunner().invoke(cli.app, args)


def test_the_command_reports_a_plan_without_deleting(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default invocation is the one an operator runs first."""
    live, dead = uuid.uuid4(), uuid.uuid4()
    reaper = FakeReaper([make_ref(str(live)), make_ref(str(dead))])

    result = _run(monkeypatch, reaper, [make_process(live)], [])

    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    assert reaper.purged == []
    assert reaper.closed is True


def test_apply_reaches_the_reaper(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flag has to survive the whole way to a purge, not just be parsed."""
    live_ids = [uuid.uuid4() for _ in range(4)]
    dead = uuid.uuid4()
    reaper = FakeReaper([make_ref(str(tid)) for tid in [*live_ids, dead]])

    result = _run(monkeypatch, reaper, [make_process(t) for t in live_ids], ["--apply"])

    assert result.exit_code == 0
    assert [ref.team_id for ref in reaper.purged] == [str(dead)]


def test_a_refused_sweep_exits_three_and_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exit 3 is what a cron job alerts on; the reason is what a human reads."""
    reaper = FakeReaper([make_ref(str(uuid.uuid4())) for _ in range(5)])

    result = _run(monkeypatch, reaper, [], ["--apply"])

    assert result.exit_code == 3
    assert "REFUSED" in result.output
    assert reaper.purged == []


def test_force_gets_past_the_guard_from_the_command_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented escape hatch has to actually be wired to the driver."""
    reaper = FakeReaper([make_ref(str(uuid.uuid4())) for _ in range(5)])

    result = _run(monkeypatch, reaper, [], ["--apply", "--force"])

    assert result.exit_code == 0
    assert len(reaper.purged) == 5


def test_the_reaper_is_closed_even_when_the_sweep_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leaked cluster connection outlives the process that opened it."""
    reaper = FakeReaper([])

    def _explode(*_args: object, **_kwargs: object) -> None:
        msg = "driver blew up"
        raise RuntimeError(msg)

    monkeypatch.setattr(cli, "sweep", _explode)
    result = _run(monkeypatch, reaper, [], [])

    assert result.exit_code != 0
    assert reaper.closed is True


def test_a_store_that_cannot_be_opened_exits_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing was scanned, so nothing may be reported as swept."""

    def _explode() -> None:
        msg = "mongo unreachable"
        raise RuntimeError(msg)

    monkeypatch.setattr(cli, "_build_event_store", _explode)
    result = CliRunner().invoke(cli.app, [])

    assert result.exit_code == 2
    assert "Cannot read the live team set" in result.output


def test_no_configured_backend_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment with no cluster and no workspace root has nothing to sweep."""
    monkeypatch.setattr(cli, "_build_event_store", lambda: FakeEventStore([]))
    monkeypatch.setattr(cli, "_build_reapers", lambda _only: [])

    result = CliRunner().invoke(cli.app, [])

    assert result.exit_code == 0
    assert "No backend configured" in result.output


def test_json_output_round_trips_as_the_report_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--json`` is for machines, so it must parse back into the model."""
    reaper = FakeReaper([make_ref(str(uuid.uuid4()))])

    result = _run(monkeypatch, reaper, [make_process(uuid.uuid4())], ["--json"])

    parsed = SweepReport.model_validate_json(result.output.strip())
    assert parsed.total_orphans == 1
