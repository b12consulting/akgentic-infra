"""Behaviour of the orphaned team-resource sweep driver."""

from __future__ import annotations

import uuid

import pytest
from akgentic.team.models import TeamStatus

from akgentic.infra.maintenance.models import ResourceKind
from akgentic.infra.maintenance.sweep import live_team_ids, sweep
from tests.maintenance.conftest import (
    FakeEventStore,
    FakeReaper,
    make_claiming_store,
    make_process,
    make_ref,
)

# ---------------------------------------------------------------------------
# The live set
# ---------------------------------------------------------------------------


def test_live_set_excludes_deleted_teams() -> None:
    """A soft-deleted team is a tombstone: its resources are reapable."""
    running, stopped, deleted = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    store = FakeEventStore(
        [
            make_process(running, TeamStatus.RUNNING),
            make_process(stopped, TeamStatus.STOPPED),
            make_process(deleted, TeamStatus.DELETED),
        ]
    )

    assert live_team_ids(store) == {str(running), str(stopped)}


def test_stopped_team_keeps_its_resources() -> None:
    """STOPPED is resumable, so its vectors and sandbox must survive."""
    stopped = uuid.uuid4()
    store = FakeEventStore([make_process(stopped, TeamStatus.STOPPED)])
    reaper = FakeReaper([make_ref(str(stopped))])

    report = sweep([reaper], store, apply=True)

    assert report.total_orphans == 0
    assert reaper.purged == []


# ---------------------------------------------------------------------------
# The ordering invariant
# ---------------------------------------------------------------------------


def test_every_reaper_is_scanned_before_the_live_set_is_read() -> None:
    """Scan-then-live is the whole safety property; assert it, don't assume it.

    Reading the live set first would let a team created mid-sweep appear in
    the scan and be absent from the live set — and be deleted minutes after
    an operator created it.
    """
    journal: list[str] = []
    store = FakeEventStore([], journal)
    reapers = [
        FakeReaper([], journal, kind=ResourceKind.WEAVIATE),
        FakeReaper([], journal, kind=ResourceKind.DOCKER),
    ]

    sweep(reapers, store)

    assert journal == ["scan", "scan", "list_teams"]


def test_live_set_is_read_once_for_the_whole_sweep() -> None:
    """All reapers decide against one snapshot, never a per-backend re-read."""
    store = FakeEventStore([])
    reapers = [
        FakeReaper([], kind=ResourceKind.WEAVIATE),
        FakeReaper([], kind=ResourceKind.DOCKER),
    ]

    sweep(reapers, store)

    assert store.calls == 1


# ---------------------------------------------------------------------------
# Dry run vs apply
# ---------------------------------------------------------------------------


def test_dry_run_is_the_default_and_deletes_nothing() -> None:
    """The plan names the orphan; nothing is purged without ``apply``."""
    dead = uuid.uuid4()
    reaper = FakeReaper([make_ref(str(dead))])

    report = sweep([reaper], FakeEventStore([]))

    assert report.applied is False
    assert [ref.team_id for ref in report.reports[0].orphans] == [str(dead)]
    assert report.total_purged == 0
    assert reaper.purged == []


def test_apply_purges_exactly_the_planned_orphans() -> None:
    """The set decided on is the set deleted — no re-query in between."""
    live, dead = uuid.uuid4(), uuid.uuid4()
    store = FakeEventStore([make_process(live)])
    reaper = FakeReaper([make_ref(str(live)), make_ref(str(dead), size=7)])

    report = sweep([reaper], store, apply=True)

    assert [ref.team_id for ref in reaper.purged] == [str(dead)]
    assert report.total_purged == 7
    assert report.reports[0].scanned == 2


# ---------------------------------------------------------------------------
# Grace period
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("age_seconds", "expected_orphans", "expected_young"),
    [(30.0, 0, 1), (7200.0, 1, 0), (None, 1, 0)],
)
def test_grace_period_holds_back_young_resources(
    age_seconds: float | None, expected_orphans: int, expected_young: int
) -> None:
    """A resource younger than the grace period survives, whatever the live set.

    An unknown age (``None``) is *not* protected: the grace period is the
    second line of defence, and the scan ordering is the first.
    """
    reaper = FakeReaper([make_ref(str(uuid.uuid4()), age_seconds=age_seconds)])

    report = sweep([reaper], FakeEventStore([]), apply=True, force=True, grace_seconds=3600.0)

    assert len(report.reports[0].orphans) == expected_orphans
    assert report.reports[0].skipped_young == expected_young


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


def test_unreachable_backend_is_reported_not_reported_clean() -> None:
    """An empty scan from a dead cluster must never read as "no orphans"."""
    reaper = FakeReaper([], scan_error=OSError("connection refused"))

    report = sweep([reaper], FakeEventStore([]))

    entry = report.reports[0]
    assert entry.available is False
    assert entry.unavailable_reason == "connection refused"
    assert entry.orphans == []


def test_one_unreachable_backend_does_not_stop_the_others() -> None:
    """A dead Weaviate must not leave Docker containers leaking too."""
    dead = uuid.uuid4()
    broken = FakeReaper([], scan_error=OSError("down"), kind=ResourceKind.WEAVIATE)
    working = FakeReaper([make_ref(str(dead))], kind=ResourceKind.DOCKER)

    report = sweep([broken, working], FakeEventStore([]), apply=True, force=True)

    assert report.reports[0].available is False
    assert [ref.team_id for ref in working.purged] == [str(dead)]


def test_a_failed_purge_is_recorded_and_the_sweep_continues() -> None:
    """One stuck container must not strand every other orphan."""
    reaper = FakeReaper(
        [make_ref(str(uuid.uuid4())), make_ref(str(uuid.uuid4()))],
        purge_error=OSError("device busy"),
    )

    report = sweep([reaper], FakeEventStore([]), apply=True, force=True)

    assert len(report.reports[0].failures) == 2
    assert report.total_failures == 2
    assert report.total_purged == 0


# ---------------------------------------------------------------------------
# Blast-radius guard
# ---------------------------------------------------------------------------


def test_an_empty_live_set_refuses_to_delete_anything() -> None:
    """An unreadable store reports no teams and condemns every resource.

    This is not hypothetical: a schema migration left every stored team
    document unparseable on a developer machine, and the first sweep there
    condemned all 39 live sandboxes.
    """
    reaper = FakeReaper([make_ref(str(uuid.uuid4())) for _ in range(39)])

    report = sweep([reaper], FakeEventStore([]), apply=True)

    assert report.refused is True
    assert report.applied is False
    assert reaper.purged == []
    assert "live team set is empty" in (report.refusal_reason or "")


def test_the_empty_live_set_guard_is_silent_when_there_is_nothing_to_reap() -> None:
    """A genuinely empty deployment sweeps clean rather than raising an alarm."""
    report = sweep([FakeReaper([])], FakeEventStore([]), apply=True)

    assert report.refused is False
    assert report.applied is True


def test_a_plan_condemning_most_of_the_scan_is_refused() -> None:
    """One live team and forty condemned is a misread store, not a backlog."""
    live = uuid.uuid4()
    refs = [make_ref(str(live)), *(make_ref(str(uuid.uuid4())) for _ in range(9))]
    reaper = FakeReaper(refs)

    report = sweep([reaper], FakeEventStore([make_process(live)]), apply=True)

    assert report.refused is True
    assert "90%" in (report.refusal_reason or "")
    assert reaper.purged == []


def test_a_proportionate_plan_applies_untouched() -> None:
    """Steady state: a few dead teams among many live ones reaps normally."""
    live_ids = [uuid.uuid4() for _ in range(9)]
    dead = uuid.uuid4()
    reaper = FakeReaper([make_ref(str(tid)) for tid in [*live_ids, dead]])
    store = FakeEventStore([make_process(tid) for tid in live_ids])

    report = sweep([reaper], store, apply=True)

    assert report.refused is False
    assert [ref.team_id for ref in reaper.purged] == [str(dead)]


def test_force_overrides_the_guard() -> None:
    """An operator who has read the dry run can still reap a large backlog."""
    reaper = FakeReaper([make_ref(str(uuid.uuid4())) for _ in range(39)])

    report = sweep([reaper], FakeEventStore([]), apply=True, force=True)

    assert report.refused is False
    assert report.applied is True
    assert len(reaper.purged) == 39


def test_the_guard_never_fires_on_a_dry_run() -> None:
    """A dry run deletes nothing, so there is no blast radius to guard."""
    reaper = FakeReaper([make_ref(str(uuid.uuid4())) for _ in range(39)])

    report = sweep([reaper], FakeEventStore([]))

    assert report.refused is False
    assert report.refusal_reason is None
    assert len(report.reports[0].orphans) == 39


# ---------------------------------------------------------------------------
# Workspace claims
# ---------------------------------------------------------------------------


def test_a_shared_workspace_id_protects_a_directory_no_team_id_names() -> None:
    """The failure this prevents is deleting a live team's files.

    ``WorkspaceTool(workspace_id=...)`` is a supported override, so a live
    team routinely owns a directory whose name is not its id. Diffing against
    team ids alone condemns it.
    """
    live = uuid.uuid4()
    store = make_claiming_store(live, "shared-docs")
    reaper = FakeReaper([make_ref("shared-docs"), make_ref(str(live))], kind=ResourceKind.WORKSPACE)

    report = sweep([reaper], store, apply=True)

    assert report.extra_claims == 1
    assert report.total_orphans == 0
    assert reaper.purged == []


def test_a_claim_a_dead_team_made_does_not_protect_anything() -> None:
    """Only *live* teams' claims count, or deletion would never reclaim a share."""
    store = make_claiming_store(uuid.uuid4(), "shared-docs", TeamStatus.DELETED)
    reaper = FakeReaper([make_ref("shared-docs")], kind=ResourceKind.WORKSPACE)

    report = sweep([reaper], store, apply=True, force=True)

    assert report.extra_claims == 0
    assert [ref.team_id for ref in reaper.purged] == ["shared-docs"]


def test_claims_protect_every_backend_not_only_the_workspace_one() -> None:
    """The protected set is one set; a reaper does not get its own rules."""
    reaper = FakeReaper([make_ref("shared-docs")], kind=ResourceKind.DOCKER)

    report = sweep([reaper], make_claiming_store(uuid.uuid4(), "shared-docs"), apply=True)

    assert report.total_orphans == 0


def test_a_card_the_store_cannot_resolve_refuses_the_whole_apply() -> None:
    """Under-protection deletes files, so a partial answer is not acted on."""
    store = make_claiming_store(uuid.uuid4(), "shared-docs")
    store.cards.clear()
    reaper = FakeReaper([make_ref(str(uuid.uuid4()))], kind=ResourceKind.WORKSPACE)

    report = sweep([reaper], store, apply=True)

    assert report.unreadable_teams > 0
    assert report.refused is True
    assert "workspace claims" in (report.refusal_reason or "")
    assert reaper.purged == []


def test_a_store_that_cannot_resolve_cards_at_all_refuses_too() -> None:
    """A raising ``load_agent_cards`` is the same danger as a missing card."""
    store = make_claiming_store(uuid.uuid4(), "shared-docs")
    store.set_card_error(RuntimeError("store down"))
    reaper = FakeReaper([make_ref(str(uuid.uuid4()))], kind=ResourceKind.WORKSPACE)

    report = sweep([reaper], store, apply=True)

    assert report.refused is True
    assert reaper.purged == []


def test_the_cards_are_resolved_in_one_round_trip() -> None:
    """``load_agent_cards`` exists to prevent an N+1 across a team's roles."""
    store = make_claiming_store(uuid.uuid4(), "shared-docs")

    sweep([FakeReaper([])], store)

    assert store.journal.count("load_agent_cards") == 1
