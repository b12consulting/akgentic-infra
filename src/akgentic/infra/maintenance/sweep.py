"""Driver for the orphaned team-resource sweep.

``TeamManager.delete_team`` purges the event store and deregisters the team. It
touches no vector row, and the workspace trees it does remove it removes only on
the path that runs to completion — so every crash between the event-store write
and the ``rmtree``, every tier whose deletion policy refuses, and every tree or
row written before those paths existed leaks by construction. A delete hook
cannot recover any of it. This is therefore a reverse sweep: enumerate what the
backends hold, diff it against the teams that are still live, remove the
difference.

**The scan happens before the live-set read, and that order is the safety
property.** Take a team created while the sweep is running:

* scan first, live-set second — the team is absent from the scan snapshot (it
  did not exist yet), so it cannot be reaped whatever the live set says;
* live-set first, scan second — the team is absent from the live set and
  present in the scan, and the sweep deletes a brand-new team's data.

Nothing else in this module protects against that, so nothing else may be
allowed to reorder it. The grace period is a second, weaker line of defence,
and it applies wherever a backend exposes an age: today that is the workspace
reaper, whose directory mtime moves on every write, so an actively used tree
reads as young and survives a sweep that misjudged it. Vector rows appear only
on ingest, long after the team is durable, and expose no cheap creation time at
the granularity a reference is cut on, so no grace period applies there.

**A soft-deleted team is dead for this purpose.** ``list_teams`` returns
``DELETED`` processes alongside live ones, so "present in the store" is not
the test — "present and not ``DELETED``" is.

**What is protected is a set of claims, not a set of team ids.** A workspace
reference is keyed on its whole ``<scope>/<kind>/<leaf>`` path, because a team's
own tree can sit in either scope and a leaf is unique only within one. So the
claim set is the resolved ``deletion_candidate_paths`` of every live team —
the *same* reader the delete path resolves its targets through, never a second
derivation of it, because a sweep that re-derived which trees a team owns would
drift from the delete path silently and in the one direction that cannot be
undone. A live team whose candidates cannot be resolved makes that set
incomplete, which on the workspace path means deleting a live team's files; the
guard refuses rather than proceed on a partial answer.

That keying is what reclaims the one leak an id-keyed set cannot see. A team
whose card once resolved per-principal has a tree at ``<owner>/_team/<id>``; if
the card now declares ``workspace_sharable``, its agents write to
``_shared/_team/<id>`` and nothing will ever write to the first tree again.
Under id-keyed protection it is protected for ever, because the team is live.
Under path-keyed protection it is not in the team's candidate set, and it is
reclaimed — which is the safety-net job this sweep exists to do.

**A thin live set is treated as a broken read, not as a mass deletion.** The
event store answers "which teams exist" on a best-effort basis: the YAML and
Mongo backends both log and skip a document they cannot parse, so a schema
migration that leaves stored documents behind makes every surviving team
invisible — and every one of its resources look orphaned. That is not
hypothetical; it was the state of a developer machine the first time this
sweep ran there, where a live set of zero would have condemned the resources
of all 39 live teams on it. The blast-radius guard below refuses to apply such
a plan. It is the last line of defence, and unlike the other two it fires on a
wrong *answer* rather than a wrong *order*.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from akgentic.infra.maintenance.models import ReaperReport, SweepReport
from akgentic.infra.server.services._workspace_paths import deletion_candidate_paths
from akgentic.team.models import TeamStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

    from akgentic.infra.maintenance.models import ResourceRef
    from akgentic.infra.maintenance.reapers import TeamResourceReaper
    from akgentic.team.models import Process
    from akgentic.team.ports import EventStore

logger = logging.getLogger(__name__)

DEFAULT_GRACE_SECONDS: float = 3600.0
"""How long a resource is protected from reaping by its age alone."""

DEFAULT_MAX_ORPHAN_FRACTION: float = 0.5
"""Share of the scan that may be orphaned before the sweep refuses to apply.

A steady-state deployment reaps a handful of teams per run. A plan that
condemns most of what it scanned is far more likely to be a misread live set
than a genuine backlog — so it stops and asks for a human.
"""


def live_team_ids(event_store: EventStore) -> set[str]:
    """Return the ids of every team that still owns its resources.

    A ``DELETED`` process is excluded: it is a tombstone, and its vectors and
    workspace files are exactly what this sweep exists to reap.

    Args:
        event_store: Store to read. Any backend satisfying the protocol works
            — the sweep is tier-agnostic by construction.

    Returns:
        Team ids as strings, matching how the backends spell them.
    """
    return _team_ids(_live_processes(event_store))


def _live_processes(event_store: EventStore) -> list[Process]:
    """Return every stored process that is not a tombstone."""
    return [p for p in event_store.list_teams() if p.status is not TeamStatus.DELETED]


def _team_ids(processes: Sequence[Process]) -> set[str]:
    """Return the team ids of *processes*, spelled as the backends spell them."""
    return {str(process.team_id) for process in processes}


def _workspace_claims(
    processes: Sequence[Process], event_store: EventStore
) -> tuple[set[str], int]:
    """Return the workspace paths live teams claim, and how many would not resolve.

    Every claim is a live team's own deletion candidate, resolved through
    ``deletion_candidate_paths`` — the **same** reader
    ``TeamService.delete_team`` resolves its targets through. Re-deriving which
    trees a team owns here would drift from that reader silently, and the
    direction of that drift is a sweep deleting what the delete path would have
    kept. The reaper's own contribution is the reverse direction only: finding
    trees no live team claims at all.

    **One card-store read per live team, where the old structural walk made one
    batched read in total.** That is the price of not re-deriving the delete
    path's judgement, and correctness on an unrecoverable delete path buys it
    easily. If a large deployment needs it batched, that is a change to
    ``_workspace_paths``' reader, never a re-derivation here.

    A failure is counted **per process** so one unreadable team cannot hide the
    rest. ``deletion_candidate_paths`` documents two raises —
    ``AgentCardNotFoundError`` for a hash the store does not hold and
    ``ValueError`` for a declaration that cannot yield a path — and anything
    else the store does on its way to answering is the same danger, an
    incomplete protected set, so it is counted rather than allowed to abort the
    sweep or, far worse, to read as "this team claims nothing".

    Args:
        processes: The live processes whose candidates to resolve.
        event_store: Store holding the cards those processes reference.

    Returns:
        The claimed paths as POSIX strings, and the number of live teams whose
        candidates could not be resolved. A non-zero count means the protected
        set is incomplete — never treat it as "no extra claims".
    """
    claims: set[str] = set()
    unreadable = 0
    for process in processes:
        try:
            candidates = deletion_candidate_paths(process=process, store=event_store)
        except Exception as exc:  # noqa: BLE001 - one bad team must not hide the rest
            logger.warning(
                "Could not resolve the workspace candidates of live team %s: %s",
                process.team_id,
                exc,
            )
            unreadable += 1
            continue
        claims.update(path.as_posix() for path in candidates)
    return claims, unreadable


def sweep(
    reapers: Sequence[TeamResourceReaper],
    event_store: EventStore,
    *,
    apply: bool = False,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
    max_orphan_fraction: float = DEFAULT_MAX_ORPHAN_FRACTION,
    force: bool = False,
) -> SweepReport:
    """Find, and optionally remove, resources whose owning team is gone.

    Args:
        reapers: Backends to sweep. Each is scanned before the live set is
            read — see the module docstring on why that order is load-bearing.
        event_store: Source of truth for which teams are live.
        apply: When ``False`` (the default) nothing is deleted and the report
            is a plan. A sweep is destructive; it opts in, never out.
        grace_seconds: Resources younger than this are held back regardless of
            the live set. Only applies where the backend exposes a creation
            time.
        max_orphan_fraction: Refuse to apply a plan condemning more than this
            share of what was scanned. ``1.0`` disables the ratio check; the
            empty-live-set check has no threshold and is lifted by ``force``
            alone.
        force: Apply the plan even when the blast-radius guard objects. For an
            operator who has looked at the dry run and knows why it is large.

    Returns:
        One :class:`~.models.ReaperReport` per reaper, in the order given, and
        a refusal reason when the guard stopped an ``apply``.
    """
    scanned = _scan_all(reapers)
    processes = _live_processes(event_store)
    live = _team_ids(processes)
    claims, unreadable = _workspace_claims(processes, event_store)
    protected = live | claims
    logger.info(
        "Sweep: %d live teams, %d extra claims, %d unreadable, apply=%s",
        len(live),
        len(claims),
        unreadable,
        apply,
    )

    report = SweepReport(
        applied=False,
        live_team_ids=len(live),
        extra_claims=len(claims),
        unreadable_teams=unreadable,
    )
    for reaper, (refs, failure) in zip(reapers, scanned, strict=True):
        report.reports.append(_classify(reaper, refs, protected, failure, grace_seconds))
    if not apply:
        return report

    refusal = None if force else _blast_radius_refusal(report, max_orphan_fraction)
    if refusal is not None:
        logger.error("Refusing to apply: %s", refusal)
        report.refusal_reason = refusal
        return report

    report.applied = True
    for reaper, entry in zip(reapers, report.reports, strict=True):
        _purge_orphans(reaper, entry)
    return report


def _scan_all(
    reapers: Sequence[TeamResourceReaper],
) -> list[tuple[list[ResourceRef], str | None]]:
    """Scan every reaper, before any live-set read.

    A backend that cannot be reached yields an empty list and a reason rather
    than raising: one unreachable cluster must not stop the other backends
    from being swept, and an empty scan must never be reported as a clean one.

    Args:
        reapers: Backends to scan.

    Returns:
        One ``(references, failure_reason)`` pair per reaper, positionally
        aligned with ``reapers``.
    """
    results: list[tuple[list[ResourceRef], str | None]] = []
    for reaper in reapers:
        try:
            results.append((reaper.scan(), None))
        except Exception as exc:  # noqa: BLE001 - one backend must not stop the rest
            logger.warning("Reaper %s could not scan: %s", reaper.backend or reaper.kind, exc)
            results.append(([], str(exc)))
    return results


def _classify(
    reaper: TeamResourceReaper,
    refs: list[ResourceRef],
    protected: set[str],
    failure: str | None,
    grace_seconds: float,
) -> ReaperReport:
    """Split one reaper's scan into claimed, too-young, and orphaned.

    Args:
        reaper: Backend the references came from.
        refs: Everything the scan found.
        protected: Every key a live team claims — its id, plus the resolved
            path of every tree its deletion would consider.
        failure: Why the scan failed, or ``None`` when it succeeded.
        grace_seconds: Age below which a resource is held back.

    Returns:
        The reaper's report, with nothing purged yet.

    Note:
        A reference is diffed on its ``claim_key`` when it has one and on its
        ``team_id`` otherwise. That is the whole of what the driver knows about
        keying: the workspace reaper protects on a path and the vector reapers
        on a team id, and putting the key on the reference is what keeps the
        driver from learning either backend's layout.
    """
    if failure is not None:
        return ReaperReport(
            kind=reaper.kind,
            backend=reaper.backend,
            available=False,
            unavailable_reason=failure,
        )

    report = ReaperReport(kind=reaper.kind, backend=reaper.backend, scanned=len(refs))
    for ref in refs:
        if (ref.claim_key or ref.team_id) in protected:
            continue
        if ref.age_seconds is not None and ref.age_seconds < grace_seconds:
            report.skipped_young += 1
            continue
        report.orphans.append(ref)
    return report


def _blast_radius_refusal(report: SweepReport, max_orphan_fraction: float) -> str | None:
    """Return why this plan is too large to apply unattended, or ``None``.

    All three rules exist to catch the same failure — a protected set that is
    short because the store could not be read, not because the teams are gone.
    None of them can distinguish that from a genuine backlog, which is the
    point: the sweep stops and shows an operator the plan instead of guessing.

    The unreadable-claims rule is checked first and is the strictest, because
    it is the one that bears on the workspace reaper: an incomplete claim set
    condemns files that a live team is still writing to, and unlike a vector row
    those cannot be rebuilt.

    Args:
        report: The classified plan.
        max_orphan_fraction: Share of the scan that may be orphaned.

    Returns:
        A refusal reason naming the numbers, or ``None`` to proceed.
    """
    orphans = report.total_orphans
    if orphans == 0:
        return None
    if report.unreadable_teams:
        return (
            f"{report.unreadable_teams} live team(s) would not yield their workspace "
            "candidates, so the workspace claims are incomplete and a tree a live team "
            "owns could be condemned. Fix the store, or re-run with --only vector to "
            "sweep the recoverable backends."
        )
    if report.live_team_ids == 0:
        return (
            f"the live team set is empty while {orphans} resources are condemned. "
            "An event store that cannot read its documents reports no teams and "
            "makes every team look deleted. Check the store, then re-run with "
            "--force if the plan is genuinely correct."
        )
    scanned = sum(entry.scanned for entry in report.reports)
    fraction = orphans / scanned if scanned else 0.0
    if fraction > max_orphan_fraction:
        return (
            f"{orphans} of {scanned} scanned resources are condemned "
            f"({fraction:.0%} > {max_orphan_fraction:.0%}). Review the dry run, "
            "then re-run with --force or a higher --max-orphan-fraction."
        )
    return None


def _purge_orphans(reaper: TeamResourceReaper, report: ReaperReport) -> None:
    """Delete every orphan in *report*, recording failures in place.

    One failure never aborts the pass: a collection the cluster refuses to
    delete from must not strand every other orphan behind it.

    Args:
        reaper: Backend to delete through.
        report: The reaper's classified report, mutated with the outcome.
    """
    for ref in report.orphans:
        try:
            report.purged += reaper.purge(ref)
        except Exception as exc:  # noqa: BLE001 - one failure must not stop the rest
            logger.warning("Failed to purge %s %s: %s", reaper.kind, ref.label, exc)
            report.failures.append(f"{ref.label}: {exc}")
