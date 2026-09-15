"""Driver for the orphaned team-resource sweep.

``TeamManager.delete_team`` purges the event store and deregisters the team.
It touches no Weaviate object and removes no Docker container, so every
deleted team leaks its side-store state by construction — and a delete hook
bolted on now would still leak on every crash between the two writes, with no
way to recover what leaked before it existed. This is therefore a reverse
sweep: enumerate what the backends hold, diff it against the teams that are
still live, remove the difference.

**The scan happens before the live-set read, and that order is the safety
property.** Take a team created while the sweep is running:

* scan first, live-set second — the team is absent from the scan snapshot (it
  did not exist yet), so it cannot be reaped whatever the live set says;
* live-set first, scan second — the team is absent from the live set and
  present in the scan, and the sweep deletes a brand-new team's data.

Nothing else in this module protects against that, so nothing else may be
allowed to reorder it. The grace period is a second, weaker line of defence
for backends whose resource is created before the team's first persisted
checkpoint — Docker, where the container starts with the team. Weaviate
objects appear only on ingest, long after the team is durable, so no grace
period applies there.

**A soft-deleted team is dead for this purpose.** ``list_teams`` returns
``DELETED`` processes alongside live ones, so "present in the store" is not
the test — "present and not ``DELETED``" is.

**What is protected is a set of claims, not a set of team ids.** A workspace
directory is named ``workspace_id or team_id``, and ``workspace_id`` is an
operator-chosen override that two teams may share — so a live team routinely
owns a directory whose name is not its id. Every ``workspace_id`` on a card a
live team references is therefore added to the protected set before anything
is condemned. A card that cannot be resolved makes that set incomplete, which
on the workspace path means deleting a live team's files; the guard refuses
rather than proceed on a partial answer.

**A thin live set is treated as a broken read, not as a mass deletion.** The
event store answers "which teams exist" on a best-effort basis: the YAML and
Mongo backends both log and skip a document they cannot parse, so a schema
migration that leaves stored documents behind makes every surviving team
invisible — and every one of its resources look orphaned. That is not
hypothetical; it was the state of a developer machine the first time this
sweep ran there, where a live set of zero would have reaped 39 live
sandboxes. The blast-radius guard below refuses to apply such a plan. It is
the last line of defence, and unlike the other two it fires on a wrong
*answer* rather than a wrong *order*.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from akgentic.infra.maintenance.models import ReaperReport, SweepReport
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
    containers are exactly what this sweep exists to reap.

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
    """Return the directory names live teams claim beyond their own ids.

    A workspace directory is named ``workspace_id or team_id``, so a team that
    configures a ``workspace_id`` — including one shared with another team —
    owns a directory its id does not name.

    The names live on tool cards, which a ``Process`` holds only as content
    hashes: the nested ``team_card`` was removed from the model in favour of
    the flat projection, so the cards have to be resolved through the store.
    All of them are resolved in **one** round trip, which is the contract
    ``load_agent_cards`` exists to offer. Each card is then walked
    structurally for ``workspace_id`` at any depth rather than by reaching
    into a particular card type — the field appears on more than one tool
    card, and a structural walk cannot miss a nesting the sweep has not been
    told about.

    Args:
        processes: The live processes whose cards to read.
        event_store: Store holding the cards those processes reference.

    Returns:
        The claimed names, and the number of card references that could not be
        resolved. A non-zero count means the protected set is incomplete —
        never treat it as "no extra claims".
    """
    hashes = sorted({ref.card_hash for process in processes for ref in process.agent_cards})
    if not hashes:
        return set(), 0
    try:
        cards = event_store.load_agent_cards(hashes)
    except Exception as exc:  # noqa: BLE001 - an unreadable store must not crash the scan
        logger.warning("Could not resolve agent cards for workspace claims: %s", exc)
        return set(), len(hashes)

    claims: set[str] = set()
    unreadable = 0
    for card_hash in hashes:
        card = cards.get(card_hash)
        if card is None:
            logger.warning("Card %s is referenced by a live team but not in the store", card_hash)
            unreadable += 1
            continue
        try:
            _collect_workspace_ids(card.model_dump(), claims)
        except Exception as exc:  # noqa: BLE001 - one bad card must not hide the rest
            logger.warning("Could not read workspace claims from card %s: %s", card_hash, exc)
            unreadable += 1
    return claims, unreadable


def _collect_workspace_ids(node: object, into: set[str]) -> None:
    """Collect every non-empty ``workspace_id`` string reachable from *node*."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "workspace_id" and isinstance(value, str) and value:
                into.add(value)
            else:
                _collect_workspace_ids(value, into)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _collect_workspace_ids(item, into)


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
            logger.warning("Reaper %s could not scan: %s", reaper.kind, exc)
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
        protected: Every name a live team claims — its id, plus any
            ``workspace_id`` it declares.
        failure: Why the scan failed, or ``None`` when it succeeded.
        grace_seconds: Age below which a resource is held back.

    Returns:
        The reaper's report, with nothing purged yet.
    """
    if failure is not None:
        return ReaperReport(kind=reaper.kind, available=False, unavailable_reason=failure)

    report = ReaperReport(kind=reaper.kind, scanned=len(refs))
    for ref in refs:
        if ref.team_id in protected:
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
    condemns files that a live team is still writing to, and unlike a container
    or a vector those cannot be rebuilt.

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
            f"{report.unreadable_teams} agent card(s) referenced by a live team could "
            "not be read, so the workspace claims are incomplete and a directory a "
            "live team owns could be condemned. Fix the store, or re-run with "
            "--only weaviate --only docker to sweep the recoverable backends."
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

    One failure never aborts the pass: a container Docker refuses to remove
    must not strand every other orphan behind it.

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
