"""Command-line entry point for the orphaned team-resource sweep.

Run it as a module, which is what a Kubernetes ``CronJob`` or a Nomad
periodic batch should invoke::

    python -m akgentic.infra.maintenance                 # dry run, prints the plan
    python -m akgentic.infra.maintenance --apply         # actually deletes
    python -m akgentic.infra.maintenance --only docker --apply

The workspace reaper deletes the team's **files**, which nothing can rebuild.
Read the dry run before applying it, and prefer ``--only weaviate --only
docker`` on a schedule if that is the only part you want unattended.

Backends are discovered from the environment, and every one of them is
optional — a sweep with Weaviate unconfigured simply reports Docker.

===========================  ==================================================
``AKGENTIC_WEAVIATE_URL``    Enables the Weaviate reaper. Unset, it is skipped.
``AKGENTIC_WEAVIATE_API_KEY``  Optional cluster credential.
``AKGENTIC_WORKSPACES_ROOT``   Workspace root to sweep (default ``./workspaces``).
``MONGO_URI`` / ``MONGO_DB``   Read the live team set from Mongo. Unset, the
                             filesystem event store is read instead.
``AKGENTIC_EVENT_STORE_PATH``  Filesystem event store root.
===========================  ==================================================

Exit codes:
    0 — swept cleanly (dry run or applied, nothing failed)
    1 — a backend was unreachable, or a purge failed
    2 — the live team set could not be read; nothing was scanned or deleted
    3 — the blast-radius guard refused the plan; nothing was deleted
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import typer

from akgentic.infra.maintenance.models import ResourceKind, SweepReport
from akgentic.infra.maintenance.reapers import (
    DockerReaper,
    TeamResourceReaper,
    WeaviateReaper,
    WorkspaceReaper,
)
from akgentic.infra.maintenance.sweep import (
    DEFAULT_GRACE_SECONDS,
    DEFAULT_MAX_ORPHAN_FRACTION,
    sweep,
)

if TYPE_CHECKING:
    from akgentic.team.ports import EventStore

logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    help="Remove Weaviate objects, Docker sandboxes and workspaces owned by deleted teams.",
)


@app.command()
def main(
    apply: bool = typer.Option(
        False,  # noqa: FBT003 - typer reads the default as the flag's off state
        "--apply",
        help="Actually delete. Without it the sweep prints the plan and changes nothing.",
    ),
    only: list[ResourceKind] = typer.Option(  # noqa: B008 - typer builds the default
        [],
        "--only",
        help="Restrict the sweep to these backends. Repeatable. Default: all configured.",
    ),
    grace_seconds: float = typer.Option(
        DEFAULT_GRACE_SECONDS,
        "--grace-seconds",
        help="Hold back resources younger than this, whatever the live set says.",
    ),
    max_orphan_fraction: float = typer.Option(
        DEFAULT_MAX_ORPHAN_FRACTION,
        "--max-orphan-fraction",
        help="Refuse to apply a plan condemning more than this share of the scan.",
    ),
    force: bool = typer.Option(
        False,  # noqa: FBT003 - typer reads the default as the flag's off state
        "--force",
        help="Apply even when the blast-radius guard objects. Read the dry run first.",
    ),
    as_json: bool = typer.Option(
        False,  # noqa: FBT003 - typer reads the default as the flag's off state
        "--json",
        help="Emit the report as JSON instead of text.",
    ),
) -> None:
    """Sweep every configured backend for resources whose team is gone."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    try:
        event_store = _build_event_store()
    except Exception as exc:  # noqa: BLE001 - reported as exit 2, never a traceback
        typer.echo(f"Cannot read the live team set: {exc}", err=True)
        raise typer.Exit(code=2) from None

    reapers = _build_reapers(set(only))
    if not reapers:
        typer.echo("No backend configured to sweep; nothing to do.", err=True)
        raise typer.Exit(code=0)

    try:
        report = sweep(
            reapers,
            event_store,
            apply=apply,
            grace_seconds=grace_seconds,
            max_orphan_fraction=max_orphan_fraction,
            force=force,
        )
    finally:
        for reaper in reapers:
            reaper.close()

    typer.echo(report.model_dump_json(indent=2) if as_json else _render(report))
    raise typer.Exit(code=_exit_code(report))


def _build_event_store() -> EventStore:
    """Open the event store holding the live team set.

    Mongo when ``MONGO_URI`` and ``MONGO_DB`` are both set — the enterprise
    and department tiers — otherwise the filesystem store the community tier
    writes. Index creation is suppressed: a maintenance job reads, and must
    never be the process that provisions a schema.

    Returns:
        A store satisfying the ``EventStore`` protocol.
    """
    uri, db_name = os.environ.get("MONGO_URI"), os.environ.get("MONGO_DB")
    if uri and db_name:
        import pymongo

        from akgentic.team.repositories.mongo import MongoEventStore

        return MongoEventStore(pymongo.MongoClient(uri)[db_name], auto_create_indexes=False)

    from akgentic.infra.server.settings import CommunitySettings
    from akgentic.team.repositories.yaml import YamlEventStore

    return YamlEventStore(data_dir=CommunitySettings().event_store_path)


def _build_reapers(only: set[ResourceKind]) -> list[TeamResourceReaper]:
    """Construct one reaper per configured, requested backend.

    A backend that cannot even be constructed — Weaviate without the
    ``weaviate-client`` extra, say — is reported and skipped rather than
    failing the sweep, so a partial install still reaps what it can.

    Args:
        only: Restrict to these kinds. Empty means every configured backend.

    Returns:
        Reapers in a stable order: Weaviate, then Docker, then the workspace
        filesystem. The workspace reaper needs no configuration — a missing
        root simply scans empty — so it is always built, and the deletions it
        proposes are the ones an operator should read most carefully.
    """
    from akgentic.tool.vector_store import weaviate_api_key, weaviate_url

    reapers: list[TeamResourceReaper] = []
    url = weaviate_url()
    if url and (not only or ResourceKind.WEAVIATE in only):
        try:
            reapers.append(WeaviateReaper(url, weaviate_api_key()))
        except Exception as exc:  # noqa: BLE001 - a missing extra is not a crash
            typer.echo(f"Weaviate reaper unavailable: {exc}", err=True)
    if not only or ResourceKind.DOCKER in only:
        reapers.append(DockerReaper())
    if not only or ResourceKind.WORKSPACE in only:
        reapers.append(WorkspaceReaper())
    return reapers


def _render(report: SweepReport) -> str:
    """Render a sweep report as operator-readable text.

    Args:
        report: The completed sweep.

    Returns:
        A multi-line summary naming every orphan.
    """
    if report.refusal_reason is not None:
        headline = f"REFUSED — nothing deleted: {report.refusal_reason}"
    elif report.applied:
        headline = "APPLIED"
    else:
        headline = "DRY RUN — nothing deleted"
    claims = f" (+{report.extra_claims} claimed workspace name(s))" if report.extra_claims else ""
    lines = [f"Sweep {headline}", f"Live teams: {report.live_team_ids}{claims}."]
    if report.unreadable_teams:
        lines.append(
            f"WARNING: {report.unreadable_teams} live team(s) would not yield their "
            "workspace claims; the protected set is incomplete."
        )
    for entry in report.reports:
        if not entry.available:
            lines.append(f"  {entry.kind}: UNAVAILABLE — {entry.unavailable_reason}")
            continue
        lines.append(
            f"  {entry.kind}: scanned {entry.scanned}, orphaned {len(entry.orphans)}, "
            f"held back (too young) {entry.skipped_young}, purged {entry.purged}"
        )
        lines.extend(
            f"    - {ref.label} (team {ref.team_id}, {ref.size_hint} {_unit(entry.kind)})"
            for ref in entry.orphans
        )
        lines.extend(f"    ! {failure}" for failure in entry.failures)
    return "\n".join(lines)


def _unit(kind: ResourceKind) -> str:
    """Return the noun for what a reaper counts, for the plan's orphan lines."""
    if kind is ResourceKind.DOCKER:
        return "container"
    if kind is ResourceKind.WORKSPACE:
        return "files"
    return "objects"


def _exit_code(report: SweepReport) -> int:
    """Map a sweep outcome onto a process exit code.

    A refusal gets its own code so a cron job can alert on "this needs a human"
    separately from "a backend was down", which usually clears itself.

    Args:
        report: The completed sweep.

    Returns:
        ``3`` when the guard refused, ``1`` when a backend was unreachable or a
        purge failed, ``0`` otherwise.
    """
    if report.refused:
        return 3
    unavailable = any(not entry.available for entry in report.reports)
    return 1 if unavailable or report.total_failures else 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(app())
