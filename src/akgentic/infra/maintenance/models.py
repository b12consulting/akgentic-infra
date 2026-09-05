"""Typed results for the orphaned team-resource sweep.

Every value that crosses a boundary in this package is a model here — the
reapers return ``ResourceRef`` lists, the driver returns a ``SweepReport``,
and the CLI renders that report rather than re-deriving anything from the
backends. A sweep is destructive, so its plan has to be inspectable before
it is applied and quotable after it is.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ResourceKind(StrEnum):
    """Which backend a reapable resource lives in."""

    WEAVIATE = "weaviate"
    DOCKER = "docker"
    WORKSPACE = "workspace"


class ResourceRef(BaseModel):
    """One reapable resource, together with the team that owns it.

    A reference is produced by a reaper's ``scan`` and handed back to the same
    reaper's ``purge``. It carries everything the purge needs, so the driver
    never has to re-query a backend between deciding and acting — the set it
    decided on is exactly the set it deletes.

    Attributes:
        kind: The backend this resource lives in.
        team_id: The key the resource is filed under, spelled exactly as the
            resource itself spells it — a Weaviate ``team_id`` property value,
            the suffix of a ``sandbox-<team_id>`` container name, a workspace
            directory name. It is a team id in every case **except** a
            workspace directory named after a shared ``workspace_id``, which
            is a supported configuration and is why the sweep compares against
            live *claims* rather than live team ids alone. Kept as ``str``
            rather than ``uuid.UUID`` because a malformed value must survive
            the scan and be reported, not crash it.
        detail: Backend handle the purge acts on — a Weaviate collection name,
            a Docker container id.
        label: Human-readable identity for the report (container name, or
            ``collection/team``). Never used to address the resource.
        size_hint: Objects behind the reference when the backend reports one
            cheaply, else ``0``. Advisory only — it sizes the report, it does
            not gate the delete.
        age_seconds: Age of the resource at scan time when the backend exposes
            a creation timestamp, else ``None``. Drives the grace period.
    """

    kind: ResourceKind
    team_id: str
    detail: str
    label: str
    size_hint: int = 0
    age_seconds: float | None = None


class ReaperReport(BaseModel):
    """What one reaper found, and what it did about it.

    Attributes:
        kind: The reaper that produced this report.
        available: Whether the backend could be reached at all. A sweep over an
            unreachable backend reports ``False`` here and leaves ``scanned``
            at zero — it never reports "no orphans", which would read as a
            clean cluster.
        unavailable_reason: Why the backend could not be reached, when
            ``available`` is ``False``.
        scanned: Resources seen, orphaned or not.
        skipped_young: Resources inside the grace period, held back from this
            sweep regardless of the live set.
        orphans: Resources whose owning team is not live.
        purged: Objects or containers actually removed. Zero on a dry run.
        failures: One line per resource whose purge raised, naming the
            resource. A failure never aborts the sweep — the remaining
            orphans are still reaped.
    """

    kind: ResourceKind
    available: bool = True
    unavailable_reason: str | None = None
    scanned: int = 0
    skipped_young: int = 0
    orphans: list[ResourceRef] = Field(default_factory=list)
    purged: int = 0
    failures: list[str] = Field(default_factory=list)


class SweepReport(BaseModel):
    """The outcome of one full sweep across every configured reaper.

    Attributes:
        applied: Whether anything was actually deleted. ``False`` for a dry
            run — the default — and also for an ``apply`` the blast-radius
            guard refused, so a caller never has to read two fields to learn
            whether the cluster changed.
        refusal_reason: Why the guard stopped an ``apply``, or ``None``. Set
            only when ``apply`` was asked for and declined; a dry run leaves it
            ``None`` because a dry run was never going to delete anything.
        live_team_ids: Size of the live set the orphan decision was made
            against, for sanity: a sweep that finds zero live teams against a
            populated cluster is a misconfigured store, not an empty product.
        extra_claims: Names protected on top of the live team ids — the
            ``workspace_id`` values live teams declare, which are directory
            names owned by a team whose id they are not.
        unreadable_teams: Agent cards referenced by a live team that the store
            could not resolve. Any value above zero means the workspace claims
            are incomplete, so the guard refuses to apply: under-protection
            here deletes data.
        reports: One entry per reaper, in the order they ran.
    """

    applied: bool
    refusal_reason: str | None = None
    live_team_ids: int
    extra_claims: int = 0
    unreadable_teams: int = 0
    reports: list[ReaperReport] = Field(default_factory=list)

    @property
    def refused(self) -> bool:
        """Whether the blast-radius guard declined to apply this plan."""
        return self.refusal_reason is not None

    @property
    def total_orphans(self) -> int:
        """Orphaned resources found across every reaper."""
        return sum(len(report.orphans) for report in self.reports)

    @property
    def total_purged(self) -> int:
        """Objects and containers actually removed across every reaper."""
        return sum(report.purged for report in self.reports)

    @property
    def total_failures(self) -> int:
        """Purges that raised across every reaper."""
        return sum(len(report.failures) for report in self.reports)
