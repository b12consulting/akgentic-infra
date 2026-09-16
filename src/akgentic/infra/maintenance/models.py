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
    """Which backend a reapable resource lives in.

    ``VECTOR`` is one kind for the whole vector-store family rather than one per
    backend: a deployment running both Weaviate and Qdrant produces two
    ``ReaperReport`` entries of this kind, told apart by their labels. The
    distinction an operator acts on is *which resources are condemned*, and that
    is per collection and team, not per vendor — and ``--only vector`` then means
    "every vector store", which is the selector a schedule actually wants.
    """

    VECTOR = "vector"
    WORKSPACE = "workspace"


class ResourceRef(BaseModel):
    """One reapable resource, together with the team that owns it.

    A reference is produced by a reaper's ``scan`` and handed back to the same
    reaper's ``purge``. It carries everything the purge needs, so the driver
    never has to re-query a backend between deciding and acting — the set it
    decided on is exactly the set it deletes.

    Attributes:
        kind: The backend this resource lives in.
        team_id: The team the resource belongs to, spelled exactly as the
            resource itself spells it — a Weaviate ``team_id`` property value, a
            Qdrant ``team_id`` payload value, the ``<leaf>`` of a workspace
            tree under the ``_team`` kind. It is a team id in every case: a
            named or metadata workspace tree is addressed by no team id and is
            therefore never scanned at all. Kept as ``str`` rather than
            ``uuid.UUID`` because a malformed value must survive the scan and be
            reported, not crash it.
        detail: Backend handle the purge acts on — a vector collection name, an
            absolute workspace directory path.
        label: Human-readable identity for the report
            (``backend:collection/team``, or a workspace's relative
            ``<scope>/<kind>/<leaf>`` path). Never used to address the resource:
            two backends share one ``ResourceKind``, so the label is the only
            thing in the report that says which cluster an orphan is in.
        claim_key: What the driver diffs against the protected set, when that is
            not ``team_id``. The vector reapers protect on a team id, because a
            row is addressed by one and nothing else; the workspace reaper
            protects on the whole three-segment path, because a ``<leaf>`` is
            unique only within a scope and a kind — and because a live team can
            own ``_shared/_team/<id>`` while an identically-leafed
            ``<owner>/_team/<id>`` beside it is a stale tree nothing will write
            to again. Defaults to ``None``, meaning "diff on ``team_id``", so a
            reaper that has nothing sharper to say says nothing.
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
    claim_key: str | None = None
    size_hint: int = 0
    age_seconds: float | None = None


class ReaperReport(BaseModel):
    """What one reaper found, and what it did about it.

    Attributes:
        kind: The reaper that produced this report.
        backend: Which backend of that kind, when a kind has more than one —
            ``"weaviate"`` or ``"qdrant"`` for a vector reaper, ``None`` for a
            reaper that is the only one of its kind. Two configured vector
            stores produce two reports of the same ``kind``, and an
            ``available=False`` report carries no orphans to name the cluster
            in, so without this the one line an operator must act on cannot say
            which cluster is down.
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
        purged: Objects or directories actually removed. Zero on a dry run.
        failures: One line per resource whose purge raised, naming the
            resource. A failure never aborts the sweep — the remaining
            orphans are still reaped.
    """

    kind: ResourceKind
    backend: str | None = None
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
        extra_claims: Claims protected on top of the bare live team ids — the
            resolved ``<scope>/<kind>/<leaf>`` paths of every live team's
            deletion candidates. A live team routinely owns a tree its id alone
            does not name, and a tree whose leaf *is* its id may still be one
            the team no longer writes to. The field name predates the change
            from names to paths and is kept because it is on the ``--json``
            surface; the meaning — claims beyond the bare live team ids — is
            unchanged.
        unreadable_teams: Live teams whose deletion candidates the store could
            not resolve. Any value above zero means the workspace claims are
            incomplete, so the guard refuses to apply: under-protection here
            deletes data.
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
        """Objects and directories actually removed across every reaper."""
        return sum(report.purged for report in self.reports)

    @property
    def total_failures(self) -> int:
        """Purges that raised across every reaper."""
        return sum(len(report.failures) for report in self.reports)
