"""Backend reapers for the orphaned team-resource sweep.

Each reaper knows one backend and answers two questions about it: *what is
here, and who owns it* (``scan``), and *remove this* (``purge``). Neither
knows what a live team is — the orphan decision belongs to the driver in
:mod:`akgentic.infra.maintenance.sweep`, which is the only place the ordering
invariant that makes the sweep safe can be enforced once.

Two families of backend leak team-scoped resources today:

* **The vector stores** — every row a team writes to a cluster backend carries
  its ``team_id``, and deleting the team removes none of them. There is one
  reaper per *configured* backend, resolved through ``akgentic-tool``'s
  registry, so a deployment running Qdrant is swept rather than reported clean.
* **Workspace filesystem** — ``<AKGENTIC_WORKSPACES_ROOT>/<scope>/<kind>/<leaf>``,
  plus the tree's ``<leaf>.git`` journal and ``<leaf>.index`` metadata sidecars.
  This one holds the team's *data*: it is the only reaper here whose deletions
  are unrecoverable, and ``WorkspaceReaper`` documents the extra rules that
  follow from that.

There is deliberately **no sandbox-container reaper**. The sandbox names its
container after random hex rather than after a team, and removes it on every
stop, so a reaper here could attribute nothing and would report every
deployment clean — worse than no reaper at all.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol

from akgentic.infra.adapters.shared.team_tree_only_policy import TeamTreeOnlyPolicy
from akgentic.infra.maintenance.models import ResourceKind, ResourceRef
from akgentic.infra.maintenance.vector_backends import (
    administrative_backend,
    team_index_factory,
)
from akgentic.infra.protocols.workspace_deletion import (
    WorkspaceDeletionContext,
    WorkspaceDeletionPolicy,
)
from akgentic.infra.server.services._workspace_paths import WORKSPACE_PATH_SEGMENTS
from akgentic.tool.workspace import TEAM_KIND, git_dir_for, meta_dir_for

if TYPE_CHECKING:
    from collections.abc import Iterator

    from akgentic.infra.maintenance.vector_backends import TeamIndex, VectorAdminBackend

logger = logging.getLogger(__name__)


class TeamResourceReaper(Protocol):
    """A backend that holds resources keyed by the team that produced them.

    Implementations are constructed by the CLI, used for exactly one sweep,
    and closed. They are deliberately ignorant of the live team set: a reaper
    that could decide what is dead would also be able to decide it at the
    wrong moment.
    """

    kind: ResourceKind
    backend: str | None
    """Which backend of that kind, or ``None`` when the kind has only one.

    A kind with two configured backends reports twice, and the driver has no
    other way to tell the two reports apart once a scan has failed and carries
    no orphans.
    """

    def scan(self) -> list[ResourceRef]:
        """Return every team-owned resource currently in the backend.

        Raises:
            OSError: If the backend cannot be reached. The driver records the
                failure and continues with the other reapers rather than
                aborting the sweep.
        """
        ...

    def purge(self, ref: ResourceRef) -> int:
        """Remove one resource previously returned by :meth:`scan`.

        Returns:
            Number of rows or directories removed.
        """
        ...

    def close(self) -> None:
        """Release the backend connection. Always called, even on failure."""
        ...


# ---------------------------------------------------------------------------
# Vector stores
# ---------------------------------------------------------------------------


class VectorStoreReaper:
    """Reaps vector-store rows whose ``team_id`` names a team that is gone.

    One instance serves **one registered backend**, named at construction, and
    a sweep builds one per configured backend — so a deployment running both
    Weaviate and Qdrant gets two reports of the same
    :attr:`~.models.ResourceKind.VECTOR` kind, told apart by their labels.

    One reference is produced per ``(collection, team_id)`` pair, because that
    is the granularity both backends delete at.

    **The write side is the backend's, the read side is this reaper's.**
    ``list_collections`` and ``delete_by_team`` are called on the backend the
    registry builds; enumerating the distinct team ids present has no backend
    primitive, so it goes through a
    :class:`~.vector_backends.TeamIndex` that owns its own connection. See
    that module for why the split is where it is.

    Args:
        backend_name: A registered vector-store backend name.

    Raises:
        ValueError: If the backend is not registered, is not provisioned, or is
            one this sweep does not reap.
    """

    kind: ResourceKind = ResourceKind.VECTOR

    def __init__(self, backend_name: str) -> None:
        factory = team_index_factory(backend_name)
        if factory is None:
            msg = f"Vector-store backend '{backend_name}' has no team-id enumerator."
            raise ValueError(msg)
        # str | None, not str: the protocol attribute is mutable and therefore
        # invariant, so a narrower type here would not satisfy it.
        self.backend: str | None = backend_name
        self._admin: VectorAdminBackend = administrative_backend(backend_name)
        self._index: TeamIndex = factory()

    def scan(self) -> list[ResourceRef]:
        """Return one reference per ``(collection, team_id)`` in the cluster.

        **A collection that is not team-scoped is never condemned.** On a
        shared collection ``team_id`` records who *wrote* a row, and both
        backends' ``delete_by_team`` refuses one before any cluster call — so
        scanning it could only ever produce orphans whose purge fails, against
        rows a live team is still reading.

        Returns:
            References carrying the row count as ``size_hint``. Rows expose no
            cheap creation time at this granularity, so ``age_seconds`` is
            always ``None`` and the grace period does not apply — see the
            module note in :mod:`~.sweep` on why the scan ordering already
            covers the race a grace period would.

        Raises:
            Exception: Whatever the cluster raises. The driver turns it into an
                ``available=False`` report rather than an empty, clean-looking
                one.
        """
        from akgentic.tool.vector_store.protocol import collection_is_team_scoped

        refs: list[ResourceRef] = []
        for collection in self._admin.list_collections():
            if not collection_is_team_scoped(collection):
                logger.info(
                    "Skipping '%s' on %s: shared across teams, so no row in it is "
                    "any one team's to reap",
                    collection,
                    self.backend,
                )
                continue
            for team_id, count in self._index.team_counts(collection).items():
                refs.append(
                    ResourceRef(
                        kind=self.kind,
                        team_id=team_id,
                        detail=collection,
                        label=f"{self.backend}:{collection}/{team_id}",
                        size_hint=count,
                    )
                )
        return refs

    def purge(self, ref: ResourceRef) -> int:
        """Delete every row in ``ref.detail`` stamped with ``ref.team_id``.

        Delegates to the backend's own ``delete_by_team``, which owns whatever
        paging its cluster needs. **The two backends disagree on the return
        value** — Weaviate counts what it deleted, Qdrant returns ``None`` — so
        a backend that reports no count falls back to the count the scan saw.

        Returns:
            Rows deleted, or ``ref.size_hint`` when the backend does not say.

        Raises:
            ValueError: If the collection is shared across teams or has gone,
                both of which the backend refuses. The driver records it as a
                failure and reaps the remaining orphans.
        """
        deleted = self._admin.delete_by_team(ref.detail, ref.team_id)
        return ref.size_hint if deleted is None else int(deleted)

    def close(self) -> None:
        """Release this reaper's read connection.

        The backend is not closed: its client comes from a process-wide cache
        this sweep does not own.
        """
        self._index.close()


# ---------------------------------------------------------------------------
# Workspace filesystem
# ---------------------------------------------------------------------------


class WorkspaceReaper:
    """Reaps team workspace trees under ``AKGENTIC_WORKSPACES_ROOT``.

    **This is the one reaper that destroys data rather than runtime.** A vector
    row can be re-ingested from the source it was derived from; the files an
    agent wrote cannot be recovered from anywhere. Every rule below exists
    because of that asymmetry, and none of them may be relaxed for convenience.

    **It is the safety net behind ``TeamService.delete_team``, not the
    mechanism.** That path removes a team's trees as the team goes, resolving
    its candidates through ``deletion_candidate_paths`` and filtering them
    through the tier's ``WorkspaceDeletionPolicy``. What is left for a sweep is
    what that path could not take: a crash between the event-store write and the
    ``rmtree``, a tier whose policy refused at the time, and every tree written
    before that path existed.

    A workspace path is three segments, ``<root>/<scope>/<kind>/<leaf>``, where
    ``<scope>`` is the owner's user id or the reserved ``_shared`` and ``<kind>``
    is one of ``_team`` / ``_id`` / ``_meta``. The tree's ``<leaf>.git`` journal
    and ``<leaf>.index`` metadata directory are reaped with it — a journal or an
    index outliving its tree is a leak whose name no longer resolves to
    anything, and ``<leaf>.index/rag/*.yaml`` holds the extracted text of every
    document the tree indexed.

    **The ``_team`` kind segment is what makes a leaf a team id.** Only
    ``<scope>/_team/<uuid>`` is ever a candidate, in **either** scope: a
    ``_shared/_team/<team_id>`` tree is still one team's own. An ``_id`` or
    ``_meta`` tree is addressed by no team id, so no team's deletion can orphan
    it and it never enters a plan at all. That is structurally sharper than the
    rule it replaces — a UUID-shaped *name* could be an operator's
    ``workspace_id``, which had to be handed to the driver's claim set to be
    safe.

    **Nothing is condemned that the deletion policy refuses.** The same
    ``WorkspaceDeletionPolicy`` the delete path consults is asked about every
    candidate before it enters the plan, so the sweep cannot drift from the
    delete path's judgement in the one direction that is unrecoverable.

    Symlinks are skipped rather than followed, at every level: a link in place
    of a scope, a kind or a leaf points at something whose ownership this reaper
    cannot reason about.

    Args:
        root: Workspace root. Defaults to the ``AKGENTIC_WORKSPACES_ROOT``
            environment variable, or ``./workspaces`` — the same resolution
            ``akgentic.tool.workspace.get_workspace`` performs.
        policy: Rule deciding which trees may go. Defaults to
            :class:`~akgentic.infra.adapters.shared.team_tree_only_policy.TeamTreeOnlyPolicy`,
            exactly as ``TierServices.workspace_deletion_policy``'s
            ``default_factory`` does.
    """

    kind: ResourceKind = ResourceKind.WORKSPACE
    backend: str | None = None
    """There is one workspace reaper per sweep, so its kind needs no qualifier."""

    def __init__(
        self, root: Path | None = None, policy: WorkspaceDeletionPolicy | None = None
    ) -> None:
        self._root = root if root is not None else default_workspace_root()
        self._policy: WorkspaceDeletionPolicy = (
            policy if policy is not None else TeamTreeOnlyPolicy()
        )

    def scan(self) -> list[ResourceRef]:
        """Return one reference per ``<scope>/_team/<uuid>`` tree the policy allows.

        A missing root is not an error — a deployment whose agents never wrote
        a file has nothing here — and yields an empty scan.

        Sorted at every level, so a plan an operator reads twice is the same
        plan twice.

        Returns:
            References whose ``detail`` is the absolute leaf path, ``label`` and
            ``claim_key`` the relative ``<scope>/<kind>/<leaf>``, ``size_hint``
            the file count, and ``age_seconds`` the directory's age by mtime. A
            busy workspace therefore looks young and is held back by the grace
            period, which is the safe bias.

        Raises:
            OSError: If the root exists but cannot be listed.
        """
        if not self._root.is_dir():
            logger.info("Workspace root %s does not exist; nothing to scan", self._root)
            return []
        refs: list[ResourceRef] = []
        for kind_dir in self._team_kind_dirs():
            for leaf in _child_directories(kind_dir):
                ref = self._candidate(leaf)
                if ref is not None:
                    refs.append(ref)
        return refs

    def purge(self, ref: ResourceRef) -> int:
        """Delete one workspace tree and both of its sidecars.

        The three targets are attempted **independently**, so a failure on one
        cannot skip the next: a ``<leaf>.git`` that will not go must not take
        ``<leaf>.index`` with it, which is the retention half.

        **The two sidecars anchor differently, deliberately.** The tree and its
        ``.git`` are bounded against this reaper's root by
        :func:`_assert_inside`. The ``.index`` is anchored to whatever
        ``meta_dir_for`` resolved, because an operator may legitimately relocate
        the metadata root with ``AKGENTIC_WORKSPACE_META_ROOT`` — and refusing
        to delete it there would re-open the retention leak in precisely the
        deployment that configured a separate root. ``TeamService``'s
        ``_remove_workspace_trees`` does the same; the two must stay the same.

        Returns:
            Directories actually removed — between ``1`` and ``3``, since
            neither sidecar exists until something creates it.

        Raises:
            OSError: If the reference is not a workspace tree of this root, or
                if any of the three removals failed. The message names every
                failure, and the driver records it against this orphan.
        """
        target = Path(ref.detail)
        relative = _assert_inside(target, self._root)
        removed = 0
        failures: list[str] = []
        for victim in (target, git_dir_for(target), meta_dir_for(str(relative))):
            try:
                removed += _remove_tree(victim)
            except OSError as exc:
                failures.append(f"{victim}: {exc}")
        if failures:
            raise OSError("; ".join(failures))
        return removed

    def close(self) -> None:
        """No handle to release — the filesystem is read per call."""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _team_kind_dirs(self) -> Iterator[Path]:
        """Yield every ``<root>/<scope>/_team`` directory, scopes in sorted order.

        Both scopes are walked: ``_shared`` holds one team's own tree under
        ``_team`` exactly as a principal scope does. Any other kind — the
        reserved ``_id`` and ``_meta``, or a name this layout does not know — is
        logged at DEBUG and skipped: it belongs to no team and is nobody's
        orphan.
        """
        for scope in _child_directories(self._root):
            for kind_dir in _child_directories(scope):
                if kind_dir.name != TEAM_KIND:
                    logger.debug(
                        "Skipping %s/%s: no team id addresses a '%s' tree",
                        scope.name,
                        kind_dir.name,
                        kind_dir.name,
                    )
                    continue
                yield kind_dir

    def _candidate(self, leaf: Path) -> ResourceRef | None:
        """Turn one ``<scope>/_team/<leaf>`` directory into a reference, or skip it.

        Skips a leaf whose name does not parse as a UUID — which is what the
        ``<leaf>.git`` and ``<leaf>.index`` sidecars are, so they are never
        candidates in their own right — and then anything the deletion policy
        refuses.

        **The refusal is recorded at INFO**, mirroring the delete path's
        "Workspace kept": a deletion nobody can explain afterwards is worse than
        one that did not happen, and so is a tree that quietly survived every
        sweep.

        **``owner_user_id`` is the ``<scope>`` segment, and that is the best
        answer available here.** The delete path reads ``Process.user_id``; this
        reaper has no process, because the team being gone is the whole signal.
        For a principal scope the segment *is* the owner's user id verbatim —
        ``user_segment`` is identity, with no encoding and no digest — and for a
        ``_shared`` tree the scope names no principal at all. A tier policy
        keying on the owner therefore fails to match a shared tree and refuses
        it, which is the safe direction for an unrecoverable delete. The default
        ``TeamTreeOnlyPolicy`` reads only ``kind`` and ``leaf`` and is
        unaffected.
        """
        try:
            team_id = uuid.UUID(leaf.name)
        except ValueError:
            logger.debug("Skipping workspace leaf '%s': not a team-id-shaped name", leaf.name)
            return None
        relative = PurePosixPath(*leaf.relative_to(self._root).parts)
        scope, kind, leaf_name = relative.parts
        ctx = WorkspaceDeletionContext(
            team_id=team_id,
            owner_user_id=scope,
            path=relative,
            scope=scope,
            kind=kind,
            leaf=leaf_name,
        )
        if not self._policy.may_delete(ctx=ctx):
            logger.info(
                "Workspace kept — candidate=%s policy=%s",
                relative,
                type(self._policy).__name__,
            )
            return None
        return ResourceRef(
            kind=self.kind,
            team_id=leaf_name,
            detail=str(leaf),
            label=str(relative),
            claim_key=relative.as_posix(),
            size_hint=_count_files(leaf),
            age_seconds=_directory_age(leaf),
        )


def default_workspace_root() -> Path:
    """Return the configured workspace root.

    Resolved exactly as ``akgentic.tool.workspace.get_workspace`` resolves it,
    so the sweep looks where the writers write.
    """
    return Path(os.environ.get("AKGENTIC_WORKSPACES_ROOT", "./workspaces"))


def _child_directories(parent: Path) -> list[Path]:
    """Every real subdirectory of *parent*, sorted by name.

    Files and symlinks are dropped, and the symlink test comes **first**:
    ``is_dir`` follows a link, so a link to a directory would otherwise pass as
    one. Sorting at every level is what makes a plan stable run to run.

    Raises:
        OSError: If *parent* cannot be listed. Unavailable is never clean — the
            driver turns it into an ``available=False`` report rather than an
            empty, clean-looking scan.
    """
    return [
        entry for entry in sorted(parent.iterdir()) if not entry.is_symlink() and entry.is_dir()
    ]


def _remove_tree(target: Path) -> int:
    """Remove *target* recursively when it is a real directory.

    A missing sidecar is a silent no-op: neither the journal nor the metadata
    directory exists until something creates it. A symlink is refused rather
    than followed, for the reason the scan skips them.

    Returns:
        ``1`` when a directory was removed, ``0`` when there was nothing there.

    Raises:
        OSError: If the removal itself failed.
    """
    if target.is_symlink() or not target.is_dir():
        return 0
    shutil.rmtree(target)
    return 1


def _assert_inside(path: Path, root: Path) -> PurePosixPath:
    """Return *path*'s workspace path below *root*, or refuse to touch it.

    The last check before the one operation in this package that removes a
    directory tree, and ADR-042 §7 rule 3 requires it even though the reference
    came from this reaper's own scan: it can only fire if something between the
    two rewrote it, which is exactly when it matters.

    **Depth is checked as well as containment, and that is the dangerous half.**
    A path *two* segments below the root is a ``<scope>/<kind>`` directory
    holding every team's tree of that kind, and an ``rmtree`` of one would take
    all of them. Only a path exactly ``WORKSPACE_PATH_SEGMENTS`` deep is a tree.

    **Depth is measured twice, on the literal reference and on what it resolves
    to, because the two can disagree.** ``a/../b`` is three parts and lands one
    segment deep, naming the *parent* of every tree beneath it; measuring only
    the literal would approve it. This is the same hazard ``_contained_target``
    in ``server/services/team_service.py`` documents for the delete path, and
    the reaper keeps its own check rather than borrowing that one because the
    two guard different inputs.

    Returns:
        The relative ``<scope>/<kind>/<leaf>`` path, which ``purge`` needs to
        locate the metadata sidecar through the tool's own resolver.

    Raises:
        OSError: If *path* is not exactly one workspace path below *root*,
            literally or once resolved.
    """
    literal = _segments_below(path.absolute(), root.absolute())
    resolved = _segments_below(path.resolve(), root.resolve())
    if literal is None or resolved is None:
        msg = (
            f"refusing to remove {path}: not a workspace tree "
            f"{WORKSPACE_PATH_SEGMENTS} segments below {root}"
        )
        raise OSError(msg)
    return resolved


def _segments_below(path: Path, root: Path) -> PurePosixPath | None:
    """*path* relative to *root*, iff it is exactly one workspace path deep.

    ``None`` for anything outside *root* and for any other depth. Neither
    argument is resolved here: the caller decides which of the two measurements
    it is taking.
    """
    if not path.is_relative_to(root):
        return None
    relative = path.relative_to(root)
    if len(relative.parts) != WORKSPACE_PATH_SEGMENTS:
        return None
    return PurePosixPath(*relative.parts)


def _count_files(directory: Path) -> int:
    """Return the number of files under *directory*, or ``0`` if unreadable.

    Advisory only — it sizes the plan for the operator reading it. A tree that
    cannot be walked still gets reaped on its name, so a count failure must
    not propagate.
    """
    try:
        return sum(1 for path in directory.rglob("*") if path.is_file())
    except OSError as exc:
        logger.debug("Could not size %s: %s", directory, exc)
        return 0


def _directory_age(directory: Path) -> float | None:
    """Return the age in seconds of *directory* by mtime, or ``None``.

    mtime rather than any creation time, because it is the only timestamp
    every platform agrees on — and it moves on every write, so an actively
    used workspace reads as young and the grace period protects it.
    """
    try:
        modified = directory.stat().st_mtime
    except OSError as exc:
        logger.debug("Could not stat %s: %s", directory, exc)
        return None
    return datetime.now(UTC).timestamp() - modified
