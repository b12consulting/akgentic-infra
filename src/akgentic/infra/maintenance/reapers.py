"""Backend reapers for the orphaned team-resource sweep.

Each reaper knows one backend and answers two questions about it: *what is
here, and who owns it* (``scan``), and *remove this* (``purge``). Neither
knows what a live team is — the orphan decision belongs to the driver in
:mod:`akgentic.infra.maintenance.sweep`, which is the only place the ordering
invariant that makes the sweep safe can be enforced once.

Two families of backend leak team-scoped resources today, because
``TeamManager``'s delete path purges the event store and nothing else:

* **The vector stores** — every row a team writes to a cluster backend carries
  its ``team_id``, and deleting the team removes none of them. There is one
  reaper per *configured* backend, resolved through ``akgentic-tool``'s
  registry, so a deployment running Qdrant is swept rather than reported clean.
* **Workspace filesystem** — ``<AKGENTIC_WORKSPACES_ROOT>/<workspace_id or
  team_id>``, plus a sibling ``<name>.git`` journal. This one holds the team's
  *data*: it is the only reaper here whose deletions are unrecoverable, and
  ``WorkspaceReaper`` documents the extra rules that follow from that.

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
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from akgentic.infra.maintenance.models import ResourceKind, ResourceRef
from akgentic.infra.maintenance.vector_backends import (
    administrative_backend,
    team_index_factory,
)

if TYPE_CHECKING:
    from akgentic.infra.maintenance.vector_backends import TeamIndex, VectorAdminBackend

logger = logging.getLogger(__name__)

GIT_DIR_SUFFIX: str = ".git"
"""Suffix of a workspace's sibling git journal: ``foo`` journals to ``foo.git``."""


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
    """Reaps workspace directories under ``AKGENTIC_WORKSPACES_ROOT``.

    **This is the one reaper that destroys data rather than runtime.** A vector
    row can be re-ingested from the source it was derived from; the files an
    agent wrote cannot be recovered from anywhere. Every rule below exists
    because of that asymmetry, and none of them may be relaxed for convenience.

    A workspace directory is ``<root>/<workspace_id or team_id>``, with the git
    journal in a sibling ``<name>.git`` (``GIT_DIR_SUFFIX``). The two are
    reaped together — a journal outliving its tree is a leak whose name no
    longer resolves to anything.

    **Only a name that parses as a UUID is ever a candidate.** ``workspace_id``
    is an operator-chosen override and ``WorkspaceTool(workspace_id="shared")``
    is a supported configuration: a named shared tree belongs to whoever
    configured it, is not addressed by any team id, and this reaper must never
    touch it. A UUID-named directory that *is* a shared workspace is protected
    the other way — the driver adds every ``workspace_id`` a live team declares
    to the protected set before anything is condemned.

    Symlinks are skipped rather than followed: the root is a directory of
    workspaces, and a link in it points at something whose ownership this
    reaper cannot reason about.

    Args:
        root: Workspace root. Defaults to the ``AKGENTIC_WORKSPACES_ROOT``
            environment variable, or ``./workspaces`` — the same resolution
            ``akgentic.tool.workspace.get_workspace`` performs.
    """

    kind: ResourceKind = ResourceKind.WORKSPACE
    backend: str | None = None
    """There is one workspace reaper per sweep, so its kind needs no qualifier."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root if root is not None else default_workspace_root()

    def scan(self) -> list[ResourceRef]:
        """Return one reference per UUID-named workspace directory.

        A missing root is not an error — a deployment whose agents never wrote
        a file has nothing here — and yields an empty scan.

        Returns:
            References whose ``detail`` is the absolute directory path,
            ``size_hint`` the file count, and ``age_seconds`` the directory's
            age by mtime. A busy workspace therefore looks young and is held
            back by the grace period, which is the safe bias.

        Raises:
            OSError: If the root exists but cannot be listed.
        """
        if not self._root.is_dir():
            logger.info("Workspace root %s does not exist; nothing to scan", self._root)
            return []
        refs: list[ResourceRef] = []
        for entry in sorted(self._root.iterdir()):
            ref = self._candidate(entry)
            if ref is not None:
                refs.append(ref)
        return refs

    def purge(self, ref: ResourceRef) -> int:
        """Delete one workspace tree and its sibling git journal.

        Returns:
            Directories removed — ``1`` for the tree, ``2`` when it had a
            journal.

        Raises:
            OSError: If either removal fails.
        """
        path = Path(ref.detail)
        _assert_inside(path, self._root)
        shutil.rmtree(path)
        removed = 1
        journal = path.parent / f"{path.name}{GIT_DIR_SUFFIX}"
        if journal.is_dir() and not journal.is_symlink():
            shutil.rmtree(journal)
            removed += 1
        return removed

    def close(self) -> None:
        """No handle to release — the filesystem is read per call."""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _candidate(self, entry: Path) -> ResourceRef | None:
        """Turn one root entry into a reference, or ``None`` to skip it.

        Skips anything that is not a plain directory, every symlink, the
        ``<name>.git`` journals (reaped with their tree, never on their own —
        their name does not parse as a UUID in any case), and every name that
        is not a UUID.
        """
        if entry.is_symlink() or not entry.is_dir():
            return None
        try:
            uuid.UUID(entry.name)
        except ValueError:
            logger.debug("Skipping workspace '%s': not a team-id-shaped name", entry.name)
            return None
        return ResourceRef(
            kind=self.kind,
            team_id=entry.name,
            detail=str(entry),
            label=entry.name,
            size_hint=_count_files(entry),
            age_seconds=_directory_age(entry),
        )


def default_workspace_root() -> Path:
    """Return the configured workspace root.

    Resolved exactly as ``akgentic.tool.workspace.get_workspace`` resolves it,
    so the sweep looks where the writers write.
    """
    return Path(os.environ.get("AKGENTIC_WORKSPACES_ROOT", "./workspaces"))


def _assert_inside(path: Path, root: Path) -> None:
    """Refuse to delete anything that is not a direct child of *root*.

    A belt-and-braces check on the one operation in this package that removes
    a directory tree: the reference came from this reaper's own scan, so this
    can only fire if something between the two rewrote it.

    Raises:
        OSError: If *path* is not a direct child of *root*.
    """
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved.parent != resolved_root:
        msg = f"refusing to remove {resolved}: not a direct child of {resolved_root}"
        raise OSError(msg)


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
