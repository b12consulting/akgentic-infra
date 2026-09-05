"""Backend reapers for the orphaned team-resource sweep.

Each reaper knows one backend and answers two questions about it: *what is
here, and who owns it* (``scan``), and *remove this* (``purge``). Neither
knows what a live team is — the orphan decision belongs to the driver in
:mod:`akgentic.infra.maintenance.sweep`, which is the only place the ordering
invariant that makes the sweep safe can be enforced once.

Two backends leak team-scoped resources today because ``TeamManager``'s delete
path purges the event store and nothing else:

* **Weaviate** — every vector object carries a ``team_id`` property; deleting a
  team removes no object.
* **Docker** — ``DockerSandboxActor`` names its container ``sandbox-<team_id>``
  and stops it on teardown, deliberately never running ``docker rm``, so an
  exited container survives its team forever.
* **Workspace filesystem** — ``<AKGENTIC_WORKSPACES_ROOT>/<workspace_id or
  team_id>``, plus a sibling ``<name>.git`` journal. This one holds the team's
  *data*: it is the only reaper here whose deletions are unrecoverable, and
  ``WorkspaceReaper`` documents the extra rules that follow from that.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from akgentic.infra.maintenance.models import ResourceKind, ResourceRef

if TYPE_CHECKING:
    import weaviate

logger = logging.getLogger(__name__)

SANDBOX_CONTAINER_PREFIX: str = "sandbox-"
"""Name prefix ``DockerSandboxActor`` gives every per-team container."""

GIT_DIR_SUFFIX: str = ".git"
"""Suffix of a workspace's sibling git journal: ``foo`` journals to ``foo.git``."""

_DOCKER_TIMEOUT_SECONDS: int = 60
_WEAVIATE_GROUP_LIMIT: int = 10_000
_WEAVIATE_DELETE_PASSES: int = 100


class TeamResourceReaper(Protocol):
    """A backend that holds resources keyed by the team that produced them.

    Implementations are constructed by the CLI, used for exactly one sweep,
    and closed. They are deliberately ignorant of the live team set: a reaper
    that could decide what is dead would also be able to decide it at the
    wrong moment.
    """

    kind: ResourceKind

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
            Number of objects or containers removed.
        """
        ...

    def close(self) -> None:
        """Release the backend connection. Always called, even on failure."""
        ...


# ---------------------------------------------------------------------------
# Weaviate
# ---------------------------------------------------------------------------


class WeaviateReaper:
    """Reaps vector objects whose ``team_id`` names a team that is gone.

    One reference is produced per ``(collection, team_id)`` pair, because that
    is the granularity Weaviate deletes at.

    **Why this owns a raw client rather than a** ``WeaviateBackend``. The
    backend already exposes the two write-side primitives a sweeper needs —
    ``list_collections`` and ``delete_by_team``, both documented as crossing
    the team boundary on purpose — but no way to enumerate the distinct
    ``team_id`` values present, which is the read side of the same job.
    Reaching into ``WeaviateBackend._client`` for it would break the
    package's own encapsulation, so this class connects for itself. The
    enumeration primitive belongs on ``WeaviateBackend`` next to its two
    siblings; that is an ``akgentic-tool`` change, and when it lands this
    class collapses into a delegation.

    Args:
        url: Weaviate cluster URL, e.g. ``http://localhost:8080``.
        api_key: Optional API key for an authenticated cluster.
    """

    kind: ResourceKind = ResourceKind.WEAVIATE

    def __init__(self, url: str, api_key: str | None = None) -> None:
        from urllib.parse import urlparse

        import weaviate as _wv
        from weaviate.auth import AuthApiKey

        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        use_https = parsed.scheme == "https"
        port = parsed.port or (443 if use_https else 8080)

        self._client: weaviate.WeaviateClient = _wv.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=use_https,
            grpc_host=host,
            grpc_port=50051,
            grpc_secure=use_https,
            auth_credentials=AuthApiKey(api_key) if api_key else None,
        )

    def scan(self) -> list[ResourceRef]:
        """Return one reference per ``(collection, team_id)`` in the cluster.

        Returns:
            References carrying the object count as ``size_hint``. Objects
            expose no cheap creation time at this granularity, so
            ``age_seconds`` is always ``None`` and the grace period does not
            apply — see the module note in :mod:`~.sweep` on why the scan
            ordering already covers the race a grace period would.
        """
        refs: list[ResourceRef] = []
        for collection in self._client.collections.list_all():
            for team_id, count in self._team_counts(collection).items():
                refs.append(
                    ResourceRef(
                        kind=self.kind,
                        team_id=team_id,
                        detail=collection,
                        label=f"{collection}/{team_id}",
                        size_hint=count,
                    )
                )
        return refs

    def purge(self, ref: ResourceRef) -> int:
        """Delete every object in ``ref.detail`` stamped with ``ref.team_id``.

        Weaviate caps a single ``delete_many`` at its configured maximum, so
        this loops until a pass deletes nothing.

        Returns:
            Total objects deleted across all passes.
        """
        from weaviate.classes.query import Filter

        collection = self._client.collections.get(ref.detail)
        where = Filter.by_property(_team_id_property()).equal(ref.team_id)
        deleted = 0
        for _ in range(_WEAVIATE_DELETE_PASSES):
            result = collection.data.delete_many(where=where)
            pass_count = int(getattr(result, "successful", 0) or 0)
            deleted += pass_count
            if pass_count == 0:
                return deleted
        logger.warning(
            "Weaviate purge of %s hit the %d-pass cap at %d objects; the next sweep will finish it",
            ref.label,
            _WEAVIATE_DELETE_PASSES,
            deleted,
        )
        return deleted

    def close(self) -> None:
        """Disconnect the Weaviate client."""
        self._client.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _team_counts(self, collection: str) -> dict[str, int]:
        """Count objects per ``team_id`` in one collection.

        Prefers a server-side ``group_by`` aggregation. Clusters that reject
        it — an older server, or a group cardinality above the configured
        maximum — fall back to a full object walk, which is slower but cannot
        silently under-report and so cannot make a live team look orphaned.

        Args:
            collection: Collection to aggregate.

        Returns:
            Object count keyed by ``team_id``. Objects with a missing or
            non-string ``team_id`` are excluded: an unattributable object is
            never anyone's orphan.
        """
        try:
            return self._aggregate_team_counts(collection)
        except Exception as exc:  # noqa: BLE001 - any client/server error degrades
            logger.info(
                "Weaviate group_by aggregation unavailable on '%s' (%s); "
                "falling back to an object walk",
                collection,
                exc,
            )
            return self._walk_team_counts(collection)

    def _aggregate_team_counts(self, collection: str) -> dict[str, int]:
        """Count per ``team_id`` via a server-side aggregation."""
        from weaviate.classes.aggregate import GroupByAggregate

        prop = _team_id_property()
        result = self._client.collections.get(collection).aggregate.over_all(
            group_by=GroupByAggregate(prop=prop, limit=_WEAVIATE_GROUP_LIMIT),
        )
        counts: dict[str, int] = {}
        for group in result.groups:
            value = group.grouped_by.value
            if isinstance(value, str) and value:
                # A group the server reports without a count is still a group:
                # size_hint falls back to 0, the orphan decision is unaffected.
                counts[value] = int(group.total_count or 0)
        return counts

    def _walk_team_counts(self, collection: str) -> dict[str, int]:
        """Count per ``team_id`` by iterating every object in the collection."""
        prop = _team_id_property()
        counts: dict[str, int] = {}
        for obj in self._client.collections.get(collection).iterator(
            return_properties=[prop],
        ):
            value = obj.properties.get(prop)
            if isinstance(value, str) and value:
                counts[value] = counts.get(value, 0) + 1
        return counts


def _team_id_property() -> str:
    """Return the Weaviate schema property carrying the owning team's id.

    Read from ``akgentic-tool`` rather than restated here, so a rename of the
    property on the write side cannot leave the sweeper filtering on a name
    nothing is stamped with — which would delete nothing and report success.
    """
    from akgentic.tool.vector_store.weaviate import TEAM_ID_PROPERTY

    return TEAM_ID_PROPERTY


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------


class DockerReaper:
    """Reaps ``sandbox-<team_id>`` containers whose team is gone.

    **Only non-running containers are scanned.** A running sandbox is a live
    process with a shell in it; removing one under an operator because a
    database row disappeared is worse than leaking it. ``DockerSandboxActor``
    stops its container on teardown, so a genuine orphan is exited and this
    sweep reaches it on the next pass regardless.

    The container's ``/workspace`` is a **bind** mount from
    ``AKGENTIC_WORKSPACES_ROOT``, so ``docker rm -v`` removes only anonymous
    volumes and never touches workspace content. Reaping the workspace
    directory itself is a separate decision — it is the team's data, not its
    runtime — and this reaper deliberately does not make it.

    Args:
        docker_binary: Executable to invoke. Overridable for tests and for
            hosts where the client is not on the default path.
    """

    kind: ResourceKind = ResourceKind.DOCKER

    def __init__(self, docker_binary: str = "docker") -> None:
        self._docker = docker_binary

    def scan(self) -> list[ResourceRef]:
        """Return one reference per stopped ``sandbox-<team_id>`` container.

        Returns:
            References whose ``detail`` is the container id and whose
            ``age_seconds`` is set when Docker's creation timestamp parses.

        Raises:
            OSError: If the Docker client is absent or the daemon is
                unreachable.
        """
        lines = self._run(
            [
                "ps",
                "--all",
                "--filter",
                f"name=^{SANDBOX_CONTAINER_PREFIX}",
                "--filter",
                "status=exited",
                "--filter",
                "status=created",
                "--filter",
                "status=dead",
                "--format",
                "{{.ID}}\t{{.Names}}\t{{.CreatedAt}}",
            ]
        )
        refs: list[ResourceRef] = []
        for line in lines:
            ref = self._parse_container(line)
            if ref is not None:
                refs.append(ref)
        return refs

    def purge(self, ref: ResourceRef) -> int:
        """Remove one container and its anonymous volumes.

        Returns:
            ``1`` — one container removed.

        Raises:
            OSError: If the removal fails.
        """
        self._run(["rm", "--volumes", ref.detail])
        return 1

    def close(self) -> None:
        """No connection to release — the Docker CLI is invoked per call."""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_container(self, line: str) -> ResourceRef | None:
        """Turn one ``docker ps`` row into a reference, or ``None`` to skip.

        A container whose name suffix is not a UUID is skipped with a warning:
        it did not come from ``DockerSandboxActor``, and an unattributable
        container is never anyone's orphan.
        """
        parts = line.split("\t")
        if len(parts) < 3:
            return None
        container_id, name, created_at = parts[0], parts[1], parts[2]
        team_id = name.removeprefix(SANDBOX_CONTAINER_PREFIX)
        try:
            uuid.UUID(team_id)
        except ValueError:
            logger.warning("Skipping container '%s': suffix '%s' is not a team id", name, team_id)
            return None
        return ResourceRef(
            kind=self.kind,
            team_id=team_id,
            detail=container_id,
            label=name,
            size_hint=1,
            age_seconds=_parse_docker_age(created_at),
        )

    def _run(self, args: list[str]) -> list[str]:
        """Run a Docker sub-command and return its non-empty output lines.

        Args:
            args: Arguments after the executable name.

        Returns:
            Output lines with surrounding whitespace stripped.

        Raises:
            OSError: If the client is missing, times out, or exits non-zero.
        """
        try:
            result = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [self._docker, *args],
                capture_output=True,
                text=True,
                timeout=_DOCKER_TIMEOUT_SECONDS,
                check=False,
            )
        except FileNotFoundError as exc:
            msg = f"Docker client '{self._docker}' not found"
            raise OSError(msg) from exc
        except subprocess.TimeoutExpired as exc:
            msg = f"docker {args[0]} timed out after {_DOCKER_TIMEOUT_SECONDS}s"
            raise OSError(msg) from exc
        if result.returncode != 0:
            msg = f"docker {args[0]} failed (exit {result.returncode}): {result.stderr.strip()}"
            raise OSError(msg)
        return [line for line in result.stdout.splitlines() if line.strip()]


def _parse_docker_age(created_at: str) -> float | None:
    """Return the age in seconds of a Docker ``CreatedAt`` timestamp.

    Docker renders it as ``2026-09-04 10:11:12 +0200 CEST``; the trailing
    zone abbreviation is dropped and the rest parsed with an explicit format.

    Args:
        created_at: The ``{{.CreatedAt}}`` field, verbatim.

    Returns:
        Age in seconds, or ``None`` when the timestamp does not parse. An
        unknown age does not protect a container — the grace period is a
        second line of defence, and the scan ordering is the first.
    """
    fields = created_at.split()
    if len(fields) < 3:
        return None
    try:
        created = datetime.strptime(" ".join(fields[:3]), "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        logger.debug("Unparseable docker CreatedAt: %r", created_at)
        return None
    return (datetime.now(UTC) - created).total_seconds()


# ---------------------------------------------------------------------------
# Workspace filesystem
# ---------------------------------------------------------------------------


class WorkspaceReaper:
    """Reaps workspace directories under ``AKGENTIC_WORKSPACES_ROOT``.

    **This is the one reaper that destroys data rather than runtime.** A
    Weaviate object can be re-ingested and a sandbox container rebuilt from
    its image; the files an agent wrote cannot be recovered from anywhere.
    Every rule below exists because of that asymmetry, and none of them may be
    relaxed for convenience.

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
