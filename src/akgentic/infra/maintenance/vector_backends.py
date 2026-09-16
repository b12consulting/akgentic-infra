"""Which vector-store backends this sweep reclaims, and how it reads them.

``akgentic-tool`` puts a registry in front of its vector-store backends, so
"the Weaviate reaper" is the wrong shape: a Qdrant deployment leaks identically
and a sweep hardcoded to one vendor would report it clean. Everything
backend-specific therefore lives here, behind two seams.

**The write side is the backend's.** ``list_collections`` and ``delete_by_team``
are already on both cluster backends, documented as crossing the team boundary
on purpose. :class:`VectorAdminBackend` is the slice of a backend this sweep
calls — not part of ``VectorStoreService``, which is why it is named here rather
than imported.

**The read side is ours.** No backend exposes "which distinct team ids are
present", which is the one question an orphan scan asks, and reaching into a
backend's private client for it would break that package's encapsulation. Each
:class:`TeamIndex` therefore opens its own connection to the cluster its
backend is configured for. That is a second connection for the duration of one
maintenance run, and the deliberate alternative to sharing the backend's cached
client — which this process does not own and must not close. When the
enumeration primitive lands next to its two siblings in ``akgentic-tool``, these
classes collapse into delegation.

**Every registered backend is accounted for, reapable or not.**
:data:`BACKEND_DISPOSITIONS` maps each one to an enumerator or to a stated
reason for leaving it alone, and a spec fails when the registry reports a name
this module has never heard of. "Every registered name has a reaper" would be
the wrong guard — two of the four registered backends hold nothing this sweep
can reclaim, so it would be red from the day it was written and the first
person to see it would weaken it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

    import weaviate
    from qdrant_client import QdrantClient
    from qdrant_client.conversions.common_types import PointId

logger = logging.getLogger(__name__)

_WEAVIATE_GROUP_LIMIT: int = 10_000
"""Cap on the groups one ``group_by`` aggregation may return."""

_QDRANT_SCROLL_BATCH: int = 1_000
"""Points read per ``scroll`` page. Payload-only, so a large page is cheap."""


# ---------------------------------------------------------------------------
# The two seams
# ---------------------------------------------------------------------------


class VectorAdminBackend(Protocol):
    """The team-scoped cleanup surface this sweep needs from a backend.

    Both cluster backends expose exactly these two methods, outside
    ``VectorStoreService`` and documented as administrative. The return types
    differ — Weaviate counts what it deleted, Qdrant returns nothing — so the
    union is spelled here and normalised by the caller.
    """

    def list_collections(self) -> list[str]:
        """Return every collection present in the cluster."""
        ...

    def delete_by_team(self, collection: str, team_id: str) -> int | None:
        """Delete every row in *collection* stamped with *team_id*."""
        ...


class TeamIndex(Protocol):
    """Reads which teams hold rows in a vector backend, and how many.

    The read half of the sweep. An implementation owns its own connection and
    releases it in :meth:`close`.
    """

    def team_counts(self, collection: str) -> dict[str, int]:
        """Return the row count per ``team_id`` in *collection*.

        Raises:
            Exception: If the cluster cannot be read. The driver turns that
                into an ``available=False`` report — never into "no orphans".
        """
        ...

    def close(self) -> None:
        """Release the connection. Always called, even on failure."""
        ...


TeamIndexFactory = Callable[[], "TeamIndex"]
"""Builds a :class:`TeamIndex` from the environment its backend is configured by."""


@dataclass(frozen=True)
class NotSwept:
    """A registered backend this sweep deliberately does not reclaim.

    Carrying the reason as a value rather than a comment is what lets the guard
    spec tell "considered and excluded" apart from "never noticed", which is the
    only distinction that matters when a new backend is registered.

    Attributes:
        reason: Why the backend holds nothing this sweep can reclaim.
    """

    reason: str


# ---------------------------------------------------------------------------
# Weaviate
# ---------------------------------------------------------------------------


class WeaviateTeamIndex:
    """Enumerates team ids in a Weaviate cluster.

    Prefers a server-side ``group_by`` aggregation and falls back to a full
    object walk, which is slower but cannot silently under-report — and an
    under-report here makes a live team look orphaned.
    """

    def __init__(self) -> None:
        """Connect to the cluster ``AKGENTIC_WEAVIATE_URL`` names.

        Raises:
            ValueError: If no cluster URL is exported.
            ImportError: If the ``weaviate-client`` extra is not installed.
        """
        from urllib.parse import urlparse

        from akgentic.tool.vector_store.backends.weaviate import weaviate_api_key, weaviate_url

        url = weaviate_url()
        if not url:
            msg = "No Weaviate cluster URL is exported; cannot enumerate team ids."
            raise ValueError(msg)

        import weaviate as _wv
        from weaviate.auth import AuthApiKey

        api_key = weaviate_api_key()
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

    def team_counts(self, collection: str) -> dict[str, int]:
        """Count objects per ``team_id`` in one collection.

        Args:
            collection: Collection to read.

        Returns:
            Object count keyed by ``team_id``. Objects with a missing or
            non-string ``team_id`` are excluded: an unattributable object is
            never anyone's orphan.
        """
        try:
            return self._aggregate(collection)
        except Exception as exc:  # noqa: BLE001 - any client/server error degrades
            logger.info(
                "Weaviate group_by aggregation unavailable on '%s' (%s); "
                "falling back to an object walk",
                collection,
                exc,
            )
            return self._walk(collection)

    def close(self) -> None:
        """Disconnect this index's own client."""
        self._client.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _aggregate(self, collection: str) -> dict[str, int]:
        """Count per ``team_id`` via a server-side aggregation."""
        from weaviate.classes.aggregate import GroupByAggregate

        prop = weaviate_team_id_key()
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

    def _walk(self, collection: str) -> dict[str, int]:
        """Count per ``team_id`` by iterating every object in the collection."""
        prop = weaviate_team_id_key()
        counts: dict[str, int] = {}
        for obj in self._client.collections.get(collection).iterator(
            return_properties=[prop],
        ):
            value = obj.properties.get(prop)
            if isinstance(value, str) and value:
                counts[value] = counts.get(value, 0) + 1
        return counts


def weaviate_team_id_key() -> str:
    """Return the Weaviate schema property carrying the owning team's id.

    Read from ``akgentic-tool`` rather than restated here, so a rename on the
    write side cannot leave this sweep filtering on a name nothing is stamped
    with — which would delete nothing and report success.
    """
    from akgentic.tool.vector_store.backends.weaviate import TEAM_ID_PROPERTY

    return TEAM_ID_PROPERTY


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------


class QdrantTeamIndex:
    """Enumerates team ids in a Qdrant cluster by scrolling point payloads.

    Qdrant offers no group-by aggregation over an arbitrary payload key, so
    there is one path rather than Weaviate's two: scroll every point, reading
    the team key alone and no vectors.
    """

    def __init__(self) -> None:
        """Connect to the cluster ``AKGENTIC_QDRANT_URL`` names.

        Raises:
            ValueError: If no cluster URL is exported.
            ImportError: If the ``qdrant-client`` extra is not installed.
        """
        from akgentic.tool.vector_store.backends.qdrant import qdrant_api_key, qdrant_url

        url = qdrant_url()
        if not url:
            msg = "No Qdrant cluster URL is exported; cannot enumerate team ids."
            raise ValueError(msg)

        from qdrant_client import QdrantClient as _QdrantClient

        self._client: QdrantClient = _QdrantClient(url=url, api_key=qdrant_api_key())

    def team_counts(self, collection: str) -> dict[str, int]:
        """Count points per ``team_id`` in one collection.

        Args:
            collection: Collection to read.

        Returns:
            Point count keyed by ``team_id``. Points with a missing or
            non-string team key are excluded, exactly as Weaviate's are.

            A scroll that hands back the offset it was given is stopped rather
            than followed: this runs unattended, and a cursor that does not
            advance would hang the sweep silently instead of failing it. The
            partial count under-reports, which leaks an orphan rather than
            condemning a live team.
        """
        key = qdrant_team_id_key()
        counts: dict[str, int] = {}
        previous: PointId | None = None
        while True:
            points, offset = self._client.scroll(
                collection_name=collection,
                limit=_QDRANT_SCROLL_BATCH,
                offset=previous,
                with_payload=[key],
                with_vectors=False,
            )
            for point in points:
                value = (point.payload or {}).get(key)
                if isinstance(value, str) and value:
                    counts[value] = counts.get(value, 0) + 1
            if offset is None:
                return counts
            if offset == previous:
                logger.warning(
                    "Qdrant scroll of '%s' returned the same offset twice (%r); "
                    "stopping the enumeration at %d team(s) rather than looping",
                    collection,
                    offset,
                    len(counts),
                )
                return counts
            previous = offset

    def close(self) -> None:
        """Disconnect this index's own client."""
        self._client.close()


def qdrant_team_id_key() -> str:
    """Return the Qdrant payload key carrying the owning team's id.

    Imported from the write side for the same reason as its Weaviate sibling.
    ``TENANT_PAYLOAD`` is a second axis, not a team key, and is never read here.
    """
    from akgentic.tool.vector_store.backends.qdrant import TEAM_ID_PAYLOAD

    return TEAM_ID_PAYLOAD


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------

BACKEND_DISPOSITIONS: dict[str, TeamIndexFactory | NotSwept] = {
    "weaviate": WeaviateTeamIndex,
    "qdrant": QdrantTeamIndex,
    "inmemory": NotSwept(
        "the in-memory index has no durable store: it lives in the vector-store "
        "actor's own state and dies with the process that held it, so a deleted "
        "team leaves nothing behind to reclaim."
    ),
    "local": NotSwept(
        "the local index lives under the workspace tree's <meta> sibling, so it "
        "is reclaimed with that tree by the workspace reaper. Reaping it "
        "separately would delete an index whose tree is still live."
    ),
}
"""Every registered vector-store backend, mapped to an enumerator or a reason.

A backend missing from this mapping is what the guard spec catches. Adding one
to ``akgentic-tool``'s registry is therefore a change that must pass through
here — either with a way to read its team ids, or with a stated reason it holds
nothing this sweep can reclaim.
"""


def unaccounted_backends(names: Iterable[str]) -> list[str]:
    """Return the *names* this module neither reaps nor states a reason for.

    Args:
        names: Registered backend names, typically ``available_backends()``.

    Returns:
        The unaccounted names, sorted. Empty is the healthy answer.
    """
    return sorted(set(names) - set(BACKEND_DISPOSITIONS))


def team_index_factory(name: str) -> TeamIndexFactory | None:
    """Return how to read *name*'s team ids, or ``None`` if it is not swept.

    Args:
        name: A registered backend name.

    Returns:
        The factory for a reapable backend, else ``None`` — both for a backend
        declared :class:`NotSwept` and for one this module has never heard of.
    """
    disposition = BACKEND_DISPOSITIONS.get(name)
    if disposition is None or isinstance(disposition, NotSwept):
        return None
    return disposition


def sweepable_backends() -> list[str]:
    """Return the registered backends that are both reapable and provisioned.

    ``spec.is_configured()`` is the seam that answers "has this deployment
    provisioned the backend" without this package naming a single environment
    variable — going around it would put two env-var vocabularies in
    ``akgentic-infra``, and the second one to go stale would do so silently.

    Returns:
        Backend names, sorted, so a sweep's report order is deterministic.
    """
    from akgentic.tool.vector_store.registry import available_backends, get_backend_spec

    names: list[str] = []
    for name in available_backends():
        if team_index_factory(name) is None:
            continue
        try:
            if get_backend_spec(name).is_configured():
                names.append(name)
        except Exception:  # noqa: BLE001 - a readiness probe must not break discovery
            logger.debug("is_configured() raised for backend '%s'", name, exc_info=True)
    return names


def administrative_backend(name: str) -> VectorAdminBackend:
    """Build *name*'s backend as a sweeper — belonging to no team.

    ``BackendContext.team_id=None`` is documented in the registry as exactly
    this case: a backend built to call ``delete_by_team`` or
    ``list_collections``, which belongs to no team by construction. It is not
    closed here; a cluster backend's client comes from a process-wide cache this
    sweep does not own.

    Args:
        name: A registered backend name.

    Returns:
        The backend, seen through the administrative slice this sweep uses.

    Raises:
        ValueError: If *name* is not registered, or is not provisioned.
    """
    from akgentic.tool.vector_store.protocol import VectorStoreConfig
    from akgentic.tool.vector_store.registry import BackendContext, get_backend_spec

    spec = get_backend_spec(name)
    backend = spec.factory(
        BackendContext(
            config=VectorStoreConfig(name="orphan-sweep", role="maintenance"),
            team_id=None,
        )
    )
    # The two cleanup methods sit outside VectorStoreService, which is what the
    # factory is typed to return, so the administrative slice is asserted here.
    return cast("VectorAdminBackend", backend)
