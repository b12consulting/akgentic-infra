"""Behaviour of the vector-store reaper.

The workspace reaper has its own module; the team-id readers and the
registered-backend declaration have theirs. What is left here is the reaper
itself: how it turns a backend's collections into references, and how it
delegates a delete back to that backend.
"""

from __future__ import annotations

import pytest

from akgentic.infra.maintenance.models import ResourceKind, ResourceRef
from akgentic.infra.maintenance.reapers import VectorStoreReaper


class _FakeAdminBackend:
    """The administrative slice of a cluster backend: list, and delete by team."""

    def __init__(
        self,
        collections: list[str],
        deleted_count: int | None = 0,
        *,
        list_error: Exception | None = None,
        delete_error: Exception | None = None,
    ) -> None:
        self._collections = collections
        self._deleted_count = deleted_count
        self._list_error = list_error
        self._delete_error = delete_error
        self.deletes: list[tuple[str, str]] = []

    def list_collections(self) -> list[str]:
        """Return the cluster's collections, or raise the configured error."""
        if self._list_error is not None:
            raise self._list_error
        return list(self._collections)

    def delete_by_team(self, collection: str, team_id: str) -> int | None:
        """Record the delete and report what the backend would report."""
        if self._delete_error is not None:
            raise self._delete_error
        self.deletes.append((collection, team_id))
        return self._deleted_count


class _FakeTeamIndex:
    """A ``TeamIndex`` stub over a fixed per-collection count map."""

    def __init__(
        self,
        counts: dict[str, dict[str, int]],
        *,
        read_error: Exception | None = None,
    ) -> None:
        self._counts = counts
        self._read_error = read_error
        self.read: list[str] = []
        self.closed = False

    def team_counts(self, collection: str) -> dict[str, int]:
        """Return the canned counts, or raise the configured error."""
        self.read.append(collection)
        if self._read_error is not None:
            raise self._read_error
        return dict(self._counts.get(collection, {}))

    def close(self) -> None:
        """Mark the read connection released."""
        self.closed = True


def _reaper(
    backend: _FakeAdminBackend,
    index: _FakeTeamIndex,
    name: str = "weaviate",
) -> VectorStoreReaper:
    """Build a reaper around fakes, skipping the registry and the connect."""
    reaper = object.__new__(VectorStoreReaper)
    reaper._name = name  # noqa: SLF001 - constructing the fake
    reaper._backend = backend  # noqa: SLF001 - constructing the fake
    reaper._index = index  # noqa: SLF001 - constructing the fake
    return reaper


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def test_scan_emits_one_reference_per_collection_and_team() -> None:
    """A delete is per ``(collection, team)``, so a reference must be too."""
    reaper = _reaper(
        _FakeAdminBackend(["Documents", "Chunks"]),
        _FakeTeamIndex(
            {"Documents": {"team-a": 3, "team-b": 5}, "Chunks": {"team-a": 2}},
        ),
    )

    refs = sorted(reaper.scan(), key=lambda ref: ref.label)

    assert [(ref.detail, ref.team_id, ref.size_hint) for ref in refs] == [
        ("Chunks", "team-a", 2),
        ("Documents", "team-a", 3),
        ("Documents", "team-b", 5),
    ]
    assert all(ref.kind is ResourceKind.VECTOR for ref in refs)


def test_a_reference_names_the_backend_it_came_from() -> None:
    """Two backends share one kind, so the label is what tells them apart."""
    backend = _FakeAdminBackend(["planning"])
    index = _FakeTeamIndex({"planning": {"team-a": 1}})

    weaviate_ref = _reaper(backend, index, "weaviate").scan()[0]
    qdrant_ref = _reaper(backend, index, "qdrant").scan()[0]

    assert weaviate_ref.label == "weaviate:planning/team-a"
    assert qdrant_ref.label == "qdrant:planning/team-a"


def test_a_shared_collection_yields_no_reference() -> None:
    """``delete_by_team`` refuses a shared collection before any cluster call.

    Scanning one could therefore only ever produce orphans whose purge fails,
    against rows a live team is still reading.
    """
    index = _FakeTeamIndex(
        {"workspace_chunks": {"team-a": 9}, "planning": {"team-a": 1}},
    )
    reaper = _reaper(_FakeAdminBackend(["workspace_chunks", "planning"]), index)

    refs = reaper.scan()

    assert [ref.detail for ref in refs] == ["planning"]
    assert index.read == ["planning"]


def test_a_vector_reference_carries_no_age() -> None:
    """The grace period needs an age, and a vector row exposes none cheaply."""
    reaper = _reaper(_FakeAdminBackend(["planning"]), _FakeTeamIndex({"planning": {"t": 1}}))

    assert reaper.scan()[0].age_seconds is None


# ---------------------------------------------------------------------------
# A backend that cannot be read
# ---------------------------------------------------------------------------


def test_an_unreadable_cluster_raises_rather_than_scanning_empty() -> None:
    """An empty scan would be classified as a clean backend, which is a lie.

    The driver turns the raise into ``available=False``; swallowing it here
    would report "no orphans" for a cluster nobody could reach.
    """
    reaper = _reaper(
        _FakeAdminBackend([], list_error=OSError("connection refused")),
        _FakeTeamIndex({}),
    )

    with pytest.raises(OSError, match="connection refused"):
        reaper.scan()


def test_an_enumeration_that_fails_raises_rather_than_counting_zero() -> None:
    """Same contract one level down: a failed read is not an empty collection."""
    reaper = _reaper(
        _FakeAdminBackend(["planning"]),
        _FakeTeamIndex({}, read_error=RuntimeError("aggregation exploded")),
    )

    with pytest.raises(RuntimeError, match="aggregation exploded"):
        reaper.scan()


# ---------------------------------------------------------------------------
# Purging
# ---------------------------------------------------------------------------


def _ref(collection: str = "planning", team_id: str = "team-a", size: int = 4) -> ResourceRef:
    """Build a vector reference the way a scan would."""
    return ResourceRef(
        kind=ResourceKind.VECTOR,
        team_id=team_id,
        detail=collection,
        label=f"weaviate:{collection}/{team_id}",
        size_hint=size,
    )


def test_purge_delegates_to_the_backends_own_delete() -> None:
    """Whatever paging a cluster needs is the backend's problem, not the reaper's."""
    backend = _FakeAdminBackend(["planning"], deleted_count=20_250)
    reaper = _reaper(backend, _FakeTeamIndex({}))

    assert reaper.purge(_ref()) == 20_250
    assert backend.deletes == [("planning", "team-a")]


def test_purge_falls_back_to_the_size_hint_when_the_backend_reports_no_count() -> None:
    """Weaviate counts what it deleted and Qdrant returns nothing; both report."""
    reaper = _reaper(_FakeAdminBackend(["planning"], deleted_count=None), _FakeTeamIndex({}))

    assert reaper.purge(_ref(size=7)) == 7


def test_a_refused_delete_propagates_as_a_purge_failure() -> None:
    """The driver records it and reaps the remaining orphans regardless."""
    reaper = _reaper(
        _FakeAdminBackend(["planning"], delete_error=ValueError("shared across teams")),
        _FakeTeamIndex({}),
    )

    with pytest.raises(ValueError, match="shared across teams"):
        reaper.purge(_ref())


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_close_releases_the_read_connection() -> None:
    """A sweep opens one connection per backend and must not leave it open."""
    index = _FakeTeamIndex({})
    _reaper(_FakeAdminBackend([]), index).close()

    assert index.closed is True


def test_a_backend_with_no_enumerator_is_refused_at_construction() -> None:
    """A reaper that cannot read team ids would report every deployment clean."""
    with pytest.raises(ValueError, match="no team-id enumerator"):
        VectorStoreReaper("inmemory")
