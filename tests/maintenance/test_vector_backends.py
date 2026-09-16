"""The vector-backend declaration, its guard, and the two team-id readers."""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import TYPE_CHECKING

import pytest
from akgentic.tool.vector_store.protocol import VectorStoreConfig
from akgentic.tool.vector_store.registry import (
    BackendContext,
    BackendSpec,
    available_backends,
    register_backend,
    unregister_backend,
)

from akgentic.infra.maintenance import vector_backends
from akgentic.infra.maintenance.vector_backends import (
    BACKEND_DISPOSITIONS,
    NotSwept,
    QdrantTeamIndex,
    WeaviateTeamIndex,
    qdrant_team_id_key,
    sweepable_backends,
    team_index_factory,
    unaccounted_backends,
    weaviate_team_id_key,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_WEAVIATE_AGGREGATE_MODULE = "weaviate.classes.aggregate"


class _StubGroupByAggregate:
    """Stands in for Weaviate's ``GroupByAggregate`` argument holder."""

    def __init__(self, *, prop: str, limit: int) -> None:
        self.prop = prop
        self.limit = limit


@pytest.fixture(autouse=True)
def _weaviate_aggregate_symbol() -> Iterator[None]:
    """Make ``GroupByAggregate`` importable when the client extra is absent.

    ``WeaviateTeamIndex._aggregate`` imports it to build the ``group_by``
    argument, and neither CI nor a community-tier install has
    ``weaviate-client``. Without this the import raises, ``team_counts``
    degrades to its object walk, and every aggregation spec below then passes or
    fails for a reason that has nothing to do with the code under test — which is
    how a green suite here and a red one in CI came apart in the first place.

    The vendor class is a plain argument holder that the fake never inspects, so
    standing in for it costs no fidelity: what these specs exercise is this
    package's grouping and filtering, not Weaviate's.
    """
    if importlib.util.find_spec("weaviate") is not None:
        yield
        return

    installed: list[str] = []
    for name in ("weaviate", "weaviate.classes", _WEAVIATE_AGGREGATE_MODULE):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            installed.append(name)
    sys.modules[_WEAVIATE_AGGREGATE_MODULE].GroupByAggregate = _StubGroupByAggregate
    try:
        yield
    finally:
        for name in installed:
            sys.modules.pop(name, None)


def _unused_factory(_context: BackendContext) -> object:
    """A factory for specs that exist only to be inspected, never built."""
    msg = "this backend is never constructed"
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# The registered-backend guard
# ---------------------------------------------------------------------------


def test_every_registered_vector_backend_is_accounted_for() -> None:
    """A backend the registry knows and this sweep does not is a silent leak.

    "Every registered name has a *reaper*" would be the wrong guard — two of
    the registered backends hold nothing reclaimable — so the rule is that
    every name is either reapable or carries a stated reason for being left
    alone. Registering a backend in ``akgentic-tool`` therefore has to pass
    through the declaration here.
    """
    assert unaccounted_backends(available_backends()) == []


def test_a_backend_nobody_declared_is_reported_as_unaccounted() -> None:
    """The guard above is only worth having if it can actually go red.

    Registers a backend the declaration has never heard of and asserts it is
    named, rather than trusting that a green guard is a discriminating one.
    """

    register_backend(BackendSpec(name="novel_store", factory=_unused_factory))
    try:
        assert "novel_store" in unaccounted_backends(available_backends())
    finally:
        unregister_backend("novel_store")

    assert "novel_store" not in unaccounted_backends(available_backends())


def test_a_backend_that_is_not_swept_states_why() -> None:
    """A reason is a value, so "considered" is distinguishable from "missed"."""
    not_swept = {
        name: disposition
        for name, disposition in BACKEND_DISPOSITIONS.items()
        if isinstance(disposition, NotSwept)
    }

    assert set(not_swept) == {"inmemory", "local"}
    assert all(entry.reason.strip() for entry in not_swept.values())


def test_a_backend_that_is_not_swept_has_no_enumerator() -> None:
    """Declaring a reason and offering a reader would contradict each other."""
    assert team_index_factory("inmemory") is None
    assert team_index_factory("local") is None
    assert team_index_factory("weaviate") is WeaviateTeamIndex
    assert team_index_factory("qdrant") is QdrantTeamIndex


def test_an_unknown_backend_has_no_enumerator() -> None:
    """A name nothing declared is not reapable, and must not look like it is."""
    assert team_index_factory("no_such_backend") is None


# ---------------------------------------------------------------------------
# Discovery through the registry
# ---------------------------------------------------------------------------


def test_only_provisioned_backends_are_swept(monkeypatch: pytest.MonkeyPatch) -> None:
    """``is_configured()`` is the seam; this package names no cluster variable."""
    monkeypatch.delenv("AKGENTIC_WEAVIATE_URL", raising=False)
    monkeypatch.setenv("AKGENTIC_QDRANT_URL", "http://qdrant:6333")

    assert sweepable_backends() == ["qdrant"]


def test_a_deployment_with_no_vector_store_sweeps_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-memory and local backends are always registered and never swept."""
    monkeypatch.delenv("AKGENTIC_WEAVIATE_URL", raising=False)
    monkeypatch.delenv("AKGENTIC_QDRANT_URL", raising=False)

    assert sweepable_backends() == []


def test_both_clusters_are_swept_in_a_deterministic_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two configured backends are two reports, and the order must not wander."""
    monkeypatch.setenv("AKGENTIC_WEAVIATE_URL", "http://weaviate:8080")
    monkeypatch.setenv("AKGENTIC_QDRANT_URL", "http://qdrant:6333")

    assert sweepable_backends() == ["qdrant", "weaviate"]


def test_a_backend_whose_readiness_probe_raises_is_skipped_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One broken probe must not take the whole sweep down with it."""

    def _explode() -> bool:
        msg = "probe blew up"
        raise RuntimeError(msg)

    monkeypatch.delenv("AKGENTIC_QDRANT_URL", raising=False)
    monkeypatch.setenv("AKGENTIC_WEAVIATE_URL", "http://weaviate:8080")
    monkeypatch.setattr(
        "akgentic.tool.vector_store.registry.get_backend_spec",
        lambda name: BackendSpec(name=name, factory=_unused_factory, is_configured=_explode),
    )

    assert sweepable_backends() == []


# ---------------------------------------------------------------------------
# The team-id keys, read from the write side
# ---------------------------------------------------------------------------


def test_the_weaviate_team_id_key_is_read_from_the_write_side() -> None:
    """Restating the property name here would let a rename silently no-op."""
    from akgentic.tool.vector_store.backends.weaviate import TEAM_ID_PROPERTY

    assert weaviate_team_id_key() == TEAM_ID_PROPERTY


def test_the_qdrant_team_id_key_is_read_from_the_write_side() -> None:
    """Same contract, second backend: the payload key has one definition."""
    from akgentic.tool.vector_store.backends.qdrant import TEAM_ID_PAYLOAD

    assert qdrant_team_id_key() == TEAM_ID_PAYLOAD


# ---------------------------------------------------------------------------
# Weaviate: reading team ids
# ---------------------------------------------------------------------------


class _FakeAggregate:
    """Aggregation surface that either answers or refuses."""

    def __init__(self, groups: list[tuple[str, int]] | None) -> None:
        self._groups = groups

    def over_all(self, *, group_by: object) -> object:  # noqa: ARG002 - shape only
        """Return grouped counts, or raise to force the walk fallback."""
        if self._groups is None:
            msg = "group_by unsupported"
            raise RuntimeError(msg)
        groups = [
            type(
                "Group",
                (),
                {"grouped_by": type("By", (), {"value": value})(), "total_count": n},
            )()
            for value, n in self._groups
        ]
        return type("Result", (), {"groups": groups})()


class _FakeCollection:
    """One Weaviate collection with canned aggregate and iterator."""

    def __init__(
        self,
        groups: list[tuple[str, int]] | None,
        objects: list[str] | None = None,
    ) -> None:
        self.aggregate = _FakeAggregate(groups)
        self._objects = objects or []

    def iterator(self, *, return_properties: list[str]) -> list[object]:
        """Yield objects carrying only the ``team_id`` property."""
        prop = return_properties[0]
        return [type("Obj", (), {"properties": {prop: value}})() for value in self._objects]


class _FakeCollections:
    """The cluster's collection registry."""

    def __init__(self, collections: dict[str, _FakeCollection]) -> None:
        self._collections = collections

    def get(self, name: str) -> _FakeCollection:
        """Return one collection by name."""
        return self._collections[name]


class _FakeWeaviateClient:
    """A Weaviate client exposing only what the index touches."""

    def __init__(self, collections: dict[str, _FakeCollection]) -> None:
        self.collections = _FakeCollections(collections)
        self.closed = False

    def close(self) -> None:
        """Record the disconnect."""
        self.closed = True


def _weaviate_index_over(collections: dict[str, _FakeCollection]) -> WeaviateTeamIndex:
    """Build a ``WeaviateTeamIndex`` around a fake client, skipping the connect."""
    index = object.__new__(WeaviateTeamIndex)
    index._client = _FakeWeaviateClient(collections)  # noqa: SLF001 - constructing the fake
    return index


def test_weaviate_counts_come_from_the_server_side_aggregation() -> None:
    """The cheap path: one round trip per collection rather than a full walk."""
    index = _weaviate_index_over({"Documents": _FakeCollection([("team-a", 3), ("team-b", 5)])})

    assert index.team_counts("Documents") == {"team-a": 3, "team-b": 5}


def test_weaviate_falls_back_to_an_object_walk_when_aggregation_is_refused() -> None:
    """An older cluster must still be swept — silently reporting none is worse."""
    index = _weaviate_index_over({"Documents": _FakeCollection(None, objects=["t1", "t1", "t2"])})

    assert index.team_counts("Documents") == {"t1": 2, "t2": 1}


def test_weaviate_objects_with_no_team_id_are_never_attributed() -> None:
    """An unattributable object belongs to no team and is never an orphan."""
    index = _weaviate_index_over({"Documents": _FakeCollection(None, objects=["t1", "", "t1"])})

    assert index.team_counts("Documents") == {"t1": 2}


def test_weaviate_close_disconnects_its_own_client() -> None:
    """The read connection is this sweep's, and it must not outlive the run."""
    index = _weaviate_index_over({})
    index.close()

    assert index._client.closed is True  # noqa: SLF001 - asserting on the fake


def test_weaviate_without_a_cluster_url_refuses_to_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unprovisioned backend is a skipped reaper, never a half-built one."""
    monkeypatch.delenv("AKGENTIC_WEAVIATE_URL", raising=False)

    with pytest.raises(ValueError, match="No Weaviate cluster URL"):
        WeaviateTeamIndex()


# ---------------------------------------------------------------------------
# Qdrant: reading team ids
# ---------------------------------------------------------------------------


class _FakePoint:
    """One scrolled point, carrying only its payload."""

    def __init__(self, payload: dict[str, object] | None) -> None:
        self.payload = payload


class _FakeQdrantClient:
    """A Qdrant client that hands back canned scroll pages."""

    def __init__(self, pages: list[tuple[list[_FakePoint], object | None]]) -> None:
        self._pages = list(pages)
        self.closed = False
        self.scrolls: list[object | None] = []
        self.with_payload: list[object] = []

    def scroll(
        self,
        *,
        collection_name: str,  # noqa: ARG002 - shape only
        limit: int,  # noqa: ARG002 - shape only
        offset: object | None,
        with_payload: object,
        with_vectors: bool,  # noqa: ARG002 - shape only
    ) -> tuple[list[_FakePoint], object | None]:
        """Return the next canned page, recording the offset it was asked for."""
        self.scrolls.append(offset)
        self.with_payload.append(with_payload)
        return self._pages.pop(0)

    def close(self) -> None:
        """Record the disconnect."""
        self.closed = True


def _qdrant_index_over(
    pages: list[tuple[list[_FakePoint], object | None]],
) -> QdrantTeamIndex:
    """Build a ``QdrantTeamIndex`` around a fake client, skipping the connect."""
    index = object.__new__(QdrantTeamIndex)
    index._client = _FakeQdrantClient(pages)  # noqa: SLF001 - constructing the fake
    return index


def test_qdrant_counts_points_by_their_team_payload() -> None:
    """Qdrant has no group-by over a payload key, so the scan scrolls."""
    key = qdrant_team_id_key()
    index = _qdrant_index_over(
        [([_FakePoint({key: "t1"}), _FakePoint({key: "t2"}), _FakePoint({key: "t1"})], None)]
    )

    assert index.team_counts("workspace_chunks") == {"t1": 2, "t2": 1}


def test_qdrant_follows_the_scroll_to_the_last_page() -> None:
    """Stopping at the first page would under-report and orphan a live team."""
    key = qdrant_team_id_key()
    index = _qdrant_index_over(
        [
            ([_FakePoint({key: "t1"})], "page-2"),
            ([_FakePoint({key: "t1"}), _FakePoint({key: "t2"})], None),
        ]
    )

    counts = index.team_counts("planning")

    assert counts == {"t1": 2, "t2": 1}
    assert index._client.scrolls == [None, "page-2"]  # noqa: SLF001 - asserting on the fake


def test_qdrant_points_with_no_team_payload_are_never_attributed() -> None:
    """Matches Weaviate: an unattributable row is never anyone's orphan."""
    key = qdrant_team_id_key()
    index = _qdrant_index_over(
        [([_FakePoint({key: "t1"}), _FakePoint(None), _FakePoint({})], None)]
    )

    assert index.team_counts("planning") == {"t1": 1}


def test_qdrant_reads_the_team_key_and_no_vectors() -> None:
    """A scan pulling vectors would move gigabytes to count strings."""
    index = _qdrant_index_over([([], None)])

    index.team_counts("planning")

    assert index._client.with_payload == [[qdrant_team_id_key()]]  # noqa: SLF001 - the fake


def test_qdrant_close_disconnects_its_own_client() -> None:
    """The read connection is this sweep's, and it must not outlive the run."""
    index = _qdrant_index_over([])
    index.close()

    assert index._client.closed is True  # noqa: SLF001 - asserting on the fake


def test_qdrant_without_a_cluster_url_refuses_to_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unprovisioned backend is a skipped reaper, never a half-built one."""
    monkeypatch.delenv("AKGENTIC_QDRANT_URL", raising=False)

    with pytest.raises(ValueError, match="No Qdrant cluster URL"):
        QdrantTeamIndex()


# ---------------------------------------------------------------------------
# Building an administrative backend
# ---------------------------------------------------------------------------


def test_an_administrative_backend_belongs_to_no_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``team_id=None`` is the registry's documented sweeper case.

    A sweeper that carried a team would be able to reap only itself — the one
    team that is never being reaped.
    """
    seen: list[BackendContext] = []

    def _factory(context: BackendContext) -> object:
        seen.append(context)
        return object()

    monkeypatch.setattr(
        "akgentic.tool.vector_store.registry.get_backend_spec",
        lambda name: BackendSpec(name=name, factory=_factory),  # type: ignore[arg-type]
    )

    vector_backends.administrative_backend("weaviate")

    assert seen[0].team_id is None
    assert isinstance(seen[0].config, VectorStoreConfig)
