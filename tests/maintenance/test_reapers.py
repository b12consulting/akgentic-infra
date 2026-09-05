"""Behaviour of the individual backend reapers."""

from __future__ import annotations

import subprocess
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from akgentic.infra.maintenance import reapers
from akgentic.infra.maintenance.models import ResourceKind, ResourceRef
from akgentic.infra.maintenance.reapers import DockerReaper, WeaviateReaper, _parse_docker_age

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------


class _FakeCompleted:
    """Minimal stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _patch_docker(
    monkeypatch: pytest.MonkeyPatch, stdout: str = "", returncode: int = 0
) -> list[list[str]]:
    """Replace ``subprocess.run`` and return the list of argv it receives."""
    seen: list[list[str]] = []

    def _fake_run(argv: list[str], **_kwargs: object) -> _FakeCompleted:
        seen.append(argv)
        return _FakeCompleted(stdout=stdout, returncode=returncode, stderr="boom")

    monkeypatch.setattr(reapers.subprocess, "run", _fake_run)
    return seen


def _row(name: str, container_id: str = "abc123", *, minutes_old: int = 600) -> str:
    """Render one ``docker ps`` row as the reaper's format string produces it."""
    created = datetime.now(UTC) - timedelta(minutes=minutes_old)
    return f"{container_id}\t{name}\t{created.strftime('%Y-%m-%d %H:%M:%S %z')} UTC"


def test_scan_maps_container_names_to_team_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """``sandbox-<team_id>`` is the only thing tying a container to a team."""
    team_id = uuid.uuid4()
    _patch_docker(monkeypatch, stdout=_row(f"sandbox-{team_id}", "cid1"))

    refs = DockerReaper().scan()

    assert len(refs) == 1
    assert refs[0].team_id == str(team_id)
    assert refs[0].detail == "cid1"
    assert refs[0].label == f"sandbox-{team_id}"
    assert refs[0].kind is ResourceKind.DOCKER


def test_scan_skips_containers_whose_suffix_is_not_a_team_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unattributable container is never anyone's orphan."""
    _patch_docker(monkeypatch, stdout=_row("sandbox-not-a-uuid", "cid1"))

    assert DockerReaper().scan() == []


def test_scan_never_lists_running_containers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reaping a live sandbox under an operator is worse than leaking it."""
    seen = _patch_docker(monkeypatch)

    DockerReaper().scan()

    argv = seen[0]
    assert "status=exited" in argv
    assert "status=created" in argv
    assert "status=dead" in argv
    assert "status=running" not in argv


def test_scan_carries_the_container_age(monkeypatch: pytest.MonkeyPatch) -> None:
    """The grace period needs an age, and Docker is the backend that has one."""
    _patch_docker(monkeypatch, stdout=_row(f"sandbox-{uuid.uuid4()}", minutes_old=120))

    age = DockerReaper().scan()[0].age_seconds

    assert age is not None
    assert 7000 < age < 7400


def test_purge_removes_the_container_and_its_anonymous_volumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``-v`` is safe here: ``/workspace`` is a bind mount, never anonymous."""
    seen = _patch_docker(monkeypatch)
    ref = ResourceRef(
        kind=ResourceKind.DOCKER, team_id=str(uuid.uuid4()), detail="cid1", label="sandbox-x"
    )

    assert DockerReaper().purge(ref) == 1
    assert seen[0] == ["docker", "rm", "--volumes", "cid1"]


def test_a_nonzero_docker_exit_raises_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """The driver turns this into an "unavailable" report, not a crash."""
    _patch_docker(monkeypatch, returncode=1)

    with pytest.raises(OSError, match="failed"):
        DockerReaper().scan()


def test_a_missing_docker_client_raises_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host without Docker sweeps its other backends and reports this one."""

    def _raise(_argv: list[str], **_kwargs: object) -> _FakeCompleted:
        raise FileNotFoundError

    monkeypatch.setattr(reapers.subprocess, "run", _raise)

    with pytest.raises(OSError, match="not found"):
        DockerReaper().scan()


def test_a_docker_timeout_raises_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wedged daemon must not wedge the sweep."""

    def _raise(_argv: list[str], **_kwargs: object) -> _FakeCompleted:
        raise subprocess.TimeoutExpired(cmd="docker", timeout=60)

    monkeypatch.setattr(reapers.subprocess, "run", _raise)

    with pytest.raises(OSError, match="timed out"):
        DockerReaper().scan()


@pytest.mark.parametrize("raw", ["", "garbage", "not-a-date 25:99:99 +0000 UTC"])
def test_an_unparseable_timestamp_yields_no_age(raw: str) -> None:
    """Unknown age is ``None`` — the grace period then simply does not apply."""
    assert _parse_docker_age(raw) is None


# ---------------------------------------------------------------------------
# Weaviate
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


class _FakeData:
    """Delete surface that reports a shrinking number of successes."""

    def __init__(self, batches: list[int]) -> None:
        self._batches = list(batches)
        self.calls = 0

    def delete_many(self, *, where: object) -> object:  # noqa: ARG002 - shape only
        """Return the next configured batch size, then zero forever."""
        self.calls += 1
        count = self._batches.pop(0) if self._batches else 0
        return type("Result", (), {"successful": count})()


class _FakeCollection:
    """One Weaviate collection with canned aggregate, iterator and data."""

    def __init__(
        self,
        groups: list[tuple[str, int]] | None,
        objects: list[str] | None = None,
        batches: list[int] | None = None,
    ) -> None:
        self.aggregate = _FakeAggregate(groups)
        self.data = _FakeData(batches or [])
        self._objects = objects or []

    def iterator(self, *, return_properties: list[str]) -> list[object]:
        """Yield objects carrying only the ``team_id`` property."""
        prop = return_properties[0]
        return [type("Obj", (), {"properties": {prop: value}})() for value in self._objects]


class _FakeCollections:
    """The cluster's collection registry."""

    def __init__(self, collections: dict[str, _FakeCollection]) -> None:
        self._collections = collections
        self.closed = False

    def list_all(self) -> dict[str, object]:
        """Return every collection name in the cluster."""
        return dict.fromkeys(self._collections, object())

    def get(self, name: str) -> _FakeCollection:
        """Return one collection by name."""
        return self._collections[name]


class _FakeClient:
    """A Weaviate client exposing only what the reaper touches."""

    def __init__(self, collections: dict[str, _FakeCollection]) -> None:
        self.collections = _FakeCollections(collections)
        self.closed = False

    def close(self) -> None:
        """Record the disconnect."""
        self.closed = True


def _reaper_over(collections: dict[str, _FakeCollection]) -> WeaviateReaper:
    """Build a ``WeaviateReaper`` around a fake client, skipping the connect."""
    reaper = object.__new__(WeaviateReaper)
    reaper._client = _FakeClient(collections)  # noqa: SLF001 - constructing the fake
    return reaper


def test_scan_emits_one_reference_per_collection_and_team() -> None:
    """A delete is per ``(collection, team)``, so a reference must be too."""
    reaper = _reaper_over(
        {
            "Documents": _FakeCollection([("team-a", 3), ("team-b", 5)]),
            "Chunks": _FakeCollection([("team-a", 2)]),
        }
    )

    refs = sorted(reaper.scan(), key=lambda ref: ref.label)

    assert [(ref.detail, ref.team_id, ref.size_hint) for ref in refs] == [
        ("Chunks", "team-a", 2),
        ("Documents", "team-a", 3),
        ("Documents", "team-b", 5),
    ]


def test_scan_falls_back_to_an_object_walk_when_aggregation_is_refused() -> None:
    """An older cluster must still be swept — silently reporting none is worse."""
    reaper = _reaper_over({"Documents": _FakeCollection(None, objects=["t1", "t1", "t2"])})

    counts = {ref.team_id: ref.size_hint for ref in reaper.scan()}

    assert counts == {"t1": 2, "t2": 1}


def test_objects_with_no_team_id_are_never_attributed() -> None:
    """An unattributable object belongs to no team and is never an orphan."""
    reaper = _reaper_over({"Documents": _FakeCollection(None, objects=["t1", "", "t1"])})

    assert {ref.team_id: ref.size_hint for ref in reaper.scan()} == {"t1": 2}


def test_purge_loops_until_a_pass_deletes_nothing() -> None:
    """Weaviate caps one ``delete_many``; a large team needs several passes."""
    collection = _FakeCollection([], batches=[10_000, 10_000, 250])
    reaper = _reaper_over({"Documents": collection})
    ref = ResourceRef(
        kind=ResourceKind.WEAVIATE,
        team_id="team-a",
        detail="Documents",
        label="Documents/team-a",
    )

    assert reaper.purge(ref) == 20_250
    assert collection.data.calls == 4


def test_close_disconnects_the_client() -> None:
    """A sweep opens one connection and must not leave it open."""
    reaper = _reaper_over({})
    reaper.close()

    assert reaper._client.closed is True  # noqa: SLF001 - asserting on the fake


def test_the_team_id_property_is_read_from_the_write_side() -> None:
    """Restating the property name here would let a rename silently no-op."""
    from akgentic.tool.vector_store.weaviate import TEAM_ID_PROPERTY

    assert reapers._team_id_property() == TEAM_ID_PROPERTY  # noqa: SLF001 - unit under test
