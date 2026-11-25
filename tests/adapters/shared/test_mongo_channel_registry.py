"""Tests for ``adapters.shared.mongo_channel_registry`` (mongomock, no server)."""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any

import pytest

pytest.importorskip("mongomock")

from akgentic.infra.adapters.shared.mongo_channel_registry import (  # noqa: E402
    MongoChannelRegistry,
)
from akgentic.infra.protocols.channels import (  # noqa: E402
    ChannelAddress,
    ChannelBinding,
    ChannelRegistry,
)

_TEST_DB = "test_akgentic_infra"


@pytest.fixture
def mongo_client() -> Any:
    # ``Any``: mongomock stands in for pymongo and is not a subclass of it.
    import mongomock

    return mongomock.MongoClient()


@pytest.fixture
def mongo_db(mongo_client: Any) -> Any:
    # ``Any``: mongomock stands in for pymongo and is not a subclass of it.
    return mongo_client[_TEST_DB]


@pytest.fixture
def registry(mongo_client: Any) -> MongoChannelRegistry:
    return MongoChannelRegistry(mongo_client[_TEST_DB])


def _binding(
    channel: str = "slack",
    user: str = "U1",
    team_id: uuid.UUID | None = None,
    agent_name: str = "@Human",
) -> ChannelBinding:
    return ChannelBinding(
        channel=channel,
        channel_user_id=user,
        team_id=team_id or uuid.uuid4(),
        agent_name=agent_name,
    )


def test_satisfies_the_channel_registry_protocol(registry: MongoChannelRegistry) -> None:
    assert isinstance(registry, ChannelRegistry)


async def test_register_then_find_binding_round_trips_the_whole_binding(
    registry: MongoChannelRegistry,
) -> None:
    """G3 — every field survives, metadata included."""
    binding = _binding().model_copy(update={"metadata": {"thread": "t-9", "n": 3}})
    await registry.register(binding)
    assert await registry.find_binding(ChannelAddress(channel="slack", channel_user_id="U1")) == (
        binding
    )


class _BindingWithExtraField(ChannelBinding):
    extra_field: str = "sentinel"


async def test_register_stores_a_field_it_has_never_heard_of(
    registry: MongoChannelRegistry,
    mongo_db: Any,
) -> None:
    """G3 — the whole binding is dumped; an enumerated ``$set`` would drop this field."""
    binding = _BindingWithExtraField(
        channel="slack", channel_user_id="U1", team_id=uuid.uuid4(), agent_name="@Human"
    )
    await registry.register(binding)
    doc = mongo_db["channel_users"].find_one({"channel": "slack", "channel_user_id": "U1"})
    assert doc["extra_field"] == "sentinel"


async def test_find_binding_unmapped_returns_none(registry: MongoChannelRegistry) -> None:
    assert await registry.find_binding(ChannelAddress(channel="slack", channel_user_id="x")) is None


async def test_register_same_address_replaces_the_binding(
    registry: MongoChannelRegistry,
    mongo_db: Any,
) -> None:
    old, new = _binding(), _binding()
    await registry.register(old)
    await registry.register(new)
    assert await registry.find_binding(old) == new
    assert registry.find_binding_sync(old.team_id, old.agent_name) is None
    assert mongo_db["channel_users"].count_documents({}) == 1


async def test_deregister_removes_the_binding_and_is_idempotent(
    registry: MongoChannelRegistry,
) -> None:
    binding = _binding()
    await registry.register(binding)
    await registry.deregister(binding)
    await registry.deregister(binding)
    assert await registry.find_binding(binding) is None
    assert registry.find_binding_sync(binding.team_id, binding.agent_name) is None


async def test_deregister_team_releases_every_channel_and_only_that_team(
    registry: MongoChannelRegistry,
) -> None:
    """G5 — one team bound on two channels loses both; another team keeps its binding."""
    team_id = uuid.uuid4()
    slack, whatsapp = _binding("slack", "U1", team_id), _binding("whatsapp", "U1", team_id)
    other = _binding("slack", "U2")
    for binding in (slack, whatsapp, other):
        await registry.register(binding)

    await registry.deregister_team(team_id)

    assert await registry.find_binding(slack) is None
    assert await registry.find_binding(whatsapp) is None
    assert await registry.find_binding(other) == other

    await registry.deregister_team(team_id)
    assert await registry.find_binding(other) == other


async def test_find_binding_sync_hit_and_miss(registry: MongoChannelRegistry) -> None:
    binding = _binding()
    await registry.register(binding)
    assert registry.find_binding_sync(binding.team_id, binding.agent_name) == binding
    assert registry.find_binding_sync(binding.team_id, "@Other") is None
    assert registry.find_binding_sync(uuid.uuid4(), binding.agent_name) is None


async def test_a_second_replica_sees_the_binding(mongo_client: Any) -> None:
    """G4 — a registry instance that never saw ``register`` answers from the collection."""
    binding = _binding()
    await MongoChannelRegistry(mongo_client[_TEST_DB]).register(binding)
    replica = MongoChannelRegistry(mongo_client[_TEST_DB])
    assert replica.find_binding_sync(binding.team_id, binding.agent_name) == binding


async def test_legacy_document_reads_as_absent(
    registry: MongoChannelRegistry,
    mongo_db: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """G6 — a binding written before ``agent_name`` existed is ignored, with a warning."""
    team_id = uuid.uuid4()
    mongo_db["channel_users"].insert_one(
        {"channel": "slack", "channel_user_id": "U1", "team_id": str(team_id)}
    )
    with caplog.at_level(logging.WARNING):
        found = await registry.find_binding(ChannelAddress(channel="slack", channel_user_id="U1"))
    assert found is None
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
        "Ignoring unreadable channel binding for channel=slack, channel_user_id=U1"
    ]

    await registry.register(_binding(team_id=team_id))
    assert await registry.find_binding(ChannelAddress(channel="slack", channel_user_id="U1")) == (
        _binding(team_id=team_id)
    )


def _index_keys(info: dict[str, Any]) -> dict[tuple[str, ...], dict[str, Any]]:
    """``index_information()`` keyed by its field tuple."""
    return {tuple(field for field, _ in spec.get("key", [])): spec for spec in info.values()}


async def test_unique_index_exists_on_channel_pair(
    registry: MongoChannelRegistry,
    mongo_db: Any,
) -> None:
    await registry.register(_binding())
    info = mongo_db["channel_users"].index_information()
    matching = [
        spec
        for spec in info.values()
        if tuple(field for field, _ in spec.get("key", [])) == ("channel", "channel_user_id")
        and spec.get("unique") is True
    ]
    assert matching, f"expected unique compound index on (channel, channel_user_id); got {info!r}"


async def test_team_agent_index_serves_the_sync_read(
    registry: MongoChannelRegistry,
    mongo_db: Any,
) -> None:
    """AC 3 — the outbound point read goes through an index, never a scan."""
    await registry.register(_binding())
    indexes = _index_keys(mongo_db["channel_users"].index_information())
    assert ("team_id", "agent_name") in indexes, indexes
    assert not indexes[("team_id", "agent_name")].get("unique", False)
    assert indexes[("channel", "channel_user_id")].get("unique") is True


def test_index_not_created_before_first_operation(
    registry: MongoChannelRegistry,
    mongo_db: Any,
) -> None:
    # Touch only the collection's index_information — never the registry — to
    # avoid triggering the lazy _ensure_index. Only the default _id_ index
    # should be present.
    info = mongo_db["channel_users"].index_information()
    assert set(info.keys()) <= {"_id_"}


class _ThreadRecordingCollection:
    """Forwards the driver calls the registry makes, recording the calling thread."""

    def __init__(self, inner: Any) -> None:
        # ``Any``: mongomock's ``Collection`` stands in for pymongo's.
        self._inner = inner
        self.calls: list[tuple[str, int]] = []

    def _record(self, name: str) -> None:
        self.calls.append((name, threading.get_ident()))

    def create_index(self, *args: Any, **kwargs: Any) -> Any:
        self._record("create_index")
        return self._inner.create_index(*args, **kwargs)

    def update_one(self, *args: Any, **kwargs: Any) -> Any:
        self._record("update_one")
        return self._inner.update_one(*args, **kwargs)

    def find_one(self, *args: Any, **kwargs: Any) -> Any:
        self._record("find_one")
        return self._inner.find_one(*args, **kwargs)

    def delete_one(self, *args: Any, **kwargs: Any) -> Any:
        self._record("delete_one")
        return self._inner.delete_one(*args, **kwargs)

    def delete_many(self, *args: Any, **kwargs: Any) -> Any:
        self._record("delete_many")
        return self._inner.delete_many(*args, **kwargs)


def _recording(registry: MongoChannelRegistry) -> _ThreadRecordingCollection:
    recorder = _ThreadRecordingCollection(registry._collection)
    registry._collection = recorder  # type: ignore[assignment]
    return recorder


async def test_async_methods_run_every_driver_call_off_the_event_loop(
    registry: MongoChannelRegistry,
) -> None:
    """AC 4 — ``register`` / ``find_binding`` / ``deregister`` / ``deregister_team``."""
    recorder = _recording(registry)
    loop_thread = threading.get_ident()
    binding = _binding()

    await registry.register(binding)
    await registry.find_binding(binding)
    await registry.deregister(binding)
    await registry.deregister_team(binding.team_id)

    names = [name for name, _ in recorder.calls]
    assert names == [
        "create_index",
        "create_index",
        "update_one",
        "find_one",
        "delete_one",
        "delete_many",
    ]
    assert all(ident != loop_thread for _, ident in recorder.calls), recorder.calls


async def test_find_binding_sync_runs_on_the_callers_thread(
    registry: MongoChannelRegistry,
) -> None:
    """AC 4 — the sync read is not offloaded: it runs on whichever thread calls it."""
    binding = _binding()
    await registry.register(binding)
    recorder = _recording(registry)
    found: list[ChannelBinding | None] = []
    caller: list[int] = []

    def read() -> None:
        caller.append(threading.get_ident())
        found.append(registry.find_binding_sync(binding.team_id, binding.agent_name))

    thread = threading.Thread(target=read)
    thread.start()
    thread.join(timeout=5)

    assert found == [binding]
    assert recorder.calls == [("find_one", caller[0])]


class _RaisingCollection:
    """A collection whose upsert fails as a racing unique-index write would."""

    def create_index(self, *args: Any, **kwargs: Any) -> str:
        return "ignored"

    def update_one(self, *args: Any, **kwargs: Any) -> None:
        from pymongo.errors import DuplicateKeyError

        raise DuplicateKeyError("E11000 duplicate key error", 11000)


async def test_a_driver_error_keeps_its_type_through_the_offload(
    registry: MongoChannelRegistry,
) -> None:
    """AC 5 — ``asyncio.to_thread`` changes when an exception is raised, not its type."""
    pytest.importorskip("pymongo")
    from pymongo.errors import DuplicateKeyError

    registry._collection = _RaisingCollection()  # type: ignore[assignment]
    with pytest.raises(DuplicateKeyError):
        await registry.register(_binding())
