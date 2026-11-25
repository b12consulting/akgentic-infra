"""Tests for ``adapters.shared.postgres_channel_registry``.

The DB-less suite replaces ``nagra.Transaction`` with a recording fake and pins
the SQL, the parameters, the threads and the stored document. The live suite
runs the same behaviour against a real database, only when
``AKGENTIC_TEST_POSTGRES_DSN`` is set; it touches only rows it wrote.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("nagra")
pytest.importorskip("psycopg")

import nagra  # noqa: E402
import psycopg  # noqa: E402

from akgentic.infra.adapters.shared import postgres_channel_registry as module  # noqa: E402
from akgentic.infra.adapters.shared.postgres_channel_registry import (  # noqa: E402
    PostgresChannelRegistry,
    init_db,
)
from akgentic.infra.protocols.channels import (  # noqa: E402
    ChannelAddress,
    ChannelBinding,
    ChannelRegistry,
)

DSN = "postgresql://akgentic:s3cret-pw@localhost:5432/akgentic"


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


class _BindingWithExtraField(ChannelBinding):
    extra_field: str = "sentinel"


# -- the recording fake -------------------------------------------------------


class _Recorder:
    """What the fake transactions saw, shared across every instance."""

    def __init__(self) -> None:
        self.dsns: list[str] = []
        self.calls: list[tuple[str, tuple[Any, ...], int]] = []
        self.rows: list[tuple[Any, ...] | None] = []
        self.error: Exception | None = None

    @property
    def statements(self) -> list[str]:
        return [sql for sql, _, _ in self.calls]


class _FakeCursor:
    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._recorder.rows.pop(0) if self._recorder.rows else None


def _fake_transaction(recorder: _Recorder) -> type:
    class FakeTransaction:
        def __init__(self, dsn: str, *args: Any, **kwargs: Any) -> None:
            recorder.dsns.append(dsn)

        def __enter__(self) -> FakeTransaction:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def execute(self, sql: str, params: tuple[Any, ...] = ()) -> _FakeCursor:
            recorder.calls.append((sql, tuple(params), threading.get_ident()))
            if recorder.error is not None:
                raise recorder.error
            return _FakeCursor(recorder)

    return FakeTransaction


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr("nagra.Transaction", _fake_transaction(rec))
    return rec


@pytest.fixture
def registry(recorder: _Recorder) -> PostgresChannelRegistry:
    return PostgresChannelRegistry(DSN)


# -- AC 1: the class ------------------------------------------------------------


class TestConstruction:
    def test_satisfies_the_channel_registry_protocol_by_explicit_base(
        self, registry: PostgresChannelRegistry
    ) -> None:
        assert isinstance(registry, ChannelRegistry)
        assert ChannelRegistry in PostgresChannelRegistry.__bases__

    def test_construction_stores_the_dsn_and_opens_nothing(self, recorder: _Recorder) -> None:
        registry = PostgresChannelRegistry(DSN)
        assert registry._dsn == DSN
        assert recorder.dsns == []
        assert recorder.calls == []


# -- AC 2: the schema -----------------------------------------------------------


class TestSchema:
    def test_the_schema_file_exists_where_the_module_resolves_it(self) -> None:
        assert module._SCHEMA_PATH.is_file()
        assert module._SCHEMA_PATH.parent == Path(module.__file__).parent

    def test_the_schema_declares_one_table_keyed_by_the_address(self) -> None:
        schema = nagra.Schema()
        schema.load_toml(module._SCHEMA_PATH)
        assert set(schema.tables) == {"channel_users"}
        assert schema.tables["channel_users"].natural_key == ["channel", "channel_user_id"]


# -- AC 3: init_db ----------------------------------------------------------------


class TestInitDb:
    def test_creates_the_table_on_a_private_schema_then_the_team_agent_index(
        self, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        schemas: list[Any] = []

        def create_tables(self: Any, trn: Any = None) -> None:
            schemas.append(self)
            recorder.calls.append(("<create_tables>", (), threading.get_ident()))

        monkeypatch.setattr(nagra.Schema, "create_tables", create_tables)
        default_tables = set(nagra.Schema.default.tables)

        init_db(DSN)

        assert recorder.dsns == [DSN]
        assert len(schemas) == 1
        (schema,) = schemas
        assert schema is not nagra.Schema.default
        assert set(schema.tables) == {"channel_users"}
        assert set(nagra.Schema.default.tables) == default_tables
        assert recorder.statements == [
            "<create_tables>",
            "CREATE INDEX IF NOT EXISTS channel_users_team_agent_idx "
            "ON channel_users (team_id, agent_name)",
        ]

    def test_runs_twice_without_loading_into_a_shared_schema(
        self, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second load into ``Schema.default`` would raise ``Table … already in schema!``."""
        monkeypatch.setattr(nagra.Schema, "create_tables", lambda self, trn=None: None)
        init_db(DSN)
        init_db(DSN)
        assert len(recorder.calls) == 2


# -- AC 4: one statement per method, every value bound ----------------------------


_CALLER_VALUES = ("slack-chan", "U-caller-42", "@Agent-caller")


class TestSql:
    async def test_register_upserts_on_the_address(
        self, registry: PostgresChannelRegistry, recorder: _Recorder
    ) -> None:
        channel, user, agent = _CALLER_VALUES
        binding = _binding(channel, user, agent_name=agent)
        await registry.register(binding)

        ((sql, params, _),) = recorder.calls
        assert sql == (
            "INSERT INTO channel_users (channel, channel_user_id, team_id, agent_name, data) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (channel, channel_user_id) DO UPDATE SET "
            "team_id = EXCLUDED.team_id, agent_name = EXCLUDED.agent_name, data = EXCLUDED.data"
        )
        assert params[:4] == (channel, user, str(binding.team_id), agent)
        assert json.loads(params[4]) == binding.model_dump(mode="json")

    async def test_find_binding_reads_by_the_address(
        self, registry: PostgresChannelRegistry, recorder: _Recorder
    ) -> None:
        channel, user, _ = _CALLER_VALUES
        await registry.find_binding(ChannelAddress(channel=channel, channel_user_id=user))
        assert [(sql, params) for sql, params, _ in recorder.calls] == [
            (
                "SELECT data FROM channel_users WHERE channel = %s AND channel_user_id = %s",
                (channel, user),
            )
        ]

    def test_find_binding_sync_is_one_point_read_on_team_and_agent(
        self, registry: PostgresChannelRegistry, recorder: _Recorder
    ) -> None:
        team_id = uuid.uuid4()
        agent = _CALLER_VALUES[2]
        registry.find_binding_sync(team_id, agent)
        ((sql, params, _),) = recorder.calls
        assert sql == (
            "SELECT data FROM channel_users WHERE team_id = %s AND agent_name = %s LIMIT 1"
        )
        where = sql.split("WHERE", 1)[1]
        assert "team_id = %s" in where
        assert "agent_name = %s" in where
        assert params == (str(team_id), agent)

    async def test_deregister_deletes_by_the_address(
        self, registry: PostgresChannelRegistry, recorder: _Recorder
    ) -> None:
        channel, user, _ = _CALLER_VALUES
        await registry.deregister(ChannelAddress(channel=channel, channel_user_id=user))
        assert [(sql, params) for sql, params, _ in recorder.calls] == [
            (
                "DELETE FROM channel_users WHERE channel = %s AND channel_user_id = %s",
                (channel, user),
            )
        ]

    async def test_deregister_team_deletes_by_team(
        self, registry: PostgresChannelRegistry, recorder: _Recorder
    ) -> None:
        team_id = uuid.uuid4()
        await registry.deregister_team(team_id)
        assert [(sql, params) for sql, params, _ in recorder.calls] == [
            ("DELETE FROM channel_users WHERE team_id = %s", (str(team_id),))
        ]

    async def test_no_caller_value_reaches_the_sql_text(
        self, registry: PostgresChannelRegistry, recorder: _Recorder
    ) -> None:
        channel, user, agent = _CALLER_VALUES
        binding = _binding(channel, user, agent_name=agent)
        await registry.register(binding)
        await registry.find_binding(binding)
        await registry.deregister(binding)
        await registry.deregister_team(binding.team_id)
        registry.find_binding_sync(binding.team_id, agent)

        assert len(recorder.calls) == 5
        for sql in recorder.statements:
            for value in (*_CALLER_VALUES, str(binding.team_id)):
                assert value not in sql, (value, sql)


# -- AC 5: the whole binding is stored --------------------------------------------


async def test_register_stores_a_field_it_has_never_heard_of(
    registry: PostgresChannelRegistry, recorder: _Recorder
) -> None:
    """An enumerated dump of the four known fields would drop ``extra_field``."""
    binding = _BindingWithExtraField(
        channel="slack", channel_user_id="U1", team_id=uuid.uuid4(), agent_name="@Human"
    )
    await registry.register(binding)
    ((_, params, _),) = recorder.calls
    assert json.loads(params[4])["extra_field"] == "sentinel"


# -- AC 6: reading back -----------------------------------------------------------


class TestReadingBack:
    @pytest.mark.parametrize("encode", [lambda doc: doc, json.dumps], ids=["decoded", "string"])
    async def test_a_row_reads_back_as_the_whole_binding(
        self,
        registry: PostgresChannelRegistry,
        recorder: _Recorder,
        encode: Callable[[dict[str, Any]], Any],
    ) -> None:
        binding = _binding().model_copy(update={"metadata": {"thread": "t-9", "n": 3}})
        stored = encode(binding.model_dump(mode="json"))
        recorder.rows = [(stored,), (stored,)]
        assert await registry.find_binding(binding) == binding
        assert registry.find_binding_sync(binding.team_id, binding.agent_name) == binding

    async def test_no_row_reads_as_none(
        self, registry: PostgresChannelRegistry, recorder: _Recorder
    ) -> None:
        address = ChannelAddress(channel="slack", channel_user_id="x")
        assert await registry.find_binding(address) is None
        assert registry.find_binding_sync(uuid.uuid4(), "@Human") is None
        assert len(recorder.calls) == 2

    async def test_an_unreadable_document_reads_as_none_with_one_warning(
        self,
        registry: PostgresChannelRegistry,
        recorder: _Recorder,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        recorder.rows = [
            ({"channel": "slack", "channel_user_id": "U1", "team_id": str(uuid.uuid4())},)
        ]
        with caplog.at_level(logging.WARNING):
            found = await registry.find_binding(
                ChannelAddress(channel="slack", channel_user_id="U1")
            )
        assert found is None
        assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
            "Ignoring unreadable channel binding for channel=slack, channel_user_id=U1"
        ]

    async def test_a_document_that_is_not_an_object_reads_as_none(
        self,
        registry: PostgresChannelRegistry,
        recorder: _Recorder,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        recorder.rows = [("[1, 2]",)]
        with caplog.at_level(logging.WARNING):
            assert registry.find_binding_sync(uuid.uuid4(), "@Human") is None
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    async def test_a_string_that_is_not_json_reads_as_none_rather_than_raising(
        self,
        registry: PostgresChannelRegistry,
        recorder: _Recorder,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """psycopg returns a stored JSON string scalar (``'"abc"'::json``) as a bare ``str``."""
        recorder.rows = [("abc",), ("abc",)]
        with caplog.at_level(logging.WARNING):
            assert registry.find_binding_sync(uuid.uuid4(), "@Human") is None
            assert (
                await registry.find_binding(ChannelAddress(channel="slack", channel_user_id="U1"))
                is None
            )
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


# -- AC 8: threads ------------------------------------------------------------------


class TestThreads:
    async def test_async_methods_run_every_statement_off_the_event_loop(
        self, registry: PostgresChannelRegistry, recorder: _Recorder
    ) -> None:
        loop_thread = threading.get_ident()
        binding = _binding()

        await registry.register(binding)
        await registry.find_binding(binding)
        await registry.deregister(binding)
        await registry.deregister_team(binding.team_id)

        assert len(recorder.calls) == 4
        assert all(ident != loop_thread for _, _, ident in recorder.calls), recorder.calls

    def test_find_binding_sync_runs_on_the_callers_thread(
        self, registry: PostgresChannelRegistry, recorder: _Recorder
    ) -> None:
        binding = _binding()
        recorder.rows = [(binding.model_dump(mode="json"),)]
        found: list[ChannelBinding | None] = []
        caller: list[int] = []

        def read() -> None:
            caller.append(threading.get_ident())
            found.append(registry.find_binding_sync(binding.team_id, binding.agent_name))

        thread = threading.Thread(target=read)
        thread.start()
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert found == [binding]
        assert [ident for _, _, ident in recorder.calls] == [caller[0]]

    async def test_a_driver_error_keeps_its_type_through_the_offload(
        self, registry: PostgresChannelRegistry, recorder: _Recorder
    ) -> None:
        recorder.error = psycopg.OperationalError("connection refused")
        with pytest.raises(psycopg.OperationalError):
            await registry.register(_binding())


# -- AC 14: live, opt-in -------------------------------------------------------------


_LIVE_DSN = os.environ.get("AKGENTIC_TEST_POSTGRES_DSN")

live = pytest.mark.skipif(not _LIVE_DSN, reason="AKGENTIC_TEST_POSTGRES_DSN is not set")


@pytest.fixture
def live_dsn() -> str:
    assert _LIVE_DSN is not None
    return _LIVE_DSN


@pytest.fixture
def channels(live_dsn: str) -> Generator[Iterator[str], None, None]:
    """Fresh channel names for this test; their rows, and only theirs, are deleted after."""
    init_db(live_dsn)
    init_db(live_dsn)
    used: list[str] = []

    def fresh() -> Iterator[str]:
        while True:
            name = f"test-{uuid.uuid4().hex}"
            used.append(name)
            yield name

    yield fresh()
    with nagra.Transaction(live_dsn) as trn:
        for name in used:
            trn.execute("DELETE FROM channel_users WHERE channel = %s", (name,))


@live
class TestLive:
    async def test_round_trip_keeps_the_metadata(
        self, live_dsn: str, channels: Iterator[str]
    ) -> None:
        registry = PostgresChannelRegistry(live_dsn)
        binding = _binding(next(channels)).model_copy(update={"metadata": {"thread": "t-9"}})
        await registry.register(binding)
        assert await registry.find_binding(binding) == binding

    async def test_same_address_replaces(self, live_dsn: str, channels: Iterator[str]) -> None:
        registry = PostgresChannelRegistry(live_dsn)
        channel = next(channels)
        old, new = _binding(channel), _binding(channel)
        await registry.register(old)
        await registry.register(new)
        assert await registry.find_binding(old) == new
        assert registry.find_binding_sync(old.team_id, old.agent_name) is None

    async def test_deregister_is_idempotent(self, live_dsn: str, channels: Iterator[str]) -> None:
        registry = PostgresChannelRegistry(live_dsn)
        binding = _binding(next(channels))
        await registry.register(binding)
        await registry.deregister(binding)
        await registry.deregister(binding)
        assert await registry.find_binding(binding) is None

    async def test_deregister_team_spans_channels_and_spares_other_teams(
        self, live_dsn: str, channels: Iterator[str]
    ) -> None:
        registry = PostgresChannelRegistry(live_dsn)
        team_id = uuid.uuid4()
        first = _binding(next(channels), "U1", team_id)
        second = _binding(next(channels), "U1", team_id)
        other = _binding(next(channels), "U2")
        for binding in (first, second, other):
            await registry.register(binding)

        await registry.deregister_team(team_id)

        assert await registry.find_binding(first) is None
        assert await registry.find_binding(second) is None
        assert await registry.find_binding(other) == other

    async def test_sync_hit_and_miss(self, live_dsn: str, channels: Iterator[str]) -> None:
        registry = PostgresChannelRegistry(live_dsn)
        binding = _binding(next(channels))
        await registry.register(binding)
        assert registry.find_binding_sync(binding.team_id, binding.agent_name) == binding
        assert registry.find_binding_sync(binding.team_id, "@Other") is None

    async def test_a_second_instance_sees_the_binding(
        self, live_dsn: str, channels: Iterator[str]
    ) -> None:
        binding = _binding(next(channels))
        await PostgresChannelRegistry(live_dsn).register(binding)
        replica = PostgresChannelRegistry(live_dsn)
        assert replica.find_binding_sync(binding.team_id, binding.agent_name) == binding

    async def test_an_invalid_stored_document_reads_as_none_with_the_warning(
        self, live_dsn: str, channels: Iterator[str], caplog: pytest.LogCaptureFixture
    ) -> None:
        channel = next(channels)
        team_id = str(uuid.uuid4())
        with nagra.Transaction(live_dsn) as trn:
            trn.execute(
                "INSERT INTO channel_users (channel, channel_user_id, team_id, agent_name, data) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    channel,
                    "U1",
                    team_id,
                    "@Human",
                    json.dumps({"channel": channel, "channel_user_id": "U1", "team_id": team_id}),
                ),
            )
        registry = PostgresChannelRegistry(live_dsn)
        with caplog.at_level(logging.WARNING):
            found = await registry.find_binding(
                ChannelAddress(channel=channel, channel_user_id="U1")
            )
        assert found is None
        assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
            f"Ignoring unreadable channel binding for channel={channel}, channel_user_id=U1"
        ]

    def test_both_indexes_exist(self, live_dsn: str, channels: Iterator[str]) -> None:
        with nagra.Transaction(live_dsn) as trn:
            rows = trn.execute(
                "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = %s",
                ("channel_users",),
            ).fetchall()
        defs = dict(rows)
        assert "UNIQUE" in defs["channel_users_idx"]
        assert defs["channel_users_idx"].endswith("(channel, channel_user_id)")
        assert "UNIQUE" not in defs["channel_users_team_agent_idx"]
        assert defs["channel_users_team_agent_idx"].endswith("(team_id, agent_name)")
