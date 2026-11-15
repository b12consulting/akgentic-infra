"""Unit tests for the PostgreSQL startup steps: single-instance lock and init_db.

No database: ``psycopg`` is replaced by a fake module and the schema calls by
recorders, so these run in CI, which installs ``.[dev]`` without the
``[postgres]`` extra. ``tests/test_postgres_live.py`` exercises the same steps
against a real server when ``AKGENTIC_TEST_POSTGRES_DSN`` is set.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from akgentic.infra import persistence, wiring
from akgentic.infra.adapters.community.postgres_instance_lock import (
    APPLICATION_NAME,
    LOCK_NAME,
    InstanceLockedError,
    PostgresInstanceLock,
)
from akgentic.infra.server.settings import CommunitySettings, PersistenceBackend

DSN = "postgresql://akgentic:s3cret-pw@localhost:5432/akgentic"


class _FakeCursor:
    def __init__(self, row: tuple[object, ...] | None) -> None:
        self._row = row

    def fetchone(self) -> tuple[object, ...] | None:
        return self._row


class _FakeConnection:
    def __init__(self, *, lock_granted: bool, unlock_raises: bool = False) -> None:
        self.lock_granted = lock_granted
        self.unlock_raises = unlock_raises
        self.statements: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    def execute(self, sql: str, params: tuple[object, ...]) -> _FakeCursor:
        self.statements.append((sql, params))
        if "pg_advisory_unlock" in sql and self.unlock_raises:
            raise RuntimeError("connection lost")
        if "pg_try_advisory_lock" in sql:
            return _FakeCursor((self.lock_granted,))
        return _FakeCursor((True,))

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def fake_psycopg(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install a fake ``psycopg`` whose ``connect`` is recorded and configurable."""
    state: dict[str, Any] = {"lock_granted": True, "unlock_raises": False, "calls": []}

    def connect(conninfo: str, **kwargs: Any) -> _FakeConnection:
        state["calls"].append((conninfo, kwargs))
        conn = _FakeConnection(
            lock_granted=state["lock_granted"], unlock_raises=state["unlock_raises"]
        )
        state["last"] = conn
        return conn

    module = types.ModuleType("psycopg")
    module.connect = connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", module)
    return state


class TestPostgresInstanceLock:
    def test_acquire_takes_the_lock_on_a_named_autocommit_connection(
        self, fake_psycopg: dict[str, Any]
    ) -> None:
        lock = PostgresInstanceLock(DSN)
        lock.acquire()

        assert lock.held
        ((conninfo, kwargs),) = fake_psycopg["calls"]
        assert conninfo == DSN
        assert kwargs == {"autocommit": True, "application_name": APPLICATION_NAME}
        sql, params = fake_psycopg["last"].statements[0]
        assert "pg_try_advisory_lock(hashtext(%s))" in sql
        assert params == (LOCK_NAME,)

    def test_acquire_twice_is_a_no_op(self, fake_psycopg: dict[str, Any]) -> None:
        lock = PostgresInstanceLock(DSN)
        lock.acquire()
        lock.acquire()
        assert len(fake_psycopg["calls"]) == 1

    def test_lock_held_elsewhere_raises_and_closes_the_connection(
        self, fake_psycopg: dict[str, Any]
    ) -> None:
        fake_psycopg["lock_granted"] = False
        lock = PostgresInstanceLock(DSN)

        with pytest.raises(InstanceLockedError) as exc_info:
            lock.acquire()

        assert not lock.held
        assert fake_psycopg["last"].closed
        message = str(exc_info.value)
        assert "one server per database" in message
        assert APPLICATION_NAME in message
        assert "s3cret-pw" not in message

    def test_release_unlocks_closes_and_is_idempotent(self, fake_psycopg: dict[str, Any]) -> None:
        lock = PostgresInstanceLock(DSN)
        lock.acquire()
        conn = fake_psycopg["last"]

        lock.release()
        lock.release()

        assert not lock.held
        assert conn.closed
        unlocks = [s for s, _ in conn.statements if "pg_advisory_unlock" in s]
        assert len(unlocks) == 1

    def test_release_still_closes_when_the_unlock_fails(self, fake_psycopg: dict[str, Any]) -> None:
        """Closing the session is what releases the lock; a failed unlock must not stop it."""
        fake_psycopg["unlock_raises"] = True
        lock = PostgresInstanceLock(DSN)
        lock.acquire()

        lock.release()

        assert fake_psycopg["last"].closed
        assert not lock.held

    def test_release_without_acquire_does_nothing(self) -> None:
        PostgresInstanceLock(DSN).release()


class _RecordingLock:
    """Stands in for PostgresInstanceLock inside ``prepare_postgres``."""

    instances: list[_RecordingLock] = []
    fail_acquire: Exception | None = None

    def __init__(self, conn_string: str) -> None:
        self.conn_string = conn_string
        self.acquired = False
        self.released = False
        _RecordingLock.instances.append(self)

    def acquire(self) -> None:
        if _RecordingLock.fail_acquire is not None:
            raise _RecordingLock.fail_acquire
        self.acquired = True

    def release(self) -> None:
        self.released = True


@pytest.fixture()
def recorded(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the lock, the schema step and the extra check in ``persistence``."""
    _RecordingLock.instances = []
    _RecordingLock.fail_acquire = None
    calls: dict[str, Any] = {"init": [], "init_error": None}

    def fake_init(conn_string: str, *, team: bool, catalog: bool) -> None:
        calls["init"].append((conn_string, team, catalog))
        if calls["init_error"] is not None:
            raise calls["init_error"]

    monkeypatch.setattr(persistence, "PostgresInstanceLock", _RecordingLock)
    monkeypatch.setattr(persistence, "_init_schema", fake_init)
    monkeypatch.setattr(persistence, "_require_postgres_modules", lambda: None)
    monkeypatch.delenv("AKGENTIC_PERSISTENCE_BACKEND", raising=False)
    monkeypatch.delenv("AKGENTIC_POSTGRES_CONN_STRING", raising=False)
    monkeypatch.delenv("AKGENTIC_POSTGRES_INIT_DB", raising=False)
    return calls


def _settings(tmp_path: Path, **overrides: Any) -> CommunitySettings:
    return CommunitySettings(
        workspaces_root=tmp_path / "workspaces",
        event_store_path=tmp_path / "event_store",
        catalog_path=tmp_path / "catalog",
        **overrides,
    )


def _pg_settings(tmp_path: Path, **overrides: Any) -> CommunitySettings:
    return _settings(
        tmp_path,
        persistence_backend=PersistenceBackend.POSTGRES,
        postgres_conn_string=DSN,
        **overrides,
    )


class TestPreparePostgres:
    def test_yaml_backend_does_nothing(self, tmp_path: Path, recorded: dict[str, Any]) -> None:
        lock = persistence.prepare_postgres(
            _settings(tmp_path), event_store_from_settings=True, catalog_from_settings=True
        )
        assert lock is None
        assert _RecordingLock.instances == []
        assert recorded["init"] == []

    def test_postgres_takes_the_lock_and_skips_init_by_default(
        self, tmp_path: Path, recorded: dict[str, Any]
    ) -> None:
        lock = persistence.prepare_postgres(
            _pg_settings(tmp_path), event_store_from_settings=True, catalog_from_settings=True
        )
        assert isinstance(lock, _RecordingLock)
        assert lock.acquired and lock.conn_string == DSN
        assert recorded["init"] == []

    def test_init_db_runs_for_both_stores_when_enabled(
        self, tmp_path: Path, recorded: dict[str, Any]
    ) -> None:
        persistence.prepare_postgres(
            _pg_settings(tmp_path, postgres_init_db=True),
            event_store_from_settings=True,
            catalog_from_settings=True,
        )
        assert recorded["init"] == [(DSN, True, True)]

    def test_injected_stores_get_neither_lock_nor_init(
        self, tmp_path: Path, recorded: dict[str, Any]
    ) -> None:
        lock = persistence.prepare_postgres(
            _pg_settings(tmp_path, postgres_init_db=True),
            event_store_from_settings=False,
            catalog_from_settings=False,
        )
        assert lock is None
        assert _RecordingLock.instances == []
        assert recorded["init"] == []

    def test_injected_event_store_means_no_lock_but_catalog_init(
        self, tmp_path: Path, recorded: dict[str, Any]
    ) -> None:
        """The lock guards the team store; with an injected one it is not ours to guard."""
        lock = persistence.prepare_postgres(
            _pg_settings(tmp_path, postgres_init_db=True),
            event_store_from_settings=False,
            catalog_from_settings=True,
        )
        assert lock is None
        assert recorded["init"] == [(DSN, False, True)]

    def test_a_failing_init_releases_the_lock(
        self, tmp_path: Path, recorded: dict[str, Any]
    ) -> None:
        recorded["init_error"] = RuntimeError("permission denied for schema public")
        with pytest.raises(RuntimeError, match="permission denied"):
            persistence.prepare_postgres(
                _pg_settings(tmp_path, postgres_init_db=True),
                event_store_from_settings=True,
                catalog_from_settings=True,
            )
        (lock,) = _RecordingLock.instances
        assert lock.released

    def test_init_db_setting_reads_from_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: dict[str, Any]
    ) -> None:
        monkeypatch.setenv("AKGENTIC_POSTGRES_INIT_DB", "true")
        assert _pg_settings(tmp_path).postgres_init_db is True


class TestWireCommunityLock:
    def test_lock_is_taken_before_assembly_and_released_by_a_shutdown_hook(
        self, tmp_path: Path, recorded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        order: list[str] = []
        fake_services = types.SimpleNamespace(shutdown_hooks=[])

        def fake_assemble(*_: Any, **__: Any) -> Any:
            (lock,) = _RecordingLock.instances
            order.append(f"assemble(lock acquired={lock.acquired})")
            return fake_services

        monkeypatch.setattr(wiring, "_assemble", fake_assemble)

        services = wiring.wire_community(_pg_settings(tmp_path))

        (lock,) = _RecordingLock.instances
        assert order == ["assemble(lock acquired=True)"]
        assert services.shutdown_hooks == [lock.release]
        assert not lock.released

    def test_a_failing_assembly_releases_the_lock(
        self, tmp_path: Path, recorded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken_assemble(*_: Any, **__: Any) -> Any:
            raise RuntimeError("wiring failed")

        monkeypatch.setattr(wiring, "_assemble", broken_assemble)

        with pytest.raises(RuntimeError, match="wiring failed"):
            wiring.wire_community(_pg_settings(tmp_path))

        (lock,) = _RecordingLock.instances
        assert lock.released

    def test_a_locked_database_stops_wiring_before_anything_is_built(
        self, tmp_path: Path, recorded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _RecordingLock.fail_acquire = InstanceLockedError("held by another server")
        assembled: list[bool] = []
        monkeypatch.setattr(wiring, "_assemble", lambda *a, **k: assembled.append(True))

        with pytest.raises(InstanceLockedError):
            wiring.wire_community(_pg_settings(tmp_path))

        assert assembled == []

    def test_yaml_wiring_registers_no_shutdown_hook(self, tmp_path: Path) -> None:
        services = wiring.wire_community(_settings(tmp_path))
        try:
            assert services.shutdown_hooks == []
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)
