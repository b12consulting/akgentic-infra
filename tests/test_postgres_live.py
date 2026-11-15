"""Live PostgreSQL tests for the community startup steps — opt-in.

Skipped unless ``AKGENTIC_TEST_POSTGRES_DSN`` points at a server the tests may
create and drop databases on (a throwaway container), and the ``[postgres]``
extra is installed::

    docker run -d --rm --name pg-test -e POSTGRES_PASSWORD=test -p 55432:5432 postgres:16
    AKGENTIC_TEST_POSTGRES_DSN=postgresql://postgres:test@localhost:55432/postgres \\
        pytest tests/test_postgres_live.py

Every test gets its own freshly created database, dropped afterwards.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

ADMIN_DSN = os.environ.get("AKGENTIC_TEST_POSTGRES_DSN")

pytestmark = pytest.mark.skipif(
    not ADMIN_DSN, reason="AKGENTIC_TEST_POSTGRES_DSN not set (opt-in live Postgres tests)"
)

if ADMIN_DSN:
    psycopg = pytest.importorskip("psycopg")
    pytest.importorskip("nagra")


def _with_database(dsn: str, database: str) -> str:
    base, _, _ = dsn.rpartition("/")
    return f"{base}/{database}"


@pytest.fixture()
def fresh_dsn() -> Iterator[str]:
    """A brand-new empty database, dropped (with its sessions) after the test."""
    assert ADMIN_DSN is not None
    name = f"akgentic_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    try:
        yield _with_database(ADMIN_DSN, name)
    finally:
        with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def _settings(tmp_path: Path, dsn: str, **overrides: object):  # noqa: ANN202
    from akgentic.infra.server.settings import CommunitySettings, PersistenceBackend

    return CommunitySettings(
        workspaces_root=tmp_path / "workspaces",
        event_store_path=tmp_path / "event_store",
        catalog_path=tmp_path / "catalog",
        persistence_backend=PersistenceBackend.POSTGRES,
        postgres_conn_string=dsn,
        description_model="",
        **overrides,
    )


def _tables(dsn: str) -> set[str]:
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    return {r[0] for r in rows}


class TestLockAgainstPostgres:
    def test_second_holder_is_refused_until_the_first_releases(self, fresh_dsn: str) -> None:
        from akgentic.infra.adapters.community.postgres_instance_lock import (
            InstanceLockedError,
            PostgresInstanceLock,
        )

        first, second = PostgresInstanceLock(fresh_dsn), PostgresInstanceLock(fresh_dsn)
        first.acquire()
        try:
            with pytest.raises(InstanceLockedError):
                second.acquire()
        finally:
            first.release()

        second.acquire()
        second.release()

    def test_holder_is_findable_and_a_killed_holder_frees_the_lock(self, fresh_dsn: str) -> None:
        """A crashed server never leaves the database locked."""
        from akgentic.infra.adapters.community.postgres_instance_lock import (
            APPLICATION_NAME,
            PostgresInstanceLock,
        )

        crashed = PostgresInstanceLock(fresh_dsn)
        crashed.acquire()
        with psycopg.connect(fresh_dsn, autocommit=True) as conn:
            (pid,) = conn.execute(
                "SELECT pid FROM pg_stat_activity WHERE application_name = %s "
                "AND datname = current_database()",
                (APPLICATION_NAME,),
            ).fetchone()
            conn.execute("SELECT pg_terminate_backend(%s)", (pid,))

        survivor = PostgresInstanceLock(fresh_dsn)
        survivor.acquire()
        survivor.release()
        crashed.release()  # its connection is already gone; must not raise

    def test_two_databases_do_not_contend(self, fresh_dsn: str) -> None:
        from akgentic.infra.adapters.community.postgres_instance_lock import (
            PostgresInstanceLock,
        )

        assert ADMIN_DSN is not None
        here, elsewhere = PostgresInstanceLock(fresh_dsn), PostgresInstanceLock(ADMIN_DSN)
        here.acquire()
        try:
            elsewhere.acquire()
            elsewhere.release()
        finally:
            here.release()


class TestCommunityServerOnPostgres:
    def test_init_db_at_startup_creates_both_schemas(self, fresh_dsn: str, tmp_path: Path) -> None:
        from akgentic.infra.wiring import wire_community

        assert "event_entries" not in _tables(fresh_dsn)
        services = wire_community(_settings(tmp_path, fresh_dsn, postgres_init_db=True))
        try:
            assert {
                "team_process_entries",
                "event_entries",
                "agent_state_entries",
                "agent_card_entries",
                "catalog_entries",
            } <= _tables(fresh_dsn)
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)
            for hook in services.shutdown_hooks:
                hook()

    def test_second_server_is_refused_and_a_restart_succeeds(
        self, fresh_dsn: str, tmp_path: Path
    ) -> None:
        """The lifespan's shutdown hook releases the lock; a concurrent second server fails."""
        from fastapi.testclient import TestClient

        from akgentic.infra.adapters.community.postgres_instance_lock import (
            InstanceLockedError,
        )
        from akgentic.infra.server.app import create_server_app
        from akgentic.infra.wiring import wire_community

        settings = _settings(tmp_path, fresh_dsn, postgres_init_db=True)
        with TestClient(create_server_app(settings)) as client:
            assert client.get("/teams").status_code == 200
            with pytest.raises(InstanceLockedError):
                wire_community(settings)

        # The first app's lifespan has run its shutdown hooks: a restart works.
        with TestClient(create_server_app(settings)) as client:
            assert client.get("/teams").status_code == 200
