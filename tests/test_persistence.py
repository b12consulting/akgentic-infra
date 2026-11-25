"""Tests for the community-tier persistence switch (YAML, MongoDB or PostgreSQL).

Covers the persistence settings, the per-backend builders ``mongo_stores`` and
``postgres_stores`` and the community ``build_stores`` in
``akgentic.infra.persistence``, and the ``event_store`` / ``catalog_repository``
overrides on ``wire_community``. Nothing here opens a database connection: the
Postgres stores only record the DSN at construction (these tests need the
``[postgres]`` extra but no server), and the Mongo stores run against ``mongomock``.
"""

from __future__ import annotations

import asyncio
import inspect
import subprocess
import sys
import uuid
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest
from akgentic.catalog import YamlEntryRepository
from akgentic.team.models import DeleteMode, TeamStatus
from akgentic.team.repositories.yaml import YamlEventStore
from pydantic import ValidationError

from akgentic.infra import persistence
from akgentic.infra.adapters.community.yaml_channel_registry import YamlChannelRegistry
from akgentic.infra.adapters.shared.channel_dispatcher import ChannelDispatcher
from akgentic.infra.persistence import (
    CATALOG_COLLECTION,
    PersistenceBackendUnavailableError,
    PersistenceStores,
    build_stores,
    mongo_stores,
    postgres_stores,
)
from akgentic.infra.protocols.channels import ChannelBinding, ChannelRegistry
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.settings import CommunitySettings, PersistenceBackend
from akgentic.infra.wiring import wire_community
from tests.conftest import _seed_catalog

DSN = "postgresql://akgentic:s3cret-pw@localhost:5432/akgentic"
MONGODB_URI = "mongodb://akgentic:s3cret-pw@localhost:27017"

# The suite-wide ``_isolate_persistence_backend`` fixture (tests/conftest.py)
# keeps an operator's AKGENTIC_PERSISTENCE_* / connection-string exports out.


def _yaml_settings(tmp_path: Path, **overrides: Any) -> CommunitySettings:
    return CommunitySettings(
        workspaces_root=tmp_path / "workspaces",
        event_store_path=tmp_path / "event_store",
        catalog_path=tmp_path / "catalog",
        **overrides,
    )


def _postgres_settings(tmp_path: Path, **overrides: Any) -> CommunitySettings:
    defaults: dict[str, Any] = {
        "persistence_backend": PersistenceBackend.POSTGRES,
        "postgres_dsn": DSN,
    }
    return _yaml_settings(tmp_path, **{**defaults, **overrides})


def _mongo_settings(tmp_path: Path, **overrides: Any) -> CommunitySettings:
    defaults: dict[str, Any] = {
        "persistence_backend": PersistenceBackend.MONGO,
        "mongodb_uri": MONGODB_URI,
    }
    return _yaml_settings(tmp_path, **{**defaults, **overrides})


class _BuilderSpy:
    """Records each call to a public builder, then delegates to the real one."""

    def __init__(self, real: Callable[..., PersistenceStores]) -> None:
        self._signature = inspect.signature(real)
        self._real = real
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> PersistenceStores:
        # Explicit arguments only: a parameter left at its default is absent.
        self.calls.append(dict(self._signature.bind(*args, **kwargs).arguments))
        return self._real(*args, **kwargs)


class _NoopOffload:
    """A distinct offload, so a test can tell it reached the registry."""

    async def __call__(self, fn: Callable[..., Any], *args: Any) -> Any:
        return fn(*args)


def _spy_on(monkeypatch: pytest.MonkeyPatch, name: str) -> _BuilderSpy:
    spy = _BuilderSpy(getattr(persistence, name))
    monkeypatch.setattr(persistence, name, spy)
    return spy


class TestPersistenceSettings:
    """The persistence fields on CommunitySettings."""

    def test_default_backend_is_yaml_without_connection_strings(self, tmp_path: Path) -> None:
        settings = _yaml_settings(tmp_path)
        assert settings.persistence_backend is PersistenceBackend.YAML
        assert settings.postgres_dsn is None
        assert settings.mongodb_uri is None
        assert settings.mongodb_database_name == "akgentic"

    def test_postgres_backend_and_dsn_read_from_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AKGENTIC_PERSISTENCE_BACKEND", "postgres")
        monkeypatch.setenv("AKGENTIC_POSTGRES_DSN", DSN)
        settings = _yaml_settings(tmp_path)
        assert settings.persistence_backend is PersistenceBackend.POSTGRES
        assert settings.postgres_dsn is not None
        assert settings.postgres_dsn.get_secret_value() == DSN

    def test_mongo_backend_uri_and_database_read_from_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AKGENTIC_PERSISTENCE_BACKEND", "mongo")
        monkeypatch.setenv("AKGENTIC_MONGODB_URI", MONGODB_URI)
        monkeypatch.setenv("AKGENTIC_MONGODB_DATABASE_NAME", "tenant_a")
        settings = _yaml_settings(tmp_path)
        assert settings.persistence_backend is PersistenceBackend.MONGO
        assert settings.mongodb_uri is not None
        assert settings.mongodb_uri.get_secret_value() == MONGODB_URI
        assert settings.mongodb_database_name == "tenant_a"

    def test_postgres_without_a_dsn_fails_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="AKGENTIC_POSTGRES_DSN"):
            _yaml_settings(tmp_path, persistence_backend=PersistenceBackend.POSTGRES)

    def test_mongo_without_a_uri_fails_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="AKGENTIC_MONGODB_URI"):
            _yaml_settings(tmp_path, persistence_backend=PersistenceBackend.MONGO)

    def test_postgres_with_an_empty_dsn_fails_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="must start with 'postgresql://'"):
            _yaml_settings(
                tmp_path, persistence_backend=PersistenceBackend.POSTGRES, postgres_dsn=""
            )

    def test_unknown_backend_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            _yaml_settings(tmp_path, persistence_backend="mongodb")

    @pytest.mark.parametrize(
        "dsn",
        [
            # Nagra dispatches on the exact ``postgresql://`` prefix and echoes
            # any other DSN, password included, in its error.
            "postgres://akgentic:s3cret-pw@localhost/akgentic",
            "mysql://akgentic:s3cret-pw@localhost/akgentic",
        ],
    )
    def test_non_postgresql_scheme_is_rejected_without_echoing_the_dsn(
        self, tmp_path: Path, dsn: str
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _yaml_settings(tmp_path, postgres_dsn=dsn)
        assert "s3cret-pw" not in str(exc_info.value)

    @pytest.mark.parametrize(
        "uri",
        [
            "http://akgentic:s3cret-pw@localhost:27017",
            "postgresql://akgentic:s3cret-pw@localhost/akgentic",
        ],
    )
    def test_non_mongodb_scheme_is_rejected_without_echoing_the_uri(
        self, tmp_path: Path, uri: str
    ) -> None:
        with pytest.raises(ValidationError, match="must start with 'mongodb://'") as exc_info:
            _yaml_settings(tmp_path, mongodb_uri=uri)
        assert "s3cret-pw" not in str(exc_info.value)

    def test_mongodb_srv_scheme_is_accepted(self, tmp_path: Path) -> None:
        uri = "mongodb+srv://akgentic:s3cret-pw@cluster.example.net"
        settings = _mongo_settings(tmp_path, mongodb_uri=uri)
        assert settings.mongodb_uri is not None
        assert settings.mongodb_uri.get_secret_value() == uri

    @pytest.mark.parametrize("name", ["", "has.dot", "has space", "a" * 64])
    def test_invalid_mongodb_database_name_is_rejected(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(ValidationError):
            _mongo_settings(tmp_path, mongodb_database_name=name)

    @pytest.mark.parametrize("factory", [_postgres_settings, _mongo_settings])
    def test_connection_string_is_a_secret_in_repr(self, tmp_path: Path, factory: Any) -> None:
        settings = factory(tmp_path)
        assert "s3cret-pw" not in repr(settings)
        assert "s3cret-pw" not in str(settings.model_dump())

    def test_connection_strings_are_allowed_alongside_the_yaml_backend(
        self, tmp_path: Path
    ) -> None:
        """A connection string left in the environment must not stop a YAML boot."""
        settings = _yaml_settings(tmp_path, postgres_dsn=DSN, mongodb_uri=MONGODB_URI)
        assert settings.persistence_backend is PersistenceBackend.YAML


def _mongomock_database(name: str = "tenant_a") -> Any:
    # ``Any``: mongomock's ``Database`` stands in for pymongo's and is not a subclass of it.
    pytest.importorskip("pymongo")
    mongomock = pytest.importorskip("mongomock")
    return mongomock.MongoClient()[name]


class TestMongoStores:
    """``mongo_stores`` builds both Mongo stores in the caller's database (mongomock)."""

    @pytest.fixture()
    def db(self) -> Any:
        # ``Any``: see ``_mongomock_database``.
        return _mongomock_database("tenant_a")

    def test_event_store_is_mongo_on_the_given_database(self, db: Any) -> None:
        from akgentic.team.repositories.mongo import MongoEventStore

        stores = mongo_stores(db, delete_mode=DeleteMode.HARD, auto_create_indexes=False)
        assert isinstance(stores, PersistenceStores)
        assert isinstance(stores.event_store, MongoEventStore)
        assert stores.event_store._db is db

    def test_catalog_repository_is_mongo_on_the_catalog_collection(self, db: Any) -> None:
        from akgentic.catalog.repositories.mongo import MongoEntryRepository

        repo = mongo_stores(db, delete_mode=DeleteMode.HARD).catalog_repository
        assert isinstance(repo, MongoEntryRepository)
        assert repo._collection.name == CATALOG_COLLECTION
        assert repo._collection.database.name == "tenant_a"

    @pytest.mark.parametrize("mode", [DeleteMode.SOFT, DeleteMode.HARD])
    def test_event_store_carries_the_delete_mode(self, db: Any, mode: DeleteMode) -> None:
        stores = mongo_stores(db, delete_mode=mode, auto_create_indexes=False)
        assert stores.event_store._delete_mode is mode  # type: ignore[attr-defined]

    def test_auto_create_indexes_true_provisions_the_teams_indexes(self, db: Any) -> None:
        mongo_stores(db, delete_mode=DeleteMode.HARD, auto_create_indexes=True)
        assert "teams_status_idx" in db["teams"].index_information()

    def test_auto_create_indexes_false_leaves_the_teams_indexes_out(self, db: Any) -> None:
        mongo_stores(db, delete_mode=DeleteMode.HARD, auto_create_indexes=False)
        assert "teams_status_idx" not in db["teams"].index_information()

    def test_channel_registry_is_mongo_on_the_same_database(self, db: Any) -> None:
        from akgentic.infra.adapters.shared.mongo_channel_registry import MongoChannelRegistry

        registry = mongo_stores(
            db, delete_mode=DeleteMode.HARD, auto_create_indexes=False
        ).channel_registry
        assert isinstance(registry, MongoChannelRegistry)
        assert registry._collection.name == "channel_users"
        assert registry._collection.database is db

    def test_building_the_channel_registry_does_no_io(self, db: Any) -> None:
        mongo_stores(db, delete_mode=DeleteMode.HARD, auto_create_indexes=False)
        assert set(db["channel_users"].index_information()) <= {"_id_"}

    def test_an_injected_offload_reaches_the_channel_registry(self, db: Any) -> None:
        offload = _NoopOffload()
        registry = mongo_stores(
            db, delete_mode=DeleteMode.HARD, auto_create_indexes=False, offload=offload
        ).channel_registry
        assert registry._offload is offload  # type: ignore[attr-defined]

    def test_without_an_offload_the_registry_uses_to_thread(self, db: Any) -> None:
        registry = mongo_stores(
            db, delete_mode=DeleteMode.HARD, auto_create_indexes=False
        ).channel_registry
        assert registry._offload is asyncio.to_thread  # type: ignore[attr-defined]


class TestPostgresStores:
    """``postgres_stores`` builds both Nagra stores on the DSN without a database."""

    @pytest.fixture(autouse=True)
    def _needs_postgres_extra(self) -> None:
        pytest.importorskip("nagra")
        pytest.importorskip("psycopg_pool")

    def test_both_stores_are_nagra_on_the_dsn(self) -> None:
        from akgentic.catalog.repositories.postgres import PostgresEntryRepository
        from akgentic.team.repositories.postgres import NagraEventStore

        stores = postgres_stores(DSN, delete_mode=DeleteMode.HARD)
        assert isinstance(stores, PersistenceStores)
        assert isinstance(stores.event_store, NagraEventStore)
        assert stores.event_store._conn_string == DSN
        assert isinstance(stores.catalog_repository, PostgresEntryRepository)
        assert stores.catalog_repository._conn_string == DSN

    @pytest.mark.parametrize("mode", [DeleteMode.SOFT, DeleteMode.HARD])
    def test_event_store_carries_the_delete_mode(self, mode: DeleteMode) -> None:
        stores = postgres_stores(DSN, delete_mode=mode)
        assert stores.event_store._delete_mode is mode  # type: ignore[attr-defined]

    def test_channel_registry_is_postgres_on_the_dsn(self) -> None:
        from akgentic.infra.adapters.shared.postgres_channel_registry import (
            PostgresChannelRegistry,
        )

        registry = postgres_stores(DSN, delete_mode=DeleteMode.HARD).channel_registry
        assert isinstance(registry, ChannelRegistry)
        assert isinstance(registry, PostgresChannelRegistry)
        assert registry._dsn == DSN
        assert registry._offload is asyncio.to_thread

    def test_an_injected_offload_reaches_the_channel_registry(self) -> None:
        offload = _NoopOffload()
        registry = postgres_stores(
            DSN, delete_mode=DeleteMode.HARD, offload=offload
        ).channel_registry
        assert registry._offload is offload  # type: ignore[attr-defined]


def test_public_builders_do_not_probe_for_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """The install hint names ``akgentic-infra[...]``, wrong advice for another tier."""
    pytest.importorskip("nagra")
    pytest.importorskip("psycopg_pool")
    db = _mongomock_database()

    def probe_forbidden(backend: PersistenceBackend) -> None:
        raise AssertionError(f"a public builder probed for the {backend} extra")

    monkeypatch.setattr(persistence, "_require_backend_modules", probe_forbidden)
    mongo_pair = mongo_stores(db, delete_mode=DeleteMode.HARD, auto_create_indexes=False)
    assert isinstance(mongo_pair, PersistenceStores)
    assert isinstance(postgres_stores(DSN, delete_mode=DeleteMode.HARD), PersistenceStores)


class TestBuildStoresGoesThroughTheSharedBuilders:
    """``build_stores`` resolves settings to the public builders, once each."""

    def test_mongo_settings_reach_mongo_stores(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("pymongo")
        mongomock = pytest.importorskip("mongomock")
        monkeypatch.setattr("pymongo.MongoClient", mongomock.MongoClient)
        spy = _spy_on(monkeypatch, "mongo_stores")
        build_stores(
            _mongo_settings(
                tmp_path, mongodb_database_name="tenant_a", team_delete_mode=DeleteMode.SOFT
            )
        )
        assert len(spy.calls) == 1
        (call,) = spy.calls
        assert call["db"].name == "tenant_a"
        assert call["delete_mode"] is DeleteMode.SOFT
        assert "auto_create_indexes" not in call

    def test_postgres_settings_reach_postgres_stores(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("nagra")
        pytest.importorskip("psycopg_pool")
        spy = _spy_on(monkeypatch, "postgres_stores")
        build_stores(_postgres_settings(tmp_path, team_delete_mode=DeleteMode.SOFT))
        assert len(spy.calls) == 1
        (call,) = spy.calls
        assert call["dsn"] == DSN
        assert call["delete_mode"] is DeleteMode.SOFT


class _FalsyEventStore(YamlEventStore):
    """A valid store that is falsy, as one defining ``__len__`` on an empty log would be."""

    def __bool__(self) -> bool:
        return False


class _FalsyEntryRepository(YamlEntryRepository):
    """A valid catalog repository that is falsy."""

    def __bool__(self) -> bool:
        return False


class _FalsyChannelRegistry(YamlChannelRegistry):
    """A valid channel registry that is falsy."""

    def __bool__(self) -> bool:
        return False


class TestInjectedStoreIsKeptEvenWhenFalsy:
    """The overlay tests ``is not None``: a falsy injected store still wins."""

    def test_a_falsy_injected_channel_registry_is_kept(self, tmp_path: Path) -> None:
        injected = _FalsyChannelRegistry(registry_path=tmp_path / "injected.yaml")
        stores = build_stores(
            _yaml_settings(tmp_path, channel_registry_path=tmp_path / "configured.yaml"),
            channel_registry=injected,
        )
        assert stores.channel_registry is injected

    def test_a_falsy_injected_event_store_is_kept(self, tmp_path: Path) -> None:
        injected = _FalsyEventStore(data_dir=tmp_path / "injected_events")
        stores = build_stores(_yaml_settings(tmp_path), event_store=injected)
        assert stores.event_store is injected

    def test_a_falsy_injected_catalog_is_kept(self, tmp_path: Path) -> None:
        injected = _FalsyEntryRepository(root=tmp_path / "injected_catalog")
        stores = build_stores(_yaml_settings(tmp_path), catalog_repository=injected)
        assert stores.catalog_repository is injected


# A plain install, simulated in a fresh interpreter: every database driver is made
# unimportable before anything loads. ``sys.modules`` absence cannot be the check,
# because ``akgentic.catalog`` and ``akgentic.team.repositories`` import their own
# Mongo / Postgres stores eagerly whenever the driver is installed, as in ``[dev]``.
_LAZY_IMPORT_SCRIPT = """
import sys
import tempfile
from pathlib import Path

for driver in ("pymongo", "nagra", "psycopg", "psycopg_pool"):
    sys.modules[driver] = None

from akgentic.infra.persistence import build_stores
from akgentic.infra.server.settings import CommunitySettings
import akgentic.infra.adapters.shared.mongo_channel_registry
import akgentic.infra.adapters.shared.postgres_channel_registry
import akgentic.infra.scripts.init_db

tmp = Path(tempfile.mkdtemp())
build_stores(
    CommunitySettings(
        workspaces_root=tmp / "ws", event_store_path=tmp / "es", catalog_path=tmp / "cat"
    )
)
print("OK")
"""


def test_a_plain_install_imports_and_builds_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without any database driver, the module imports and the YAML default builds."""
    for name in ("AKGENTIC_PERSISTENCE_BACKEND", "AKGENTIC_POSTGRES_DSN", "AKGENTIC_MONGODB_URI"):
        monkeypatch.delenv(name, raising=False)
    result = subprocess.run(
        [sys.executable, "-c", _LAZY_IMPORT_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip().endswith("OK"), result.stdout


class TestBuildStoresYaml:
    """The default branch builds exactly what wire_community built before."""

    def test_event_store_is_yaml_at_the_configured_path(self, tmp_path: Path) -> None:
        store = build_stores(_yaml_settings(tmp_path)).event_store
        assert isinstance(store, YamlEventStore)
        assert store._data_dir == tmp_path / "event_store"

    def test_event_store_carries_the_team_delete_mode(self, tmp_path: Path) -> None:
        stores = build_stores(_yaml_settings(tmp_path, team_delete_mode=DeleteMode.SOFT))
        assert stores.event_store._delete_mode is DeleteMode.SOFT  # type: ignore[attr-defined]

    def test_catalog_repository_is_yaml_at_the_configured_path(self, tmp_path: Path) -> None:
        repo = build_stores(_yaml_settings(tmp_path)).catalog_repository
        assert isinstance(repo, YamlEntryRepository)

    def test_channel_registry_is_yaml_at_the_configured_path(self, tmp_path: Path) -> None:
        path = tmp_path / "channels.yaml"
        registry = build_stores(
            _yaml_settings(tmp_path, channel_registry_path=path)
        ).channel_registry
        assert isinstance(registry, YamlChannelRegistry)
        assert registry._path == path

    def test_channel_registry_is_disabled_without_a_path(self, tmp_path: Path) -> None:
        registry = build_stores(_yaml_settings(tmp_path)).channel_registry
        assert isinstance(registry, YamlChannelRegistry)
        assert registry._path is None


class TestBuildStoresPostgres:
    """The postgres branch builds the Nagra stores without touching a database."""

    @pytest.fixture(autouse=True)
    def _needs_postgres_extra(self) -> None:
        pytest.importorskip("nagra")
        pytest.importorskip("psycopg_pool")

    def test_event_store_is_nagra_with_the_dsn(self, tmp_path: Path) -> None:
        from akgentic.team.repositories.postgres import NagraEventStore

        store = build_stores(_postgres_settings(tmp_path)).event_store
        assert isinstance(store, NagraEventStore)
        assert store._conn_string == DSN

    def test_event_store_carries_the_team_delete_mode(self, tmp_path: Path) -> None:
        stores = build_stores(_postgres_settings(tmp_path, team_delete_mode=DeleteMode.SOFT))
        assert stores.event_store._delete_mode is DeleteMode.SOFT  # type: ignore[attr-defined]

    def test_catalog_repository_is_postgres_with_the_dsn(self, tmp_path: Path) -> None:
        from akgentic.catalog.repositories.postgres import PostgresEntryRepository

        repo = build_stores(_postgres_settings(tmp_path)).catalog_repository
        assert isinstance(repo, PostgresEntryRepository)
        assert repo._conn_string == DSN

    def test_postgres_backend_writes_nothing_under_the_yaml_paths(self, tmp_path: Path) -> None:
        build_stores(_postgres_settings(tmp_path))
        assert not (tmp_path / "event_store").exists()
        assert not (tmp_path / "catalog").exists()

    def test_channel_registry_is_postgres_and_ignores_the_yaml_path(self, tmp_path: Path) -> None:
        from akgentic.infra.adapters.shared.postgres_channel_registry import (
            PostgresChannelRegistry,
        )

        path = tmp_path / "channels.yaml"
        registry = build_stores(
            _postgres_settings(tmp_path, channel_registry_path=path)
        ).channel_registry
        assert isinstance(registry, PostgresChannelRegistry)
        assert registry._dsn == DSN
        assert not path.exists()

    def test_an_injected_catalog_replaces_the_one_the_pair_builds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from akgentic.team.repositories.postgres import NagraEventStore

        spy = _spy_on(monkeypatch, "postgres_stores")
        injected = YamlEntryRepository(root=tmp_path / "injected_catalog")
        stores = build_stores(
            _postgres_settings(tmp_path, team_delete_mode=DeleteMode.SOFT),
            catalog_repository=injected,
        )
        assert stores.catalog_repository is injected
        assert isinstance(stores.event_store, NagraEventStore)
        assert stores.event_store._delete_mode is DeleteMode.SOFT
        assert len(spy.calls) == 1


class TestBuildStoresMongo:
    """The mongo branch builds both Mongo stores over one client (mongomock)."""

    @pytest.fixture(autouse=True)
    def _mongomock_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pymongo")
        mongomock = pytest.importorskip("mongomock")
        monkeypatch.setattr("pymongo.MongoClient", mongomock.MongoClient)

    def test_event_store_is_mongo_in_the_configured_database(self, tmp_path: Path) -> None:
        from akgentic.team.repositories.mongo import MongoEventStore

        store = build_stores(_mongo_settings(tmp_path, mongodb_database_name="tenant_a"))
        assert isinstance(store.event_store, MongoEventStore)
        assert store.event_store._db.name == "tenant_a"

    def test_event_store_carries_the_team_delete_mode(self, tmp_path: Path) -> None:
        stores = build_stores(_mongo_settings(tmp_path, team_delete_mode=DeleteMode.SOFT))
        assert stores.event_store._delete_mode is DeleteMode.SOFT  # type: ignore[attr-defined]

    def test_catalog_repository_is_mongo_on_the_catalog_collection(self, tmp_path: Path) -> None:
        from akgentic.catalog.repositories.mongo import MongoEntryRepository

        repo = build_stores(_mongo_settings(tmp_path)).catalog_repository
        assert isinstance(repo, MongoEntryRepository)
        assert repo._collection.name == CATALOG_COLLECTION
        assert repo._collection.database.name == "akgentic"

    def test_both_stores_share_one_client(self, tmp_path: Path) -> None:
        stores = build_stores(_mongo_settings(tmp_path))
        event_client = stores.event_store._db.client  # type: ignore[attr-defined]
        catalog_client = stores.catalog_repository._collection.database.client  # type: ignore[attr-defined]
        assert event_client is catalog_client

    def test_mongo_backend_writes_nothing_under_the_yaml_paths(self, tmp_path: Path) -> None:
        build_stores(_mongo_settings(tmp_path))
        assert not (tmp_path / "event_store").exists()
        assert not (tmp_path / "catalog").exists()

    def test_channel_registry_shares_the_event_stores_client_and_database(
        self, tmp_path: Path
    ) -> None:
        from akgentic.infra.adapters.shared.mongo_channel_registry import MongoChannelRegistry

        stores = build_stores(
            _mongo_settings(
                tmp_path,
                mongodb_database_name="tenant_a",
                channel_registry_path=tmp_path / "ignored.yaml",
            )
        )
        registry = stores.channel_registry
        assert isinstance(registry, MongoChannelRegistry)
        event_db = stores.event_store._db  # type: ignore[attr-defined]
        assert registry._collection.database.name == "tenant_a"
        assert registry._collection.database.client is event_db.client
        assert not (tmp_path / "ignored.yaml").exists()

    def test_an_injected_event_store_replaces_the_one_the_pair_builds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from akgentic.catalog.repositories.mongo import MongoEntryRepository

        spy = _spy_on(monkeypatch, "mongo_stores")
        injected = YamlEventStore(data_dir=tmp_path / "injected_events")
        stores = build_stores(_mongo_settings(tmp_path), event_store=injected)
        assert stores.event_store is injected
        assert isinstance(stores.catalog_repository, MongoEntryRepository)
        assert stores.catalog_repository._collection.name == CATALOG_COLLECTION
        assert len(spy.calls) == 1


_POSTGRES_MODULES = ("nagra", "psycopg", "psycopg_pool")
_ALL_BACKEND_MODULES = (*_POSTGRES_MODULES, "pymongo")


def _hide_modules(monkeypatch: pytest.MonkeyPatch, hidden: tuple[str, ...]) -> None:
    """Make ``persistence``'s dependency probe see exactly ``hidden`` as missing.

    The probe answers for every backend module itself — missing if hidden,
    present otherwise — and never reaches the real import, so the result does not
    depend on what the running environment has installed.
    """

    def fake_import_module(name: str) -> object:
        if name in hidden:
            raise ImportError(f"No module named {name!r}")
        if name in _ALL_BACKEND_MODULES:
            return object()
        raise AssertionError(f"unexpected probe of {name!r}")

    monkeypatch.setattr(persistence.importlib, "import_module", fake_import_module)


class TestMissingBackendExtra:
    """Selecting a database without its extra fails at wire time, naming the fix."""

    @pytest.fixture()
    def without_extras(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _hide_modules(monkeypatch, _ALL_BACKEND_MODULES)

    @pytest.mark.usefixtures("without_extras")
    @pytest.mark.parametrize(
        ("factory", "missing", "extra"),
        [
            (_postgres_settings, "nagra, psycopg, psycopg_pool are", "akgentic-infra[postgres]"),
            (_mongo_settings, "pymongo is", "akgentic-infra[mongo]"),
        ],
    )
    def test_build_raises_with_install_hint(
        self, tmp_path: Path, factory: Any, missing: str, extra: str
    ) -> None:
        with pytest.raises(PersistenceBackendUnavailableError) as exc_info:
            build_stores(factory(tmp_path))
        message = str(exc_info.value)
        assert f"{missing} not installed" in message
        assert extra in message
        assert "s3cret-pw" not in message

    def test_names_only_the_module_that_is_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A half-installed extra (Nagra present, no pool) is caught at wire time too."""
        _hide_modules(monkeypatch, ("psycopg_pool",))
        with pytest.raises(PersistenceBackendUnavailableError) as exc_info:
            build_stores(_postgres_settings(tmp_path))
        assert "but psycopg_pool is not installed" in str(exc_info.value)

    @pytest.mark.usefixtures("without_extras")
    @pytest.mark.parametrize("factory", [_postgres_settings, _mongo_settings])
    def test_wire_community_raises_before_starting_anything(
        self, tmp_path: Path, factory: Any
    ) -> None:
        with pytest.raises(PersistenceBackendUnavailableError):
            wire_community(factory(tmp_path))

    @pytest.mark.usefixtures("without_extras")
    def test_yaml_backend_needs_no_extra(self, tmp_path: Path) -> None:
        assert isinstance(build_stores(_yaml_settings(tmp_path)).event_store, YamlEventStore)

    @pytest.mark.usefixtures("without_extras")
    def test_both_stores_injected_needs_no_extra(self, tmp_path: Path) -> None:
        """All three stores injected: nothing is built and no extra is probed."""
        store = YamlEventStore(data_dir=tmp_path / "injected_events")
        repo = YamlEntryRepository(root=tmp_path / "injected_catalog")
        registry = YamlChannelRegistry(registry_path=tmp_path / "injected.yaml")
        stores = build_stores(
            _mongo_settings(tmp_path),
            event_store=store,
            catalog_repository=repo,
            channel_registry=registry,
        )
        assert stores.event_store is store
        assert stores.catalog_repository is repo
        assert stores.channel_registry is registry

    @pytest.mark.usefixtures("without_extras")
    @pytest.mark.parametrize("factory", [_postgres_settings, _mongo_settings])
    def test_two_stores_injected_still_builds_the_registry(
        self, tmp_path: Path, factory: Any
    ) -> None:
        """Without an injected registry the backend must build one, so its extra is needed."""
        store = YamlEventStore(data_dir=tmp_path / "injected_events")
        repo = YamlEntryRepository(root=tmp_path / "injected_catalog")
        with pytest.raises(PersistenceBackendUnavailableError):
            build_stores(factory(tmp_path), event_store=store, catalog_repository=repo)


class _RecordingEventStore(YamlEventStore):
    """A real store with a distinct type, so the test can tell it was the one used."""


def _dispatcher(services: CommunityServices) -> ChannelDispatcher:
    return next(
        s for s in services.team_manager._shared_subscribers if isinstance(s, ChannelDispatcher)
    )


class TestWireCommunityOverrides:
    """Injected stores win over the setting, one store at a time."""

    @pytest.fixture()
    def injected(self, tmp_path: Path) -> Generator[tuple[CommunityServices, Any, Any], None, None]:
        store = _RecordingEventStore(data_dir=tmp_path / "injected_events")
        repo = YamlEntryRepository(root=tmp_path / "injected_catalog")
        services = wire_community(
            _yaml_settings(tmp_path), event_store=store, catalog_repository=repo
        )
        yield services, store, repo
        services.team_manager._actor_system.shutdown(timeout=5)

    def test_injected_event_store_reaches_container_and_team_manager(
        self, injected: tuple[CommunityServices, Any, Any]
    ) -> None:
        services, store, _ = injected
        assert services.event_store is store
        assert services.team_manager._event_store is store

    def test_injected_catalog_repository_backs_the_catalog(
        self, injected: tuple[CommunityServices, Any, Any]
    ) -> None:
        services, _, repo = injected
        assert services.catalog._repository is repo

    @pytest.mark.parametrize("factory", [_postgres_settings, _mongo_settings])
    def test_injected_stores_win_over_a_database_setting(
        self, tmp_path: Path, factory: Any
    ) -> None:
        """With all three stores injected, no database store is built."""
        store = _RecordingEventStore(data_dir=tmp_path / "injected_events")
        repo = YamlEntryRepository(root=tmp_path / "injected_catalog")
        registry = YamlChannelRegistry(registry_path=tmp_path / "injected.yaml")
        services = wire_community(
            factory(tmp_path),
            event_store=store,
            catalog_repository=repo,
            channel_registry=registry,
        )
        try:
            assert services.event_store is store
            assert services.catalog._repository is repo
            assert services.channel_registry is registry
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)

    def test_injected_channel_registry_reaches_container_and_dispatcher(
        self, tmp_path: Path
    ) -> None:
        registry = YamlChannelRegistry(registry_path=tmp_path / "injected.yaml")
        services = wire_community(
            _yaml_settings(tmp_path, channel_registry_path=tmp_path / "configured.yaml"),
            channel_registry=registry,
        )
        try:
            assert services.channel_registry is registry
            assert _dispatcher(services)._registry is registry
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)

    def test_without_overrides_the_setting_decides(self, tmp_path: Path) -> None:
        services = wire_community(_yaml_settings(tmp_path))
        try:
            assert type(services.event_store) is YamlEventStore
            assert isinstance(services.catalog._repository, YamlEntryRepository)
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)


class TestWireCommunityOnPostgresBindsInTheDatabase:
    """Community on ``postgres``: the bindings live in the database, one instance serves both legs.

    The event store and catalog are injected so ``LocalRuntimeCache.warm`` stays
    off the database; the registry is still built from the backend's builder.
    """

    @pytest.fixture(autouse=True)
    def _needs_postgres_extra(self) -> None:
        pytest.importorskip("nagra")
        pytest.importorskip("psycopg_pool")

    def test_registry_is_postgres_on_the_dsn(self, tmp_path: Path) -> None:
        from akgentic.infra.adapters.shared.postgres_channel_registry import (
            PostgresChannelRegistry,
        )

        services = wire_community(
            _postgres_settings(tmp_path),
            event_store=YamlEventStore(data_dir=tmp_path / "injected_events"),
            catalog_repository=YamlEntryRepository(root=tmp_path / "injected_catalog"),
        )
        try:
            registry = services.channel_registry
            assert isinstance(registry, PostgresChannelRegistry)
            assert registry._dsn == DSN
            assert _dispatcher(services)._registry is registry
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)


class TestWireCommunityOnMongoBindsInTheDatabase:
    """Community on ``mongo``: the bindings live beside the teams, one instance serves both legs."""

    @pytest.fixture(autouse=True)
    def _mongomock_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pymongo")
        mongomock = pytest.importorskip("mongomock")
        monkeypatch.setattr("pymongo.MongoClient", mongomock.MongoClient)

    async def test_registry_is_mongo_on_the_event_stores_client(self, tmp_path: Path) -> None:
        from akgentic.infra.adapters.shared.mongo_channel_registry import MongoChannelRegistry

        services = wire_community(_mongo_settings(tmp_path, mongodb_database_name="tenant_a"))
        try:
            registry = services.channel_registry
            assert isinstance(registry, MongoChannelRegistry)
            assert registry._collection.database.name == "tenant_a"
            event_client = services.event_store._db.client  # type: ignore[attr-defined]
            assert registry._collection.database.client is event_client
            assert _dispatcher(services)._registry is registry

            binding = ChannelBinding(
                channel="telegram",
                channel_user_id="U1",
                team_id=uuid.uuid4(),
                agent_name="@HumanProxy_0",
            )
            await registry.register(binding)
            assert registry.find_binding_sync(binding.team_id, binding.agent_name) == binding
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)


_SettingsFactory = Callable[..., CommunitySettings]


class TestTeamDeleteModeOnEveryBackend:
    """``team_delete_mode`` retires or purges on whichever store the backend builds.

    The team is created, then deleted through ``TeamService.delete_team`` on the
    wired stack, so the mode travels the operator's path: settings → store →
    service. The catalog is a seeded YAML repository, injected so the Mongo case
    needs no catalog of its own; the event store is the backend's. Postgres is
    covered by the store's own contract tests in ``akgentic-team``, which need a
    database; here it is covered down to the mode reaching ``NagraEventStore``.
    """

    @pytest.fixture(params=["yaml", "mongo"])
    def backend(
        self, request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[_SettingsFactory, str]:
        if request.param == "yaml":
            return _yaml_settings, "YamlEventStore"
        pytest.importorskip("pymongo")
        mongomock = pytest.importorskip("mongomock")
        monkeypatch.setattr("pymongo.MongoClient", mongomock.MongoClient)
        return _mongo_settings, "MongoEventStore"

    @staticmethod
    def _create_then_delete(
        tmp_path: Path, backend: tuple[_SettingsFactory, str], mode: DeleteMode
    ) -> tuple[CommunityServices, Any]:
        factory, store_type = backend
        catalog_root = tmp_path / "seeded_catalog"
        _seed_catalog(catalog_root)
        services = wire_community(
            factory(tmp_path, team_delete_mode=mode),
            catalog_repository=YamlEntryRepository(root=catalog_root),
        )
        assert type(services.event_store).__name__ == store_type
        assert services.team_service is not None
        process = services.team_service.create_team(catalog_namespace="test-team", user_id="u")
        services.team_service.delete_team(process.team_id)
        return services, process.team_id

    def test_soft_retires_the_team(
        self, tmp_path: Path, backend: tuple[_SettingsFactory, str]
    ) -> None:
        services, team_id = self._create_then_delete(tmp_path, backend, DeleteMode.SOFT)
        try:
            assert services.team_service is not None
            retired = services.event_store.load_team(team_id)
            assert retired is not None
            assert retired.status == TeamStatus.DELETED
            assert services.team_service.get_team(team_id) is None
            listed, _total = services.team_service.list_teams(user_id="u")
            assert team_id not in [p.team_id for p in listed]
            retired_only = services.event_store.list_teams(status=TeamStatus.DELETED)
            assert [p.team_id for p in retired_only] == [team_id]
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)

    def test_hard_purges_the_team(
        self, tmp_path: Path, backend: tuple[_SettingsFactory, str]
    ) -> None:
        services, team_id = self._create_then_delete(tmp_path, backend, DeleteMode.HARD)
        try:
            assert services.team_service is not None
            assert services.event_store.load_team(team_id) is None
            assert services.team_service.get_team(team_id) is None
            assert services.event_store.list_teams(status=TeamStatus.DELETED) == []
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)
