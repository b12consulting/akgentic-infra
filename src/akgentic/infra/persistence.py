"""Durable-store builders — one per database backend, plus the community selection.

Two layers live here.

- ``mongo_stores`` and ``postgres_stores`` define what a database backend means:
  which team ``EventStore``, which catalog ``EntryRepository``, which catalog
  collection, and the delete mode reaching the event store. Each takes a
  connection the caller owns (a Mongo ``Database`` or a Postgres DSN), so any
  tier can use them while keeping its own connection, client tuning, index
  stance and closing.
- ``build_stores`` resolves ``CommunitySettings`` to those builders.
  ``persistence_backend`` picks the backend for both stores at once — a
  deployment with teams in a database and the catalog in YAML (or the reverse)
  is not a configuration this module offers.

The ``yaml`` default imports nothing new, so a community install without an
extra behaves exactly as before. The database builders import their stores
lazily, and ``build_stores`` turns a missing extra into one clear error at wire
time, rather than an ``ImportError`` from deep inside the first request.

Neither database builder creates tables. Postgres schema creation is a
deployment step, kept out of the constructors by both packages
(``akgentic.team.scripts.init_db`` and ``akgentic.catalog.scripts.init_db``).
Mongo needs none: ``MongoEventStore`` provisions its indexes at construction (the
``teams`` ones unless ``auto_create_indexes`` turns them off) and
``MongoEntryRepository`` on its first write.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import SecretStr

from akgentic.catalog import YamlEntryRepository
from akgentic.catalog.repositories.base import EntryRepository
from akgentic.infra.server.settings import CommunitySettings, PersistenceBackend
from akgentic.team.models import DeleteMode
from akgentic.team.ports import EventStore
from akgentic.team.repositories.yaml import YamlEventStore

if TYPE_CHECKING:
    from pymongo.database import Database

logger = logging.getLogger(__name__)

#: Modules each database backend needs at runtime. ``psycopg_pool`` is on the
#: Postgres list because Nagra imports it only when the first ``Transaction``
#: opens — a missing pool would otherwise pass wiring and fail on the first request.
_BACKEND_MODULES: dict[PersistenceBackend, tuple[str, ...]] = {
    PersistenceBackend.MONGO: ("pymongo",),
    PersistenceBackend.POSTGRES: ("nagra", "psycopg", "psycopg_pool"),
}

#: The catalog collection — the one department uses.
CATALOG_COLLECTION = "catalog_entries"

__all__ = [
    "CATALOG_COLLECTION",
    "PersistenceBackendUnavailableError",
    "PersistenceStores",
    "build_stores",
    "mongo_stores",
    "postgres_stores",
]


class PersistenceBackendUnavailableError(RuntimeError):
    """The selected persistence backend's dependencies are not installed."""


class PersistenceStores(NamedTuple):
    """The two durable stores a tier wires: the team event store and the catalog repository."""

    event_store: EventStore
    catalog_repository: EntryRepository


def mongo_stores(
    # ``Any``: a BSON document has no static shape at the driver boundary.
    db: Database[Any],
    *,
    delete_mode: DeleteMode,
    auto_create_indexes: bool | None = None,
) -> PersistenceStores:
    """Both Mongo stores in ``db``, over the caller's client.

    Creates, closes and configures no client. ``MongoEventStore`` always creates
    its ``events`` / ``agent_states`` / ``agent_cards`` indexes at construction;
    ``MongoEntryRepository`` touches nothing until its first write.

    Args:
        db: The database both stores live in.
        delete_mode: How the event store deletes a team (soft retire or purge).
        auto_create_indexes: Passed straight to ``MongoEventStore``, where it
            gates the ``teams`` collection's indexes only; ``None`` defers to
            its ``MONGO_TEAM_AUTO_INDEX`` default.

    Returns:
        ``MongoEventStore`` on ``db`` and ``MongoEntryRepository`` on
        ``db[CATALOG_COLLECTION]``.
    """
    from akgentic.catalog.repositories.mongo import MongoEntryRepository
    from akgentic.team.repositories.mongo import MongoEventStore

    return PersistenceStores(
        MongoEventStore(db, auto_create_indexes=auto_create_indexes, delete_mode=delete_mode),
        MongoEntryRepository(db[CATALOG_COLLECTION]),
    )


def postgres_stores(dsn: str, *, delete_mode: DeleteMode) -> PersistenceStores:
    """Both Nagra stores against ``dsn``. Opens no connection.

    Args:
        dsn: A ``postgresql://`` connection string.
        delete_mode: How the event store deletes a team (soft retire or purge).

    Returns:
        ``NagraEventStore`` and ``PostgresEntryRepository``, both on ``dsn``.
    """
    from akgentic.catalog.repositories.postgres import PostgresEntryRepository
    from akgentic.team.repositories.postgres import NagraEventStore

    return PersistenceStores(
        NagraEventStore(dsn, delete_mode=delete_mode),
        PostgresEntryRepository(dsn),
    )


def build_stores(
    settings: CommunitySettings,
    *,
    event_store: EventStore | None = None,
    catalog_repository: EntryRepository | None = None,
) -> PersistenceStores:
    """Build the stores the settings select, keeping any the caller injected.

    An injected store wins over the setting for that store only. With both
    injected nothing is built or imported. With one injected, the backend still
    builds its pair and the injected store replaces its counterpart; on Mongo
    that means the unused event store's idempotent index builds still run.
    On Mongo, the two stores share one client.

    Args:
        settings: Community settings carrying ``persistence_backend`` and the
            backend's location.
        event_store: A team event store to use instead of building one.
        catalog_repository: A catalog repository to use instead of building one.

    Returns:
        The event store and the catalog repository.

    Raises:
        PersistenceBackendUnavailableError: A database backend is selected, a
            store must be built, and the backend's extra is not installed.
    """
    if event_store is not None and catalog_repository is not None:
        return PersistenceStores(event_store, catalog_repository)
    backend = settings.persistence_backend
    _require_backend_modules(backend)
    logger.info("Community persistence backend: %s", backend)
    built = _BACKENDS[backend](settings)
    return PersistenceStores(
        event_store if event_store is not None else built.event_store,
        catalog_repository if catalog_repository is not None else built.catalog_repository,
    )


def _yaml_from_settings(settings: CommunitySettings) -> PersistenceStores:
    """The stores ``wire_community`` built before the setting existed."""
    return PersistenceStores(
        YamlEventStore(data_dir=settings.event_store_path, delete_mode=settings.team_delete_mode),
        YamlEntryRepository(root=settings.catalog_path),
    )


def _mongo_from_settings(settings: CommunitySettings) -> PersistenceStores:
    """Both stores in ``mongodb_database_name``, over one client."""
    from pymongo import MongoClient

    # ``Any``: a BSON document has no static shape at the driver boundary.
    client: MongoClient[Any] = MongoClient(_secret(settings.mongodb_uri, "mongodb_uri"))
    return mongo_stores(
        client[settings.mongodb_database_name], delete_mode=settings.team_delete_mode
    )


def _postgres_from_settings(settings: CommunitySettings) -> PersistenceStores:
    """The Nagra stores against ``postgres_dsn``."""
    return postgres_stores(
        _secret(settings.postgres_dsn, "postgres_dsn"), delete_mode=settings.team_delete_mode
    )


_BACKENDS: dict[PersistenceBackend, Callable[[CommunitySettings], PersistenceStores]] = {
    PersistenceBackend.YAML: _yaml_from_settings,
    PersistenceBackend.MONGO: _mongo_from_settings,
    PersistenceBackend.POSTGRES: _postgres_from_settings,
}


def _require_backend_modules(backend: PersistenceBackend) -> None:
    """Fail at wire time, naming the extra, when a backend dependency is missing."""
    missing: list[str] = []
    for name in _BACKEND_MODULES.get(backend, ()):
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise PersistenceBackendUnavailableError(
            f"persistence_backend is '{backend}' but {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} not installed. "
            f'Install with: pip install "akgentic-infra[{backend}]"'
        )


def _secret(value: SecretStr | None, field: str) -> str:
    """The connection string, unwrapped. ``CommunitySettings`` guarantees it is set."""
    if value is None:  # pragma: no cover — the settings model validator forbids it
        raise ValueError(f"{field} is required for this persistence backend")
    return value.get_secret_value()
