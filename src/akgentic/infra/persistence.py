"""Community-tier persistence selection — YAML files, MongoDB or PostgreSQL.

``CommunitySettings.persistence_backend`` picks the backend for both durable
stores the community tier owns: the team ``EventStore`` and the catalog
``EntryRepository``. They move together on purpose — a deployment with teams in
a database and the catalog in YAML (or the reverse) is not a configuration this
module offers.

The ``yaml`` default imports nothing new, so a community install without an
extra behaves exactly as before. The database branches import their stores
lazily and turn a missing extra into one clear error at wire time, rather than
an ``ImportError`` from deep inside the first request.

Neither database branch creates tables. Postgres schema creation is a
deployment step, kept out of the constructors by both packages
(``akgentic.team.scripts.init_db`` and ``akgentic.catalog.scripts.init_db``).
Mongo needs none: ``MongoEventStore`` provisions its indexes at construction and
``MongoEntryRepository`` on its first write.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any, NamedTuple

from pydantic import SecretStr

from akgentic.catalog import YamlEntryRepository
from akgentic.catalog.repositories.base import EntryRepository
from akgentic.infra.server.settings import CommunitySettings, PersistenceBackend
from akgentic.team.ports import EventStore
from akgentic.team.repositories.yaml import YamlEventStore

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
    "CommunityStores",
    "PersistenceBackendUnavailableError",
    "build_stores",
]


class PersistenceBackendUnavailableError(RuntimeError):
    """The selected persistence backend's dependencies are not installed."""


class CommunityStores(NamedTuple):
    """The two durable stores the community tier wires."""

    event_store: EventStore
    catalog_repository: EntryRepository


def build_stores(
    settings: CommunitySettings,
    *,
    event_store: EventStore | None = None,
    catalog_repository: EntryRepository | None = None,
) -> CommunityStores:
    """Build the stores the settings select, keeping any the caller injected.

    An injected store wins over the setting for that store only; the other is
    built from the settings. With both injected nothing is built or imported.
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
        return CommunityStores(event_store, catalog_repository)
    backend = settings.persistence_backend
    _require_backend_modules(backend)
    logger.info("Community persistence backend: %s", backend)
    return _BACKENDS[backend](settings, event_store, catalog_repository)


def _yaml_stores(
    settings: CommunitySettings,
    event_store: EventStore | None,
    catalog_repository: EntryRepository | None,
) -> CommunityStores:
    """The stores ``wire_community`` built before the setting existed."""
    return CommunityStores(
        event_store
        if event_store is not None
        else YamlEventStore(
            data_dir=settings.event_store_path, delete_mode=settings.team_delete_mode
        ),
        catalog_repository
        if catalog_repository is not None
        else YamlEntryRepository(root=settings.catalog_path),
    )


def _mongo_stores(
    settings: CommunitySettings,
    event_store: EventStore | None,
    catalog_repository: EntryRepository | None,
) -> CommunityStores:
    """Both stores in ``mongodb_database_name``, over one client."""
    from pymongo import MongoClient

    from akgentic.catalog.repositories.mongo import MongoEntryRepository
    from akgentic.team.repositories.mongo import MongoEventStore

    # ``Any``: a BSON document has no static shape at the driver boundary.
    client: MongoClient[Any] = MongoClient(_secret(settings.mongodb_uri, "mongodb_uri"))
    db = client[settings.mongodb_database_name]
    return CommunityStores(
        event_store
        if event_store is not None
        else MongoEventStore(db, delete_mode=settings.team_delete_mode),
        catalog_repository
        if catalog_repository is not None
        else MongoEntryRepository(db[CATALOG_COLLECTION]),
    )


def _postgres_stores(
    settings: CommunitySettings,
    event_store: EventStore | None,
    catalog_repository: EntryRepository | None,
) -> CommunityStores:
    """The Nagra stores against ``postgres_dsn``."""
    from akgentic.catalog.repositories.postgres import PostgresEntryRepository
    from akgentic.team.repositories.postgres import NagraEventStore

    dsn = _secret(settings.postgres_dsn, "postgres_dsn")
    return CommunityStores(
        event_store
        if event_store is not None
        else NagraEventStore(dsn, delete_mode=settings.team_delete_mode),
        catalog_repository if catalog_repository is not None else PostgresEntryRepository(dsn),
    )


_BACKENDS: dict[
    PersistenceBackend,
    Callable[[CommunitySettings, EventStore | None, EntryRepository | None], CommunityStores],
] = {
    PersistenceBackend.YAML: _yaml_stores,
    PersistenceBackend.MONGO: _mongo_stores,
    PersistenceBackend.POSTGRES: _postgres_stores,
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
