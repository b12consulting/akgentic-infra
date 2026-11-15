"""Community-tier persistence selection — YAML files or PostgreSQL.

``CommunitySettings.persistence_backend`` picks the backend for both durable
stores the community tier owns: the team ``EventStore`` and the catalog
``EntryRepository``. They move together on purpose — a deployment with teams in
Postgres and the catalog in YAML (or the reverse) is not a configuration this
module offers.

The ``yaml`` default imports nothing new, so a community install without the
``[postgres]`` extra behaves exactly as before. The ``postgres`` branch imports
the Nagra-backed stores lazily and turns a missing extra into one clear error at
wire time, rather than an ``ImportError`` from deep inside the first request.

Neither branch creates tables. Schema creation is a deployment step, kept out of
the constructors by both packages (``akgentic.team.scripts.init_db`` and
``akgentic.catalog.scripts.init_db``).
"""

from __future__ import annotations

import importlib
import logging

from akgentic.catalog import YamlEntryRepository
from akgentic.catalog.repositories.base import EntryRepository
from akgentic.infra.adapters.community.postgres_instance_lock import PostgresInstanceLock
from akgentic.infra.server.settings import CommunitySettings, PersistenceBackend
from akgentic.team.ports import EventStore
from akgentic.team.repositories.yaml import YamlEventStore

logger = logging.getLogger(__name__)

#: Modules the ``postgres`` backend needs at runtime. ``psycopg_pool`` is on the
#: list because Nagra imports it only when the first ``Transaction`` opens — a
#: missing pool would otherwise pass wiring and fail on the first request.
_POSTGRES_MODULES = ("nagra", "psycopg", "psycopg_pool")

_INSTALL_HINT = 'pip install "akgentic-infra[postgres]"'

__all__ = [
    "PersistenceBackendUnavailableError",
    "build_catalog_repository",
    "build_event_store",
    "prepare_persistence",
]


class PersistenceBackendUnavailableError(RuntimeError):
    """The selected persistence backend's dependencies are not installed."""


def build_event_store(settings: CommunitySettings) -> EventStore:
    """Build the team event store the settings select.

    Args:
        settings: Community settings carrying ``persistence_backend`` and the
            backend's location (``event_store_path`` or ``postgres_conn_string``).

    Returns:
        A ``YamlEventStore`` or a ``NagraEventStore``, built with
        ``settings.team_delete_mode``.

    Raises:
        PersistenceBackendUnavailableError: ``postgres`` is selected and the
            ``[postgres]`` extra is not installed.
    """
    if settings.persistence_backend is PersistenceBackend.YAML:
        return YamlEventStore(
            data_dir=settings.event_store_path, delete_mode=settings.team_delete_mode
        )

    _require_postgres_modules()
    from akgentic.team.repositories.postgres import NagraEventStore

    logger.info("Team event store: PostgreSQL")
    return NagraEventStore(_conn_string(settings), delete_mode=settings.team_delete_mode)


def build_catalog_repository(settings: CommunitySettings) -> EntryRepository:
    """Build the catalog repository the settings select.

    Args:
        settings: Community settings carrying ``persistence_backend`` and the
            backend's location (``catalog_path`` or ``postgres_conn_string``).

    Returns:
        A ``YamlEntryRepository`` or a ``PostgresEntryRepository``.

    Raises:
        PersistenceBackendUnavailableError: ``postgres`` is selected and the
            ``[postgres]`` extra is not installed.
    """
    if settings.persistence_backend is PersistenceBackend.YAML:
        return YamlEntryRepository(root=settings.catalog_path)

    _require_postgres_modules()
    from akgentic.catalog.repositories.postgres import PostgresEntryRepository

    logger.info("Catalog repository: PostgreSQL")
    return PostgresEntryRepository(_conn_string(settings))


def prepare_persistence(
    settings: CommunitySettings,
    *,
    event_store_from_settings: bool,
    catalog_from_settings: bool,
) -> PostgresInstanceLock | None:
    """Run the startup steps the selected persistence backend needs.

    Called by ``wire_community`` before it builds anything that reads the
    store — in particular before ``LocalRuntimeCache.warm`` restores teams.

    **YAML** needs none: it has no schema, and its directories are created on
    first write. This returns ``None`` without importing anything.

    **PostgreSQL** needs two:

    1. **Single-instance lock** — when the team store is built from the
       settings, take :class:`PostgresInstanceLock`, so a second server on the
       same database fails here instead of running the same teams twice.
    2. **Schema** — when ``postgres_init_db`` is set, run ``init_db`` for each
       store built from the settings. Both are idempotent.

    A store the caller injected is the caller's business: neither step runs
    for it. With the YAML backend this does nothing and returns ``None``.

    Args:
        settings: Community settings.
        event_store_from_settings: The team store comes from the settings.
        catalog_from_settings: The catalog repository comes from the settings.

    Returns:
        The held lock — the caller releases it at shutdown — or ``None`` when
        no lock was taken.

    Raises:
        PersistenceBackendUnavailableError: The ``[postgres]`` extra is missing.
        InstanceLockedError: Another community server holds the database.
    """
    if settings.persistence_backend is not PersistenceBackend.POSTGRES:
        return None
    if not (event_store_from_settings or catalog_from_settings):
        return None
    _require_postgres_modules()
    conn_string = _conn_string(settings)

    lock: PostgresInstanceLock | None = None
    if event_store_from_settings:
        lock = PostgresInstanceLock(conn_string)
        lock.acquire()
    try:
        if settings.postgres_init_db:
            _init_schema(
                conn_string,
                team=event_store_from_settings,
                catalog=catalog_from_settings,
            )
    except BaseException:
        if lock is not None:
            lock.release()
        raise
    return lock


def _init_schema(conn_string: str, *, team: bool, catalog: bool) -> None:
    """Create missing tables and indexes — the same calls the init_db scripts make."""
    if team:
        from akgentic.team.repositories.postgres import init_db as init_team_db

        init_team_db(conn_string)
        logger.info("Team event store schema initialised")
    if catalog:
        from akgentic.catalog.repositories.postgres import PostgresCatalogConfig
        from akgentic.catalog.repositories.postgres import init_db as init_catalog_db

        init_catalog_db(PostgresCatalogConfig(connection_string=conn_string))
        logger.info("Catalog schema initialised")


def _require_postgres_modules() -> None:
    """Fail at wire time, naming the extra, when a Postgres dependency is missing."""
    missing: list[str] = []
    for name in _POSTGRES_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise PersistenceBackendUnavailableError(
            f"persistence_backend is 'postgres' but {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} not installed. "
            f"Install with: {_INSTALL_HINT}"
        )


def _conn_string(settings: CommunitySettings) -> str:
    """The DSN, unwrapped. ``CommunitySettings`` guarantees it is set for postgres."""
    secret = settings.postgres_conn_string
    if secret is None:  # pragma: no cover — the settings model validator forbids it
        raise ValueError("postgres_conn_string is required for the postgres backend")
    return secret.get_secret_value()
