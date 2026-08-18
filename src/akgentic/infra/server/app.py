"""FastAPI application factory for the akgentic-infra server.

Per ADR-023: the ``/admin/catalog/*`` HTTP surface IS the v2 unified router
(``akgentic.catalog.api.router.build_router``) with ``AuthStrategy`` wired
as a FastAPI dependency. The catalog package itself owns request/response
validation, error mapping, and CRUD semantics; infra owns only the mount
point, the auth gate, and the structured mutation log middleware — all of
which now live in ``CoreModule`` (ADR-039, epic 57).

``create_app`` is the community tier's entry point: process-global pre-steps
(logging, catalog prefix policy), ``TeamService`` construction and ingestion
wiring, then composition of ``CoreModule`` through the modular ``build_app``
builder, plus the wrapper-level legacy responsibilities documented on
``_build_app``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from akgentic.catalog import allowed_prefixes, set_allowed_prefixes
from akgentic.catalog.api import add_exception_handlers
from akgentic.infra.server.assembly import build_app
from akgentic.infra.server.deps import TierServices
from akgentic.infra.server.errors import add_server_exception_handlers
from akgentic.infra.server.logging_config import configure_logging
from akgentic.infra.server.modules import CoreModule
from akgentic.infra.server.routes.ws import ConnectionManager, shutdown_reader_pool
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import ServerSettings
from akgentic.infra.server.state_keys import (
    CHANNEL_PARSERS,
    CHANNEL_REGISTRY,
    CONNECTION_MANAGER,
    DRAINING,
    INGESTION,
    SERVICES,
    SETTINGS,
    TEAM_SERVICE,
)

logger = logging.getLogger(__name__)


def create_app(
    services: TierServices,
    settings: ServerSettings | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Entry-point factory: constructs ``TeamService``, completes deferred
    ``LocalIngestion`` wiring, and composes the app from ``CoreModule``.

    Args:
        services: Pre-wired tier services container.
        settings: Server settings. Defaults to ``ServerSettings()``.

    Returns:
        Configured FastAPI application instance.
    """
    settings = settings or ServerSettings()
    configure_logging(settings.log_level)
    logger.info("Logging configured: level=%s", settings.log_level)
    # Make the passed settings authoritative over the catalog's own lazy read of
    # AKGENTIC_CATALOG_MODEL_TYPE_PREFIXES, before any route can accept an entry.
    # Deliberately here and not in ``_build_app``: that is the shared test-facing
    # assembler and must not acquire a process-global side effect.
    set_allowed_prefixes(settings.catalog_model_type_prefixes)
    logger.info("Catalog model_type allowlist: %s", allowed_prefixes())
    # ``workspaces_root`` is declared on ``CommunitySettings``; base
    # ``ServerSettings`` callers fall back to the same default the field
    # declares so ``TeamService`` always has a valid FS-cleanup root.
    workspaces_root = getattr(settings, "workspaces_root", Path("workspaces"))
    team_service = TeamService(services, workspaces_root=workspaces_root)
    _wire_ingestion(services, team_service)
    return _build_app(services, team_service, settings)


def _wire_ingestion(services: TierServices, team_service: TeamService) -> None:
    """Complete deferred LocalIngestion wiring with the constructed TeamService.

    Community tier needs this deferred wiring because LocalIngestion holds a
    direct in-process reference to TeamService, creating a circular construction
    dependency (wire_community -> CommunityServices -> LocalIngestion, but
    TeamService needs CommunityServices). Department/enterprise tiers don't
    need this — their ingestion adapters communicate over the network (HTTP or
    Dapr), so they arrive fully wired from their own wire_*() functions.
    """
    from akgentic.infra.adapters.community.local_ingestion import LocalIngestion

    if isinstance(services.ingestion, LocalIngestion):
        services.ingestion.team_service = team_service


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan handler implementing the ADR-013 graceful shutdown sequence.

    Invoked through ``CoreModule.lifespan`` (the composed app's only lifespan
    contributor); the implementation stays in this module so the drain
    collaborators (``shutdown_reader_pool``, ``logger``) keep their
    long-standing patch targets.

    Startup: sets ``app.state.draining = False``.
    Shutdown: sets draining flag, waits pre-drain delay, disconnects all
    WebSocket clients, then stops all teams via ``worker_handle.stop_all()``.
    """
    DRAINING.set(app, value=False)
    logger.info("Lifespan startup: draining=False")
    yield
    # --- Shutdown sequence (ADR-013 Decision 2) ---
    DRAINING.set(app, value=True)
    logger.info("Lifespan shutdown: draining=True")

    delay = SETTINGS.require(app).shutdown_pre_drain_delay
    if delay > 0:
        logger.info("Pre-drain delay: sleeping %ds", delay)
        await asyncio.sleep(delay)

    logger.info("Disconnecting all WebSocket clients")
    await CONNECTION_MANAGER.require(app).disconnect_all()

    timeout = SETTINGS.require(app).shutdown_drain_timeout
    logger.info("Stopping all teams (timeout=%ds)", timeout)
    try:
        await asyncio.wait_for(
            asyncio.to_thread(SERVICES.require(app).worker_handle.stop_all),
            timeout=timeout,
        )
        logger.info("stop_all() completed successfully")
    except TimeoutError:
        logger.warning(
            "stop_all() exceeded shutdown_drain_timeout=%ds, proceeding with exit",
            timeout,
        )

    # Shut down the dedicated WS reader thread pool — see issue #227.
    shutdown_reader_pool()
    logger.info("WebSocket reader pool shut down")


def _build_app(
    services: TierServices,
    team_service: TeamService,
    settings: ServerSettings,
) -> FastAPI:
    """Assemble the app over the modular builder (shared by create_app and tests).

    Composes ``CoreModule`` through ``build_app``, then applies the wrapper
    responsibilities that deliberately stay OUTSIDE the module contract
    (Story 57.2 adjudicated exemptions):

    - ``_store_state`` populates the community state keys at **build time**,
      exactly as before, so non-lifespan test clients keep working
      (``CoreModule``'s lifespan writes only ``draining``);
    - exception handlers register through the two package helpers because
      their handler callables are package-private, not cleanly expressible as
      ``ExceptionHandlerSpec`` pairs.

    Args:
        services: Wired tier services container.
        team_service: Pre-built team service.
        settings: Server settings.

    Returns:
        Configured FastAPI application instance.
    """
    app = build_app(settings, services, [CoreModule(services=services, settings=settings)])
    _store_state(app, services, team_service, settings)
    add_exception_handlers(app)
    add_server_exception_handlers(app)
    return app


def _store_state(
    app: FastAPI,
    services: TierServices,
    team_service: TeamService,
    settings: ServerSettings,
) -> None:
    """Store services and configuration on app.state for dependency injection."""
    SERVICES.set(app, services)
    TEAM_SERVICE.set(app, team_service)
    SETTINGS.set(app, settings)
    CONNECTION_MANAGER.set(app, ConnectionManager())
    # ``channel_parser_registry`` is optional on the services container (only the
    # community tier declares it). CHANNEL_PARSERS is a required key, so the slot
    # is only set when the services container actually exposes a registry; a tier
    # without one leaves the slot unset and any webhook request fails loud
    # (``require()`` → LookupError → 500) instead of reading back a silent None.
    channel_parsers = getattr(services, "channel_parser_registry", None)
    if channel_parsers is not None:
        CHANNEL_PARSERS.set(app, channel_parsers)
    CHANNEL_REGISTRY.set(app, services.channel_registry)
    INGESTION.set(app, services.ingestion)
