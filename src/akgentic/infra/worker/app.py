"""FastAPI application factory for the akgentic-infra worker."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from akgentic.infra.server.logging_config import configure_logging
from akgentic.infra.worker.deps import WorkerServices
from akgentic.infra.worker.routes.readiness import router as readiness_router
from akgentic.infra.worker.routes.teams import router as teams_router
from akgentic.infra.worker.settings import WorkerSettings

logger = logging.getLogger(__name__)


def create_worker_app(
    services: WorkerServices,
    settings: WorkerSettings | None = None,
) -> FastAPI:
    """Create and configure the worker FastAPI application.

    Args:
        services: Pre-wired worker services container.
        settings: Worker settings. Defaults to ``WorkerSettings()``.

    Returns:
        Configured FastAPI application instance.
    """
    settings = settings or WorkerSettings()
    configure_logging(settings.log_level)
    logger.info("Logging configured: level=%s", settings.log_level)

    app = FastAPI(title="Akgentic Worker", lifespan=_lifespan)
    app.state.services = services
    app.state.settings = settings
    app.include_router(readiness_router)
    app.include_router(teams_router)

    logger.info("Worker app built: routes mounted")
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan handler for worker graceful shutdown.

    Startup: sets ``app.state.draining = False``.
    Shutdown: sets draining flag, waits pre-drain delay, then stops all
    teams via ``worker_handle.stop_all()``.
    """
    app.state.draining = False
    logger.info("Worker lifespan startup: draining=False")
    yield
    # --- Shutdown sequence ---
    app.state.draining = True
    logger.info("Worker lifespan shutdown: draining=True")

    settings: WorkerSettings = app.state.settings
    delay = settings.shutdown_pre_drain_delay
    if delay > 0:
        logger.info("Pre-drain delay: sleeping %ds", delay)
        await asyncio.sleep(delay)

    timeout = settings.shutdown_drain_timeout
    logger.info("Stopping all teams (timeout=%ds)", timeout)
    try:
        await asyncio.wait_for(
            asyncio.to_thread(app.state.services.worker_handle.stop_all),
            timeout=timeout,
        )
        logger.info("stop_all() completed successfully")
    except TimeoutError:
        logger.warning(
            "stop_all() exceeded shutdown_drain_timeout=%ds, proceeding with exit",
            timeout,
        )
