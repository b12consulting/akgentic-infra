"""FastAPI application factory for the akgentic-infra server.

Per ADR-023: the ``/admin/catalog/*`` HTTP surface IS the v2 unified router
(``akgentic.catalog.api.router.build_router``) with ``AuthStrategy`` wired
as a FastAPI dependency. The catalog package itself owns request/response
validation, error mapping, and CRUD semantics; infra owns only the mount
point, the auth gate, and the structured mutation log middleware — all of
which live in ``CoreModule`` (ADR-039, epic 57).

Uniform tier bootstrap (ADR-039 §5): :func:`server_modules` declares the
community composition (the ONE place it exists — tiers append their modules
to it), :func:`create_server_app` is the settings-only factory a bare
``uvicorn --factory`` target can call, and :func:`create_app` is the
pre-wired-services entry point carrying the assembly sequence exactly once —
process-global pre-steps (logging, catalog prefix policy), the community-only
ingestion backref, then ``build_app``. ``CoreModule`` is self-contained:
state, exception handlers, and the drain lifespan are its contributions, and
nothing mutates the app after ``build_app`` returns.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from akgentic.catalog import allowed_prefixes, set_allowed_prefixes
from akgentic.infra.server.assembly import AppModule, build_app
from akgentic.infra.server.deps import TierServices
from akgentic.infra.server.logging_config import configure_logging
from akgentic.infra.server.modules import CoreModule
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import CommunitySettings, ServerSettings

logger = logging.getLogger(__name__)


def server_modules(services: TierServices, settings: ServerSettings) -> list[AppModule]:
    """Declare the community composition: ``CoreModule`` alone.

    The one place the community module list exists. Tier assembly functions
    build their lists around it — hence the deliberately wide
    ``list[AppModule]`` return type.
    """
    return [CoreModule(services=services, settings=settings)]


def create_app(
    services: TierServices,
    settings: ServerSettings | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application from pre-wired services.

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
    # Deliberately a process-global pre-step of this wrapper, never a build_app
    # concern: the shared assembler must not acquire a process-global side effect.
    set_allowed_prefixes(settings.catalog_model_type_prefixes)
    logger.info("Catalog model_type allowlist: %s", allowed_prefixes())
    modules = server_modules(services, settings)
    core = next(m for m in modules if isinstance(m, CoreModule))
    _wire_ingestion(services, core.team_service)
    return build_app(settings, services, modules)


def create_server_app(settings: CommunitySettings | None = None) -> FastAPI:
    """Settings-only community factory: default settings, wire, compose.

    The uniform tier-bootstrap entry point (department and enterprise expose
    the same name). Constructs the default settings itself so a bare
    ``uvicorn --factory`` target works, wires the community services, then
    delegates to :func:`create_app` — the assembly sequence exists once.
    """
    # Function-local: ``wiring`` reaches back into server internals
    # (auth_loader → adapters), so a module-level import here is circular.
    from akgentic.infra.wiring import wire_community

    settings = settings or CommunitySettings()
    services = wire_community(settings)
    return create_app(services, settings)


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
