"""FastAPI application factory for the akgentic-infra server."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from akgentic.catalog.api import (
    add_exception_handlers,
    agent_router,
    team_router,
    template_router,
    tool_router,
)
from akgentic.catalog.api.agent_router import set_catalog as set_agent_catalog
from akgentic.catalog.api.team_router import set_catalog as set_team_catalog
from akgentic.catalog.api.template_router import set_catalog as set_template_catalog
from akgentic.catalog.api.tool_router import set_catalog as set_tool_catalog
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.routes.frontend_adapter import load_frontend_adapter
from akgentic.infra.server.routes.teams import router as teams_router
from akgentic.infra.server.routes.webhook import router as webhook_router
from akgentic.infra.server.routes.workspace import router as workspace_router
from akgentic.infra.server.routes.ws import ConnectionManager
from akgentic.infra.server.routes.ws import router as ws_router
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import ServerSettings
from akgentic.infra.wiring import wire_community


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Single entry-point factory: wires community services, constructs
    ``TeamService``, completes deferred ``LocalIngestion`` wiring, and
    mounts all routes.

    Args:
        settings: Server settings. Defaults to ``ServerSettings()``.

    Returns:
        Configured FastAPI application instance.
    """
    settings = settings or ServerSettings()
    services = wire_community(settings)
    team_service = _build_team_service(services)
    _wire_ingestion(services, team_service)
    return _build_app(services, team_service, settings)


def _build_team_service(services: CommunityServices) -> TeamService:
    """Construct TeamService from wired community services."""
    return TeamService(
        services=services,
        team_catalog=services.team_catalog,
        agent_catalog=services.agent_catalog,
        tool_catalog=services.tool_catalog,
        template_catalog=services.template_catalog,
    )


def _wire_ingestion(services: CommunityServices, team_service: TeamService) -> None:
    """Complete deferred LocalIngestion wiring with the constructed TeamService."""
    from akgentic.infra.adapters.local_ingestion import LocalIngestion

    if isinstance(services.ingestion, LocalIngestion):
        services.ingestion.team_service = team_service


def _build_app(
    services: CommunityServices,
    team_service: TeamService,
    settings: ServerSettings,
) -> FastAPI:
    """Assemble the FastAPI app from pre-built services (shared by create_app and tests).

    Args:
        services: Wired community services container.
        team_service: Pre-built team service.
        settings: Server settings.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(title="Akgentic Platform API")
    _add_cors(app, settings.cors_origins)
    _store_state(app, services, team_service, settings)
    _inject_catalogs(services)
    _mount_routes(app, settings)
    add_exception_handlers(app)
    return app


def _add_cors(app: FastAPI, cors_origins: list[str]) -> None:
    """Add CORS middleware to the application."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _store_state(
    app: FastAPI,
    services: CommunityServices,
    team_service: TeamService,
    settings: ServerSettings,
) -> None:
    """Store services and configuration on app.state for dependency injection."""
    app.state.services = services
    app.state.team_service = team_service
    app.state.settings = settings
    app.state.connection_manager = ConnectionManager()
    app.state.channel_parser_registry = services.channel_parser_registry
    app.state.channel_registry = services.channel_registry
    app.state.ingestion = services.ingestion


def _inject_catalogs(services: CommunityServices) -> None:
    """Inject catalog service instances into akgentic-catalog router modules."""
    set_agent_catalog(services.agent_catalog)
    set_team_catalog(services.team_catalog)
    set_template_catalog(services.template_catalog)
    set_tool_catalog(services.tool_catalog)


def _mount_routes(app: FastAPI, settings: ServerSettings) -> None:
    """Mount all API routers and optional frontend adapter."""
    app.include_router(teams_router)
    app.include_router(agent_router, prefix="/catalog")
    app.include_router(team_router, prefix="/catalog")
    app.include_router(template_router, prefix="/catalog")
    app.include_router(tool_router, prefix="/catalog")
    app.include_router(workspace_router)
    app.include_router(ws_router)
    app.include_router(webhook_router)

    if settings.frontend_adapter:
        adapter = load_frontend_adapter(settings.frontend_adapter)
        adapter.register_routes(app)
        app.state.frontend_adapter = adapter
