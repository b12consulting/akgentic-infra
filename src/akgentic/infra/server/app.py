"""FastAPI application factory for the akgentic-infra server."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from akgentic.infra.adapters.channel_parser_registry import ChannelParserRegistry
from akgentic.infra.protocols.channels import ChannelRegistry, InteractionChannelIngestion
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.routes.catalog import router as catalog_router
from akgentic.infra.server.routes.frontend_adapter import load_frontend_adapter
from akgentic.infra.server.routes.teams import router as teams_router
from akgentic.infra.server.routes.webhook import router as webhook_router
from akgentic.infra.server.routes.workspace import router as workspace_router
from akgentic.infra.server.routes.ws import ConnectionManager
from akgentic.infra.server.routes.ws import router as ws_router
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import ServerSettings


def create_app(
    services: CommunityServices,
    team_service: TeamService,
    settings: ServerSettings | None = None,
    cors_origins: list[str] | None = None,
    channel_parser_registry: ChannelParserRegistry | None = None,
    channel_registry: ChannelRegistry | None = None,
    ingestion: InteractionChannelIngestion | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        services: Wired community services container.
        team_service: Pre-built team service for dependency injection.
        settings: Server settings. Defaults to ServerSettings().
        cors_origins: Allowed CORS origins. Defaults to ["*"] for community tier.
        channel_parser_registry: Optional channel parser registry for webhook support.
        channel_registry: Required when channel_parser_registry is provided.
        ingestion: Required when channel_parser_registry is provided.

    Returns:
        Configured FastAPI application instance.

    Raises:
        ValueError: If channel_parser_registry is provided without channel_registry/ingestion.
    """
    app = FastAPI(title="Akgentic Platform API")

    if cors_origins is None:
        cors_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.services = services
    app.state.team_service = team_service
    app.state.settings = settings or ServerSettings()
    app.state.connection_manager = ConnectionManager()

    app.include_router(teams_router)
    app.include_router(catalog_router)
    app.include_router(workspace_router)
    app.include_router(ws_router)

    if channel_parser_registry is not None:
        if channel_registry is None or ingestion is None:
            msg = (
                "channel_registry and ingestion are required "
                "when channel_parser_registry is provided"
            )
            raise ValueError(msg)
        app.state.channel_parser_registry = channel_parser_registry
        app.state.channel_registry = channel_registry
        app.state.ingestion = ingestion
        app.include_router(webhook_router)

    if app.state.settings.frontend_adapter:
        adapter = load_frontend_adapter(app.state.settings.frontend_adapter)
        adapter.register_routes(app)
        app.state.frontend_adapter = adapter

    return app
