"""FastAPI application factory for the akgentic-infra server."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.routes.teams import router as teams_router
from akgentic.infra.server.routes.ws import ConnectionManager
from akgentic.infra.server.routes.ws import router as ws_router
from akgentic.infra.server.services.team_service import TeamService


def create_app(
    services: CommunityServices,
    team_service: TeamService,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        services: Wired community services container.
        team_service: Pre-built team service for dependency injection.
        cors_origins: Allowed CORS origins. Defaults to ["*"] for community tier.

    Returns:
        Configured FastAPI application instance.
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
    app.state.connection_manager = ConnectionManager()

    app.include_router(teams_router)
    app.include_router(ws_router)

    return app
