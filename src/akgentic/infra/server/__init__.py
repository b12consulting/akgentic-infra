"""Server module — FastAPI application, models, routes, services, and assembly contract."""

from __future__ import annotations

from akgentic.infra.server.app import create_app
from akgentic.infra.server.assembly import (
    APPLICATION,
    IDENTITY,
    OBSERVABILITY,
    POLICY,
    PROXY,
    SESSION,
    TRANSPORT,
    AllowlistSpec,
    AppManifest,
    AppModule,
    AssemblyError,
    BaseAppModule,
    BuildContext,
    DuplicateModuleNameError,
    DuplicateStateProviderError,
    ExceptionHandlerSpec,
    MiddlewareSpec,
    MissingStateProviderError,
    RouteSpec,
    UnpopulatedStateError,
    build_app,
    build_manifest,
)
from akgentic.infra.server.deps import CommunityServices, TierServices
from akgentic.infra.server.models import (
    CreateTeamRequest,
    EventListResponse,
    EventResponse,
    HumanInputRequest,
    SendMessageRequest,
    TeamListResponse,
    TeamResponse,
)
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import CommunitySettings, ServerSettings

__all__ = [
    "APPLICATION",
    "IDENTITY",
    "OBSERVABILITY",
    "POLICY",
    "PROXY",
    "SESSION",
    "TRANSPORT",
    "AllowlistSpec",
    "AppManifest",
    "AppModule",
    "AssemblyError",
    "BaseAppModule",
    "BuildContext",
    "CommunitySettings",
    "CommunityServices",
    "CreateTeamRequest",
    "DuplicateModuleNameError",
    "DuplicateStateProviderError",
    "EventListResponse",
    "EventResponse",
    "ExceptionHandlerSpec",
    "HumanInputRequest",
    "MiddlewareSpec",
    "MissingStateProviderError",
    "RouteSpec",
    "SendMessageRequest",
    "ServerSettings",
    "TeamListResponse",
    "TeamResponse",
    "TeamService",
    "TierServices",
    "UnpopulatedStateError",
    "build_app",
    "build_manifest",
    "create_app",
]
