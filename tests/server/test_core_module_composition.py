"""Department-shaped composition proof for the self-contained ``CoreModule``.

Story 57.5's reason to exist: ``build_app(settings, services, [CoreModule(...),
FakeTierModule()])`` — WITHOUT ever calling ``create_app`` — produces a fully
working community app plus the tier's contributions. A tier author composes the
community surface directly; nothing forces decorating the wrapper's result.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send

from akgentic.infra.server.app import create_app
from akgentic.infra.server.assembly import (
    IDENTITY,
    BaseAppModule,
    BuildContext,
    MiddlewareSpec,
    RouteSpec,
    StateEntry,
    build_app,
    build_manifest,
)
from akgentic.infra.server.modules import CoreModule
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.state_keys import (
    CHANNEL_PARSERS,
    CHANNEL_REGISTRY,
    CONNECTION_MANAGER,
    INGESTION,
    SERVICES,
    SETTINGS,
    TEAM_SERVICE,
)
from akgentic.infra.utils import StateKey

if TYPE_CHECKING:
    from akgentic.infra.server.deps import CommunityServices
    from akgentic.infra.server.settings import CommunitySettings

_TIER_FLAG: StateKey[str] = StateKey("tier_flag", required=True)


class _TierMiddleware:
    """Pass-through ASGI middleware standing in for a tier's IDENTITY layer."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app(scope, receive, send)


class FakeTierModule(BaseAppModule):
    """Minimal tier module: one route, one IDENTITY middleware, one state key."""

    name = "fake-tier"

    def contribute_routes(self) -> list[RouteSpec]:
        router = APIRouter()

        @router.get("/tier/ping")
        def _ping() -> dict[str, str]:
            return {"tier": "fake"}

        return [RouteSpec(router=router)]

    def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
        return [MiddlewareSpec(middleware_class=_TierMiddleware, layer=IDENTITY)]

    def contribute_state(self) -> Sequence[StateEntry[Any]]:
        return [_TIER_FLAG.entry("fake-tier-value")]


def _compose(
    settings: CommunitySettings,
    services: CommunityServices,
    *tier_modules: BaseAppModule,
) -> FastAPI:
    """Department-shaped assembly: CoreModule + tier modules, never create_app."""
    return build_app(
        settings,
        services,
        [CoreModule(services=services, settings=settings), *tier_modules],
    )


class TestDepartmentShapedComposition:
    """AC 1 (Story 57.5): the four composition-proof assertions."""

    def test_composed_app_serves_community_surface_plus_tier_contributions(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """(a) full community surface served; (c) tier route + middleware present."""
        app = _compose(seeded_settings, community_services, FakeTierModule())
        manifest = build_manifest(app)
        reference = build_manifest(create_app(community_services, seeded_settings))
        assert set(reference.routes) <= set(manifest.routes)
        assert "GET /tier/ping" in manifest.routes
        assert "_TierMiddleware" in manifest.middleware
        with TestClient(app) as client:
            assert client.get("/readiness").status_code == 200
            assert client.get("/tier/ping").json() == {"tier": "fake"}

    def test_all_community_state_keys_readable_at_build_time(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """(b) every community state key readable with no lifespan entered; (c) tier key too."""
        app = _compose(seeded_settings, community_services, FakeTierModule())
        assert SERVICES.require(app) is community_services
        assert isinstance(TEAM_SERVICE.require(app), TeamService)
        assert SETTINGS.require(app) is seeded_settings
        assert CONNECTION_MANAGER.require(app) is not None
        assert CHANNEL_PARSERS.require(app) is community_services.channel_parser_registry
        assert CHANNEL_REGISTRY.require(app) is community_services.channel_registry
        assert INGESTION.require(app) is community_services.ingestion
        assert _TIER_FLAG.require(app) == "fake-tier-value"

    def test_core_module_only_manifest_equals_create_app_manifest(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """(d) the CoreModule-ONLY composition is manifest-identical to create_app's app."""
        composed = _compose(seeded_settings, community_services)
        reference = create_app(community_services, seeded_settings)
        assert build_manifest(composed) == build_manifest(reference)
