"""Epic-57 golden-manifest identity gate for ``create_app`` (Story 57.2).

This file is deliberately a change-detector. Every constant below was captured
from the PRE-refactor factory (capture-first) and pins the community app's
observable surface: route table, middleware order, admin-route dependency
names, ``app.state`` key set, exception-handler set, and the
CORS-skipped-when-``cors_origins``-empty branch. It is the migration's
no-regression gate for re-expressing ``create_app`` over ``CoreModule`` — a
future *intentional* change to the app surface must update these constants
consciously, citing the decision that sanctions it.

Single sanctioned delta (pipeline-lead adjudication of the story's Open
Question 1): the outermost-to-innermost middleware order. The pre-refactor
factory registered CORS first and the mutation log last, which under
Starlette's reverse-add stacked ``[AdminCatalogMutationLogMiddleware,
CORSMiddleware]``. The layered builder pins CORS at TRANSPORT (outermost of
the functional stack) and the mutation log at APPLICATION (innermost), so the
composed order is ``[CORSMiddleware, AdminCatalogMutationLogMiddleware]``.
The swap changes no response bytes — CORSMiddleware only decorates responses
and short-circuits preflights; the mutation log only observes
``/admin/catalog`` mutations — and is the decision doc's stated intent
(every rejection carries ``Access-Control-*``). Every OTHER middleware
difference is a defect. Everything else in this file is pinned verbatim to
the pre-refactor capture.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import akgentic.infra
import akgentic.infra.server
from akgentic.infra.server.app import configure_process, create_app
from akgentic.infra.server.assembly import build_manifest

if TYPE_CHECKING:
    from akgentic.infra.server.deps import CommunityServices
    from akgentic.infra.server.settings import CommunitySettings

# --- Captured constants (pre-refactor factory; see module docstring) --- #

_EXPECTED_ROUTES = [
    "DELETE /admin/catalog/namespace/{namespace}",
    "DELETE /admin/catalog/{kind}/{id}",
    "DELETE /teams/{team_id}",
    "GET /admin/catalog/model_types",
    "GET /admin/catalog/namespace/{namespace}/export",
    "GET /admin/catalog/namespace/{namespace}/meta",
    "GET /admin/catalog/namespace/{namespace}/validate",
    "GET /admin/catalog/namespaces",
    "GET /admin/catalog/schema",
    "GET /admin/catalog/team/{namespace}/resolve",
    "GET /admin/catalog/{kind}",
    "GET /admin/catalog/{kind}/{id}",
    "GET /admin/catalog/{kind}/{id}/references",
    "GET /admin/catalog/{kind}/{id}/resolve",
    "GET /readiness",
    "GET /teams",
    "GET /teams/{team_id}",
    "GET /teams/{team_id}/agent-states",
    "GET /teams/{team_id}/events",
    "GET /workspace/{team_id}/file",
    "GET /workspace/{team_id}/tree",
    "GET,HEAD /docs",
    "GET,HEAD /docs/oauth2-redirect",
    "GET,HEAD /openapi.json",
    "GET,HEAD /redoc",
    "PATCH /teams/{team_id}/metadata",
    "POST /admin/catalog/clone",
    "POST /admin/catalog/namespace/import",
    "POST /admin/catalog/namespace/validate",
    "POST /admin/catalog/{kind}",
    "POST /admin/catalog/{kind}/search",
    "POST /teams",
    "POST /teams/{team_id}/human-input",
    "POST /teams/{team_id}/message",
    "POST /teams/{team_id}/message/from/{sender_name}/to/{recipient_name}",
    "POST /teams/{team_id}/message/{agent_name}",
    "POST /teams/{team_id}/notification",
    "POST /teams/{team_id}/restore",
    "POST /teams/{team_id}/stop",
    "POST /webhook/{channel}",
    "POST /workspace/{team_id}/file",
    "PUT /admin/catalog/namespace/{namespace}/meta",
    "PUT /admin/catalog/{kind}/{id}",
    "WS /ws/{team_id}",
]

# The story's single sanctioned delta (see module docstring): the layered
# builder pins CORS outermost. The pre-refactor capture read
# ["AdminCatalogMutationLogMiddleware", "CORSMiddleware"]; this expectation
# was verified green against the untouched factory with that order, then
# flipped — and ONLY this constant — under the adjudication.
_EXPECTED_MIDDLEWARE = [
    "CORSMiddleware",
    "AdminCatalogMutationLogMiddleware",
]

_EXPECTED_MIDDLEWARE_NO_CORS = [
    "AdminCatalogMutationLogMiddleware",
]

_EXPECTED_STATE_KEYS = {
    "channel_parser_registry",
    "channel_registry",
    "connection_manager",
    "draining",
    "ingestion",
    "services",
    "settings",
    "team_service",
}

_EXPECTED_EXCEPTION_HANDLERS = {
    "CatalogValidationError",
    "EntryNotFoundError",
    "HTTPException",
    "RequestValidationError",
    "ServerError",
    "ValidationError",
    "WebSocketRequestValidationError",
}

# Router-level dependencies mounted on EVERY /admin catalog route, in order.
_ROUTER_LEVEL_DEPS = [
    "require_authenticated_principal",
    "scope_catalog_caller_identity",
]

# Per-route additive gates: the four path/query owner-or-admin mutation routes
# and the single body-carried import route.
_PER_ROUTE_GATES = {
    ("PUT", "/admin/catalog/{kind}/{id}"): "require_namespace_owner_or_admin",
    ("DELETE", "/admin/catalog/{kind}/{id}"): "require_namespace_owner_or_admin",
    ("PUT", "/admin/catalog/namespace/{namespace}/meta"): "require_namespace_owner_or_admin",
    ("DELETE", "/admin/catalog/namespace/{namespace}"): "require_namespace_owner_or_admin",
    ("POST", "/admin/catalog/namespace/import"): "require_import_owner_or_admin",
}


class TestGoldenManifest:
    """Manifest identity against the pre-refactor capture."""

    def test_route_table_identity(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """The full method+path route table is byte-identical to the capture."""
        app = create_app(community_services, seeded_settings)
        assert build_manifest(app).routes == _EXPECTED_ROUTES

    def test_middleware_order(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """Outermost-to-innermost middleware order matches expectation exactly."""
        app = create_app(community_services, seeded_settings)
        assert build_manifest(app).middleware == _EXPECTED_MIDDLEWARE

    def test_cors_absent_when_origins_empty(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """Empty ``cors_origins`` skips CORSMiddleware entirely; routes unchanged."""
        no_cors = seeded_settings.model_copy(update={"cors_origins": []})
        app = create_app(community_services, no_cors)
        manifest = build_manifest(app)
        assert manifest.middleware == _EXPECTED_MIDDLEWARE_NO_CORS
        assert manifest.routes == _EXPECTED_ROUTES

    def test_admin_route_dependency_names(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """Every /admin catalog route carries the router-level auth pair, and
        exactly the five gated routes carry their additive per-route gate."""
        app = create_app(community_services, seeded_settings)
        admin_routes = [
            route
            for route in app.routes
            if isinstance(route, APIRoute) and route.path.startswith("/admin")
        ]
        assert admin_routes, "no /admin routes mounted"
        for route in admin_routes:
            for method in sorted(route.methods or set()):
                expected = list(_ROUTER_LEVEL_DEPS)
                gate = _PER_ROUTE_GATES.get((method, route.path))
                if gate is not None:
                    expected.append(gate)
                declared = [
                    getattr(dep.dependency, "__name__", str(dep.dependency))
                    for dep in route.dependencies
                ]
                live = [
                    getattr(d.call, "__name__", str(d.call))
                    for d in route.dependant.dependencies
                ]
                assert declared == expected, f"{method} {route.path}: {declared}"
                assert live == expected, f"{method} {route.path}: {live}"

    def test_state_key_set_inside_lifespan(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """The ``app.state`` key set inside a running lifespan is the capture.

        Asserted inside ``with TestClient(app)`` so the check is timing-agnostic:
        it holds whether a key is populated at build time or at startup.
        """
        app = create_app(community_services, seeded_settings)
        with TestClient(app):
            assert set(vars(app.state)["_state"].keys()) == _EXPECTED_STATE_KEYS

    def test_exception_handler_set(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """The registered exception-handler key set (by class name) is the capture."""
        app = create_app(community_services, seeded_settings)
        names = {getattr(k, "__name__", str(k)) for k in app.exception_handlers}
        assert names == _EXPECTED_EXCEPTION_HANDLERS


class TestSignatureAndImports:
    """``create_app`` stays importable with its public signature (AC 4)."""

    def test_importable_from_both_packages(self) -> None:
        """Both historical import paths resolve to the same factory."""
        assert akgentic.infra.create_app is create_app
        assert akgentic.infra.server.create_app is create_app

    def test_signature_unchanged(self) -> None:
        """``create_app(services, settings=None, modules=None, configure_process=<no-op>)``.

        Retargeted exactly once, by Story 57.7 (its AC 2 sanctioned retarget):
        the two additive parameters extend the pinned list; names, order, and
        defaults of the original pair are unchanged.
        """
        params = list(inspect.signature(create_app).parameters.values())
        assert [p.name for p in params] == ["services", "settings", "modules", "configure_process"]
        assert params[0].default is inspect.Parameter.empty
        assert params[1].default is None
        assert params[2].default is None
        assert params[3].default is configure_process


class TestLifespanShutdownOrder:
    """Full shutdown ordering: pre-drain delay, disconnect, stop, pool (AC 3).

    Existing lifespan tests assert pairwise facts (disconnect before stop,
    sleep called, pool released); this pins the complete sequence.
    """

    async def test_full_shutdown_sequence_order(self) -> None:
        from akgentic.infra.server.modules.core import _drain_lifespan as _lifespan

        call_order: list[str] = []

        settings = SimpleNamespace(shutdown_pre_drain_delay=5, shutdown_drain_timeout=30)

        async def _disconnect_all() -> None:
            call_order.append("disconnect_all")

        connection_manager = AsyncMock()
        connection_manager.disconnect_all = _disconnect_all

        worker_handle = MagicMock()
        worker_handle.stop_all = MagicMock()
        services = SimpleNamespace(worker_handle=worker_handle)

        app = MagicMock()
        app.state = SimpleNamespace(
            settings=settings,
            connection_manager=connection_manager,
            services=services,
        )
        # StateKey resolves ``source.app.state`` for non-FastAPI sources.
        app.app = app

        async def _fake_sleep(seconds: float) -> None:
            call_order.append("pre_drain_delay")

        async def _fake_to_thread(fn: object) -> None:
            call_order.append("stop_all")

        with (
            patch.object(asyncio, "sleep", _fake_sleep),
            patch.object(asyncio, "to_thread", _fake_to_thread),
            patch(
                "akgentic.infra.server.modules.core.shutdown_reader_pool",
                side_effect=lambda: call_order.append("shutdown_reader_pool"),
            ),
        ):
            ctx = _lifespan(app)
            await ctx.__aenter__()
            assert app.state.draining is False
            await ctx.__aexit__(None, None, None)

        assert app.state.draining is True
        assert call_order == [
            "pre_drain_delay",
            "disconnect_all",
            "stop_all",
            "shutdown_reader_pool",
        ]
