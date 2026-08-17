"""Unit tests for the AppModule contract, build_app builder, and build_manifest.

The demo modules below are small in-test compositions built on ``BaseAppModule``
(the research prototype is a reference, never an import target). They compose
the REAL ``RequireAuthMiddleware`` and ``StateKey`` primitives unchanged.

``build_app`` threads ``settings``/``services`` without reading them — modules
receive what they need at construction time — so the tests pass a default
``ServerSettings`` and an unvalidated ``TierServices`` container.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from akgentic.infra.server.assembly import (
    IDENTITY,
    POLICY,
    TRANSPORT,
    AllowlistSpec,
    AppModule,
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
from akgentic.infra.server.auth import RequestUser, get_request_user
from akgentic.infra.server.deps import TierServices
from akgentic.infra.server.middleware.require_auth import RequireAuthMiddleware
from akgentic.infra.server.settings import ServerSettings
from akgentic.infra.utils import StateKey

_ORIGIN = "http://testclient.local"
_KEY = "ak_test_secret"
_API_KEYS = {_KEY: "geoff"}


class _ApiKeyAuth:
    """Fake auth strategy: X-API-Key header resolves a RequestUser, else 401."""

    def __init__(self, api_keys: dict[str, str]) -> None:
        self._api_keys = api_keys

    async def resolve_request_user(self, connection: HTTPConnection) -> RequestUser:
        user_id = self._api_keys.get(connection.headers.get("X-API-Key", ""))
        if user_id is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        return RequestUser(user_id=user_id, roles=["admin"])


class _FakeAuthServices:
    """Minimal stand-in exposing the ``.auth`` slot RequireAuthMiddleware reads."""

    def __init__(self, auth: _ApiKeyAuth) -> None:
        self.auth = auth


_SERVICES: StateKey[_FakeAuthServices] = StateKey("services", required=True)
_RATE_COUNTERS: StateKey[dict[str, int]] = StateKey("rate_counters", required=True)


class _CoreModule(BaseAppModule):
    """Demo core: readiness + /teams routes, CORS at TRANSPORT."""

    name = "core"

    def __init__(self, *, events: list[str] | None = None) -> None:
        self._events = events if events is not None else []

    def contribute_routes(self) -> list[RouteSpec]:
        readiness = APIRouter()

        @readiness.get("/readiness")
        def _readiness() -> dict[str, str]:
            return {"status": "ok"}

        teams = APIRouter(prefix="/teams")

        @teams.get("")
        def _list_teams(user: RequestUser = Depends(get_request_user)) -> dict[str, str]:
            return {"user": user.user_id, "source": "core"}

        @teams.delete("/{team_id}")
        def _delete_team(team_id: str) -> dict[str, str]:
            return {"team_id": team_id, "source": "core"}

        return [RouteSpec(router=readiness), RouteSpec(router=teams)]

    def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
        return [
            MiddlewareSpec(
                middleware_class=CORSMiddleware,
                layer=TRANSPORT,
                options={
                    "allow_origins": [_ORIGIN],
                    "allow_credentials": True,
                    "allow_methods": ["*"],
                    "allow_headers": ["*"],
                },
            )
        ]

    def contribute_allowlist(self) -> AllowlistSpec:
        return AllowlistSpec(exact=frozenset({"/readiness"}))

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        self._events.append("start:core")
        yield
        self._events.append("stop:core")


class _AuthModule(BaseAppModule):
    """Demo auth: the REAL RequireAuthMiddleware over an async-built strategy."""

    name = "auth"
    provides_state = ("services",)

    def __init__(self, *, events: list[str] | None = None) -> None:
        self._events = events if events is not None else []

    def contribute_routes(self) -> list[RouteSpec]:
        router = APIRouter(prefix="/auth")

        @router.get("/me")
        def _me(user: RequestUser = Depends(get_request_user)) -> dict[str, str]:
            return {"user": user.user_id}

        return [RouteSpec(router=router)]

    def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
        return [
            MiddlewareSpec(
                middleware_class=RequireAuthMiddleware,
                layer=IDENTITY,
                options={
                    "exact_allowlist": context.allowlist.exact,
                    "prefix_allowlist": context.allowlist.prefixes,
                },
                requires_state=(_SERVICES.name,),
            )
        ]

    def contribute_allowlist(self) -> AllowlistSpec:
        return AllowlistSpec(prefixes=("/auth/",))

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        # The enterprise constraint: the strategy needs an ASYNC step (a
        # sidecar secret fetch) that cannot run in the sync factory body.
        api_keys = await self._fetch_keys()
        _SERVICES.set(app, _FakeAuthServices(_ApiKeyAuth(api_keys)))
        self._events.append("start:auth")
        yield
        self._events.append("stop:auth")

    async def _fetch_keys(self) -> dict[str, str]:
        await asyncio.sleep(0)  # stands in for the sidecar secret-store call
        return dict(_API_KEYS)


class _RateCounterMiddleware:
    """Pure-ASGI counter resolving its counters dict from app.state per request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            counters = _RATE_COUNTERS.require(HTTPConnection(scope))
            counters[scope["path"]] = counters.get(scope["path"], 0) + 1
        await self.app(scope, receive, send)


class _PolicyModule(BaseAppModule):
    """Demo POLICY-band module: provides the counters its own middleware requires."""

    name = "policy"
    provides_state = ("rate_counters",)

    def __init__(self, *, events: list[str] | None = None) -> None:
        self._events = events if events is not None else []

    def contribute_routes(self) -> list[RouteSpec]:
        router = APIRouter(prefix="/admin")

        @router.get("/rate-report")
        def _report(request: Request) -> dict[str, int]:
            return _RATE_COUNTERS.require(request)

        return [RouteSpec(router=router)]

    def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
        return [
            MiddlewareSpec(
                middleware_class=_RateCounterMiddleware,
                layer=POLICY,
                requires_state=(_RATE_COUNTERS.name,),
            )
        ]

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        await asyncio.sleep(0)  # stands in for the sidecar connect
        _RATE_COUNTERS.set(app, {})
        self._events.append("start:policy")
        yield
        self._events.append("stop:policy")


class _TeamsOverrideModule(BaseAppModule):
    """Overrides DELETE /teams/{id} — listed BEFORE core, so its route wins."""

    name = "teams-override"

    def contribute_routes(self) -> list[RouteSpec]:
        router = APIRouter(prefix="/teams")

        @router.delete("/{team_id}")
        def _delete_team(team_id: str) -> dict[str, str]:
            return {"team_id": team_id, "source": "override"}

        return [RouteSpec(router=router)]


class _NoopMiddleware:
    """Pass-through ASGI middleware for validation-path tests."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app(scope, receive, send)


def _build(modules: list[BaseAppModule]) -> FastAPI:
    """Build an app from demo modules with pass-through composition inputs."""
    return build_app(ServerSettings(), TierServices.model_construct(), modules)


def _demo_modules(events: list[str] | None = None) -> list[BaseAppModule]:
    return [_CoreModule(events=events), _AuthModule(events=events)]


class TestCompositionBehaviorParity:
    """AC 1.1: the composed demo app behaves like today's tiers."""

    def test_allowlisted_path_reachable_without_principal(self) -> None:
        with TestClient(_build(_demo_modules())) as client:
            assert client.get("/readiness").status_code == 200

    def test_gated_path_rejected_pre_routing_without_principal(self) -> None:
        with TestClient(_build(_demo_modules())) as client:
            response = client.get("/teams")
            assert response.status_code == 401
            assert response.json() == {"detail": "Not authenticated"}

    def test_authenticated_request_resolves_principal(self) -> None:
        with TestClient(_build(_demo_modules())) as client:
            response = client.get("/teams", headers={"X-API-Key": _KEY})
            assert response.status_code == 200
            assert response.json() == {"user": "geoff", "source": "core"}

    def test_allowlisted_auth_route_falls_back_to_anonymous(self) -> None:
        with TestClient(_build(_demo_modules())) as client:
            assert client.get("/auth/me").json() == {"user": "anonymous"}


class TestCorsOutermost:
    """AC 1.2: CORS position depends on its layer, never on module-list order."""

    @pytest.mark.parametrize("auth_first", [False, True])
    def test_pre_routing_401_carries_cors_headers(self, auth_first: bool) -> None:
        modules: list[BaseAppModule] = _demo_modules()
        if auth_first:
            modules.reverse()
        app = _build(modules)
        assert build_manifest(app).middleware == ["CORSMiddleware", "RequireAuthMiddleware"]
        with TestClient(app) as client:
            rejected = client.get("/teams", headers={"Origin": _ORIGIN})
            assert rejected.status_code == 401
            assert rejected.headers["access-control-allow-origin"] == _ORIGIN


class TestRouteCollision:
    """AC 1.3: route collisions resolve by module-list order — earlier wins."""

    def test_earlier_module_wins_route_collision(self) -> None:
        modules: list[BaseAppModule] = [_TeamsOverrideModule(), *_demo_modules()]
        with TestClient(_build(modules)) as client:
            response = client.delete("/teams/t1", headers={"X-API-Key": _KEY})
            assert response.json() == {"team_id": "t1", "source": "override"}

    def test_later_module_loses_route_collision(self) -> None:
        modules: list[BaseAppModule] = [*_demo_modules(), _TeamsOverrideModule()]
        with TestClient(_build(modules)) as client:
            response = client.delete("/teams/t1", headers={"X-API-Key": _KEY})
            assert response.json() == {"team_id": "t1", "source": "core"}


class TestAsyncStartupCollaborators:
    """AC 1.4: sync-registered middleware resolves async-startup state via StateKey."""

    def test_sync_registered_middleware_reads_lifespan_populated_state(self) -> None:
        modules: list[BaseAppModule] = [*_demo_modules(), _PolicyModule()]
        with TestClient(_build(modules)) as client:
            headers = {"X-API-Key": _KEY}
            client.get("/teams", headers=headers)
            client.get("/teams", headers=headers)
            report = client.get("/admin/rate-report", headers=headers).json()
            # The route reads the SAME dict the POLICY middleware increments,
            # through the StateKey — no stack walk, no lazy wrapper.
            assert report["/teams"] == 2


class TestLifespanComposition:
    """AC 1.5: startup runs in module-list order, shutdown in reverse."""

    def test_startup_in_list_order_and_shutdown_in_reverse(self) -> None:
        events: list[str] = []
        modules: list[BaseAppModule] = [
            _CoreModule(events=events),
            _AuthModule(events=events),
            _PolicyModule(events=events),
        ]
        with TestClient(_build(modules)):
            assert events == ["start:core", "start:auth", "start:policy"]
        assert events[3:] == ["stop:policy", "stop:auth", "stop:core"]


class _ForgetfulModule(BaseAppModule):
    """Declares a state key its lifespan never populates — must abort boot."""

    name = "forgetful"
    provides_state = ("orphan_slot",)

    def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
        return [
            MiddlewareSpec(
                middleware_class=_NoopMiddleware,
                layer=POLICY,
                requires_state=("orphan_slot",),
            )
        ]


class TestBootTimeStateVerification:
    """AC 1.6: an unpopulated required state key aborts startup, naming module and key."""

    def test_unpopulated_required_state_aborts_boot_naming_module_and_key(self) -> None:
        app = _build([_CoreModule(), _ForgetfulModule()])
        with pytest.raises(UnpopulatedStateError, match="orphan_slot.*forgetful"):
            with TestClient(app):
                pass

    def test_populated_required_state_boots(self) -> None:
        class _HealthyModule(_ForgetfulModule):
            name = "healthy"
            provides_state = ("orphan_slot",)

            @asynccontextmanager
            async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
                app.state.orphan_slot = object()
                yield

        with TestClient(_build([_CoreModule(), _HealthyModule()])) as client:
            assert client.get("/readiness").status_code == 200


class TestManifestSnapshot:
    """AC 1.7: build_manifest reports the route table and middleware order."""

    def test_manifest_reports_routes_and_middleware_order(self) -> None:
        manifest = build_manifest(_build(_demo_modules()))
        assert manifest.middleware == ["CORSMiddleware", "RequireAuthMiddleware"]
        for entry in ("GET /readiness", "GET /teams", "DELETE /teams/{team_id}", "GET /auth/me"):
            assert entry in manifest.routes

    def test_manifest_marks_websocket_routes(self) -> None:
        class _WsModule(BaseAppModule):
            name = "ws"

            def contribute_routes(self) -> list[RouteSpec]:
                router = APIRouter()

                @router.websocket("/ws-demo")
                async def _ws(websocket: object) -> None: ...

                return [RouteSpec(router=router)]

        manifest = build_manifest(_build([_WsModule()]))
        assert "WS /ws-demo" in manifest.routes


class TestBuildTimeValidation:
    """AC 2: build-time failures are typed errors naming the offense."""

    def test_duplicate_module_name_rejected(self) -> None:
        with pytest.raises(DuplicateModuleNameError, match="core"):
            _build([_CoreModule(), _CoreModule()])

    def test_duplicate_state_provider_rejected(self) -> None:
        class _SecondProvider(BaseAppModule):
            name = "second"
            provides_state = ("services",)

        with pytest.raises(DuplicateStateProviderError, match="services.*auth.*second"):
            _build([*_demo_modules(), _SecondProvider()])

    def test_requires_state_without_producer_rejected(self) -> None:
        class _OrphanRequirer(BaseAppModule):
            name = "orphan-requirer"

            def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
                return [
                    MiddlewareSpec(
                        middleware_class=_NoopMiddleware,
                        layer=POLICY,
                        requires_state=("never_provided",),
                    )
                ]

        with pytest.raises(MissingStateProviderError, match="orphan-requirer.*never_provided"):
            _build([_CoreModule(), _OrphanRequirer()])


class TestRealRequireAuthComposition:
    """AC 3: the real RequireAuthMiddleware composes at IDENTITY with zero edits."""

    def test_real_require_auth_composes_at_identity(self) -> None:
        app = _build(_demo_modules())
        # The manifest names the real class — not a wrapper, not a copy.
        assert "RequireAuthMiddleware" in build_manifest(app).middleware
        with TestClient(app) as client:
            assert client.get("/teams").status_code == 401
            assert client.get("/teams", headers={"X-API-Key": _KEY}).status_code == 200


def _public_paths_are_exempt(connection: HTTPConnection) -> bool:
    """Pure predicate (configuration in function form): gate everything but /public."""
    path = str(connection.scope.get("path", ""))
    return not path.startswith("/public")


class TestMiddlewareOptions:
    """AC 4: options carries settings values and PURE callables — never a live service.

    A ``requires_principal`` predicate is configuration in function form: it is
    stateless and build-time-known, so it travels through ``options``. Runtime
    collaborators (the auth strategy) still go through ``requires_state`` +
    ``StateKey.require``, populated by a module lifespan.
    """

    def test_pure_predicate_callable_passes_through_options(self) -> None:
        class _PredicateAuthModule(BaseAppModule):
            name = "predicate-auth"
            provides_state = ("services",)

            def contribute_routes(self) -> list[RouteSpec]:
                router = APIRouter(prefix="/public")

                @router.get("/ping")
                def _ping() -> dict[str, str]:
                    return {"status": "public"}

                return [RouteSpec(router=router)]

            def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
                return [
                    MiddlewareSpec(
                        middleware_class=RequireAuthMiddleware,
                        layer=IDENTITY,
                        options={"requires_principal": _public_paths_are_exempt},
                        requires_state=(_SERVICES.name,),
                    )
                ]

            @asynccontextmanager
            async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
                _SERVICES.set(app, _FakeAuthServices(_ApiKeyAuth(dict(_API_KEYS))))
                yield

        with TestClient(_build([_CoreModule(), _PredicateAuthModule()])) as client:
            assert client.get("/public/ping").status_code == 200
            assert client.get("/teams").status_code == 401


class _DemoError(Exception):
    """Demo domain error for exception-handler contribution tests."""


def _first_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(status_code=418, content={"handler": "first"})


def _second_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(status_code=418, content={"handler": "second"})


class _RaisingModule(BaseAppModule):
    """Contributes a route that raises ``_DemoError`` and a handler for it."""

    name = "raising"

    def contribute_routes(self) -> list[RouteSpec]:
        router = APIRouter()

        @router.get("/boom")
        def _boom() -> dict[str, str]:
            raise _DemoError("boom")

        return [RouteSpec(router=router)]

    def contribute_exception_handlers(self) -> list[ExceptionHandlerSpec]:
        return [ExceptionHandlerSpec(exception_class=_DemoError, handler=_first_handler)]


class TestExceptionHandlers:
    """AC 2 scope (Task 2): ExceptionHandlerSpec registration in module-list order."""

    def test_contributed_handler_shapes_the_response(self) -> None:
        with TestClient(_build([_RaisingModule()])) as client:
            response = client.get("/boom")
            assert response.status_code == 418
            assert response.json() == {"handler": "first"}

    def test_same_class_re_registration_last_wins(self) -> None:
        class _OverridingHandlerModule(BaseAppModule):
            name = "overriding"

            def contribute_exception_handlers(self) -> list[ExceptionHandlerSpec]:
                return [
                    ExceptionHandlerSpec(exception_class=_DemoError, handler=_second_handler)
                ]

        modules: list[BaseAppModule] = [_RaisingModule(), _OverridingHandlerModule()]
        with TestClient(_build(modules)) as client:
            assert client.get("/boom").json() == {"handler": "second"}


class TestContractSurface:
    """Task 3: runtime-checkable protocol and BaseAppModule no-op defaults."""

    def test_demo_module_satisfies_runtime_checkable_protocol(self) -> None:
        assert isinstance(_CoreModule(), AppModule)
        assert isinstance(BaseAppModule(), AppModule)

    def test_base_module_defaults_contribute_nothing(self) -> None:
        base = BaseAppModule()
        context = BuildContext(allowlist=AllowlistSpec())
        assert base.contribute_routes() == []
        assert base.contribute_middleware(context) == []
        assert base.contribute_allowlist() == AllowlistSpec()
        assert base.contribute_exception_handlers() == []

    def test_all_defaults_module_composes_and_boots(self) -> None:
        with TestClient(_build([BaseAppModule()])):
            pass

    def test_merged_allowlist_deduplicates_and_preserves_order(self) -> None:
        class _CapturingModule(BaseAppModule):
            name = "capturing"
            captured: BuildContext | None = None

            def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
                self.captured = context
                return []

            def contribute_allowlist(self) -> AllowlistSpec:
                return AllowlistSpec(prefixes=("/auth/", "/extra/"))

        capturing = _CapturingModule()
        _build([*_demo_modules(), capturing])
        assert capturing.captured is not None
        # "/auth/" appears once (auth module contributed it first), then "/extra/".
        assert capturing.captured.allowlist.prefixes == ("/auth/", "/extra/")
        assert "/readiness" in capturing.captured.allowlist.exact
