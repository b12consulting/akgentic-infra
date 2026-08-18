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
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi import APIRouter, Depends, FastAPI, Request, WebSocket
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

import akgentic.infra.server
from akgentic.infra.server.app import create_app, server_modules
from akgentic.infra.server.assembly import (
    EXTENSION,
    IDENTITY,
    POLICY,
    TRANSPORT,
    AllowlistSpec,
    AppManifest,
    AppModule,
    AssemblyError,
    BaseAppModule,
    BuildContext,
    DuplicateModuleNameError,
    DuplicateStateProviderError,
    ExceptionHandlerRegistrar,
    ExceptionHandlerSpec,
    ManifestDelta,
    MiddlewareSpec,
    MissingStateProviderError,
    RouteCollisionError,
    RouteSpec,
    StateEntry,
    UnpopulatedStateError,
    build_app,
    build_manifest,
    manifest_delta,
)
from akgentic.infra.server.auth import RequestUser, get_request_user
from akgentic.infra.server.deps import CommunityServices, TierServices
from akgentic.infra.server.middleware.require_auth import RequireAuthMiddleware
from akgentic.infra.server.settings import CommunitySettings, ServerSettings
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


_DELETE_TEAM = "DELETE /teams/{team_id}"


class _TeamsOverrideModule(BaseAppModule):
    """Overrides DELETE /teams/{id} — listed BEFORE core, so its route wins.

    Winning is what obliges it to declare the shadow, so the declaration is the
    default here; ``overrides=()`` builds the undeclared-collision case.
    """

    name = "teams-override"

    def __init__(self, *, overrides: tuple[str, ...] = (_DELETE_TEAM,)) -> None:
        self._overrides = overrides

    def contribute_routes(self) -> list[RouteSpec]:
        router = APIRouter(prefix="/teams")

        @router.delete("/{team_id}")
        def _delete_team(team_id: str) -> dict[str, str]:
            return {"team_id": team_id, "source": "override"}

        return [RouteSpec(router=router, overrides=self._overrides)]


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
    """AC 1.3 / Story 63.2 AC 3-4: earlier wins at runtime, and must have declared it."""

    def test_earlier_module_wins_route_collision(self) -> None:
        modules: list[BaseAppModule] = [_TeamsOverrideModule(), *_demo_modules()]
        with TestClient(_build(modules)) as client:
            response = client.delete("/teams/t1", headers={"X-API-Key": _KEY})
            assert response.json() == {"team_id": "t1", "source": "override"}

    def test_later_module_declaring_the_shadow_still_raises(self) -> None:
        """The declaration belongs to the winner; a loser cannot declare it away."""
        modules: list[BaseAppModule] = [*_demo_modules(), _TeamsOverrideModule()]
        with pytest.raises(RouteCollisionError, match=r"DELETE /teams/\{team_id\}"):
            _build(modules)


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


_BUILD_TIME: StateKey[object] = StateKey("build_time_slot", required=True)


class _BuildTimeStateModule(BaseAppModule):
    """Contributes one build-time state entry through the typed StateEntry pair."""

    name = "build-time-state"

    def __init__(self, value: object) -> None:
        self._value = value

    def contribute_state(self) -> Sequence[StateEntry[Any]]:
        return [_BUILD_TIME.entry(self._value)]


class TestContributeState:
    """AC 3 (Story 57.5): build-time state entries and the two-phase single-writer rule."""

    def test_entry_readable_immediately_after_build_without_lifespan(self) -> None:
        sentinel = object()
        app = _build([_CoreModule(), _BuildTimeStateModule(sentinel)])
        # No TestClient, no lifespan — the builder applied the entry at build time.
        assert _BUILD_TIME.require(app) is sentinel

    def test_two_build_time_writers_rejected(self) -> None:
        class _SecondWriter(BaseAppModule):
            name = "second-writer"

            def contribute_state(self) -> Sequence[StateEntry[Any]]:
                return [_BUILD_TIME.entry(object())]

        with pytest.raises(
            DuplicateStateProviderError,
            match="build_time_slot.*build-time-state.*second-writer",
        ):
            _build([_BuildTimeStateModule(object()), _SecondWriter()])

    def test_build_time_key_clashing_with_lifespan_provider_rejected(self) -> None:
        class _LifespanWriter(BaseAppModule):
            name = "lifespan-writer"
            provides_state = ("build_time_slot",)

        with pytest.raises(
            DuplicateStateProviderError,
            match="build_time_slot.*build-time-state.*lifespan-writer",
        ):
            _build([_BuildTimeStateModule(object()), _LifespanWriter()])

    def test_same_module_writing_both_phases_rejected(self) -> None:
        class _BothPhasesModule(BaseAppModule):
            name = "both-phases"
            provides_state = ("build_time_slot",)

            def contribute_state(self) -> Sequence[StateEntry[Any]]:
                return [_BUILD_TIME.entry(object())]

        with pytest.raises(
            DuplicateStateProviderError,
            match="build_time_slot.*both-phases.*both-phases",
        ):
            _build([_BothPhasesModule()])

    def test_requires_state_satisfied_by_build_time_entry(self) -> None:
        class _BuildTimeRequirer(BaseAppModule):
            name = "build-time-requirer"

            def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
                return [
                    MiddlewareSpec(
                        middleware_class=_NoopMiddleware,
                        layer=POLICY,
                        requires_state=(_BUILD_TIME.name,),
                    )
                ]

        # Build-time validation passes (the entry is the producer) AND startup
        # verification passes (the slot is populated before startup runs).
        modules: list[BaseAppModule] = [_BuildTimeStateModule(object()), _BuildTimeRequirer()]
        with TestClient(_build(modules)):
            pass


class _PathModule(BaseAppModule):
    """One route at a configurable method/path/prefix, answering with its own name.

    The response body carries the module name, so a request proves *which*
    module's handler served a shadowed path — never ``app.routes`` order.
    """

    def __init__(
        self,
        name: str,
        path: str,
        *,
        method: str = "GET",
        prefix: str = "",
        overrides: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self._path = path
        self._method = method
        self._prefix = prefix
        self._overrides = overrides

    def contribute_routes(self) -> list[RouteSpec]:
        router = APIRouter()
        source = self.name

        @router.api_route(self._path, methods=[self._method])
        def _handler() -> dict[str, str]:
            return {"source": source}

        return [RouteSpec(router=router, prefix=self._prefix, overrides=self._overrides)]


class _WsPathModule(BaseAppModule):
    """One websocket route on a shared path — websocket specs collide too."""

    def __init__(self, name: str, *, overrides: tuple[str, ...] = ()) -> None:
        self.name = name
        self._overrides = overrides

    def contribute_routes(self) -> list[RouteSpec]:
        router = APIRouter()

        @router.websocket("/ws-shared")
        async def _ws(websocket: WebSocket) -> None: ...

        return [RouteSpec(router=router, overrides=self._overrides)]


class _TwoSpecModule(BaseAppModule):
    """One module contributing two specs that collide with each other."""

    name = "twice"

    def __init__(self, *, overrides: tuple[str, ...] = ()) -> None:
        self._overrides = overrides

    def contribute_routes(self) -> list[RouteSpec]:
        first = APIRouter()

        @first.get("/duplicated")
        def _first() -> dict[str, str]:
            return {"spec": "first"}

        second = APIRouter()

        @second.get("/duplicated")
        def _second() -> dict[str, str]:
            return {"spec": "second"}

        return [RouteSpec(router=first, overrides=self._overrides), RouteSpec(router=second)]


class _CountingRoutesModule(BaseAppModule):
    """Counts ``contribute_routes`` invocations — the builder must call it once."""

    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def contribute_routes(self) -> list[RouteSpec]:
        self.calls += 1
        router = APIRouter(prefix="/counted")

        @router.get("/ping")
        def _ping() -> dict[str, str]:
            return {"status": "ok"}

        return [RouteSpec(router=router)]


class TestUndeclaredRouteCollision:
    """Story 63.2 AC 2: an undeclared shadow is a named build-time error."""

    def test_undeclared_collision_names_both_modules_and_the_literal_entry(self) -> None:
        modules: list[BaseAppModule] = [_TeamsOverrideModule(overrides=()), *_demo_modules()]
        with pytest.raises(RouteCollisionError) as excinfo:
            _build(modules)
        message = str(excinfo.value)
        # The message IS the deliverable: it must carry the winner, the loser,
        # the method, the path, and the literal line to paste. Asserted by
        # fragment so a wording improvement is not a test edit.
        for fragment in (
            "'teams-override'",
            "'core'",
            "DELETE",
            "/teams/{team_id}",
            'overrides=("DELETE /teams/{team_id}",)',
        ):
            assert fragment in message

    def test_same_module_colliding_with_itself_is_detected_identically(self) -> None:
        with pytest.raises(RouteCollisionError, match=r"twice.*GET /duplicated") as excinfo:
            _build([_TwoSpecModule()])
        assert 'overrides=("GET /duplicated",)' in str(excinfo.value)

    def test_earlier_spec_of_one_module_declares_and_wins(self) -> None:
        with TestClient(_build([_TwoSpecModule(overrides=("GET /duplicated",))])) as client:
            assert client.get("/duplicated").json() == {"spec": "first"}

    def test_websocket_collision_is_keyed_ws(self) -> None:
        modules: list[BaseAppModule] = [_WsPathModule("ws-a"), _WsPathModule("ws-b")]
        with pytest.raises(RouteCollisionError, match=r"WS /ws-shared"):
            _build(modules)

    def test_declared_websocket_collision_builds(self) -> None:
        modules: list[BaseAppModule] = [
            _WsPathModule("ws-a", overrides=("WS /ws-shared",)),
            _WsPathModule("ws-b"),
        ]
        assert "WS /ws-shared" in build_manifest(_build(modules)).routes


class TestRouteCollisionKeying:
    """Story 63.2 AC 3, 7-9: what counts as the same route, and what does not."""

    def test_prefix_is_resolved_before_comparison(self) -> None:
        modules: list[BaseAppModule] = [
            _PathModule("acme-prefixed", "/reports", prefix="/acme"),
            _PathModule("bare", "/acme/reports"),
        ]
        with pytest.raises(RouteCollisionError, match=r"GET /acme/reports"):
            _build(modules)

    def test_prefixed_route_does_not_collide_with_the_unprefixed_path(self) -> None:
        modules: list[BaseAppModule] = [
            _PathModule("acme-prefixed", "/reports", prefix="/acme"),
            _PathModule("bare", "/reports"),
        ]
        with TestClient(_build(modules)) as client:
            # Both survive as distinct routes — proven by two distinct bodies,
            # which no shadowing composition could produce.
            assert client.get("/acme/reports").json() == {"source": "acme-prefixed"}
            assert client.get("/reports").json() == {"source": "bare"}

    def test_declaring_module_serves_the_shadowed_path(self) -> None:
        modules: list[BaseAppModule] = [
            _PathModule(
                "acme-prefixed", "/reports", prefix="/acme", overrides=("GET /acme/reports",)
            ),
            _PathModule("bare", "/acme/reports"),
        ]
        with TestClient(_build(modules)) as client:
            assert client.get("/acme/reports").json() == {"source": "acme-prefixed"}

    def test_different_methods_on_one_path_do_not_collide(self) -> None:
        modules: list[BaseAppModule] = [
            _PathModule("reader", "/acme/reports"),
            _PathModule("writer", "/acme/reports", method="POST"),
        ]
        with TestClient(_build(modules)) as client:
            assert client.get("/acme/reports").json() == {"source": "reader"}
            assert client.post("/acme/reports").json() == {"source": "writer"}

    def test_declaration_matching_nothing_is_inert(self) -> None:
        """Over-declaring is the client's problem, never a build failure.

        A client that keeps a declaration through a framework upgrade which
        removed the stock route it shadowed must not be punished for it.
        """
        modules: list[BaseAppModule] = [
            _PathModule("careful", "/acme/reports", overrides=("GET /gone-in-the-upgrade",)),
        ]
        with TestClient(_build(modules)) as client:
            assert client.get("/acme/reports").status_code == 200


class TestPublishedCollisionError:
    """Story 63.2 AC 12: a client catches the error by the name it imports."""

    def test_error_resolves_from_the_server_package_and_is_exported(self) -> None:
        assert akgentic.infra.server.RouteCollisionError is RouteCollisionError
        assert "RouteCollisionError" in akgentic.infra.server.__all__

    def test_error_is_an_assembly_error(self) -> None:
        """A client already catching ``AssemblyError`` keeps catching this one."""
        assert issubclass(RouteCollisionError, AssemblyError)


class TestRoutesCollectedOnce:
    """Story 63.2 AC 10: detection and mounting consume ONE collection pass."""

    def test_contribute_routes_invoked_once_and_its_result_is_what_mounts(self) -> None:
        module = _CountingRoutesModule()
        app = _build([_CoreModule(), module])
        # A module may construct its routers inside contribute_routes; calling
        # it twice would double-construct and the two results could differ.
        assert module.calls == 1
        assert "GET /counted/ping" in build_manifest(app).routes
        with TestClient(app) as client:
            assert client.get("/counted/ping").json() == {"status": "ok"}

    def test_a_rejected_composition_never_calls_contribute_routes(self) -> None:
        """Collection happens only once the composition is known-good.

        A module may take side effects inside ``contribute_routes`` — the
        stock ``CoreModule`` sets the process-global unified catalog there —
        so a composition rejected for a duplicate name or state provider must
        not have run them.
        """
        first, second = _CountingRoutesModule(), _CountingRoutesModule()
        with pytest.raises(DuplicateModuleNameError):
            _build([first, second])
        assert (first.calls, second.calls) == (0, 0)


class TestExceptionHandlerRegistrarForm:
    """AC 3 (Story 57.5): the builder-mediated registrar handler form."""

    def test_registrar_invoked_by_builder_and_never_outside_composition(self) -> None:
        installed: list[FastAPI] = []

        def _install(app: FastAPI) -> None:
            installed.append(app)
            app.add_exception_handler(_DemoError, _first_handler)

        class _RegistrarModule(_RaisingModule):
            name = "registrar"

            def contribute_exception_handlers(
                self,
            ) -> list[ExceptionHandlerSpec | ExceptionHandlerRegistrar]:
                return [ExceptionHandlerRegistrar(install=_install)]

        module = _RegistrarModule()
        assert installed == []  # never invoked before/outside composition
        app = _build([module])
        assert installed == [app]  # invoked exactly once, by the builder, on the composed app
        with TestClient(app) as client:
            response = client.get("/boom")
            assert response.status_code == 418
            assert response.json() == {"handler": "first"}


# --- Story 63.3: manifest_delta, the client's no-regression instrument --- #

# A stand-in tier surface. The unit cases below diff hand-built manifest pairs
# rather than composed apps: ``manifest_delta`` is a pure function of two
# models, and routing a purity question through a FastAPI build would only make
# the failures harder to read.
_STOCK_ROUTES = ["GET /readiness", "GET /teams", "POST /teams"]
_STOCK_STACK = ["CORSMiddleware", "RequireAuthMiddleware", "MutationLogMiddleware"]

_ACME_ROUTE = "GET /acme/reports/{report_id}"


def _stock() -> AppManifest:
    """A fresh manifest standing in for the tier as the framework ships it."""
    return AppManifest(routes=list(_STOCK_ROUTES), middleware=list(_STOCK_STACK))


class TestManifestDeltaRoutes:
    """AC 3-5, 10: route membership, reported as sorted lists."""

    def test_identical_manifests_report_no_change(self) -> None:
        delta = manifest_delta(_stock(), _stock())
        assert delta.routes_added == []
        assert delta.routes_removed == []
        assert delta.middleware_added == []
        assert delta.middleware_removed == []
        assert delta.stock_middleware_reordered is False

    def test_client_route_is_the_only_addition(self) -> None:
        composed = AppManifest(routes=[*_STOCK_ROUTES, _ACME_ROUTE], middleware=list(_STOCK_STACK))
        delta = manifest_delta(_stock(), composed)
        assert delta.routes_added == [_ACME_ROUTE]
        assert delta.routes_removed == []
        assert delta.middleware_added == []
        assert delta.middleware_removed == []
        assert delta.stock_middleware_reordered is False

    def test_missing_stock_route_is_reported_as_removed(self) -> None:
        composed = AppManifest(
            routes=["GET /readiness", "GET /teams"], middleware=list(_STOCK_STACK)
        )
        delta = manifest_delta(_stock(), composed)
        assert delta.routes_removed == ["POST /teams"]
        assert delta.routes_added == []
        assert delta.middleware_added == []
        assert delta.middleware_removed == []
        assert delta.stock_middleware_reordered is False

    def test_route_lists_are_sorted_even_from_unsorted_manifests(self) -> None:
        """A hand-built manifest need not be sorted; the delta always is."""
        stock = AppManifest(routes=["GET /zeta", "GET /alpha", "GET /mike"], middleware=[])
        composed = AppManifest(routes=["GET /yankee", "GET /alpha", "GET /bravo"], middleware=[])
        delta = manifest_delta(stock, composed)
        assert delta.routes_added == ["GET /bravo", "GET /yankee"]
        assert delta.routes_removed == ["GET /mike", "GET /zeta"]


class TestManifestDeltaMiddleware:
    """AC 6-9: what counts as a middleware change, and what counts as a reorder."""

    def test_middleware_inserted_innermost_is_an_addition_not_a_reorder(self) -> None:
        composed = AppManifest(
            routes=list(_STOCK_ROUTES), middleware=[*_STOCK_STACK, "AcmeMiddleware"]
        )
        delta = manifest_delta(_stock(), composed)
        assert delta.middleware_added == ["AcmeMiddleware"]
        assert delta.middleware_removed == []
        assert delta.stock_middleware_reordered is False

    def test_middleware_inserted_outermost_is_an_addition_not_a_reorder(self) -> None:
        """Where the client slots in is irrelevant — only stock's own order matters."""
        composed = AppManifest(
            routes=list(_STOCK_ROUTES), middleware=["AcmeMiddleware", *_STOCK_STACK]
        )
        delta = manifest_delta(_stock(), composed)
        assert delta.middleware_added == ["AcmeMiddleware"]
        assert delta.middleware_removed == []
        assert delta.stock_middleware_reordered is False

    def test_two_swapped_stock_entries_are_a_reorder(self) -> None:
        outer, middle, inner = _STOCK_STACK
        composed = AppManifest(routes=list(_STOCK_ROUTES), middleware=[middle, outer, inner])
        delta = manifest_delta(_stock(), composed)
        assert delta.stock_middleware_reordered is True
        assert delta.middleware_added == []
        assert delta.middleware_removed == []

    def test_removed_stock_middleware_is_not_also_a_reorder(self) -> None:
        """The trap case: a removal is reported once, by ``middleware_removed``.

        Filtering only the composed list would leave the removed entry in the
        stock sequence, so every removal would ALSO raise the flag — and a
        client could no longer tell "the framework dropped a middleware" from
        "the framework restacked them".
        """
        outer, _middle, inner = _STOCK_STACK
        composed = AppManifest(routes=list(_STOCK_ROUTES), middleware=[outer, inner])
        delta = manifest_delta(_stock(), composed)
        assert delta.middleware_removed == ["RequireAuthMiddleware"]
        assert delta.stock_middleware_reordered is False
        assert delta.middleware_added == []

    def test_added_and_removed_middleware_keep_manifest_order(self) -> None:
        """Order is the whole content of a middleware name — never sorted away."""
        composed = AppManifest(
            routes=list(_STOCK_ROUTES),
            middleware=["ZebraMiddleware", "CORSMiddleware", "AcmeMiddleware"],
        )
        delta = manifest_delta(_stock(), composed)
        # Composed order (outermost first), NOT alphabetical.
        assert delta.middleware_added == ["ZebraMiddleware", "AcmeMiddleware"]
        # Stock order, NOT alphabetical.
        assert delta.middleware_removed == ["RequireAuthMiddleware", "MutationLogMiddleware"]
        # An addition and a removal in the same diff, and still no reorder: the
        # one surviving stock entry holds its relative position. Both filters
        # have to fire for this — it is AC 6 and AC 9 in a single manifest.
        assert delta.stock_middleware_reordered is False


class TestManifestDeltaIsPure:
    """AC 2: a pure function of its two arguments."""

    def test_repeated_calls_agree_and_neither_argument_is_mutated(self) -> None:
        stock = AppManifest(routes=["GET /b", "GET /a"], middleware=["Outer", "Inner"])
        composed = AppManifest(routes=["GET /a", "GET /c"], middleware=["Inner", "Outer", "Extra"])
        first = manifest_delta(stock, composed)
        second = manifest_delta(stock, composed)
        assert first == second
        assert stock.routes == ["GET /b", "GET /a"]
        assert stock.middleware == ["Outer", "Inner"]
        assert composed.routes == ["GET /a", "GET /c"]
        assert composed.middleware == ["Inner", "Outer", "Extra"]

    def test_argument_order_decides_the_direction(self) -> None:
        """``(stock, composed)`` is part of the contract — swapping inverts it."""
        stock = _stock()
        composed = AppManifest(routes=[*_STOCK_ROUTES, _ACME_ROUTE], middleware=list(_STOCK_STACK))
        assert manifest_delta(stock, composed).routes_added == [_ACME_ROUTE]
        assert manifest_delta(composed, stock).routes_removed == [_ACME_ROUTE]


class TestPublishedManifestDelta:
    """AC 1, 12: the shape a client compiles against, and where it imports it from."""

    def test_both_names_resolve_from_the_server_package_and_are_exported(self) -> None:
        assert akgentic.infra.server.ManifestDelta is ManifestDelta
        assert akgentic.infra.server.manifest_delta is manifest_delta
        assert "ManifestDelta" in akgentic.infra.server.__all__
        assert "manifest_delta" in akgentic.infra.server.__all__

    def test_the_five_field_names_are_the_whole_contract(self) -> None:
        """Client repos read these five names; nothing else may appear.

        Every other case here reads the fields, so a rename breaks the suite —
        but a *sixth* field, or one of the five gaining a default, would slip
        through green. Both are breaking changes to a published model: a sixth
        field a client is not asserting on hides a regression, and a default
        makes a partial delta literal constructible by accident.
        """
        fields = ManifestDelta.model_fields
        assert list(fields) == [
            "routes_added",
            "routes_removed",
            "middleware_added",
            "middleware_removed",
            "stock_middleware_reordered",
        ]
        assert [name for name, field in fields.items() if not field.is_required()] == []


class _AcmeReportsModule(BaseAppModule):
    """A third-party module, standing in for one a client package would ship.

    Subclasses ``BaseAppModule`` rather than structurally implementing
    ``AppModule``: a structural implementer silently stops satisfying the
    contract the day a seventh verb is added. The name is vendor-prefixed, so
    it cannot collide with a stock module's.
    """

    name = "acme-reports"

    def contribute_routes(self) -> list[RouteSpec]:
        router = APIRouter(prefix="/acme/reports")

        @router.get("/{report_id}")
        def _report(report_id: str) -> dict[str, str]:
            return {"report_id": report_id}

        return [RouteSpec(router=router)]

    def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
        return [MiddlewareSpec(middleware_class=_NoopMiddleware, layer=EXTENSION)]


class TestManifestDeltaOnRealComposition:
    """AC 11: the three-line client assertion, against apps the framework builds.

    Real apps are what make this case worth its cost: FastAPI's own built-ins
    (``/openapi.json``, ``/docs``, ``/redoc``, ``/docs/oauth2-redirect``) appear
    in BOTH manifests and therefore cancel out of the delta. That cancellation
    is precisely why a client can assert ``routes_added == [<one entry>]`` at
    all, and no hand-built pair would ever exercise it.
    """

    def test_client_module_adds_exactly_its_route_and_perturbs_nothing_stock(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        tier_app = create_app(community_services, seeded_settings)
        acme_app = create_app(
            community_services,
            seeded_settings,
            modules=[
                *server_modules(community_services, seeded_settings),
                _AcmeReportsModule(),
            ],
        )
        stock, composed = build_manifest(tier_app), build_manifest(acme_app)
        # Derived, not guessed: build_manifest comma-joins a route's methods,
        # so confirm the entry really has this shape before pinning it.
        assert [entry for entry in composed.routes if "/acme/" in entry] == [_ACME_ROUTE]

        delta = manifest_delta(stock, composed)
        assert delta.routes_removed == [] and delta.middleware_removed == []
        assert delta.stock_middleware_reordered is False
        assert delta.routes_added == [_ACME_ROUTE]
        # The stock stack is untouched and the client's middleware is innermost.
        assert delta.middleware_added == ["_NoopMiddleware"]
        assert composed.middleware == [*stock.middleware, "_NoopMiddleware"]
