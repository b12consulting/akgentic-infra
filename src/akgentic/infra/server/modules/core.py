"""``CoreModule`` — the community tier's base composition module (epic 57).

Re-expresses the legacy factory as a **self-contained** ``AppModule``:
readiness, ``/teams``, ``/workspace``, ``/ws``, ``/webhook`` and the
``/admin`` catalog surface (router-level auth pair plus the per-route
owner-or-admin gates, built exactly as before); CORS at TRANSPORT only when
origins are configured; the admin-catalog mutation log at APPLICATION; the
``/readiness`` allowlist. Self-contained means the full contract, with no
wrapper residue:

- ``__init__`` requires the wired ``services.team_service`` and fails loud
  at composition time when the container arrives unwired;
- the community ``app.state`` keys are contributed at **build time** via
  ``contribute_state`` (builder-applied, readable without a lifespan); the
  lifespan writes only ``draining``;
- exception handlers register through ``ExceptionHandlerRegistrar``
  contributions wrapping the catalog package's ``add_exception_handlers`` and
  infra's ``add_server_exception_handlers`` — package-private handler
  callables stay private, the builder invokes the helpers during composition;
- the graceful-shutdown drain lifespan is owned here outright
  (:func:`_drain_lifespan`).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, Depends, FastAPI
from fastapi.dependencies.utils import get_parameterless_sub_dependant
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends as DependsParam
from fastapi.routing import APIRoute

from akgentic.catalog.api import add_exception_handlers
from akgentic.catalog.api._settings import CatalogRouterSettings
from akgentic.catalog.api.router import build_router as build_catalog_router
from akgentic.catalog.api.router import set_catalog as set_unified_catalog
from akgentic.infra.server.assembly import (
    APPLICATION,
    TRANSPORT,
    AllowlistSpec,
    BaseAppModule,
    BuildContext,
    ExceptionHandlerRegistrar,
    ExceptionHandlerSpec,
    MiddlewareSpec,
    RouteSpec,
    StateEntry,
)
from akgentic.infra.server.deps import TierServices
from akgentic.infra.server.errors import add_server_exception_handlers
from akgentic.infra.server.routes._admin_mutation_log import AdminCatalogMutationLogMiddleware
from akgentic.infra.server.routes._auth_dep import require_authenticated_principal
from akgentic.infra.server.routes._catalog_authz import (
    require_import_owner_or_admin,
    require_namespace_owner_or_admin,
)
from akgentic.infra.server.routes._catalog_caller_identity import scope_catalog_caller_identity
from akgentic.infra.server.routes.readiness import router as readiness_router
from akgentic.infra.server.routes.teams import router as teams_router
from akgentic.infra.server.routes.webhook import router as webhook_router
from akgentic.infra.server.routes.workspace import router as workspace_router
from akgentic.infra.server.routes.ws import ConnectionManager, shutdown_reader_pool
from akgentic.infra.server.routes.ws import router as ws_router
from akgentic.infra.server.settings import ServerSettings
from akgentic.infra.server.state_keys import (
    CHANNEL_PARSERS,
    CHANNEL_REGISTRY,
    CONNECTION_MANAGER,
    DRAINING,
    INGESTION,
    SERVICES,
    SETTINGS,
    TEAM_SERVICE,
)

logger = logging.getLogger(__name__)


class CoreModule(BaseAppModule):
    """Community base module: routes, CORS + mutation log, state, drain lifespan.

    Constructed with the concrete objects it needs — ``build_app``
    deliberately reads nothing from its own ``settings``/``services``
    parameters.
    """

    name = "core"
    provides_state: tuple[str, ...] = ("draining",)

    def __init__(self, services: TierServices, settings: ServerSettings) -> None:
        self._services = services
        self._settings = settings
        if services.team_service is None:
            msg = (
                "services.team_service is not set — wire_* must construct "
                "TeamService and assign it before composing CoreModule"
            )
            raise ValueError(msg)
        self.team_service = services.team_service

    def contribute_routes(self) -> list[RouteSpec]:
        """Core routers in the legacy mount order, then the ``/admin`` catalog.

        Admin-catalog comes last among API routers so its ``/admin`` prefix
        cannot accidentally shadow any other route. The v2 catalog is injected
        into the unified router's module-level slot here — build time, before
        any request can arrive — so every composition that lists ``CoreModule``
        gets the slot set by construction.
        """
        set_unified_catalog(self._services.catalog)
        return [
            RouteSpec(router=readiness_router),
            RouteSpec(router=teams_router),
            RouteSpec(router=workspace_router),
            RouteSpec(router=ws_router),
            RouteSpec(router=webhook_router),
            RouteSpec(router=_build_admin_catalog_router()),
        ]

    def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
        """CORS at TRANSPORT (only when origins are configured), mutation log at APPLICATION.

        When ``cors_origins`` is empty the CORS spec is not contributed at
        all, letting an external gateway (e.g. Azure App Service) manage CORS
        — the module decides presence, a build-time settings branch.
        """
        specs: list[MiddlewareSpec] = []
        if self._settings.cors_origins:
            specs.append(
                MiddlewareSpec(
                    middleware_class=CORSMiddleware,
                    layer=TRANSPORT,
                    options={
                        "allow_origins": self._settings.cors_origins,
                        "allow_credentials": True,
                        "allow_methods": ["*"],
                        "allow_headers": ["*"],
                    },
                )
            )
        specs.append(
            MiddlewareSpec(
                middleware_class=AdminCatalogMutationLogMiddleware,
                layer=APPLICATION,
            )
        )
        return specs

    def contribute_allowlist(self) -> AllowlistSpec:
        """Only ``/readiness`` is reachable without an authenticated principal."""
        return AllowlistSpec(exact=frozenset({"/readiness"}))

    def contribute_state(self) -> Sequence[StateEntry[Any]]:
        """The seven community state keys, applied by the builder at build time."""
        entries: list[StateEntry[Any]] = [
            SERVICES.entry(self._services),
            TEAM_SERVICE.entry(self.team_service),
            SETTINGS.entry(self._settings),
            CONNECTION_MANAGER.entry(ConnectionManager()),
        ]
        # ``channel_parser_registry`` is optional on the services container (only the
        # community tier declares it). CHANNEL_PARSERS is a required key, so the slot
        # is only set when the services container actually exposes a registry; a tier
        # without one leaves the slot unset and any webhook request fails loud
        # (``require()`` → LookupError → 500) instead of reading back a silent None.
        channel_parsers = getattr(self._services, "channel_parser_registry", None)
        if channel_parsers is not None:
            entries.append(CHANNEL_PARSERS.entry(channel_parsers))
        entries.append(CHANNEL_REGISTRY.entry(self._services.channel_registry))
        entries.append(INGESTION.entry(self._services.ingestion))
        return entries

    def contribute_exception_handlers(
        self,
    ) -> list[ExceptionHandlerSpec | ExceptionHandlerRegistrar]:
        """The catalog and ``ServerError`` handler families, via registrars.

        Both helpers' handler callables are package-private; registrars let
        the builder invoke the packages' own registration helpers without
        coupling this module to another package's private names.
        """
        return [
            ExceptionHandlerRegistrar(install=add_exception_handlers),
            ExceptionHandlerRegistrar(install=add_server_exception_handlers),
        ]

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        """Drain lifespan: ``draining`` flag on startup, graceful shutdown sequence.

        Delegates to the module-level :func:`_drain_lifespan` so the lifespan
        unit tests keep their construct-and-drive shape without instantiating
        ``CoreModule`` (whose ``__init__`` requires a wired services container).
        """
        async with _drain_lifespan(app):
            yield


@asynccontextmanager
async def _drain_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Drain lifespan implementing the ADR-013 graceful shutdown sequence.

    Startup: sets ``app.state.draining = False``.
    Shutdown: sets draining flag, waits pre-drain delay, disconnects all
    WebSocket clients, then stops all teams via ``worker_handle.stop_all()``.
    """
    DRAINING.set(app, value=False)
    logger.info("Lifespan startup: draining=False")
    yield
    # --- Shutdown sequence (ADR-013 Decision 2) ---
    DRAINING.set(app, value=True)
    logger.info("Lifespan shutdown: draining=True")

    delay = SETTINGS.require(app).shutdown_pre_drain_delay
    if delay > 0:
        logger.info("Pre-drain delay: sleeping %ds", delay)
        await asyncio.sleep(delay)

    logger.info("Disconnecting all WebSocket clients")
    await CONNECTION_MANAGER.require(app).disconnect_all()

    timeout = SETTINGS.require(app).shutdown_drain_timeout
    logger.info("Stopping all teams (timeout=%ds)", timeout)
    try:
        await asyncio.wait_for(
            asyncio.to_thread(SERVICES.require(app).worker_handle.stop_all),
            timeout=timeout,
        )
        logger.info("stop_all() completed successfully")
    except TimeoutError:
        logger.warning(
            "stop_all() exceeded shutdown_drain_timeout=%ds, proceeding with exit",
            timeout,
        )

    # Shut down the dedicated WS reader thread pool — see issue #227.
    shutdown_reader_pool()
    logger.info("WebSocket reader pool shut down")


def _build_admin_catalog_router() -> APIRouter:
    """Build the ``/admin`` catalog wrapper router exactly as the legacy factory did.

    The v2 unified catalog router is built with the generic kind-CRUD family
    enabled, the per-route owner-or-admin gates are attached in place, and the
    result is wrapped in an intermediate ``APIRouter`` carrying the ``/admin``
    prefix and the router-level authentication + caller-identity dependencies —
    ``RouteSpec`` carries only ``(router, prefix)``, so the wrapper is what
    preserves the router-level dependency pair.
    """
    catalog_router = build_catalog_router(
        CatalogRouterSettings(expose_generic_kind_crud=True),
    )
    # Additionally gate the modify + delete routes with the resource-level
    # owner-or-admin dependency, per-route (NOT a blanket include_router
    # dependency — that would fire on reads and break the body-carried create
    # routes). The router-level authentication gate below stays exactly as-is;
    # this gate is additive.
    _attach_owner_or_admin_gate(catalog_router)
    # The import route's target namespace is body-carried, so the path/query
    # route gate above cannot see it. Attach a separate body-reading gate to
    # exactly POST /catalog/namespace/import — the one mutating route the
    # route gate deliberately does NOT body-peek. Kept as a sibling helper
    # (not folded into _OWNER_OR_ADMIN_GATED_ROUTES) so the no-body-peek route
    # gate and the body-peek import gate stay visibly distinct.
    _attach_import_owner_or_admin_gate(catalog_router)
    # Scope each /admin/catalog/* request inside
    # Catalog.as_caller(request_user.user_id), derived server-side from the
    # get_request_user seam. Attached once here so department (composes
    # CoreModule) and enterprise (transplants these routes/state) inherit it
    # from the same place. The authentication gate is router-level; both
    # dependencies are additive and never read a spoofable inbound header.
    wrapper = APIRouter()
    wrapper.include_router(
        catalog_router,
        prefix="/admin",
        dependencies=[
            Depends(require_authenticated_principal),
            Depends(scope_catalog_caller_identity),
        ],
    )
    return wrapper


# Catalog-router-relative paths of the four owner-or-admin-gated mutation
# routes. Paths are the router-local form (the router's own ``/catalog``
# prefix), each paired with the HTTP method that mutates.
# ``DELETE /catalog/namespace/{namespace}`` is the namespace-delete route
# introduced by the catalog package's namespace-delete capability. The
# attachment is forward-compatible: it gates the route only when present on
# the built router, and skips silently (no hard-fail) if a catalog version is
# pinned that does not yet expose it. See issue #297 for the cross-package
# sequencing.
_OWNER_OR_ADMIN_GATED_ROUTES: tuple[tuple[str, str], ...] = (
    ("PUT", "/catalog/{kind}/{id}"),
    ("DELETE", "/catalog/{kind}/{id}"),
    ("PUT", "/catalog/namespace/{namespace}/meta"),
    ("DELETE", "/catalog/namespace/{namespace}"),
)


def _attach_owner_or_admin_gate(router: APIRouter) -> None:
    """Append the owner-or-admin dependency to the four gated mutation routes.

    Augments each matching ``APIRoute`` on the already-built catalog router in
    place, by method + router-local path. The dependency is added to both
    ``route.dependencies`` (so it is part of the route *definition* and
    travels with the route when enterprise transplants it) and the live
    ``route.dependant`` (so it fires at request time). This is additive — the
    router-level authentication dependency is untouched, reads/creates are not
    gated, and the dependency is per-route.

    Routes absent from the currently-pinned catalog (notably the
    namespace-delete route from the catalog package) are simply not found and
    silently skipped — the attachment is forward-compatible.
    """
    dep: DependsParam = Depends(require_namespace_owner_or_admin)
    for method, path in _OWNER_OR_ADMIN_GATED_ROUTES:
        for route in router.routes:
            if not isinstance(route, APIRoute):
                continue
            if route.path == path and route.methods is not None and method in route.methods:
                route.dependencies.append(dep)
                route.dependant.dependencies.insert(
                    0,
                    get_parameterless_sub_dependant(depends=dep, path=route.path_format),
                )


# Router-local (method, path) of the single body-carried mutation route gated
# by the import-specific owner-or-admin dependency. Kept separate from
# ``_OWNER_OR_ADMIN_GATED_ROUTES`` because that constant drives the
# no-body-peek route gate; this route's namespace lives in the YAML body and
# needs the body-reading ``require_import_owner_or_admin`` instead.
_IMPORT_OWNER_OR_ADMIN_GATED_ROUTE: tuple[str, str] = ("POST", "/catalog/namespace/import")


def _attach_import_owner_or_admin_gate(router: APIRouter) -> None:
    """Append the body-reading import gate to ``POST /catalog/namespace/import``.

    Mirrors ``_attach_owner_or_admin_gate``'s in-place two-line augmentation
    (append to ``route.dependencies`` AND insert into ``route.dependant``) but
    for the single import route and with ``require_import_owner_or_admin`` —
    the dependency that reads the YAML body to find the target namespace, then
    applies the same owner-or-admin predicate. Adding it to
    ``route.dependencies`` makes it part of the route *definition* so it
    travels with the ``APIRoute`` when enterprise transplants it.

    This is additive and per-route: the router-level authentication and
    caller-identity dependencies are untouched, the four path/query routes
    keep ``require_namespace_owner_or_admin`` (which still does not
    body-peek), and no other route is gated by the import dependency.
    """
    dep: DependsParam = Depends(require_import_owner_or_admin)
    method, path = _IMPORT_OWNER_OR_ADMIN_GATED_ROUTE
    for route in router.routes:
        if not isinstance(route, APIRoute):
            continue
        if route.path == path and route.methods is not None and method in route.methods:
            route.dependencies.append(dep)
            route.dependant.dependencies.insert(
                0,
                get_parameterless_sub_dependant(depends=dep, path=route.path_format),
            )
