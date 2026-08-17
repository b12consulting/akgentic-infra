"""Modular FastAPI app assembly — the ``AppModule`` contract and layered builder (ADR-039).

The FastAPI app is composed from an **explicit ordered list of modules**. Each
module alters the app exclusively through a closed verb vocabulary —
``contribute_routes`` / ``contribute_middleware`` / ``contribute_allowlist`` /
``contribute_exception_handlers`` / ``lifespan`` — and only :func:`build_app`
ever touches the ``FastAPI`` object. No module (and no tier) calls
``app.add_middleware``, ``app.include_router``, ``app.add_exception_handler``,
or writes ``app.state`` outside its own lifespan.

Key semantics, fixed here and nowhere else:

- **Middleware position is a declared layer ordinal**, not a registration call
  order. Lower ordinal = outermost. The builder sorts specs by
  ``(layer, module index, spec index)`` and performs Starlette's LIFO
  reverse-add itself — the only place in the codebase where that inversion
  exists.
- **Routes mount in module-list order**; on a path collision the earlier
  module's route wins (Starlette matches first), so an override module is
  simply listed before the module it overrides.
- **Lifespans compose** via ``AsyncExitStack``: startup in module-list order,
  shutdown in reverse.
- **State follows a single-writer rule**: exactly one module may list a key in
  ``provides_state``. The builder validates at build time that no key has two
  producers and that every middleware ``requires_state`` key has a producing
  module, then re-verifies at the end of startup that every required key was
  actually populated — a missed wire is a named startup crash, not an
  ``AttributeError`` on the first unlucky request.

``build_manifest`` snapshots a composed app's route table and middleware order;
migration stories pin it as a golden-manifest regression test.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import Any, Final, Protocol, runtime_checkable

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, ConfigDict, Field, InstanceOf
from starlette.requests import Request
from starlette.responses import Response

from akgentic.infra.server.deps import TierServices
from akgentic.infra.server.settings import ServerSettings

logger = logging.getLogger(__name__)

# --- Middleware layer vocabulary --- #
# Named integer anchors, deliberately NOT an enum: any integer is a legal
# layer, so a tier may slot between bands (the POLICY band conventionally uses
# sub-slots 510-570 as plain integers). Lower ordinal = outermost.

OBSERVABILITY: Final = 50  # OTel ASGI instrumentation
TRANSPORT: Final = 100  # CORS — outermost of the functional stack
PROXY: Final = 200  # trusted-CIDR proxy-header rewriting
SESSION: Final = 300  # cookie / Redis session decode — outside IDENTITY
IDENTITY: Final = 400  # RequireAuth — resolve once, stash, 401 pre-routing
POLICY: Final = 500  # rate limit, content security, payload, idempotency
APPLICATION: Final = 600  # route-adjacent concerns (mutation log) — innermost

# Typed ``Any`` sentinel (same idiom as ``akgentic.infra.utils``) so the
# ``getattr(state, name, _MISSING)`` read stays ``Any`` under mypy strict.
_MISSING: Any = object()

# Exception handlers are Starlette-shaped: sync or async, HTTP-request based.
ExceptionHandler = Callable[[Request, Exception], Response | Awaitable[Response]]


# --- Typed assembly errors --- #


class AssemblyError(Exception):
    """Base class for app-composition contract violations."""


class DuplicateModuleNameError(AssemblyError):
    """Two modules in one composition share a ``name`` (build time)."""


class DuplicateStateProviderError(AssemblyError):
    """Two modules declare the same key in ``provides_state`` (build time)."""


class MissingStateProviderError(AssemblyError):
    """A middleware ``requires_state`` key has no producing module (build time)."""


class UnpopulatedStateError(AssemblyError):
    """A required state key was never populated by its producer's lifespan (boot time)."""


# --- Contribution specs --- #


class AllowlistSpec(BaseModel):
    """Paths a module needs reachable without an authenticated principal."""

    model_config = ConfigDict(frozen=True)

    exact: frozenset[str] = Field(
        default=frozenset(),
        description="Paths that bypass the identity gate by exact match",
    )
    prefixes: tuple[str, ...] = Field(
        default=(),
        description="Path prefixes that bypass the identity gate",
    )


class BuildContext(BaseModel):
    """Merged build-time facts the builder hands to every ``contribute_middleware`` call.

    ``allowlist`` is the union of ALL modules' allowlist contributions, so an
    identity-gate module configures itself with the full merged allowlist
    without knowing which modules contributed which entries.
    """

    model_config = ConfigDict(frozen=True)

    allowlist: AllowlistSpec = Field(
        description="Union of every module's allowlist contribution",
    )


class MiddlewareSpec(BaseModel):
    """One middleware contribution: class + layer + config-only options.

    ``options`` carries **configuration only**: settings values and *pure,
    stateless callables* (a ``requires_principal`` predicate, an ``on_reject``
    response shaper — configuration in function form). It must **never carry a
    live service**: a middleware that needs a runtime collaborator names its
    ``app.state`` slot in ``requires_state`` and resolves it per request
    (``StateKey.require``); the slot is populated by some module's lifespan and
    verified populated before the first request is accepted.
    """

    model_config = ConfigDict(frozen=True)

    middleware_class: type = Field(
        description="ASGI middleware class the builder instantiates with options",
    )
    layer: int = Field(
        description="Layer ordinal deciding stack position; lower is outermost",
    )
    options: dict[str, object] = Field(
        default_factory=dict,
        description="Config-only constructor kwargs; never a live service",
    )
    requires_state: tuple[str, ...] = Field(
        default=(),
        description="StateKey names this middleware resolves per request",
    )


class RouteSpec(BaseModel):
    """One router contribution, mounted in module-list order.

    The router arrives pre-built: router-level dependencies (auth gates,
    caller-identity scoping) are the contributing module's business,
    constructed before contribution.
    """

    model_config = ConfigDict(frozen=True)

    router: InstanceOf[APIRouter] = Field(
        description="Pre-built APIRouter, mounted as contributed",
    )
    prefix: str = Field(
        default="",
        description="Mount prefix passed to include_router",
    )


class ExceptionHandlerSpec(BaseModel):
    """One exception-handler contribution, registered in module-list order.

    Starlette resolves handlers by exception-class lookup (dict semantics), so
    a later registration for the *same* exception class replaces an earlier
    one — see :func:`build_app` for the last-wins rule.
    """

    model_config = ConfigDict(frozen=True)

    exception_class: type[Exception] = Field(
        description="Exception class this handler is registered for",
    )
    handler: ExceptionHandler = Field(
        description="Starlette-shaped handler (sync or async) building the response",
    )


# --- The module contract --- #


@runtime_checkable
class AppModule(Protocol):
    """The only legal way anything alters the app — a closed verb vocabulary.

    The vocabulary is CLOSED: exactly the five contribution verbs plus ``name``
    and ``provides_state``. A module that needs anything else is a
    contract-change discussion, not a workaround.
    """

    name: str
    """Unique-per-composition module identifier (kebab-case slug)."""

    provides_state: tuple[str, ...]
    """StateKey names this module's lifespan populates (single-writer rule)."""

    def contribute_routes(self) -> list[RouteSpec]:
        """Return routers to mount, in contribution order."""
        ...

    def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
        """Return middleware specs, given the merged build-time context."""
        ...

    def contribute_allowlist(self) -> AllowlistSpec:
        """Return paths this module needs reachable without a principal."""
        ...

    def contribute_exception_handlers(self) -> list[ExceptionHandlerSpec]:
        """Return exception handlers to register, in contribution order."""
        ...

    def lifespan(self, app: FastAPI) -> AbstractAsyncContextManager[None]:
        """Return this module's startup/shutdown context for the composed app."""
        ...


class BaseAppModule:
    """No-op defaults for every ``AppModule`` member.

    Subclass and override only what the module contributes; the composed
    behavior of an all-defaults module is exactly nothing.
    """

    name: str = "unnamed"
    provides_state: tuple[str, ...] = ()

    def contribute_routes(self) -> list[RouteSpec]:
        """Contribute no routes."""
        return []

    def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
        """Contribute no middleware."""
        return []

    def contribute_allowlist(self) -> AllowlistSpec:
        """Contribute an empty allowlist."""
        return AllowlistSpec()

    def contribute_exception_handlers(self) -> list[ExceptionHandlerSpec]:
        """Contribute no exception handlers."""
        return []

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        """No-op lifespan: no startup work, no shutdown work."""
        yield


# --- The builder --- #


class _OrderedMiddleware(BaseModel):
    """A middleware spec tagged with its deterministic sort position."""

    model_config = ConfigDict(frozen=True)

    module_index: int = Field(description="Position of the contributing module in the list")
    spec_index: int = Field(description="Position of the spec within the module's contribution")
    module_name: str = Field(description="Name of the contributing module")
    spec: MiddlewareSpec = Field(description="The contributed middleware spec")


class _StateRequirement(BaseModel):
    """One required state key with its declared producer and requirer, for boot checks."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(description="StateKey name that must be populated at end of startup")
    required_by: str = Field(description="Module whose middleware requires the key")
    provided_by: str = Field(description="Module that declared the key in provides_state")


def build_app(
    settings: ServerSettings,
    services: TierServices,
    modules: Sequence[AppModule],
) -> FastAPI:
    """Assemble one FastAPI app from an ordered module composition.

    Order semantics (fixed, documented, tier-independent):

    - **Routes** mount in module-list order; on a path collision the earlier
      module's route wins (Starlette matches first).
    - **Middleware** sort by ``(layer, module index, spec index)`` and are all
      added after all routes; the lowest layer ends up outermost regardless of
      module-list order.
    - **Lifespans** compose via ``AsyncExitStack``: startup in module-list
      order, shutdown in reverse; every middleware ``requires_state`` key is
      verified populated at the end of startup, aborting boot otherwise.
    - **Exception handlers** register in module-list order. Starlette keeps
      handlers in a per-exception-class dict, so when two modules register a
      handler for the *same* exception class the later module's handler wins
      (last-wins, plain Starlette semantics).

    Args:
        settings: The tier's server settings. Threaded for a uniform tier
            entrypoint; the builder itself reads nothing from it — modules are
            constructed by the tier assembly function with whatever
            configuration they need.
        services: The tier's wired service container. Threaded like
            ``settings``; state slots are populated only by module lifespans
            (single-writer rule), never by the builder.
        modules: The ordered composition. The order IS the composition.

    Returns:
        The composed FastAPI application.

    Raises:
        DuplicateModuleNameError: Two modules share a ``name``.
        DuplicateStateProviderError: Two modules provide the same state key.
        MissingStateProviderError: A ``requires_state`` key has no producer.
    """
    module_list = list(modules)
    providers = _validate_composition(module_list)
    context = BuildContext(allowlist=_merge_allowlists(module_list))
    ordered = _collect_middleware(module_list, context)
    requirements = _validate_required_state(ordered, providers)
    app = FastAPI(
        title="Akgentic Platform API",
        lifespan=_compose_lifespan(module_list, requirements),
    )
    _mount_routes(app, module_list)
    _register_exception_handlers(app, module_list)
    # Middleware are added strictly AFTER all routes so an APPLICATION-layer
    # middleware wraps every mounted response (mutation-log requirement).
    _add_middleware(app, ordered)
    logger.debug(
        "build_app: composed %d modules, %d middleware specs",
        len(module_list),
        len(ordered),
    )
    return app


def _validate_composition(modules: list[AppModule]) -> dict[str, str]:
    """Reject duplicate module names and duplicate state producers (build time).

    Returns:
        Mapping of each provided state key to its single producing module.
    """
    names: set[str] = set()
    for module in modules:
        if module.name in names:
            raise DuplicateModuleNameError(
                f"duplicate module name '{module.name}' in composition"
            )
        names.add(module.name)
    providers: dict[str, str] = {}
    for module in modules:
        for key in module.provides_state:
            if key in providers:
                raise DuplicateStateProviderError(
                    f"state key '{key}' has two producers: "
                    f"modules '{providers[key]}' and '{module.name}'"
                )
            providers[key] = module.name
    return providers


def _merge_allowlists(modules: list[AppModule]) -> AllowlistSpec:
    """Union every module's allowlist, preserving first-seen prefix order."""
    exact: set[str] = set()
    prefixes: list[str] = []
    for module in modules:
        spec = module.contribute_allowlist()
        exact |= spec.exact
        prefixes.extend(p for p in spec.prefixes if p not in prefixes)
    return AllowlistSpec(exact=frozenset(exact), prefixes=tuple(prefixes))


def _collect_middleware(
    modules: list[AppModule], context: BuildContext
) -> list[_OrderedMiddleware]:
    """Gather all middleware specs, sorted outermost-first by the layer triple."""
    entries = [
        _OrderedMiddleware(
            module_index=module_index,
            spec_index=spec_index,
            module_name=module.name,
            spec=spec,
        )
        for module_index, module in enumerate(modules)
        for spec_index, spec in enumerate(module.contribute_middleware(context))
    ]
    return sorted(entries, key=lambda e: (e.spec.layer, e.module_index, e.spec_index))


def _validate_required_state(
    ordered: list[_OrderedMiddleware], providers: dict[str, str]
) -> list[_StateRequirement]:
    """Check every ``requires_state`` key has a declared producer (build time).

    Returns:
        The requirements to re-verify at the end of composed startup.
    """
    requirements: list[_StateRequirement] = []
    for entry in ordered:
        for key in entry.spec.requires_state:
            if key not in providers:
                raise MissingStateProviderError(
                    f"middleware {entry.spec.middleware_class.__name__} from module "
                    f"'{entry.module_name}' requires state key '{key}', "
                    f"but no module in the composition provides it"
                )
            requirements.append(
                _StateRequirement(
                    key=key,
                    required_by=entry.module_name,
                    provided_by=providers[key],
                )
            )
    return requirements


def _compose_lifespan(
    modules: list[AppModule], requirements: list[_StateRequirement]
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build the composed lifespan: startups in list order, shutdowns in reverse.

    A startup failure in module N unwinds modules 1..N-1 cleanly —
    ``AsyncExitStack`` semantics give this for free.
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            for module in modules:
                await stack.enter_async_context(module.lifespan(app))
            _verify_startup_state(app, requirements)
            yield
        # AsyncExitStack unwinds here — module shutdowns run in reverse order.

    return _lifespan


def _verify_startup_state(app: FastAPI, requirements: list[_StateRequirement]) -> None:
    """Abort boot when a declared collaborator slot was never populated."""
    missing = [
        f"app.state.{req.key} was never populated after startup: "
        f"declared by module '{req.provided_by}', required by module '{req.required_by}'"
        for req in requirements
        if getattr(app.state, req.key, _MISSING) is _MISSING
    ]
    if missing:
        raise UnpopulatedStateError("; ".join(missing))


def _mount_routes(app: FastAPI, modules: list[AppModule]) -> None:
    """Mount every module's routers in module-list order (earlier wins collisions)."""
    for module in modules:
        for route_spec in module.contribute_routes():
            app.include_router(route_spec.router, prefix=route_spec.prefix)


def _register_exception_handlers(app: FastAPI, modules: list[AppModule]) -> None:
    """Register handlers in module-list order (same-class re-registration: last wins)."""
    for module in modules:
        for spec in module.contribute_exception_handlers():
            app.add_exception_handler(spec.exception_class, spec.handler)


def _add_middleware(app: FastAPI, ordered: list[_OrderedMiddleware]) -> None:
    """Add sorted middleware so the lowest layer ends up outermost.

    Starlette's ``add_middleware`` inserts at index 0 (last added = outermost),
    so the outermost-first sorted list is added in reverse. This is the ONLY
    place in the codebase where that inversion exists.
    """
    for entry in reversed(ordered):
        # A plain ``type`` + unpacked options cannot satisfy Starlette's
        # ParamSpec'd _MiddlewareFactory protocol statically; the runtime
        # contract (cls(app, **options) -> ASGI app) is exactly what specs carry.
        app.add_middleware(entry.spec.middleware_class, **entry.spec.options)  # type: ignore[arg-type]


# --- Manifest — the no-regression instrument --- #


class AppManifest(BaseModel):
    """Snapshot of a composed app, for golden-manifest regression tests."""

    model_config = ConfigDict(frozen=True)

    routes: list[str] = Field(
        description="Sorted 'METHODS /path' entries; websocket routes use WS",
    )
    middleware: list[str] = Field(
        description="Middleware class names, outermost to innermost",
    )


def build_manifest(app: FastAPI) -> AppManifest:
    """Snapshot the route table and the outermost-to-innermost middleware order."""
    entries: list[str] = []
    for route in app.routes:
        path = getattr(route, "path", None)
        if path is None:
            continue
        methods = getattr(route, "methods", None) or {"WS"}
        entries.append(f"{','.join(sorted(methods))} {path}")
    # ``Middleware.cls`` is Starlette's ParamSpec'd factory protocol, which has
    # no static ``__name__``; every concrete middleware is a class and has one.
    middleware = [str(getattr(m.cls, "__name__", m.cls)) for m in app.user_middleware]
    return AppManifest(routes=sorted(entries), middleware=middleware)
