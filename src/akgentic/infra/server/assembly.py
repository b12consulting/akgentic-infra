"""Modular FastAPI app assembly — the ``AppModule`` contract and layered builder (ADR-039).

The FastAPI app is composed from an **explicit ordered list of modules**. Each
module alters the app exclusively through a closed verb vocabulary —
``contribute_routes`` / ``contribute_middleware`` / ``contribute_allowlist`` /
``contribute_exception_handlers`` / ``contribute_state`` / ``lifespan`` — and
only :func:`build_app` ever touches the ``FastAPI`` object. No module (and no
tier) calls ``app.add_middleware``, ``app.include_router``,
``app.add_exception_handler``, or writes ``app.state`` outside builder-applied
``contribute_state`` entries and its own lifespan.

Key semantics, fixed here and nowhere else:

- **Middleware position is a declared layer ordinal**, not a registration call
  order. Lower ordinal = outermost. The builder sorts specs by
  ``(layer, module index, spec index)`` and performs Starlette's LIFO
  reverse-add itself — the only place in the codebase where that inversion
  exists.
- **Routes mount in module-list order**; on a path collision the earlier
  module's route wins (Starlette matches first), so an override module is
  simply listed before the module it overrides. That runtime rule is
  unchanged, but shadowing must now be **stated by the party doing it**: the
  earlier (winning) ``RouteSpec`` lists the shadowed ``"METHOD /path"`` in
  ``overrides`` or the build fails with :class:`RouteCollisionError`. A client
  colliding with a framework route by accident, and a framework upgrade that
  steals a client's route, both surface at build time instead of silently
  changing behaviour in production.
- **Lifespans compose** via ``AsyncExitStack``: startup in module-list order,
  shutdown in reverse.
- **State follows a single-writer rule spanning two phases**: a key is written
  either at build time (a ``contribute_state`` entry, applied by the builder
  after routes so it is readable without a lifespan) or at startup (listed in
  ``provides_state`` and populated by the declaring module's lifespan) — never
  both, and never by two modules. The builder validates at build time that no
  key has two producers across both phases and that every middleware
  ``requires_state`` key has a producing module in either phase, then
  re-verifies at the end of startup that every required key was actually
  populated — a missed wire is a named startup crash, not an
  ``AttributeError`` on the first unlucky request.

``build_manifest`` snapshots a composed app's route table and middleware order;
migration stories pin it as a golden-manifest regression test. ``manifest_delta``
is its client-side counterpart, pointed the other way: it diffs two manifests so
a client package splicing its own modules into a tier's composition can assert,
from its own repo and in its own CI, that it added exactly what it meant to and
disturbed nothing stock.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import Any, Final, Generic, Protocol, TypeVar, runtime_checkable

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, ConfigDict, Field, InstanceOf
from starlette.requests import Request
from starlette.responses import Response

from akgentic.infra.server.deps import TierServices
from akgentic.infra.server.settings import ServerSettings
from akgentic.infra.utils import StateKey

T = TypeVar("T")

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


class RouteCollisionError(AssemblyError):
    """Two route specs mount the same METHOD /path without a declared override (build time)."""


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

    ``overrides`` is how a spec *states* that it deliberately shadows another
    module's route. It belongs to the winner — the earliest contributing spec
    for that path — because the loser declaring it would invert the guarantee:
    a client could declare away a collision it is in fact losing.
    """

    model_config = ConfigDict(frozen=True)

    router: InstanceOf[APIRouter] = Field(
        description="Pre-built APIRouter, mounted as contributed",
    )
    prefix: str = Field(
        default="",
        description="Mount prefix passed to include_router",
    )
    overrides: tuple[str, ...] = Field(
        default=(),
        description="'METHOD /path' entries this spec deliberately shadows (WS for websockets)",
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


class ExceptionHandlerRegistrar(BaseModel):
    """Builder-mediated exception-handler registration (the registrar form).

    For handler families whose callables are package-private (the catalog's
    ``add_exception_handlers``, infra's ``add_server_exception_handlers``),
    decomposing into ``ExceptionHandlerSpec`` pairs would couple the module to
    another package's private names. A registrar instead hands the builder the
    package's own registration helper; the builder invokes it during
    composition, in module-list order — the module never holds the app outside
    ``lifespan``.
    """

    model_config = ConfigDict(frozen=True)

    # Named ``install`` (not ``register``): pydantic warns that ``register``
    # shadows a ``BaseModel`` attribute.
    install: Callable[[FastAPI], None] = Field(
        description="Registration helper the builder invokes on the composed app",
    )


class StateEntry(Generic[T]):  # noqa: UP046  # mirrors StateKey's ADR-030 classic Generic[T] form
    """One build-time state contribution: a ``StateKey`` paired with its value.

    Deliberately a plain generic class (``StateKey``'s own idiom), NOT a
    Pydantic model: ``value`` carries live services (``TierServices``,
    ``TeamService``) that have no serialized form. The key/value pairing is
    mypy-checked at the construction site through the shared ``T`` exactly as
    ``key.set(app, value)`` would be — no ``object``-typed mapping, no cast.
    """

    __slots__ = ("key", "value")

    def __init__(self, key: StateKey[T], value: T) -> None:
        self.key = key
        self.value = value

    def apply(self, app: FastAPI) -> None:
        """Write ``value`` into ``key``'s ``app.state`` slot (builder-invoked)."""
        self.key.set(app, self.value)


# --- The module contract --- #


@runtime_checkable
class AppModule(Protocol):
    """The only legal way anything alters the app — a closed verb vocabulary.

    The vocabulary is CLOSED: exactly the six contribution verbs plus ``name``
    and ``provides_state``. A module that needs anything else is a
    contract-change discussion, not a workaround. No module writes
    ``app.state`` outside builder-applied ``contribute_state`` entries and its
    own lifespan.
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

    def contribute_exception_handlers(
        self,
    ) -> list[ExceptionHandlerSpec | ExceptionHandlerRegistrar]:
        """Return exception handlers to register, in contribution order."""
        ...

    def contribute_state(self) -> Sequence[StateEntry[Any]]:
        """Return build-time state entries, applied by the builder after routes."""
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

    def contribute_exception_handlers(
        self,
    ) -> list[ExceptionHandlerSpec | ExceptionHandlerRegistrar]:
        """Contribute no exception handlers."""
        return []

    def contribute_state(self) -> Sequence[StateEntry[Any]]:
        """Contribute no build-time state."""
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


class _ContributedRoute(BaseModel):
    """One route spec tagged with the module that contributed it."""

    model_config = ConfigDict(frozen=True)

    module_name: str = Field(description="Name of the contributing module")
    spec: RouteSpec = Field(description="The contributed route spec")


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
      module's route wins (Starlette matches first). The winning spec must
      declare the shadowed ``"METHOD /path"`` in its ``overrides``; an
      undeclared collision is a build-time error, so neither an accidental
      client collision nor a framework upgrade stealing a client's route can
      change behaviour silently.
    - **Middleware** sort by ``(layer, module index, spec index)`` and are all
      added after all routes; the lowest layer ends up outermost regardless of
      module-list order.
    - **Lifespans** compose via ``AsyncExitStack``: startup in module-list
      order, shutdown in reverse; every middleware ``requires_state`` key is
      verified populated at the end of startup, aborting boot otherwise.
    - **Exception handlers** register in module-list order. Starlette keeps
      handlers in a per-exception-class dict, so when two modules register a
      handler for the *same* exception class the later module's handler wins
      (last-wins, plain Starlette semantics). ``ExceptionHandlerRegistrar``
      contributions are invoked in place, preserving the same order.
    - **Build-time state entries** apply after routes and exception handlers,
      before middleware registration (pinned here; no request exists before
      return, so any post-route point is behaviorally equivalent) — module-list
      order, each module's entries in contribution order — so contributed
      state is readable without a lifespan.

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
        DuplicateStateProviderError: Two modules provide the same state key,
            in either phase or across the two.
        MissingStateProviderError: A ``requires_state`` key has no producer.
        RouteCollisionError: Two specs mount the same ``METHOD /path`` and the
            earlier (winning) one does not declare it in ``overrides``.
    """
    module_list = list(modules)
    contributions = [list(module.contribute_state()) for module in module_list]
    providers = _validate_composition(module_list, contributions)
    # Routes are collected only once the composition is known-good: a module is
    # entitled to side effects inside contribute_routes (CoreModule sets the
    # process-global unified catalog there), and a composition rejected for a
    # duplicate name or a duplicate state provider must not have run them.
    contributed = _collect_routes(module_list)
    _validate_route_collisions(contributed)
    context = BuildContext(allowlist=_merge_allowlists(module_list))
    ordered = _collect_middleware(module_list, context)
    requirements = _validate_required_state(ordered, providers)
    app = FastAPI(
        title="Akgentic Platform API",
        lifespan=_compose_lifespan(module_list, requirements),
    )
    _mount_routes(app, contributed)
    _register_exception_handlers(app, module_list)
    _apply_state_entries(app, contributions)
    # Middleware are added strictly AFTER all routes so an APPLICATION-layer
    # middleware wraps every mounted response (mutation-log requirement).
    _add_middleware(app, ordered)
    logger.debug(
        "build_app: composed %d modules, %d middleware specs",
        len(module_list),
        len(ordered),
    )
    return app


def _validate_composition(
    modules: list[AppModule],
    contributions: list[list[StateEntry[Any]]],
) -> dict[str, str]:
    """Reject duplicate module names and duplicate state producers (build time).

    The single-writer namespace spans both phases: a key may be written by
    exactly one ``contribute_state`` entry OR one ``provides_state``
    declaration, never twice in either phase and never once in each.

    Returns:
        Mapping of each state key (both phases) to its single producing module.
    """
    names: set[str] = set()
    for module in modules:
        if module.name in names:
            raise DuplicateModuleNameError(
                f"duplicate module name '{module.name}' in composition"
            )
        names.add(module.name)
    providers: dict[str, str] = {}
    for module, entries in zip(modules, contributions, strict=True):
        for entry in entries:
            if entry.key.name in providers:
                raise DuplicateStateProviderError(
                    f"state key '{entry.key.name}' has two producers: "
                    f"modules '{providers[entry.key.name]}' and '{module.name}'"
                )
            providers[entry.key.name] = module.name
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


def _collect_routes(modules: list[AppModule]) -> list[_ContributedRoute]:
    """Call ``contribute_routes`` once per module, in composition order.

    The ONLY call site of ``contribute_routes`` in the builder. A module is
    entitled to construct its routers inside it, so calling it a second time
    for mounting would double-construct and could yield routers other than the
    ones the collision check just cleared.

    Returns:
        Every contributed spec, tagged with its module, in composition order.
    """
    return [
        _ContributedRoute(module_name=module.name, spec=spec)
        for module in modules
        for spec in module.contribute_routes()
    ]


def _route_keys(spec: RouteSpec) -> list[str]:
    """Full mounted ``"METHOD /path"`` keys for one spec — one entry per method.

    The key is per method (unlike ``build_manifest``'s comma-joined form), so
    ``GET /x`` and ``POST /x`` are different routes. Route attributes are read
    with the same defensive ``getattr`` idiom ``build_manifest`` uses rather
    than by ``isinstance`` on FastAPI/Starlette route classes: a version bump
    reshaping that hierarchy then costs nothing here. Websocket routes carry no
    methods and key as ``WS``.

    Returns:
        The spec's mounted keys, in router order then method-sorted order.
    """
    return [
        f"{method} {spec.prefix}{path}"
        for route in spec.router.routes
        if (path := getattr(route, "path", None)) is not None
        for method in sorted(getattr(route, "methods", None) or {"WS"})
    ]


def _validate_route_collisions(contributed: list[_ContributedRoute]) -> None:
    """Reject a shadowed ``METHOD /path`` its winning spec never declared.

    Detection covers module-contributed routes only. FastAPI's own built-ins
    (``/openapi.json``, ``/docs``, ``/redoc``) are added by the ``FastAPI(...)``
    constructor, belong to no module, and could not be declared against — so
    raising on them would be a build failure the client cannot fix.

    One declaration on the earliest spec satisfies the key however many later
    specs collide on it; the check reports the first offending pair per key
    rather than accumulating.

    Raises:
        RouteCollisionError: A later spec shadows an earlier one on the same
            ``METHOD /path`` and the earlier (winning) spec does not list that
            entry in ``overrides``.
    """
    winners: dict[str, _ContributedRoute] = {}
    for entry in contributed:
        # Keys are deduplicated per spec: one router registering the same
        # METHOD /path twice is that module's own business, not the
        # cross-module shadowing this check exists to surface.
        for key in dict.fromkeys(_route_keys(entry.spec)):
            winner = winners.get(key)
            if winner is None:
                winners[key] = entry
            elif key not in winner.spec.overrides:
                raise RouteCollisionError(
                    f"route collision on '{key}': module '{entry.module_name}' shadows "
                    f"module '{winner.module_name}'; the earlier module "
                    f"'{winner.module_name}' wins at runtime and must declare it — "
                    f'add overrides=("{key}",) to its RouteSpec'
                )


def _mount_routes(app: FastAPI, contributed: list[_ContributedRoute]) -> None:
    """Mount every collected router in composition order (earlier wins collisions)."""
    for entry in contributed:
        app.include_router(entry.spec.router, prefix=entry.spec.prefix)


def _register_exception_handlers(app: FastAPI, modules: list[AppModule]) -> None:
    """Register handlers in module-list order (same-class re-registration: last wins).

    ``ExceptionHandlerRegistrar`` contributions are invoked in place, so a
    registrar's registrations interleave with sibling ``ExceptionHandlerSpec``
    contributions exactly in contribution order.
    """
    for module in modules:
        for spec in module.contribute_exception_handlers():
            if isinstance(spec, ExceptionHandlerRegistrar):
                spec.install(app)
            else:
                app.add_exception_handler(spec.exception_class, spec.handler)


def _apply_state_entries(app: FastAPI, contributions: list[list[StateEntry[Any]]]) -> None:
    """Apply build-time state entries: module-list order, contribution order within."""
    for entries in contributions:
        for entry in entries:
            entry.apply(app)


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


class ManifestDelta(BaseModel):
    """What one composition adds to, removes from, and disturbs in another.

    The client-side counterpart of :class:`AppManifest`: the framework pins a
    manifest to prove a refactor changed nothing, a client diffs two manifests
    to prove its own module added exactly what it meant to.

    Membership only, per list. ``routes_added`` / ``routes_removed`` are sorted
    so a client can assert plain list equality. ``middleware_added`` /
    ``middleware_removed`` deliberately are NOT sorted: they keep manifest
    order — composed order (outermost→innermost) for additions, stock order for
    removals — because order is the only information a middleware name carries.

    ``stock_middleware_reordered`` reports the **relative order of the stock
    entries inside the composed list**, and is deliberately independent of both
    additions and removals. A client module slotting in at any layer changes
    absolute positions and must not raise the flag; a stock entry that
    disappeared is reported by ``middleware_removed`` alone, never also as a
    reorder.
    """

    model_config = ConfigDict(frozen=True)

    routes_added: list[str] = Field(
        description="Sorted route entries in composed but not in stock",
    )
    routes_removed: list[str] = Field(
        description="Sorted route entries in stock but not in composed",
    )
    middleware_added: list[str] = Field(
        description="Middleware names new in composed, in composed (outermost-first) order",
    )
    middleware_removed: list[str] = Field(
        description="Middleware names gone from composed, in stock order",
    )
    stock_middleware_reordered: bool = Field(
        description="Whether the stock entries' relative order differs inside composed",
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


def manifest_delta(stock: AppManifest, composed: AppManifest) -> ManifestDelta:
    """Diff a stock tier manifest against a composed one — the client's gate.

    The argument order is part of the contract: ``stock`` first, so
    ``manifest_delta(build_manifest(tier_app), build_manifest(client_app))``
    reads the way a client means it. Swapping the two inverts every list
    silently.

    Both manifests are compared as **opaque entry strings** — no path parsing,
    no method splitting. A stock ``"GET /x"`` that becomes ``"GET,POST /x"``
    therefore reports as one removal plus one addition rather than as a single
    addition; that is the honest reading of a manifest entry, and it fails in
    the direction that makes a client look.

    This function reports and never enforces: it does not raise, does not warn,
    and has no strict mode. What counts as an acceptable delta is the client's
    own assertion.

    Args:
        stock: Manifest of the tier app as the framework ships it.
        composed: Manifest of that same app with the client's modules spliced
            in.

    Returns:
        The route and middleware membership differences, plus whether the stock
        middleware entries kept their relative order inside ``composed``.
    """
    stock_names = set(stock.middleware)
    composed_names = set(composed.middleware)
    # The filter is symmetric on purpose. Filtering only the composed side
    # would leave a REMOVED stock entry in the stock sequence, so every removal
    # would also raise the flag — and a client could no longer tell "the
    # framework dropped a middleware" from "the framework restacked them".
    shared_in_composed = [name for name in composed.middleware if name in stock_names]
    shared_in_stock = [name for name in stock.middleware if name in composed_names]
    return ManifestDelta(
        routes_added=sorted(set(composed.routes) - set(stock.routes)),
        routes_removed=sorted(set(stock.routes) - set(composed.routes)),
        middleware_added=[n for n in composed.middleware if n not in stock_names],
        middleware_removed=[n for n in stock.middleware if n not in composed_names],
        stock_middleware_reordered=shared_in_composed != shared_in_stock,
    )
