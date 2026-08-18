"""Story 63.4: the ``EXTENSION`` band and the published extension surface.

Two halves of one contract, tested here because a client package cannot test
them from its own repo until they exist:

- ``EXTENSION`` is the documented ordinal for third-party middleware — inside
  identity, inside policy, inside the application band. It is proven innermost
  through a composed manifest and proven *behind the identity gate* through a
  request, never by reading the constant back.
- ``akgentic.infra.server.__all__`` is the published set. It is pinned by an
  explicit literal list so an accidental export, a dropped export or a reorder
  is a failing test rather than a silent surface change.

No test here asserts on docstring or comment text: the compatibility policy is
prose for humans, and what it promises is pinned by the assertions below.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

import akgentic.infra.server
import akgentic.infra.server.auth
from akgentic.infra.server.app import create_app, server_modules
from akgentic.infra.server.assembly import (
    APPLICATION,
    EXTENSION,
    IDENTITY,
    TRANSPORT,
    BaseAppModule,
    BuildContext,
    MiddlewareSpec,
    RouteSpec,
    build_app,
    build_manifest,
)
from akgentic.infra.server.deps import CommunityServices, TierServices
from akgentic.infra.server.settings import CommunitySettings, ServerSettings
from akgentic.infra.utils import StateKey

# Ordinals with no named anchor: 510 sits inside the POLICY band's conventional
# sub-slot range, 999 outside every band the framework knows about. Both are
# legal layers — the bands are anchors, not a closed vocabulary.
_POLICY_SUB_SLOT = 510
_UNBANDED = 999

# The published surface, written out. Deliberately a literal and not a
# derivation: a sorted() or a doc-scrape would re-derive whatever the module
# happens to say, which pins nothing. ``__all__`` groups by case — constants,
# then classes, then functions — and is sorted within each group, so the
# "CommunitySettings" before "CommunityServices" inversion below is real,
# pre-dates this story and is preserved rather than quietly repaired.
_EXPECTED_ALL = [
    "APPLICATION",
    "EXTENSION",
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
    "ExceptionHandlerRegistrar",
    "ExceptionHandlerSpec",
    "HumanInputRequest",
    "ManifestDelta",
    "MiddlewareSpec",
    "MissingStateProviderError",
    "RequestUser",
    "RouteCollisionError",
    "RouteSpec",
    "SendMessageRequest",
    "ServerSettings",
    "StateEntry",
    "TeamListResponse",
    "TeamResponse",
    "TeamService",
    "TierServices",
    "UnpopulatedStateError",
    "build_app",
    "build_manifest",
    "configure_process",
    "create_app",
    "create_server_app",
    "get_request_user",
    "manifest_delta",
    "server_modules",
]

# The names the compatibility policy promises a client package: the entrypoints,
# the spec models, the contract types and every band anchor. Checked as a subset
# so a failure names the missing export, independent of the exact-order pin.
_POLICY_SURFACE = [
    "create_app",
    "server_modules",
    "build_app",
    "build_manifest",
    "manifest_delta",
    "get_request_user",
    "RouteSpec",
    "MiddlewareSpec",
    "AllowlistSpec",
    "ExceptionHandlerSpec",
    "ExceptionHandlerRegistrar",
    "BuildContext",
    "StateEntry",
    "AppModule",
    "BaseAppModule",
    "OBSERVABILITY",
    "TRANSPORT",
    "PROXY",
    "SESSION",
    "IDENTITY",
    "POLICY",
    "APPLICATION",
    "EXTENSION",
]


class _AcmeExtensionMiddleware:
    """Pass-through ASGI middleware standing in for a client's own."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app(scope, receive, send)


class _AcmePolicySlotMiddleware(_AcmeExtensionMiddleware):
    """Pass-through at an unnamed ordinal inside the POLICY band."""


class _AcmeUnbandedMiddleware(_AcmeExtensionMiddleware):
    """Pass-through at an ordinal outside every named band."""


class _AcmeTransportMiddleware(_AcmeExtensionMiddleware):
    """Pass-through at the outermost functional anchor."""


class _AcmeLayeredModule(BaseAppModule):
    """Contributes one pass-through middleware per requested (class, layer) pair."""

    def __init__(self, name: str, specs: list[tuple[type, int]]) -> None:
        self.name = name
        self._specs = specs

    def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
        return [MiddlewareSpec(middleware_class=cls, layer=layer) for cls, layer in self._specs]


class _AcmeExtensionModule(BaseAppModule):
    """A third-party module contributing one middleware at the documented default.

    Subclasses ``BaseAppModule`` rather than implementing ``AppModule``
    structurally, and carries a vendor-prefixed name so it cannot collide with a
    stock module's.
    """

    name = "acme-extension"

    def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
        return [MiddlewareSpec(middleware_class=_AcmeExtensionMiddleware, layer=EXTENSION)]


_TOKEN_HEADER = b"x-acme-token"
_TOKEN = b"open-sesame"

_SEEN_PATHS: StateKey[list[str]] = StateKey("acme_seen_paths", required=True)


class _AcmeGateMiddleware:
    """Stands in for the identity gate: 401 pre-routing without the token header.

    Written the way the framework's own middleware are — a plain ASGI class the
    builder instantiates as ``cls(app, **options)`` — because that is the
    contract ``MiddlewareSpec`` documents.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and dict(scope["headers"]).get(_TOKEN_HEADER) != _TOKEN:
            await JSONResponse({"detail": "Unauthorized"}, status_code=401)(scope, receive, send)
            return
        await self.app(scope, receive, send)


class _AcmeRecorderMiddleware:
    """Records the path of every request that reaches it, from its state slot."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            _SEEN_PATHS.require(HTTPConnection(scope)).append(scope["path"])
        await self.app(scope, receive, send)


class _AcmeGateModule(BaseAppModule):
    """Contributes the gate at IDENTITY plus the one route the accepted case hits."""

    name = "acme-gate"

    def contribute_routes(self) -> list[RouteSpec]:
        router = APIRouter()

        @router.get("/acme/ping")
        def _ping() -> dict[str, str]:
            return {"pong": "acme"}

        return [RouteSpec(router=router)]

    def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
        return [MiddlewareSpec(middleware_class=_AcmeGateMiddleware, layer=IDENTITY)]


class _AcmeRecorderModule(BaseAppModule):
    """Contributes the recorder at EXTENSION and owns the list it appends to.

    The list is resolved per request from ``app.state`` (``requires_state``)
    rather than handed over in ``options``, because ``options`` carries
    configuration and never a live collaborator.
    """

    name = "acme-recorder"
    provides_state = (_SEEN_PATHS.name,)

    def __init__(self, seen: list[str]) -> None:
        self._seen = seen

    def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
        return [
            MiddlewareSpec(
                middleware_class=_AcmeRecorderMiddleware,
                layer=EXTENSION,
                requires_state=(_SEEN_PATHS.name,),
            )
        ]

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        _SEEN_PATHS.set(app, self._seen)
        yield


def _build(modules: list[BaseAppModule]) -> FastAPI:
    """Build an app from stand-in modules with pass-through composition inputs."""
    return build_app(ServerSettings(), TierServices.model_construct(), modules)


class TestExtensionBand:
    """AC 1, 4: a named anchor innermost of the framework's own, and nothing more."""

    def test_extension_is_seven_hundred_and_inside_application(self) -> None:
        """The ordinal is fixed by the decision, not chosen by the implementation."""
        assert EXTENSION == 700
        assert APPLICATION < EXTENSION

    def test_unnamed_ordinals_compose_and_sort_by_integer_value(self) -> None:
        """The bands are anchors, not an enum: any integer stays a legal layer.

        A client integrating a signed webhook legitimately needs POLICY, so the
        builder must not grow a band-membership check. This composes ordinals
        inside a band's sub-slot range (510) and outside every band (999)
        alongside two anchors, and asserts the stack is ordered by the plain
        integers.
        """
        app = _build(
            [
                _AcmeLayeredModule(
                    "acme-unsorted",
                    [
                        (_AcmeUnbandedMiddleware, _UNBANDED),
                        (_AcmeExtensionMiddleware, EXTENSION),
                        (_AcmeTransportMiddleware, TRANSPORT),
                        (_AcmePolicySlotMiddleware, _POLICY_SUB_SLOT),
                    ],
                )
            ]
        )
        assert build_manifest(app).middleware == [
            "_AcmeTransportMiddleware",
            "_AcmePolicySlotMiddleware",
            "_AcmeExtensionMiddleware",
            "_AcmeUnbandedMiddleware",
        ]


class TestExtensionIsInnermostOfTheStockStack:
    """AC 2: proven through a composed manifest, not by reading the constant.

    Goes through ``create_app`` because "innermost of a stock community stack"
    is only meaningful against the real composition — the stock stack is what
    the client's middleware has to end up inside of.
    """

    def test_client_middleware_at_extension_lands_last(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        stock = build_manifest(create_app(community_services, seeded_settings))
        composed = build_manifest(
            create_app(
                community_services,
                seeded_settings,
                modules=[
                    *server_modules(community_services, seeded_settings),
                    _AcmeExtensionModule(),
                ],
            )
        )
        # Every stock entry in its stock position, the client's entry innermost.
        assert composed.middleware == [*stock.middleware, "_AcmeExtensionMiddleware"]


class TestExtensionSitsInsideTheIdentityGate:
    """AC 3: the property that justifies the band existing at all.

    A middleware at ``EXTENSION`` must see a request only once the identity gate
    would have admitted it — that is the whole reason a client author is told
    700 rather than left to guess a number that could land outside the gate.
    Both halves live here on purpose: without the accepted case, a recorder that
    never records would pass the rejected case trivially.
    """

    def test_rejected_request_never_reaches_the_extension_middleware(self) -> None:
        seen: list[str] = []
        app = _build([_AcmeGateModule(), _AcmeRecorderModule(seen)])
        with TestClient(app) as client:
            assert client.get("/acme/ping").status_code == 401
        assert seen == []

    def test_accepted_request_reaches_the_extension_middleware(self) -> None:
        seen: list[str] = []
        app = _build([_AcmeGateModule(), _AcmeRecorderModule(seen)])
        with TestClient(app) as client:
            response = client.get("/acme/ping", headers={"x-acme-token": "open-sesame"})
        assert response.status_code == 200
        assert response.json() == {"pong": "acme"}
        assert seen == ["/acme/ping"]


class TestIdentitySeamIsPublished:
    """AC 5, 6: one documented import path, and the old one keeps working."""

    def test_public_names_are_the_same_objects_as_the_module_path(self) -> None:
        """A re-export, not a re-implementation — identity, not equality."""
        assert akgentic.infra.server.get_request_user is akgentic.infra.server.auth.get_request_user
        assert akgentic.infra.server.RequestUser is akgentic.infra.server.auth.RequestUser

    def test_public_get_request_user_resolves_the_anonymous_default(self) -> None:
        """The behaviour a client gets through the public name, not just the name."""
        from akgentic.infra.server import RequestUser, get_request_user

        user = get_request_user(HTTPConnection({"type": "http", "headers": []}))
        assert isinstance(user, RequestUser)
        assert user.user_id == "anonymous"
        assert user.roles == []
        assert user.scopes == []

    def test_stashed_principal_is_returned_through_the_public_name(self) -> None:
        """The re-export carries the stash-reading behaviour, not only the fallback."""
        from akgentic.infra.server import RequestUser, get_request_user

        stashed = RequestUser(user_id="acme-user", roles=["admin"])
        connection = HTTPConnection({"type": "http", "headers": []})
        connection.state.request_user = stashed
        assert get_request_user(connection) is stashed


class TestPublishedSurface:
    """AC 7-10: ``__all__`` is the contract, pinned name by name and in order."""

    def test_all_is_exactly_the_pinned_list(self) -> None:
        """Order included: the case grouping is the convention this file keeps."""
        assert list(akgentic.infra.server.__all__) == _EXPECTED_ALL

    def test_every_exported_name_resolves(self) -> None:
        """No entry is a typo or a stale export left behind by a move."""
        server = akgentic.infra.server
        unresolved = [name for name in server.__all__ if not hasattr(server, name)]
        assert not unresolved, f"exported but unresolvable: {unresolved}"

    def test_every_policy_name_is_exported(self) -> None:
        """A future contributor who forgets an export learns which one."""
        exported = set(akgentic.infra.server.__all__)
        missing = [name for name in _POLICY_SURFACE if name not in exported]
        assert not missing, f"missing from __all__: {missing}"

    def test_function_tail_is_sorted(self) -> None:
        """The function group sorts plainly — the invariant 63-1 pinned.

        Deliberately NOT ``list(__all__) == sorted(__all__)``: that is red on
        untouched master because of the pre-existing CommunitySettings /
        CommunityServices inversion, which this story does not reorder.
        """
        functions = [name for name in akgentic.infra.server.__all__ if name[:1].islower()]
        assert functions == sorted(functions)
