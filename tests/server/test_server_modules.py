"""Story 63.1: ``server_modules`` is the community tier's public composition.

The point of the function is a client package in a repo we do not own: it must
be able to splice its own module into the tier's list without retyping that list
(which drifts silently at the next upgrade) or forking the tier. So the tests
below exercise the *client's* usage shape, not an internal helper's:

- the returned list is **fresh** on every call, so a client that appends to it
  cannot poison the next caller's composition;
- routing ``create_app``'s default through it changes **nothing** — the manifest
  is identical to the untouched community app (epic 57's golden manifest);
- ``modules=[]`` still yields the community app, pinning the load-bearing
  ``modules or server_modules(...)`` truthiness choice;
- the name resolves from ``akgentic.infra.server`` and is in ``__all__``;
- an appended ``BaseAppModule`` subclass contributing one route and one
  middleware serves the full community surface **plus** its own contributions,
  with the stock middleware order untouched.

Compositions are compared through ``build_manifest`` only — never by reading
constants or source text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.testclient import TestClient

import akgentic.infra.server
from akgentic.infra.server.app import create_app, server_modules
from akgentic.infra.server.assembly import (
    APPLICATION,
    BaseAppModule,
    BuildContext,
    MiddlewareSpec,
    RouteSpec,
    build_manifest,
)
from akgentic.infra.server.modules import CoreModule
from tests.server.test_assembly import _NoopMiddleware

if TYPE_CHECKING:
    from akgentic.infra.server.deps import CommunityServices
    from akgentic.infra.server.settings import CommunitySettings

_ACME_ROUTE = "GET /acme/reports/{report_id}"


class AcmeAuditModule(BaseAppModule):
    """Client-shaped extension module: one vendor-prefixed route, one middleware.

    Deliberately shaped like third-party code — a ``BaseAppModule`` subclass
    (so a seventh contribution verb would arrive as an inherited no-op rather
    than silently breaking the contract), a vendor-prefixed module ``name``, and
    routes under a vendor prefix the framework does not own.
    """

    name = "acme-audit"

    def contribute_routes(self) -> list[RouteSpec]:
        router = APIRouter(prefix="/acme")

        @router.get("/reports/{report_id}")
        def _report(report_id: str) -> dict[str, str]:
            return {"report_id": report_id, "vendor": "acme"}

        return [RouteSpec(router=router)]

    def contribute_middleware(self, context: BuildContext) -> list[MiddlewareSpec]:
        # Any existing band anchor serves here; the extension band is Story 63.4.
        return [MiddlewareSpec(middleware_class=_NoopMiddleware, layer=APPLICATION)]


class TestFreshCommunityComposition:
    """AC 1: one ``CoreModule``, freshly built, independent per call."""

    def test_returns_one_core_module_built_from_the_arguments(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """Exactly one entry, a ``CoreModule``, carrying the passed services."""
        modules = server_modules(community_services, seeded_settings)
        assert len(modules) == 1
        core = modules[0]
        assert isinstance(core, CoreModule)
        # Built from the passed container, not from a wiring call of its own.
        assert core.team_service is community_services.team_service

    def test_two_calls_return_independent_lists_and_instances(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """No shared module-level list, no shared ``CoreModule`` instance."""
        first = server_modules(community_services, seeded_settings)
        second = server_modules(community_services, seeded_settings)
        assert first is not second
        assert first[0] is not second[0]

    def test_client_appending_does_not_leak_into_the_next_call(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """Mutating a returned list cannot poison a later caller's composition."""
        spliced = server_modules(community_services, seeded_settings)
        spliced.append(AcmeAuditModule())
        assert len(spliced) == 2
        assert len(server_modules(community_services, seeded_settings)) == 1


class TestManifestIdentity:
    """AC 2, AC 5: routing the default through the public name changes nothing."""

    def test_explicit_server_modules_matches_the_default(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """``modules=server_modules(...)`` is manifest-identical to ``modules=None``."""
        default = create_app(community_services, seeded_settings)
        explicit = create_app(
            community_services,
            seeded_settings,
            modules=server_modules(community_services, seeded_settings),
        )
        assert build_manifest(explicit) == build_manifest(default)

    def test_empty_module_list_still_builds_the_community_app(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """The ``or`` truthiness choice: a falsy ``modules`` means community.

        Switching the default to ``is not None`` would silently turn this app
        from the community surface into an empty one — pinned here so the
        change cannot happen unnoticed.
        """
        default = create_app(community_services, seeded_settings)
        empty = create_app(community_services, seeded_settings, modules=[])
        assert build_manifest(empty) == build_manifest(default)


class TestPublishedName:
    """AC 4: the name is reachable the way a client imports it."""

    def test_resolves_from_the_server_package(self) -> None:
        """``from akgentic.infra.server import server_modules`` is the same object."""
        assert akgentic.infra.server.server_modules is server_modules

    def test_exported_in_sorted_all(self) -> None:
        """The name is in ``__all__``, sorted into the function group.

        ``__all__`` groups by case — constant anchors, then classes, then
        functions — so the invariant this story must keep is that the function
        tail it adds to stays sorted, which places ``server_modules`` directly
        after ``create_server_app``.
        """
        exported = list(akgentic.infra.server.__all__)
        assert "server_modules" in exported
        functions = [name for name in exported if name[:1].islower()]
        assert functions == sorted(functions)


class TestClientShapedComposition:
    """AC 3: append a client module to the public list; stock surface intact."""

    def test_appended_module_adds_its_surface_and_perturbs_nothing_stock(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """Community routes all still served, ``/acme`` route added, order kept."""
        community = build_manifest(create_app(community_services, seeded_settings))
        app = create_app(
            community_services,
            seeded_settings,
            modules=[*server_modules(community_services, seeded_settings), AcmeAuditModule()],
        )
        composed = build_manifest(app)

        assert set(community.routes) <= set(composed.routes)
        assert _ACME_ROUTE in composed.routes
        assert _NoopMiddleware.__name__ in composed.middleware
        # The stock middleware, in the order they appear inside the composed
        # stack, are exactly the community stack — the client module added a
        # layer without reordering anything it does not own.
        stock_in_composed = [name for name in composed.middleware if name in community.middleware]
        assert stock_in_composed == community.middleware

        with TestClient(app) as client:
            assert client.get("/readiness").status_code == 200
            assert client.get("/acme/reports/r-1").json() == {
                "report_id": "r-1",
                "vendor": "acme",
            }
