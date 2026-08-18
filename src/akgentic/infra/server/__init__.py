"""Server module — FastAPI application, models, routes, services, and assembly contract.

**This module is the published extension surface** (ADR-040 §7, §10). A client
package writing its own ``AppModule`` in a repo the framework does not own
imports everything it needs from ``akgentic.infra.server`` and from nowhere
else — the deeper module paths (``akgentic.infra.server.assembly``,
``.auth``, ``.app``) are where the code happens to live today, not a contract.
The package root ``akgentic.infra`` deliberately stays as it is: it re-exports
the tier entrypoints (``create_app``, ``create_server_app``,
``configure_process``) for a deployment, not the module-authoring surface, and
widening it would create a second published set needing its own policy.

Compatibility policy for that surface:

- **What is public is exactly ``__all__``** — every name listed there, and no
  name that is not. It carries the contract types (``AppModule``,
  ``BaseAppModule``) and, as members of those types rather than as exports of
  their own, the six contribution verbs (``contribute_routes``,
  ``contribute_middleware``, ``contribute_allowlist``,
  ``contribute_exception_handlers``, ``contribute_state``, ``lifespan``)
  together with ``name`` and ``provides_state``; the spec models
  (``RouteSpec``, ``MiddlewareSpec``, ``AllowlistSpec``,
  ``ExceptionHandlerSpec``, ``ExceptionHandlerRegistrar``, ``BuildContext``,
  ``StateEntry``); the band anchors; the entry points ``create_app``,
  ``server_modules``, ``build_app``, ``build_manifest`` and ``manifest_delta``
  with the result types a client asserts on (``AppManifest``,
  ``ManifestDelta``); the errors a composition can raise (``AssemblyError``
  and its subclasses, ``RouteCollisionError`` among them); and
  ``get_request_user`` with ``RequestUser``. A breaking change to any of them
  requires a major version and a decision record.
- **New contribution verbs arrive with a ``BaseAppModule`` default**, so a
  subclass keeps composing across the upgrade that adds them. That is why a
  third-party module subclasses ``BaseAppModule`` rather than implementing the
  ``AppModule`` Protocol structurally: a structural implementer silently stops
  satisfying the contract the day a seventh verb appears.
- **New band anchors may be added between existing ones; existing ordinals do
  not move.** ``EXTENSION = 700`` was itself added this way, inside
  ``APPLICATION = 600``, and no client middleware ordinal changed meaning.
- **Everything not in ``__all__`` is internal** and may move, be renamed or
  disappear without notice, in any release.
"""

from __future__ import annotations

from akgentic.infra.server.app import (
    configure_process,
    create_app,
    create_server_app,
    server_modules,
)
from akgentic.infra.server.assembly import (
    APPLICATION,
    EXTENSION,
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
