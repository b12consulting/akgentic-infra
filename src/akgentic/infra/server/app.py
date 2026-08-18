"""FastAPI application factory for the akgentic-infra server.

Per ADR-023: the ``/admin/catalog/*`` HTTP surface IS the v2 unified router
(``akgentic.catalog.api.router.build_router``) with ``AuthStrategy`` wired
as a FastAPI dependency. The catalog package itself owns request/response
validation, error mapping, and CRUD semantics; infra owns only the mount
point, the auth gate, and the structured mutation log middleware — all of
which live in ``CoreModule`` (ADR-039, epic 57).

Uniform tier bootstrap (ADR-039 §5): :func:`create_app` is the
pre-wired-services entry point carrying the assembly sequence exactly once —
the invariant process globals (logging, catalog prefix policy) hardwired
first, then the additive ``configure_process`` tier hook, then ``build_app``
over the ``modules`` argument (``None`` means the community composition:
``CoreModule`` alone). A tier hands its module list and its process hook
directly to ``create_app``; :func:`create_server_app` is the settings-only
factory a bare ``uvicorn --factory`` target can call. ``CoreModule`` is
self-contained: state, exception handlers, and the drain lifespan are its
contributions, and nothing mutates the app after ``build_app`` returns.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from fastapi import FastAPI

from akgentic.catalog import allowed_prefixes, set_allowed_prefixes
from akgentic.infra.server.assembly import AppModule, build_app
from akgentic.infra.server.deps import TierServices
from akgentic.infra.server.logging_config import configure_logging
from akgentic.infra.server.modules import CoreModule
from akgentic.infra.server.settings import CommunitySettings, ServerSettings

logger = logging.getLogger(__name__)


def configure_process(settings: ServerSettings) -> None:
    """Default tier process hook: do nothing.

    ``create_app`` invokes its ``configure_process`` argument AFTER the
    invariant process globals (``configure_logging``,
    ``set_allowed_prefixes``) — extension semantics: a tier passes its own
    callable to ADD process configuration (e.g. OTel setup) on top of the
    invariants, never to replace them. App contributions are the modules'
    business, never this hook's.
    """


def create_app(
    services: TierServices,
    settings: ServerSettings | None = None,
    modules: Sequence[AppModule] | None = None,
    configure_process: Callable[[ServerSettings], None] = configure_process,
) -> FastAPI:
    """Create and configure the FastAPI application from pre-wired services.

    The assembly sequence, fixed here and nowhere else: ``configure_logging``
    and ``set_allowed_prefixes`` run invariant and hardwired — a tier can
    neither forget nor override them — then the ``configure_process`` hook
    runs (additive tier process config, after both by construction), then
    ``build_app`` composes the module list.

    Args:
        services: Pre-wired tier services container.
        settings: Server settings. Defaults to ``ServerSettings()``.
        modules: Ordered module composition. ``None`` selects the community
            composition — ``CoreModule`` alone.
        configure_process: Additive tier process hook, invoked after the
            invariant process globals. Defaults to the module-level named
            no-op of the same name.

    Returns:
        Configured FastAPI application instance.
    """
    settings = settings or ServerSettings()

    # configure_logging and set_allowed_prefixes are invariant, hardwired, and
    # run first. A tier cannot forget or override them. configure_process is
    # additive, invoked after the invariants, so a tier can add to the process
    # configuration but cannot replace the invariants.
    configure_logging(settings.log_level)
    logger.info("Logging configured: level=%s", settings.log_level)
    set_allowed_prefixes(settings.catalog_model_type_prefixes)
    logger.info("Catalog model_type allowlist: %s", allowed_prefixes())
    configure_process(settings)

    return build_app(
        settings, services, modules or [CoreModule(services=services, settings=settings)]
    )


def create_server_app(settings: CommunitySettings | None = None) -> FastAPI:
    """Settings-only community factory: default settings, wire, compose.

    The uniform tier-bootstrap entry point (department and enterprise expose
    the same name). Constructs the default settings itself so a bare
    ``uvicorn --factory`` target works, wires the community services, then
    delegates to :func:`create_app` — the assembly sequence exists once.
    """
    # Function-local: ``wiring`` reaches back into server internals
    # (auth_loader → adapters), so a module-level import here is circular.
    from akgentic.infra.wiring import wire_community

    settings = settings or CommunitySettings()
    services = wire_community(settings)
    return create_app(services, settings)
