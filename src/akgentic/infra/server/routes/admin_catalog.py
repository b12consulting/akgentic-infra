"""Admin catalog router — generic CRUD over templates, tools, agents, teams.

Implements ADR-022 §D1 (five CRUD verbs per entity), §D2 (repository-exception
→ HTTP-status mapping delegated to the global handlers registered by
``akgentic.catalog.api._errors.add_exception_handlers``), and §D7 (structured
INFO-level log line on every mutation; reads stay silent at INFO).

One generic helper :func:`_register_entity_routes` is applied four times — once
per entity — with per-entity Pydantic entry models and per-entity ``?q=``
mappings. Adding a fifth entity is a one-line additional call.

`?q=<s>` per-entity mapping (mirrors AC #6):

* ``templates`` → ``TemplateQuery(placeholder=s)``
* ``tools`` → ``ToolQuery(name=s)``
* ``agents`` → ``AgentQuery(description=s)``
* ``teams`` → ``TeamQuery(name=s)``

The router relies entirely on middleware-level auth (ADR-022 §D3) — no
``Depends(AuthStrategy)`` is attached to any route, and no RBAC role is
introduced. The only auth contact is defensive, inside the mutation-log
emitter, where ``services.auth.authenticate(request)`` feeds the
``principal_id`` log field.
"""

import logging
from collections.abc import Callable
from typing import cast

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from akgentic.catalog.models import (
    AgentEntry,
    AgentQuery,
    EntryNotFoundError,
    TeamEntry,
    TeamQuery,
    TemplateEntry,
    TemplateQuery,
    ToolEntry,
    ToolQuery,
)
from akgentic.catalog.services import (
    AgentCatalog,
    TeamCatalog,
    TemplateCatalog,
    ToolCatalog,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/catalog", tags=["admin-catalog"])


# --- Per-entity dependency accessors (module-private) -------------------------


def _get_template_catalog(request: Request) -> TemplateCatalog:
    """Read the wired TemplateCatalog from ``app.state.services``."""
    return cast(TemplateCatalog, request.app.state.services.template_catalog)


def _get_tool_catalog(request: Request) -> ToolCatalog:
    """Read the wired ToolCatalog from ``app.state.services``."""
    return cast(ToolCatalog, request.app.state.services.tool_catalog)


def _get_agent_catalog(request: Request) -> AgentCatalog:
    """Read the wired AgentCatalog from ``app.state.services``."""
    return cast(AgentCatalog, request.app.state.services.agent_catalog)


def _get_team_catalog(request: Request) -> TeamCatalog:
    """Read the wired TeamCatalog from ``app.state.services``."""
    return cast(TeamCatalog, request.app.state.services.team_catalog)


# --- Logging helper -----------------------------------------------------------


def _emit_mutation_log(
    request: Request, *, entity_name: str, entity_id: str, operation: str
) -> None:
    """Emit one structured INFO log line per mutation (create/update/delete).

    The only field sourced from auth is ``principal_id``, resolved defensively
    via the already-wired ``services.auth`` — no ``Depends(AuthStrategy)``
    (ADR-022 §D3, middleware-level auth).
    """
    principal_id = request.app.state.services.auth.authenticate(request) or "anonymous"
    logger.info(
        "catalog admin mutation",
        extra={
            "principal_id": principal_id,
            "entity_type": entity_name,
            "entity_id": entity_id,
            "operation": operation,
        },
    )


# --- Generic route registration ----------------------------------------------


def _register_entity_routes(
    router: APIRouter,
    *,
    entity_name: str,
    get_catalog: Callable[..., object],
    entry_model: type[BaseModel],
    query_from_q: Callable[[str], BaseModel],
) -> None:
    """Register the five CRUD verbs for one entity on ``router``.

    Generic over entity: delegates to the passed-in ``get_catalog`` dependency
    and to ``entry_model`` as request body / response type. Uses
    ``query_from_q`` to translate the optional ``?q=<s>`` query parameter into
    the entity's typed query model (per AC #6).

    Mutation routes never wrap the catalog call in try/except — they let
    ``CatalogValidationError`` (→ 409) and ``EntryNotFoundError`` (→ 404)
    propagate to the global handlers registered in
    ``akgentic.catalog.api._errors.add_exception_handlers``.
    """
    singular = entity_name.rstrip("s").title()
    prefix = f"/{entity_name}"

    @router.get(prefix, response_model=list[entry_model])  # type: ignore[valid-type]
    def _list_entries(
        q: str | None = None,
        catalog: object = Depends(get_catalog),
    ) -> list[BaseModel]:
        if q is None:
            return catalog.list()  # type: ignore[attr-defined, no-any-return]
        return catalog.search(query_from_q(q))  # type: ignore[attr-defined, no-any-return]

    @router.get(prefix + "/{entry_id}", response_model=entry_model)
    def _get_entry(
        entry_id: str,
        catalog: object = Depends(get_catalog),
    ) -> BaseModel:
        result = catalog.get(entry_id)  # type: ignore[attr-defined]
        if result is None:
            raise EntryNotFoundError(f"{singular} id '{entry_id}' not found")
        return cast(BaseModel, result)

    @router.post(prefix, response_model=entry_model, status_code=201)
    def _create_entry(
        entry: entry_model,  # type: ignore[valid-type]
        request: Request,
        catalog: object = Depends(get_catalog),
    ) -> BaseModel:
        catalog.create(entry)  # type: ignore[attr-defined]
        entry_id: str = entry.id  # type: ignore[attr-defined]
        _emit_mutation_log(request, entity_name=entity_name, entity_id=entry_id, operation="create")
        return cast(BaseModel, entry)

    @router.put(prefix + "/{entry_id}", response_model=entry_model)
    def _update_entry(
        entry_id: str,
        entry: entry_model,  # type: ignore[valid-type]
        request: Request,
        catalog: object = Depends(get_catalog),
    ) -> BaseModel:
        catalog.update(entry_id, entry)  # type: ignore[attr-defined]
        _emit_mutation_log(request, entity_name=entity_name, entity_id=entry_id, operation="update")
        return cast(BaseModel, entry)

    @router.delete(prefix + "/{entry_id}", status_code=204)
    def _delete_entry(
        entry_id: str,
        request: Request,
        catalog: object = Depends(get_catalog),
    ) -> None:
        catalog.delete(entry_id)  # type: ignore[attr-defined]
        _emit_mutation_log(request, entity_name=entity_name, entity_id=entry_id, operation="delete")


# --- Four registration calls — one per entity -------------------------------


_register_entity_routes(
    router,
    entity_name="templates",
    get_catalog=_get_template_catalog,
    entry_model=TemplateEntry,
    query_from_q=lambda s: TemplateQuery(placeholder=s),
)
_register_entity_routes(
    router,
    entity_name="tools",
    get_catalog=_get_tool_catalog,
    entry_model=ToolEntry,
    query_from_q=lambda s: ToolQuery(name=s),
)
_register_entity_routes(
    router,
    entity_name="agents",
    get_catalog=_get_agent_catalog,
    entry_model=AgentEntry,
    query_from_q=lambda s: AgentQuery(description=s),
)
_register_entity_routes(
    router,
    entity_name="teams",
    get_catalog=_get_team_catalog,
    entry_model=TeamEntry,
    query_from_q=lambda s: TeamQuery(name=s),
)


__all__ = ["router"]
