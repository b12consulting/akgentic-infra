"""Resource-level owner-or-admin authorization gate for catalog mutations.

Implements ADR-028 §Decision 1: a single FastAPI dependency,
``require_namespace_owner_or_admin``, attached per-route to the catalog
modify + delete routes at the shared ``/admin/catalog/*`` mount. Every
deployment tier inherits the same policy from one place; the catalog never
learns about roles (ADR-013 stays role-agnostic).

The rule is resource-level, not static-role::

    allow iff caller.user_id == namespace.owner_user_id OR "admin" in caller.roles

Identity comes from the ADR-023 ``RequestUser`` seam (``get_request_user``)
exactly as each tier already resolves it — this module adds no new auth path
and never branches on ``auth_method``. ``"admin"`` is the literal role string
from the existing three-role model (``admin``/``operator``/``viewer``).

The namespace owner is resolved the same way the catalog anchors ownership
(see ``Catalog._check_ownership``): the ``kind="team"`` entry's ``user_id`` if
a team entry exists in the namespace, else the ``kind="meta"`` (``_meta``)
entry's ``user_id``; neither → unresolvable. An unresolvable owner fails
**closed** — a non-admin caller gets 403 — while an admin still bypasses
because the admin branch returns before any owner lookup.
"""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, Request

from akgentic.catalog.models.queries import EntryQuery
from akgentic.infra.server.auth import RequestUser, get_request_user

logger = logging.getLogger(__name__)

__all__ = ["require_namespace_owner_or_admin"]


def require_namespace_owner_or_admin(
    namespace: str,
    request: Request,
    user: RequestUser = Depends(get_request_user),
) -> None:
    """Authorize a catalog mutation: caller must own ``namespace`` or be admin.

    FastAPI binds the ``namespace`` parameter from the route **path** *or*
    the **query** by name, so the same dependency covers both shapes — the
    two ``/namespace/{namespace}/...`` routes (path) and the two
    ``/{kind}/{id}`` routes (``namespace`` query).

    Args:
        namespace: The target namespace (path or query, bound by name).
        request: The inbound request — used to reach the wired catalog via
            ``request.app.state.services.catalog``.
        user: The authenticated principal resolved through the ADR-023 seam.

    Raises:
        HTTPException: 403 when the caller is neither an admin nor the
            resolved namespace owner (including the fail-closed case where
            the owner is unresolvable).
    """
    if "admin" in user.roles:
        return
    owner = _resolve_namespace_owner(request, namespace)
    if owner is None or owner != user.user_id:
        logger.info(
            "owner-or-admin gate denied mutation",
            extra={"namespace": namespace, "user_id": user.user_id, "owner": owner},
        )
        raise HTTPException(status_code=403, detail="not authorized to modify this namespace")


def _resolve_namespace_owner(request: Request, namespace: str) -> str | None:
    """Return the namespace owner's ``user_id`` via the team→meta anchor rule.

    Reads the already-wired catalog instance from
    ``request.app.state.services.catalog`` (the same object infra injects via
    ``set_catalog(services.catalog)`` in ``create_app``) and applies the
    ownership anchor identical to ``Catalog._check_ownership``: a
    ``kind="team"`` entry's ``user_id`` if one exists in the namespace, else
    the ``kind="meta"`` entry's ``user_id``. Returns ``None`` when neither
    anchor exists (owner unresolvable → caller must fail closed).

    The lookup uses the public ``Catalog.list`` API filtered by kind +
    namespace (mirroring the router's ``put_namespace_meta`` handler) so the
    anchor is found by **kind**, never by a presumed entry id.

    Args:
        request: The inbound request carrying ``app.state.services.catalog``.
        namespace: The namespace whose owner is being resolved.

    Returns:
        The anchor entry's ``user_id``, or ``None`` if the namespace has
        neither a team nor a meta entry.
    """
    catalog = request.app.state.services.catalog
    teams = catalog.list(EntryQuery(kind="team", namespace=namespace))
    if teams:
        return str(teams[0].user_id)
    metas = catalog.list(EntryQuery(kind="meta", namespace=namespace))
    if metas:
        return str(metas[0].user_id)
    return None
