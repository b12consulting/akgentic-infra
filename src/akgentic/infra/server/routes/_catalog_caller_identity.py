"""Per-request caller-identity wiring for the ``/admin/catalog/*`` mount.

Implements ADR-028 §Decision 7 (infra side) and ADR-023 §Decision 2/4/5: each
``/admin/catalog/*`` request runs inside the catalog's caller-identity scope
``Catalog.as_caller(request_user.user_id)``, where ``request_user`` is resolved
through the ADR-023 identity seam (``get_request_user``). Department/enterprise
override that seam via ``app.dependency_overrides`` so the real authenticated
``user_id`` flows through unchanged; community resolves the default
``"anonymous"`` and behaviour is byte-unchanged.

The identity is **server-derived** from the authenticated ``RequestUser`` —
never read from a spoofable inbound client header (ADR-023 §Alternative B). The
mechanism reuses the catalog's already-present ``Catalog.as_caller`` contextvar
affordance (``akgentic.catalog.catalog``); infra does NOT reimplement the
contextvar and the catalog never imports infra.

Wiring shape: a FastAPI generator dependency. Entering opens
``Catalog.as_caller(user_id)`` (which sets a ``ContextVar`` token); the
``finally`` of the generator guarantees the contextvar is reset on exit, so a
request resolving ``"gpiroux"`` cannot bleed its caller identity into a
subsequent request that resolves ``"anonymous"`` in the shared app
(ADR-028 §Decision 7 contextvar semantics).

The catalog *reading* this contextvar on write to stamp ``entry.user_id`` is
the companion catalog half (akgentic-catalog Epic 27 / Story 27.2) — out of
scope here. Until that lands and the ``akgentic-catalog`` submodule pointer is
bumped, this wiring correctly *sets* the caller identity, but writes still
persist ``"anonymous"`` because the catalog does not yet read it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends

from akgentic.catalog.catalog import Catalog
from akgentic.infra.server.auth import RequestUser, get_request_user

__all__ = ["scope_catalog_caller_identity"]


async def scope_catalog_caller_identity(
    user: RequestUser = Depends(get_request_user),
) -> AsyncIterator[None]:
    """Run the request inside ``Catalog.as_caller(user.user_id)``.

    FastAPI resolves ``user`` through the ADR-023 seam (honouring each tier's
    ``dependency_overrides``), then enters the catalog's caller-identity
    context manager for the request body. The ``finally`` in
    ``Catalog.as_caller`` resets the contextvar token on exit, so the identity
    is scoped to a single request and never leaks across requests.

    This dependency is **async on purpose**: an async generator dependency is
    entered on the request's own asyncio task, so the ``ContextVar`` set by
    ``Catalog.as_caller`` lives in the same ``contextvars`` context that
    FastAPI snapshots when it dispatches the path operation — whether that
    handler is ``async`` (same task) or ``sync`` (run in a threadpool with a
    *copied* context). A sync generator dependency would instead set the
    contextvar in a throwaway threadpool thread whose context never reaches the
    handler, so the identity would be invisible.

    The caller ``user_id`` is taken from the authenticated ``RequestUser``
    (server-side); no inbound client header is consulted, so the spoof surface
    rejected by ADR-023 §Alternative B does not exist here.

    Yields:
        ``None`` — the dependency exists purely for its context-manager side
        effect of setting and resetting the caller-identity contextvar.
    """
    with Catalog.as_caller(user.user_id):
        yield
