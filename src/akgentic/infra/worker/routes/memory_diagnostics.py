"""Shared, temporary, opt-in ``/debug/memory`` leak-diagnostics router.

A sibling of the readiness/teams worker routers. Stateless: the baseline census
is held by the client (posted back to ``/diff``); the server only ever reads its
own heap. The pure primitives are imported from ``akgentic.core.diagnostics``
(core owns them and the self-referential ``ReferrerNode`` rebuild); only the
FastAPI surface lives here. Diagnostic only — mounted opt-in per tier behind
``AKGENTIC_WORKER_DEBUG_MEMORY``; remove before GA (ADR-015 §2).
"""

from __future__ import annotations

import gc

from fastapi import APIRouter

from akgentic.core.diagnostics import (
    ObjectCensus,
    ReferrerNode,
    ReferrerReport,
    TypeGrowth,
)

router = APIRouter(prefix="/debug/memory", tags=["debug"])

# Referrer types that are never the leak root — the walk's own plumbing and
# generic frames — so the chain shows the real holder, not our call stack.
_SKIP_REFERRER_TYPES = frozenset({"frame", "ReferrerNode", "list"})


@router.get("/census", response_model=ObjectCensus)
async def census(label: str = "") -> ObjectCensus:
    """Return the worker's current live per-class object census (post gc.collect)."""
    return ObjectCensus.capture(label=label)


@router.post("/census/diff", response_model=list[TypeGrowth])
async def census_diff(baseline: ObjectCensus, top: int | None = None) -> list[TypeGrowth]:
    """Diff a client-supplied baseline census against the worker's current census.

    Returns the per-class growth (current - baseline), positive deltas only,
    ranked worst-first — the classes this worker retained since the baseline.
    """
    return ObjectCensus.diff(baseline, ObjectCensus.capture(label="current"), top=top)


def _short(obj: object) -> str:
    """Trimmed repr for a referrer (never raises, never huge)."""
    try:
        rep = repr(obj)
    except Exception:  # noqa: BLE001 — a broken __repr__ must not break the drill
        return f"<{type(obj).__name__} repr failed>"
    return rep[:120] + "…" if len(rep) > 120 else rep


def _walk(obj: object, depth: int, fanout: int, seen: set[int]) -> list[ReferrerNode]:
    """Walk up gc.get_referrers from ``obj`` to ``depth``, ``fanout`` per level."""
    if depth <= 0:
        return []
    nodes: list[ReferrerNode] = []
    for ref in gc.get_referrers(obj):
        if type(ref).__name__ in _SKIP_REFERRER_TYPES or id(ref) in seen:
            continue
        seen.add(id(ref))
        nodes.append(
            ReferrerNode(
                type_name=type(ref).__name__,
                detail=_short(ref),
                referrers=_walk(ref, depth - 1, fanout, seen),
            )
        )
        if len(nodes) >= fanout:
            break
    return nodes


@router.get("/referrers", response_model=ReferrerReport)
async def referrers(
    type_name: str,
    depth: int = 4,
    fanout: int = 3,
    samples: int = 3,
    newest: bool = True,
) -> ReferrerReport:
    """Trace who still holds live instances of ``type_name`` — names the leak root.

    Picks ``samples`` live instances of the named class and walks up
    ``gc.get_referrers`` ``depth`` hops (``fanout`` referrers per hop) so the
    chain ends at the long-lived root pinning them. Run it on a class from the
    census diff, e.g. ``?type_name=_ProxyHistogram``.

    ``newest=True`` (default) samples the LAST instances in heap order — the ones
    most likely allocated during the load run, i.e. the leaked ones — so a class
    with a large legitimate baseline (e.g. ModelMetaclass, whose first instance is
    ``BaseModel``) still surfaces a leaked instance rather than an imported one.
    """
    gc.collect()
    instances = [o for o in gc.get_objects() if type(o).__name__ == type_name]
    chosen = instances[-samples:] if newest else instances[:samples]
    trees = [
        ReferrerNode(
            type_name=type_name,
            detail=_short(inst),
            referrers=_walk(inst, depth, fanout, {id(inst), id(instances), id(chosen)}),
        )
        for inst in chosen
    ]
    return ReferrerReport(type_name=type_name, live_count=len(instances), samples=trees)
