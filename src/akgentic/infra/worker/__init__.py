"""Worker module — typed configuration, DI container, and shared routers for akgentic-infra workers.

After Epic 28 the dead worker FastAPI shell (``create_worker_app``,
``WorkerLifecycle``, the ``_lifespan`` factory) is gone. The public surface is
exactly: settings, the DI container, the three shared routers (readiness,
teams, and the opt-in ``/debug/memory`` diagnostics router) that every tier
mounts on its own FastAPI worker, and the team description generator each
tier's wiring hands to its runtime cache.
"""

from akgentic.infra.worker.deps import WorkerServices
from akgentic.infra.worker.description import TeamDescriptionGenerator
from akgentic.infra.worker.routes.memory_diagnostics import router as memory_diagnostics_router
from akgentic.infra.worker.routes.readiness import router as readiness_router
from akgentic.infra.worker.routes.teams import router as teams_router
from akgentic.infra.worker.settings import WorkerSettings

__all__ = [
    "TeamDescriptionGenerator",
    "WorkerServices",
    "WorkerSettings",
    "memory_diagnostics_router",
    "readiness_router",
    "teams_router",
]
