"""Operational maintenance jobs for a running akgentic deployment.

Today this is one job: the orphaned team-resource sweep, which removes the
vector-store rows and workspace directories left behind by teams that no longer
exist. See :mod:`akgentic.infra.maintenance.sweep` for why it is a reverse sweep
rather than a delete-time hook, and
:mod:`akgentic.infra.maintenance.vector_backends` for which vector backends it
reclaims and which it deliberately leaves alone.
"""

from __future__ import annotations

from akgentic.infra.maintenance.models import (
    ReaperReport,
    ResourceKind,
    ResourceRef,
    SweepReport,
)
from akgentic.infra.maintenance.reapers import (
    TeamResourceReaper,
    VectorStoreReaper,
    WorkspaceReaper,
    default_workspace_root,
)
from akgentic.infra.maintenance.sweep import (
    DEFAULT_GRACE_SECONDS,
    DEFAULT_MAX_ORPHAN_FRACTION,
    live_team_ids,
    sweep,
)
from akgentic.infra.maintenance.vector_backends import (
    BACKEND_DISPOSITIONS,
    NotSwept,
    sweepable_backends,
    unaccounted_backends,
)

__all__ = [
    "BACKEND_DISPOSITIONS",
    "DEFAULT_GRACE_SECONDS",
    "DEFAULT_MAX_ORPHAN_FRACTION",
    "NotSwept",
    "ReaperReport",
    "ResourceKind",
    "ResourceRef",
    "SweepReport",
    "TeamResourceReaper",
    "VectorStoreReaper",
    "WorkspaceReaper",
    "default_workspace_root",
    "live_team_ids",
    "sweep",
    "sweepable_backends",
    "unaccounted_backends",
]
