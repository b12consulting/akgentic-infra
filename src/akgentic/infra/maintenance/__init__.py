"""Operational maintenance jobs for a running akgentic deployment.

Today this is one job: the orphaned team-resource sweep, which removes the
Weaviate objects, Docker sandbox containers and workspace directories left
behind by teams that no longer exist. See :mod:`akgentic.infra.maintenance.sweep`
for why it is a reverse sweep rather than a delete-time hook.
"""

from __future__ import annotations

from akgentic.infra.maintenance.models import (
    ReaperReport,
    ResourceKind,
    ResourceRef,
    SweepReport,
)
from akgentic.infra.maintenance.reapers import (
    GIT_DIR_SUFFIX,
    SANDBOX_CONTAINER_PREFIX,
    DockerReaper,
    TeamResourceReaper,
    WeaviateReaper,
    WorkspaceReaper,
    default_workspace_root,
)
from akgentic.infra.maintenance.sweep import (
    DEFAULT_GRACE_SECONDS,
    DEFAULT_MAX_ORPHAN_FRACTION,
    live_team_ids,
    sweep,
)

__all__ = [
    "DEFAULT_GRACE_SECONDS",
    "DEFAULT_MAX_ORPHAN_FRACTION",
    "GIT_DIR_SUFFIX",
    "SANDBOX_CONTAINER_PREFIX",
    "DockerReaper",
    "ReaperReport",
    "ResourceKind",
    "ResourceRef",
    "SweepReport",
    "TeamResourceReaper",
    "WeaviateReaper",
    "WorkspaceReaper",
    "default_workspace_root",
    "live_team_ids",
    "sweep",
]
