"""The request-boundary half of the workspace seam, and the door to the other half.

Two things live here, and they are here together on purpose: the
request-boundary segment guard for the ``?workspace_id=`` query parameter, and
the per-request slots the gate and the route hand values through. Both are
needed by ``routes/workspace.py`` (which opens the directory) *and* by
``routes/_team_access.py`` (which refuses the ones the team never declared), and
``workspace.py`` already imports ``_team_access``. A second copy in either
module is the duplication ADR-048 exists to remove, one scale down.

**The card-reading half now lives in**
:mod:`akgentic.infra.server.services._workspace_paths`, because
``TeamService.delete_team`` needs it too and the direction of dependency in this
package is ``routes -> services``. :func:`declared_workspace_paths` and
:func:`default_workspace_path` are re-exported here — listed in ``__all__``,
which is what makes the re-export legal under mypy's ``no_implicit_reexport`` —
so every route and every existing test that imports them from this module keeps
resolving unchanged. That module's docstring carries the layout table and the
scope/kind/leaf rules.

The seam that remains is clean: card-reading path resolution on one side, with
no FastAPI import; ``HTTPException`` and ``conn.state`` on this one.

Only the ``<leaf>`` is ever on the wire (ADR-048 Decision 8): the scope and the
kind are not the client's to choose, and the server recomputes both from the
matching card.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath

from fastapi import HTTPException
from starlette.requests import HTTPConnection

from akgentic.infra.server.services._workspace_paths import (
    declared_workspace_paths,
    default_workspace_path,
)
from akgentic.team.models import Process
from akgentic.tool.workspace import leaf_segment

logger = logging.getLogger(__name__)

__all__ = [
    "declared_workspace_paths",
    "default_workspace_path",
    "stash_team_process",
    "stash_workspace_path",
    "stashed_team_process",
    "stashed_workspace_path",
    "validate_workspace_id",
]

# Per-request slot holding the ONE resolved path the access gate authorized, on
# either branch. The route that opens the directory opens this path and nothing
# else, so a path the gate's scope check never saw cannot be opened. Named once,
# here, because both sides of the seam address it.
_AUTHORIZED_PATH_SLOT = "akgentic_authorized_workspace_path"

# Per-request slot holding the ``Process`` the team-access gate already loaded.
# ``get_team`` is ``EventStore.load_team`` on the department and enterprise
# tiers — a database read, not an in-process lookup — so the gate, the workspace
# gate and the route reading the same team three times is three queries where
# the request only ever concerns one team. Same idea as the authorized path
# stashed above: resolve once, read back.
_TEAM_PROCESS_SLOT = "akgentic_authorized_team_process"


def validate_workspace_id(workspace_id: str) -> str:
    """Reject any workspace_id that cannot be a workspace leaf, with HTTP 400.

    ``?workspace_id=`` is a **leaf selector**, so the guard is the tool's own
    :func:`akgentic.tool.workspace.leaf_segment` (ADR-052 Decision 6), applied
    to the value the caller sent. It refuses the empty string, a leading dot
    (``.`` and ``..`` included), ``/``, ``\\``, NUL, the three kind names and
    the sidecar suffixes, in any letter case, raising **before** any card is
    read or any ``Filesystem`` is constructed. It caps no length and admits
    ``%``, so every leaf a card can declare, a percent-encoded metadata leaf
    included, gets through.

    This is a traversal guard, not an access policy. Authorization is
    membership: the selector is served only when it is byte-equal to a leaf in
    :func:`declared_workspace_paths`. A path-safe value no card declares is
    refused there with 404.

    The detail is fixed: the tool's message reflects the caller's value and
    names internal directories, so it stays out of the response.

    Returns:
        The value unchanged when it is a valid leaf.
    """
    try:
        return leaf_segment(workspace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid workspace_id") from None


def stash_workspace_path(conn: HTTPConnection, path: PurePosixPath) -> None:
    """Record the one path the gate authorized, for the route that opens it."""
    setattr(conn.state, _AUTHORIZED_PATH_SLOT, path)


def stashed_workspace_path(conn: HTTPConnection) -> PurePosixPath | None:
    """The path the gate authorized for this request, or ``None`` if it never ran.

    ``None`` is not a licence to resolve the path some other way, on either
    branch: the route answers 404. A request that reaches the directory without
    the gate having resolved and checked the path is the hole the single
    convergence point closes.
    """
    stashed = getattr(conn.state, _AUTHORIZED_PATH_SLOT, None)
    if isinstance(stashed, PurePosixPath):
        return stashed
    return None


def stash_team_process(conn: HTTPConnection, process: Process) -> None:
    """Record the team ``require_team_access`` authorized, for the rest of the request.

    Only that gate writes this slot, and it writes the team named in the route
    path — never the foreign team ``_deny_foreign_named_team`` looks up, which
    is a different team and must not be mistaken for the authorized one.
    """
    setattr(conn.state, _TEAM_PROCESS_SLOT, process)


def stashed_team_process(conn: HTTPConnection) -> Process | None:
    """The authorized team the gate loaded, or ``None`` if it never ran.

    ``None`` is safe to fall back on here, unlike
    :func:`stashed_workspace_path`: re-reading the team is a redundant query,
    not a skipped authorization. The gate answered that question already, and
    the value it stashed is the same team the fallback would fetch.
    """
    stashed = getattr(conn.state, _TEAM_PROCESS_SLOT, None)
    if isinstance(stashed, Process):
        return stashed
    return None
