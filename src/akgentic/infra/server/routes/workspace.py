"""Workspace file access endpoints — tree listing, file read, file upload.

All three routes (GET ``.../tree``, GET ``.../file``, POST ``.../file``) accept an
optional ``workspace_id`` **query** parameter that selects which workspace of the
team is served. Every directory they open is a three-segment
``<scope>/<kind>/<leaf>`` path under ``workspaces_root``, produced by the single
tool-side :func:`akgentic.tool.workspace.resolve_workspace_path` (ADR-052
Decision 1, ADR-048 Decision 5) — this module composes none of it:

- When omitted, the directory is the team's own tree, the one its default-layout
  card binds to: ``<workspaces_root>/<team owner's user_id>/_team/<team_id>``,
  or ``_shared/_team/<team_id>`` when that card declares ``workspace_sharable``.
  A team with no default-layout card gets the per-principal form.
- When present, it must be a single safe path segment matching
  ``[A-Za-z0-9._-]{1,128}``; anything else (empty, ``.``/``..``, separators,
  absolute paths, over-length) is rejected with HTTP 400 *before* any
  ``Filesystem`` is constructed. It must also be a workspace one of the
  authorized team's own cards declares, or it is refused with 404 by
  ``require_workspace_access``. The matching card supplies the kind and the
  scope: a named workspace resolves under ``<owner>/_id/`` and a metadata-keyed
  one under ``<owner>/_meta/``, or under ``_shared/`` in place of the owner when
  the card declares ``workspace_sharable``.

**A principal ``<scope>`` is always the team owner's ``Process.user_id``, never
the calling principal's.** The caller's identity governs authorization — that is
what ``require_team_access`` and the workspace gate are for — and the owner
governs path resolution, because the owner is who the team's agents write
under (ADR-048 Decision 6). Resolving the caller's scope instead would send an
admin who has already *passed* authorization to a different, empty directory,
which is worse than a refusal: nothing signals it, and the caller concludes the
agent wrote nothing.

Bob cannot reach ``<alice>/_id/notes`` by declaring ``workspace_id="notes"`` on a
team of his own, because his team resolves under *his* ``process.user_id``;
reaching Alice's tree requires a team Alice owns, which ``require_team_access``
refuses him. A ``_shared`` tree has no owner, so every principal whose team
declares the same kind and leaf resolves to it, and ownership is the wrong
question there.

**Who may reach the tree is decided from the path's scope segment**
(ADR-052 Decision 4), in one place for all three routes:
``require_workspace_access`` selects the single path on either branch, runs
``check_workspace_scope`` on it, and stashes it. A user scope is put to the wired
``TeamAccessPolicy`` with that scope as the owner; ``_shared`` is refused with 403
for every caller until an entitlement policy exists. :func:`_get_workspace`
opens the stashed path and nothing else, so no route can open a path the check
never saw.

The segment guard remains a route-boundary traversal/correctness invariant, not
an access policy — it proves the value is a safe segment.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path, PurePosixPath
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile
from fastapi.params import File, Form

from akgentic.infra.server.models import (
    WorkspaceFileEntry,
    WorkspaceFileUploadResponse,
    WorkspaceTreeResponse,
)
from akgentic.infra.server.routes._team_access import (
    require_team_access,
    require_workspace_access,
)
from akgentic.infra.server.routes._workspace_resolution import (
    stashed_workspace_path,
    validate_workspace_id,
)
from akgentic.infra.server.settings import ServerSettings
from akgentic.infra.server.state_keys import SETTINGS
from akgentic.tool.workspace import Filesystem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspace", tags=["workspace"])

_MAX_FILE_SIZE = 10_485_760  # 10 MB


def _get_workspace(
    team_id: uuid.UUID,
    settings: ServerSettings,
    *,
    request: Request,
    workspace_id: str | None = None,
) -> Filesystem:
    """Instantiate a Filesystem over the one path the access gate authorized.

    Nothing is resolved here, on either branch, and nothing is authorized:
    ``require_workspace_access`` selected the path, checked its scope, and
    stashed it. This opens that path and **only** that path. A request with no
    stashed path is **404**, never a fallback, for an omitted ``workspace_id``
    as much as for a named one: a route that opens a path the scope check never
    saw is the hole the single convergence point closes.

    The stashed path must also be the one this request's selector names: the
    requested ``workspace_id``, or the team id when it is omitted. The gate and
    the route bind the same query, so they agree by construction; a stash that
    disagrees is refused with the same 404 rather than served. This compares
    the leaf with the selector and nothing more. It does not parse the leaf,
    and it decides nothing about who may reach the tree.

    ``workspaces_root`` is declared on ``CommunitySettings``; a base
    ``ServerSettings`` deployment falls back to the same default the field
    declares, mirroring ``create_app``'s own defensive read (byte-identical
    behaviour to the historical ``cast(CommunitySettings, ...)``).
    """
    # Distinguish an *omitted* param (None → the team's own tree) from an
    # *empty* one ("" → 400): any present value, including "", goes through
    # the guard.
    if workspace_id is not None:
        validate_workspace_id(workspace_id)
    path = stashed_workspace_path(request)
    selector = str(team_id) if workspace_id is None else workspace_id
    if path is None or path.name != selector:
        raise HTTPException(status_code=404, detail="Team not found")
    workspaces_root = getattr(settings, "workspaces_root", Path("workspaces"))
    return Filesystem(base_path=str(workspaces_root), workspace_name=str(path))


@router.get(
    "/{team_id}/tree",
    response_model=WorkspaceTreeResponse,
    dependencies=[Depends(require_team_access), Depends(require_workspace_access)],
)
def list_workspace_tree(
    team_id: uuid.UUID,
    request: Request,
    path: str = "",
    workspace_id: str | None = None,
) -> WorkspaceTreeResponse:
    """List files in a team's workspace directory."""
    logger.debug("GET /workspace/%s/tree path=%s", team_id, path)
    settings = SETTINGS.require(request)
    ws = _get_workspace(team_id, settings, request=request, workspace_id=workspace_id)
    try:
        entries = ws.list(path)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Path access denied") from None
    return WorkspaceTreeResponse(
        team_id=str(team_id),
        path=path,
        entries=[WorkspaceFileEntry(name=e.name, is_dir=e.is_dir, size=e.size) for e in entries],
    )


@router.get(
    "/{team_id}/file",
    dependencies=[Depends(require_team_access), Depends(require_workspace_access)],
)
def read_workspace_file(
    team_id: uuid.UUID,
    request: Request,
    path: str,
    workspace_id: str | None = None,
) -> Response:
    """Read a file from a team's workspace."""
    logger.debug("GET /workspace/%s/file path=%s", team_id, path)
    settings = SETTINGS.require(request)
    ws = _get_workspace(team_id, settings, request=request, workspace_id=workspace_id)
    try:
        data = ws.read(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found") from None
    except PermissionError:
        raise HTTPException(status_code=403, detail="Path access denied") from None
    if len(data) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB size limit")
    filename = PurePosixPath(path).name
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "/{team_id}/file",
    status_code=201,
    response_model=WorkspaceFileUploadResponse,
    dependencies=[Depends(require_team_access), Depends(require_workspace_access)],
)
async def upload_workspace_file(
    team_id: uuid.UUID,
    request: Request,
    path: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    workspace_id: str | None = None,
) -> WorkspaceFileUploadResponse:
    """Upload a file to a team's workspace."""
    settings = SETTINGS.require(request)
    ws = _get_workspace(team_id, settings, request=request, workspace_id=workspace_id)
    data = await file.read()
    if len(data) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB size limit")
    logger.info("POST /workspace/%s/file path=%s, size=%d", team_id, path, len(data))
    try:
        await asyncio.to_thread(ws.write, path, data)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Path access denied") from None
    return WorkspaceFileUploadResponse(path=path, size=len(data))
