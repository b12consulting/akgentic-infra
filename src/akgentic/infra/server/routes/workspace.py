"""Workspace file access endpoints — tree listing, file read, file upload.

All three routes (GET ``.../tree``, GET ``.../file``, POST ``.../file``) accept an
optional ``workspace_id`` **query** parameter that selects which workspace of the
team is served. Every directory they open is a three-segment
``<scope>/<kind>/<leaf>`` path under ``workspaces_root``, produced by the single
tool-side :func:`akgentic.tool.workspace.resolve_workspace_path` (ADR-052
Decision 1, ADR-048 Decision 5) — this module composes none of it:

- When omitted, the directory is the team's own per-principal default,
  ``<workspaces_root>/<team owner's user_id>/_team/<team_id>``.
- When present, it must be a single safe path segment matching
  ``[A-Za-z0-9._-]{1,128}``; anything else (empty, ``.``/``..``, separators,
  absolute paths, over-length) is rejected with HTTP 400 *before* any
  ``Filesystem`` is constructed. It must also be a workspace one of the
  authorized team's own cards declares, or it is refused with 404 by
  ``require_workspace_access``. The matching card supplies the kind and the
  scope: a named workspace resolves under ``<owner>/_id/`` and a metadata-keyed
  one under ``<owner>/_meta/``, or under ``_shared/`` in place of the owner when
  the card declares ``workspace_sharable``.

**The ``<scope>`` is always the team owner's ``Process.user_id``, never the
calling principal's.** The caller's identity governs authorization — that is
what ``require_team_access`` and the workspace gate are for — and the owner
governs path resolution, because the owner is who the team's agents write
under (ADR-048 Decision 6). Resolving the caller's scope instead would send an
admin who has already *passed* authorization to a different, empty directory,
which is worse than a refusal: nothing signals it, and the caller concludes the
agent wrote nothing.

That needs no extra check to be safe. Bob cannot reach ``<alice>/_id/notes`` by
declaring ``workspace_id="notes"`` on a team of his own, because his team
resolves under *his* ``process.user_id``; reaching Alice's tree requires a team
Alice owns, which ``require_team_access`` refuses him.

The segment guard remains a route-boundary traversal/correctness invariant, not
an access policy — it proves the value is a safe segment. **Ownership
authorization is no longer deferred:** it is the declared-workspace check in
``require_workspace_access``, and the three-segment layout means a workspace
cannot be reached by naming it even so.
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
    get_team_service,
    require_team_access,
    require_workspace_access,
)
from akgentic.infra.server.routes._workspace_resolution import (
    stashed_team_process,
    stashed_workspace_paths,
    validate_workspace_id,
)
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import ServerSettings
from akgentic.infra.server.state_keys import SETTINGS
from akgentic.tool.workspace import Filesystem, resolve_workspace_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspace", tags=["workspace"])

_MAX_FILE_SIZE = 10_485_760  # 10 MB


def _team_own_path(team_id: uuid.UUID, service: TeamService, request: Request) -> PurePosixPath:
    """The team's own tree, ``<owner>/_team/<team_id>``, through the one resolver.

    Reached only when ``workspace_id`` was omitted, which
    ``require_workspace_access`` passes through because
    ``require_team_access`` has already authorized the team. The team's metadata
    is threaded in even though the default layout never consults it, so this
    call site stays the same call the card makes at bind time.

    The team is read back from the request rather than fetched again — the gate
    that authorized it put it there, and ``get_team`` is a database read on the
    department and enterprise tiers. The fallback exists for a caller outside a
    route; unlike the declared map, re-reading the team skips no authorization.

    The scope is ``process.user_id``, not the calling principal's: the agent
    writes under the owner, so scoping on the caller would send an authorized
    admin to a different, empty directory and let them conclude the agent wrote
    nothing.
    """
    process = stashed_team_process(request) or service.get_team(team_id)
    if process is None:  # pragma: no cover — require_team_access 404s first
        raise HTTPException(status_code=404, detail="Team not found")
    try:
        return resolve_workspace_path(
            workspace_id=None,
            workspace_metadata_keys=[],
            team_id=str(team_id),
            user_id=process.user_id,
            metadata=process.metadata,
            # No card is resolved on this branch: it has always served the team's
            # per-principal default tree, whatever the team's cards declare, and it
            # still does. A literal is honest only because no card is consulted.
            # TODO(#456): resolve the team's default card inside the access gate and
            # pass its own workspace_sharable, which retires this function.
            workspace_sharable=False,
        )
    except ValueError as exc:
        # ADR-048 Decision 4, read-path row: nobody supplied this user_id
        # through the request, so 400 tells someone to fix a field that does not
        # exist and 403 asserts an access decision nobody made. An owner id that
        # cannot be a directory name is a defect in the identity producer or in
        # stored data, which is what 5xx means.
        logger.error(
            "workspace path resolution failed — team_id=%s owner=%r: %s",
            team_id,
            process.user_id,
            exc,
        )
        raise HTTPException(status_code=500, detail="Workspace path could not be resolved") from exc


def _get_workspace(
    team_id: uuid.UUID,
    settings: ServerSettings,
    *,
    request: Request,
    service: TeamService,
    workspace_id: str | None = None,
) -> Filesystem:
    """Instantiate a Filesystem over the team's three-segment workspace path.

    The path is never composed here, and the calling principal is not an input:
    the caller's identity governed *authorization*, which the two gates have
    already applied, and every path is scoped on the team owner's
    ``Process.user_id``. An omitted ``workspace_id`` resolves the team's own tree
    through the resolver; a present one is looked up in the map
    ``require_workspace_access`` already resolved and authorized for this
    request, so the card store is read once per request rather than twice — and
    so is the team, which the access gate stashed alongside it.

    A present ``workspace_id`` with no such map is **404**, never a fallback: a
    request that reached the directory without the gate having authorized the id
    is exactly the fail-open ADR-048 removes, and falling back to an unscoped
    path is the bug itself.

    ``workspaces_root`` is declared on ``CommunitySettings``; a base
    ``ServerSettings`` deployment falls back to the same default the field
    declares, mirroring ``create_app``'s own defensive read (byte-identical
    behaviour to the historical ``cast(CommunitySettings, ...)``).
    """
    # Distinguish an *omitted* param (None → the team's own tree) from an
    # *empty* one ("" → 400): only None falls back; any present value,
    # including "", goes through the guard.
    if workspace_id is None:
        path = _team_own_path(team_id, service, request)
    else:
        validate_workspace_id(workspace_id)
        declared = stashed_workspace_paths(request)
        resolved = None if declared is None else declared.get(workspace_id)
        if resolved is None:
            raise HTTPException(status_code=404, detail="Team not found")
        path = resolved
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
    service: TeamService = Depends(get_team_service),
) -> WorkspaceTreeResponse:
    """List files in a team's workspace directory."""
    logger.debug("GET /workspace/%s/tree path=%s", team_id, path)
    settings = SETTINGS.require(request)
    ws = _get_workspace(
        team_id,
        settings,
        request=request,
        service=service,
        workspace_id=workspace_id,
    )
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
    service: TeamService = Depends(get_team_service),
) -> Response:
    """Read a file from a team's workspace."""
    logger.debug("GET /workspace/%s/file path=%s", team_id, path)
    settings = SETTINGS.require(request)
    ws = _get_workspace(
        team_id,
        settings,
        request=request,
        service=service,
        workspace_id=workspace_id,
    )
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
    service: TeamService = Depends(get_team_service),
) -> WorkspaceFileUploadResponse:
    """Upload a file to a team's workspace."""
    settings = SETTINGS.require(request)
    ws = _get_workspace(
        team_id,
        settings,
        request=request,
        service=service,
        workspace_id=workspace_id,
    )
    data = await file.read()
    if len(data) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB size limit")
    logger.info("POST /workspace/%s/file path=%s, size=%d", team_id, path, len(data))
    try:
        await asyncio.to_thread(ws.write, path, data)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Path access denied") from None
    return WorkspaceFileUploadResponse(path=path, size=len(data))
