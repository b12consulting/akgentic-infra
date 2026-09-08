"""Resource-ownership authorization gate for the per-``team_id`` routes.

Implements ADR-034 §Layered authorization and ADR-035 Decision 8: a single
infra-owned FastAPI dependency, ``require_team_access``, attached per-route to
the team routes that resolve a specific ``team_id``. It is **authorization, not
authentication** — a ``Depends`` at the route consuming the principal the auth
seam already resolved, mirroring ADR-028's ``require_namespace_owner_or_admin``.

Infra keeps the team lookup and the load-bearing 404-over-403
no-existence-leak machinery non-overridable; only the allow/deny *rule* is
pluggable. The rule is a
:class:`~akgentic.infra.protocols.authz.TeamAccessPolicy` carried on the
services bundle (``TierServices.team_access_policy``); the base default is
:class:`~akgentic.infra.adapters.shared.owner_or_admin_policy.OwnerOrAdminPolicy`::

    allow iff process.user_id == caller.user_id OR "admin" in caller.roles

A missing team and a denied caller both raise **404** — 404-over-403, so the
gate never leaks the existence of a team the caller may not see. The missing
team is rejected *before* the policy is consulted. The team-access seam is the
existing infra ``TeamService.get_team`` (resolved from ``app.state``). The gate
is ``async`` so a tier policy may consult an external membership / RBAC store.

``require_workspace_access`` applies the same 404-over-403 answer to the
``?workspace_id=`` query parameter, and since ADR-048 Decision 7 it applies it
to *every* value: a workspace the authorized team's own cards do not declare is
refused rather than passed through to be served as a directory name. The
declared set is resolved through the single tool-side resolver in
``_workspace_resolution`` — the gate composes no path of its own.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import PurePosixPath

from fastapi import Depends, HTTPException, Request

from akgentic.infra.protocols.authz import TeamAccessContext, TeamAccessPolicy
from akgentic.infra.server.auth import RequestUser, get_request_user
from akgentic.infra.server.routes._workspace_resolution import (
    declared_workspace_paths,
    stash_team_process,
    stash_workspace_paths,
    stashed_team_process,
    validate_workspace_id,
)
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.state_keys import SERVICES, TEAM_SERVICE
from akgentic.team.models import Process
from akgentic.team.ports import AgentCardNotFoundError, EventStore

logger = logging.getLogger(__name__)

__all__ = [
    "get_event_store",
    "get_team_access_policy",
    "get_team_service",
    "require_team_access",
    "require_workspace_access",
]


def get_team_service(request: Request) -> TeamService:
    """Team-access seam: the wired ``TeamService`` from ``app.state``."""
    return TEAM_SERVICE.require(request)


def get_event_store(request: Request) -> EventStore:
    """Card-store seam: the wired ``EventStore`` from ``app.state``.

    The workspace gate resolves the authorized team's ``agent_cards`` against
    it. Read off the services bundle rather than through ``TeamService``'s
    private handle, alongside the other two seams below.
    """
    return SERVICES.require(request).event_store


def get_team_access_policy(request: Request) -> TeamAccessPolicy:
    """Per-team authorization seam: the wired ``TeamAccessPolicy`` from ``app.state``."""
    return SERVICES.require(request).team_access_policy


async def require_team_access(
    request: Request,
    team_id: uuid.UUID,
    user: RequestUser = Depends(get_request_user),
    service: TeamService = Depends(get_team_service),
    policy: TeamAccessPolicy = Depends(get_team_access_policy),
) -> RequestUser:
    """Authorize a per-team route: the caller must satisfy the wired policy.

    The authorized ``Process`` is recorded on the request, because the workspace
    gate and the route that opens the directory both need the same team and
    ``get_team`` is a database read on the department and enterprise tiers.
    Nothing downstream re-authorizes from it — it is the team this gate already
    said yes to, kept so one request is one team read.

    Args:
        request: The live request, carrying the slot the team lands in.
        team_id: The target team, bound from the route path.
        user: The authenticated principal (always populated by the seam).
        service: The team-access seam resolving the ``Process`` by ``team_id``.
        policy: The wired per-team authorization rule (owner-or-admin default).

    Returns:
        The authenticated ``RequestUser`` on success.

    Raises:
        HTTPException: 404 when the team is missing (raised *before* the policy
            is consulted) OR the wired policy denies the caller (404-over-403,
            no existence leak).
    """
    process = service.get_team(team_id)
    if process is None:
        raise HTTPException(status_code=404, detail="Team not found")
    ctx = TeamAccessContext(team_id=team_id, owner_user_id=process.user_id)
    if not await policy.is_allowed(ctx=ctx, user=user):
        logger.info(
            "team-access gate denied",
            extra={"team_id": str(team_id), "user_id": user.user_id, "owner": process.user_id},
        )
        raise HTTPException(status_code=404, detail="Team not found")
    stash_team_process(request, process)
    return user


async def _deny_foreign_named_team(
    workspace_id: str,
    user: RequestUser,
    service: TeamService,
    policy: TeamAccessPolicy,
) -> None:
    """Raise 404 when ``workspace_id`` names an existing team the policy denies.

    Kept from the original gate and still first, but **no longer the branch
    isolation rests on.** It was written when the served directory was the
    ``workspace_id`` itself, so naming a foreign team's id reached that team's
    tree. Under the two-segment layout it cannot: the id is a *leaf*, resolved
    under the authorized team's own owner scope, and the declared-workspace
    check below refuses it in any case. What survives is the sharper answer —
    a 404 carrying the foreign owner in the log record — for the one team that
    really declares another team's id as a ``workspace_id``, which would
    otherwise be served its own directory of that name.

    A value that is not a team id, or names no team, falls through to the
    declared-workspace check — it is no longer a pass-through.
    """
    try:
        named_team_id = uuid.UUID(workspace_id)
    except ValueError:
        return
    process = service.get_team(named_team_id)
    if process is None:
        return
    ctx = TeamAccessContext(team_id=named_team_id, owner_user_id=process.user_id)
    if not await policy.is_allowed(ctx=ctx, user=user):
        logger.info(
            "workspace-access gate denied",
            extra={
                "workspace_id": workspace_id,
                "user_id": user.user_id,
                "owner": process.user_id,
            },
        )
        raise HTTPException(status_code=404, detail="Team not found")


def _resolve_declared(
    team_id: uuid.UUID,
    process: Process,
    store: EventStore,
) -> dict[str, PurePosixPath]:
    """Resolve the team's declared workspaces, turning both failure modes into 5xx.

    Both are server-side integrity failures rather than anything the caller can
    restate — there is no request field to change — so neither may become a 400
    that blames the client or a 404 that quietly shrinks the allowed set.
    ``AgentCardNotFoundError`` is a ``LookupError``, deliberately never a
    ``ValueError``, and is caught in its own arm accordingly.
    """
    try:
        return declared_workspace_paths(process=process, store=store)
    except AgentCardNotFoundError as exc:
        # The message names the unresolved ref's role AND hash.
        logger.error("workspace-access card resolution failed — team_id=%s: %s", team_id, exc)
        raise HTTPException(status_code=500, detail="Workspace cards could not be read") from exc
    except ValueError as exc:
        logger.error(
            "workspace path resolution failed — team_id=%s owner=%r: %s",
            team_id,
            process.user_id,
            exc,
        )
        raise HTTPException(status_code=500, detail="Workspace path could not be resolved") from exc


async def require_workspace_access(
    request: Request,
    team_id: uuid.UUID,
    workspace_id: str | None = None,
    user: RequestUser = Depends(get_request_user),
    service: TeamService = Depends(get_team_service),
    policy: TeamAccessPolicy = Depends(get_team_access_policy),
    store: EventStore = Depends(get_event_store),
) -> RequestUser:
    """Authorize the optional ``?workspace_id=`` query param (ADR-048 Decision 7).

    **A ``workspace_id`` no card of the authorized team declares is refused with
    404** — ADR-034's 404-over-403 no-existence-leak answer, and the reason the
    two ``return user`` pass-throughs this gate used to end in are gone. They
    let any value that was not a foreign team's id through to be served as a
    directory name, which is how one caller reached another's tree by naming it.

    The allowed set is the team's own declared cards, resolved through the same
    :func:`akgentic.tool.workspace.resolve_workspace_path` the agent side uses,
    so the matching card's layout supplies the scope and the gate never infers
    it from the string. The resolved map is recorded on the request for the
    route that opens the directory, so the card store is read once per request.

    **The caller's identity governs authorization here; the team owner's governs
    the paths.** ``user`` decides whether this request may touch the team at all
    — that is what the policy is for — and every resolved path is scoped on
    ``process.user_id``, so what the caller reaches is the tree the team's own
    agents write to.

    An omitted ``workspace_id`` is still a pass-through: it selects the team's
    own tree, which ``require_team_access`` has already authorized.

    Args:
        request: The live request, carrying the slot the resolved map lands in.
        team_id: The **authorized** team, bound from the route path. The cards
            of this team are the authority — not of whatever team the
            ``workspace_id`` may happen to name.
        workspace_id: The target workspace, bound from the route query by name.
        user: The authenticated principal (always populated by the seam).
        service: The team-access seam resolving the ``Process`` by team id.
        policy: The wired per-team authorization rule (owner-or-admin default).
        store: The card store the team's ``agent_cards`` resolve against.

    Returns:
        The authenticated ``RequestUser`` on success.

    Raises:
        HTTPException: **400** when the value is not a single safe path segment;
            **404** when it names an existing team the wired policy denies, or
            when no card of the authorized team declares it (both 404-over-403,
            no existence leak); **500** when a card cannot be read or the path
            cannot be resolved (ADR-048 Decision 4's read-path row).
    """
    if workspace_id is None:
        return user
    validate_workspace_id(workspace_id)
    await _deny_foreign_named_team(workspace_id, user, service, policy)
    process = stashed_team_process(request) or service.get_team(team_id)
    if process is None:
        raise HTTPException(status_code=404, detail="Team not found")
    declared = _resolve_declared(team_id, process, store)
    if workspace_id not in declared:
        logger.info(
            "workspace-access gate denied",
            extra={
                "workspace_id": workspace_id,
                "user_id": user.user_id,
                "owner": process.user_id,
            },
        )
        raise HTTPException(status_code=404, detail="Team not found")
    stash_workspace_paths(request, declared)
    return user
