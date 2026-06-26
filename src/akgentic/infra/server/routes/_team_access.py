"""Resource-ownership authorization gate for the per-``team_id`` routes.

Implements ADR-034 §Layered authorization: a single infra-owned FastAPI
dependency, ``require_team_access``, attached per-route to the team routes that
resolve a specific ``team_id``. It is **authorization, not authentication** — a
``Depends`` at the route consuming the principal the auth seam already
resolved, mirroring ADR-028's ``require_namespace_owner_or_admin``.

The rule, keyed on the team owner ``Process.user_id``::

    allow iff process.user_id == caller.user_id OR "admin" in caller.roles

A missing team and a non-owner non-admin both raise **404** — 404-over-403, so
the gate never leaks the existence of a team the caller may not see. The
``admin`` role bypasses ownership. The team-access seam is the existing infra
``TeamService.get_team`` (resolved from ``app.state``); the RBAC role
*vocabulary* stays tier-side — this gate only checks ``"admin"`` membership.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import Depends, HTTPException, Request

from akgentic.infra.server.auth import RequestUser, get_request_user
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.state_keys import TEAM_SERVICE

logger = logging.getLogger(__name__)

__all__ = ["get_team_service", "require_team_access"]

_ADMIN_ROLE = "admin"


def get_team_service(request: Request) -> TeamService:
    """Team-access seam: the wired ``TeamService`` from ``app.state``."""
    return TEAM_SERVICE.require(request)


def require_team_access(
    team_id: uuid.UUID,
    user: RequestUser = Depends(get_request_user),
    service: TeamService = Depends(get_team_service),
) -> RequestUser:
    """Authorize a per-team route: caller must own ``team_id`` or be admin.

    Args:
        team_id: The target team, bound from the route path.
        user: The authenticated principal (always populated by the seam).
        service: The team-access seam resolving the ``Process`` by ``team_id``.

    Returns:
        The authenticated ``RequestUser`` on success.

    Raises:
        HTTPException: 404 when the team is missing OR the caller is neither
            the owner nor an admin (404-over-403, no existence leak).
    """
    process = service.get_team(team_id)
    if process is None or (process.user_id != user.user_id and _ADMIN_ROLE not in user.roles):
        owner = None if process is None else process.user_id
        logger.info(
            "team-access gate denied",
            extra={"team_id": str(team_id), "user_id": user.user_id, "owner": owner},
        )
        raise HTTPException(status_code=404, detail="Team not found")
    return user
