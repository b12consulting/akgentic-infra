"""Request-user identity seam for tier-agnostic server routes (ADR-023, ADR-034)."""

from __future__ import annotations

from fastapi import Request
from pydantic import BaseModel, Field


class RequestUser(BaseModel):
    """The authenticated principal for a server request, tier-agnostic.

    Deployment tiers adapt their own auth identity into this shape.
    Community supplies the anonymous default. Routes never see a
    tier-specific identity type.
    """

    user_id: str
    email: str = ""
    roles: list[str] = Field(default_factory=list)


def get_request_user(request: Request) -> RequestUser:
    """Resolve the authenticated principal for the current request.

    Reads the ``RequestUser`` the shared ``RequireAuthMiddleware`` stashes on
    ``request.state.request_user`` (ADR-034 Decision 2b — auth runs once per
    request). When no middleware has populated the stash (the community tier,
    which mounts none), falls back to the anonymous default — never ``None``,
    never raises.
    """
    user = getattr(request.state, "request_user", None)
    if isinstance(user, RequestUser):
        return user
    return RequestUser(user_id="anonymous")
