"""Tier-agnostic per-team authorization contract (ADR-035 Decision 8).

Authorization (may this principal access this team?) is kept separate from
authentication (who is this principal?, ``protocols/auth.py`` ``AuthStrategy``).
:class:`TeamAccessPolicy` is a **sibling** of ``AuthStrategy``, not a member of
it. Infra owns the team lookup and the load-bearing 404-over-403
no-existence-leak machinery; only the allow/deny *rule* is pluggable. A policy
consumes the neutral :class:`TeamAccessContext` — never the team ``Process`` —
so a tier or library policy never has to import ``akgentic.team.models``.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from akgentic.infra.server.auth import RequestUser


class TeamListFilter(BaseModel):
    """One store query used to build the set a policy allows listing.

    Fields inside one filter AND-combine. Separate filters returned by the
    policy OR-combine. An empty list denies all; one empty filter is
    unrestricted.
    """

    user_id: str | None = None
    metadata: dict[str, list[str]] | None = None

    @model_validator(mode="after")
    def _reject_ambiguous_empty_constraints(self) -> TeamListFilter:
        if self.user_id == "":
            raise ValueError("TeamListFilter.user_id must not be empty")
        if self.metadata is not None and (
            not self.metadata
            or any(
                not key or not values or any(not value for value in values)
                for key, values in self.metadata.items()
            )
        ):
            raise ValueError(
                "TeamListFilter.metadata must contain non-empty keys and value lists"
            )
        return self


class TeamAccessContext(BaseModel):
    """Neutral team-access decision context handed to a :class:`TeamAccessPolicy`.

    Carries the target ``team_id``, owner, and canonical metadata index entries
    so a tier or library policy can decide from persisted team facts without
    importing the team ``Process``. It round-trips through Pydantic (construct,
    ``model_dump``, re-validate) unchanged.
    """

    team_id: uuid.UUID
    owner_user_id: str
    metadata_indexes: list[str] = Field(default_factory=list)


@runtime_checkable
class TeamAccessPolicy(Protocol):
    """Tier-agnostic per-team authorization contract (ADR-035 Decision 8).

    The single member is ``async`` so a tier policy may consult an external
    membership / RBAC store. The community default
    (``adapters/shared/owner_or_admin_policy.py``) reproduces the historical
    owner-or-admin rule; a tier substitutes its own without re-implementing the
    infra-owned team lookup or the 404-over-403 gate.
    """

    async def list_filters(self, *, user: RequestUser) -> list[TeamListFilter]:
        """Return store-query clauses whose results OR-combine for listing."""
        ...

    async def can_create(
        self,
        *,
        metadata_indexes: list[str],
        user: RequestUser,
    ) -> bool:
        """Return whether ``user`` may create a team with the validated metadata."""
        ...

    async def is_allowed(self, *, ctx: TeamAccessContext, user: RequestUser) -> bool:
        """Return ``True`` iff ``user`` may access the team described by ``ctx``."""
        ...
