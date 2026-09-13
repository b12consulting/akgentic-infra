"""Tier-agnostic contract for *which* of a team's trees its deletion takes with it.

Deleting a team removes the tree the team owns. Which trees those are is not a
constant, because deployments differ: one wants a closed case's named workspace
to go with the last team that used it, another shares that workspace across
every team of the principal and would lose data if it went. So the rule is a
policy object with a safe default, wired exactly as
:class:`~akgentic.infra.protocols.authz.TeamAccessPolicy` is — a tier supplies
its own without this package knowing the tier's rules.

**The default refuses everything but the team's own tree** (``adapters/shared/``,
:class:`~akgentic.infra.adapters.shared.team_tree_only_policy.TeamTreeOnlyPolicy`),
and a deployment that wires nothing gets it, because the field on
``TierServices`` carries a ``default_factory``.

**Two things no policy can do, enforced by the caller rather than documented
here.** It cannot name a tree the team never bound: it is handed candidates
built from the team's own cards and answers delete-or-keep, so it returns a
verdict and never a path. And it cannot reach outside the server's workspaces
root: the caller refuses a candidate that does not resolve strictly inside it,
whatever the policy answered.

A policy consumes the neutral :class:`WorkspaceDeletionContext` — never the team
``Process`` — for ``TeamAccessContext``'s stated reason: a tier or library policy
then never has to import ``akgentic.team.models``.
"""

from __future__ import annotations

import uuid
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class WorkspaceDeletionContext(BaseModel):
    """Neutral decision context for **one** candidate tree of the team being deleted.

    Carries the team, its owner, and the candidate path with its three segments
    already split out — a policy decides on ``kind`` and ``leaf`` far more often
    than on the joined string, and splitting here means no policy re-implements
    the layout's shape. ``path`` is the resolved three-segment path **relative
    to the workspaces root**, never an absolute one: where that root is, is the
    deployment's business and not the rule's. It round-trips through Pydantic
    (construct, ``model_dump``, re-validate) unchanged.
    """

    team_id: uuid.UUID
    owner_user_id: str
    path: PurePosixPath
    scope: str
    kind: str
    leaf: str


@runtime_checkable
class WorkspaceDeletionPolicy(Protocol):
    """Tier-agnostic rule answering *may this tree go with the team?*

    **The single member is synchronous, and that is a deliberate divergence from
    ``TeamAccessPolicy.is_allowed``.** Both of its inputs are already in hand —
    the team's own ``Process`` and a path the team itself declared — so there is
    nothing external to consult, while ``TeamService.delete_team`` is a sync
    method. An ``async`` member would push ``async`` through that service's
    public shape and every caller of it for no gain. A tier whose rule genuinely
    needs I/O should load it at wiring time and close over the result.
    """

    def may_delete(self, *, ctx: WorkspaceDeletionContext) -> bool:
        """Return ``True`` iff the candidate in ``ctx`` should be removed with the team."""
        ...
