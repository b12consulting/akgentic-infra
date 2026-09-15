"""TeamTreeOnlyPolicy — the default rule for which trees a deletion takes.

The base-bundle default
:class:`~akgentic.infra.protocols.workspace_deletion.WorkspaceDeletionPolicy`
shared by all deployment tiers via ``TierServices.workspace_deletion_policy``'s
``default_factory``. Placed in ``adapters/shared`` (not ``adapters/community``)
because it is the base default for *every* tier, not community-specific —
mirroring ``OwnerOrAdminPolicy`` beside it.
"""

from __future__ import annotations

from akgentic.infra.protocols.workspace_deletion import WorkspaceDeletionContext
from akgentic.tool.workspace import TEAM_KIND


class TeamTreeOnlyPolicy:
    """Approve the team's own tree, in either scope, and nothing else.

    **The rule is the kind and the leaf, not the sharing.** A candidate is
    approved iff its ``<kind>`` is ``_team`` *and* its ``<leaf>`` is the deleted
    team's own id. Both ``<owner>/_team/<team_id>`` and
    ``_shared/_team/<team_id>`` qualify, and that is the point: a ``_team`` leaf
    is a team id the resolver derived from the **binding** team, so no other
    team's card can produce it. Once the team is gone nothing can address that
    directory again — leaving it is not sharing, it is an orphan holding that
    team's files and, in its ``.index`` sibling, the extracted text of its
    documents.

    ``_id`` and ``_meta`` trees are refused in **both** scopes: a named tree is
    reachable by every team of that principal, and a metadata tree by every team
    carrying those values, so the team being deleted is not the only party that
    can address them. Deciding *when* such a tree is safe to take needs a
    reference count this package does not have; a deployment that knows the
    answer for its own case wires its own policy.

    Stating the rule this way also means a later change that makes deletion
    resolve a named or metadata tree is **refused** rather than silently wiping
    a workspace other teams still use. ``workspace_sharable`` then only decides
    *where* the team's own tree is, which is all it should decide.

    Constructible with no arguments so it can serve as
    ``TierServices.workspace_deletion_policy``'s ``default_factory``.
    """

    def may_delete(self, *, ctx: WorkspaceDeletionContext) -> bool:
        """Approve iff the candidate is the deleted team's own ``_team`` tree."""
        return ctx.kind == TEAM_KIND and ctx.leaf == str(ctx.team_id)
