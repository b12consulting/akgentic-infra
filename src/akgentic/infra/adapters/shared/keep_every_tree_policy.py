"""KeepEveryTreePolicy — the rule that a deletion takes no tree with it.

What ``workspace_delete_mode=soft`` wires: a
:class:`~akgentic.infra.protocols.workspace_deletion.WorkspaceDeletionPolicy`
that refuses every candidate, so deleting a team leaves all of its trees on
disk. Placed beside ``TeamTreeOnlyPolicy`` in ``adapters/shared`` because every
tier selects between the two the same way.
"""

from __future__ import annotations

from akgentic.infra.protocols.workspace_deletion import (
    WorkspaceDeletionContext,
    WorkspaceDeletionPolicy,
)


class KeepEveryTreePolicy(WorkspaceDeletionPolicy):
    """Refuse every candidate, in any scope, of any kind, for any leaf.

    Soft workspace deletion is this policy, not a branch in the deletion path:
    ``TeamService.delete_team`` still resolves the team's candidates and asks
    the policy about each, and every refusal is still logged by
    ``_remove_workspace_trees`` (``Workspace kept — … policy=KeepEveryTreePolicy``),
    so what was kept is on the record.

    Whether the team's record survives is the other half of the deletion,
    decided by ``team_delete_mode`` rather than here. Paired with a hard record
    delete, the kept trees have no record pointing at them, and nothing
    reclaims them later.

    Constructible with no arguments, like ``TeamTreeOnlyPolicy``.
    """

    def may_delete(self, *, ctx: WorkspaceDeletionContext) -> bool:
        """Refuse: no tree goes with the team."""
        return False
