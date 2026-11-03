"""The workspace deletion policy a ``DeleteMode`` selects.

Every tier turns ``ServerSettings.workspace_delete_mode`` into a
``TierServices.workspace_deletion_policy`` the same way, so the mapping lives
here once and the tiers import it from ``akgentic.infra.adapters.shared``.
"""

from __future__ import annotations

from typing import assert_never

from akgentic.infra.adapters.shared.keep_every_tree_policy import KeepEveryTreePolicy
from akgentic.infra.adapters.shared.team_tree_only_policy import TeamTreeOnlyPolicy
from akgentic.infra.protocols.workspace_deletion import WorkspaceDeletionPolicy
from akgentic.team.models import DeleteMode


def workspace_deletion_policy_for(mode: DeleteMode) -> WorkspaceDeletionPolicy:
    """Return the policy for *mode*: ``HARD`` removes the team's own tree, ``SOFT`` keeps all.

    Args:
        mode: The workspace half of a team deletion.

    Returns:
        A ``TeamTreeOnlyPolicy`` for ``HARD``, a ``KeepEveryTreePolicy`` for ``SOFT``.
    """
    match mode:
        case DeleteMode.HARD:
            return TeamTreeOnlyPolicy()
        case DeleteMode.SOFT:
            return KeepEveryTreePolicy()
        case _:
            assert_never(mode)
