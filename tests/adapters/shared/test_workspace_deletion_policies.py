"""The keep-every-tree policy, and the selector that turns a ``DeleteMode`` into a policy."""

from __future__ import annotations

import uuid
from pathlib import PurePosixPath

import pytest
from akgentic.team.models import DeleteMode
from akgentic.tool.workspace import ID_KIND, METADATA_KIND, SHARED_SCOPE, TEAM_KIND

from akgentic.infra.adapters.shared.keep_every_tree_policy import KeepEveryTreePolicy
from akgentic.infra.adapters.shared.team_tree_only_policy import TeamTreeOnlyPolicy
from akgentic.infra.adapters.shared.workspace_deletion_mode import workspace_deletion_policy_for
from akgentic.infra.protocols.workspace_deletion import (
    WorkspaceDeletionContext,
    WorkspaceDeletionPolicy,
)

TEAM_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
OWNER = "alice"


def _ctx(scope: str, kind: str, leaf: str) -> WorkspaceDeletionContext:
    return WorkspaceDeletionContext(
        team_id=TEAM_ID,
        owner_user_id=OWNER,
        path=PurePosixPath(scope) / kind / leaf,
        scope=scope,
        kind=kind,
        leaf=leaf,
    )


_EVERY_KIND = [
    pytest.param(TEAM_KIND, str(TEAM_ID), id="own-team-tree"),
    pytest.param(TEAM_KIND, str(uuid.uuid4()), id="foreign-team-tree"),
    pytest.param(ID_KIND, "case-42", id="named-tree"),
    pytest.param(METADATA_KIND, "acme", id="metadata-tree"),
]


class TestKeepEveryTreePolicy:
    """``workspace_delete_mode=soft``: no tree goes with the team."""

    @pytest.mark.parametrize("scope", [OWNER, SHARED_SCOPE])
    @pytest.mark.parametrize(("kind", "leaf"), _EVERY_KIND)
    def test_refuses_every_candidate(self, scope: str, kind: str, leaf: str) -> None:
        assert KeepEveryTreePolicy().may_delete(ctx=_ctx(scope, kind, leaf)) is False

    def test_is_a_workspace_deletion_policy(self) -> None:
        assert isinstance(KeepEveryTreePolicy(), WorkspaceDeletionPolicy)

    def test_refuses_the_tree_the_default_policy_approves(self) -> None:
        """The one candidate the two policies disagree on: the team's own tree."""
        ctx = _ctx(OWNER, TEAM_KIND, str(TEAM_ID))
        assert TeamTreeOnlyPolicy().may_delete(ctx=ctx) is True
        assert KeepEveryTreePolicy().may_delete(ctx=ctx) is False


class TestWorkspaceDeletionPolicyFor:
    """The selector every tier calls with ``settings.workspace_delete_mode``."""

    def test_hard_selects_the_team_tree_only_policy(self) -> None:
        assert type(workspace_deletion_policy_for(DeleteMode.HARD)) is TeamTreeOnlyPolicy

    def test_soft_selects_the_keep_every_tree_policy(self) -> None:
        assert type(workspace_deletion_policy_for(DeleteMode.SOFT)) is KeepEveryTreePolicy


def test_the_three_names_are_public_from_the_shared_package() -> None:
    from akgentic.infra.adapters import shared

    assert shared.KeepEveryTreePolicy is KeepEveryTreePolicy
    assert shared.TeamTreeOnlyPolicy is TeamTreeOnlyPolicy
    assert shared.workspace_deletion_policy_for is workspace_deletion_policy_for
    assert {"KeepEveryTreePolicy", "TeamTreeOnlyPolicy", "workspace_deletion_policy_for"} <= set(
        shared.__all__
    )
