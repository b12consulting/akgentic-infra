"""The pre-dispatch workspace resolver (Story 68.2, AC #1, #2, #3).

``declared_workspaces`` works from the four inputs ``TeamService.create_team``
holds before any worker is contacted — the card, the owner's ``user_id``, the
optional ``team_id`` and the validated metadata — and nothing else. The
load-bearing guard is the agreement spec: two resolvers over one rule that
disagree is the whole failure mode of this story, and the profile-only card is
what makes that spec falsifiable.
"""

from __future__ import annotations

import inspect
import uuid
from pathlib import PurePosixPath

import pytest
from akgentic.team import derive_team_projection
from akgentic.team.models import TeamCard, TeamCardMember
from akgentic.tool.workspace import METADATA_SCOPE, WorkspaceTool

from akgentic.infra.errors import ServerError, WorkspaceDeclarationError
from akgentic.infra.protocols.placement import DeclaredWorkspaces
from akgentic.infra.server.routes._workspace_resolution import declared_workspace_paths
from akgentic.infra.server.services._workspace_paths import (
    declared_layouts,
    declared_workspaces,
)
from tests.server.routes._workspace_cards import (
    CaseMetadata,
    RecordingCardStore,
    bare_card,
    process_with_cards,
    tool_card,
)

_META_LEAF = "customer_id-ACME__case_id-42"


def _three_layout_team(*, with_profile: bool = False) -> TeamCard:
    """A team whose projection carries a named, a metadata and a default card.

    With ``with_profile`` a fourth workspace-declaring card is reachable **only**
    through ``agent_profiles`` — the card ``TeamCard.agent_cards`` omits and the
    projection includes.
    """
    return TeamCard(
        name="Three layouts",
        entry_point=TeamCardMember(card=tool_card("Human", WorkspaceTool(workspace_id="notes"))),
        members=[
            TeamCardMember(
                card=tool_card(
                    "Analyst", WorkspaceTool(workspace_metadata_keys=["customer_id", "case_id"])
                )
            ),
            TeamCardMember(card=tool_card("Writer", WorkspaceTool())),
            TeamCardMember(card=bare_card("Manager")),
        ],
        agent_profiles=[tool_card("Reviewer", WorkspaceTool(workspace_id="drafts"))]
        if with_profile
        else [],
    )


# ---------------------------------------------------------------------------
# AC #1 — the three layouts from the request's inputs, no Process, no store
# ---------------------------------------------------------------------------


def test_the_three_layouts_resolve_from_the_requests_inputs_with_a_team_id() -> None:
    """AC #1: named → ``shared``, metadata → ``shared``, default → ``own``."""
    team_id = uuid.uuid4()

    value = declared_workspaces(
        _three_layout_team(), user_id="alice", team_id=team_id, metadata=CaseMetadata()
    )

    assert value == DeclaredWorkspaces(
        shared={PurePosixPath("alice/notes"), PurePosixPath(METADATA_SCOPE) / _META_LEAF},
        own=PurePosixPath("alice") / str(team_id),
    )


def test_without_a_team_id_the_default_card_resolves_into_nothing() -> None:
    """AC #1, second half: ``shared`` is identical and ``own`` is ``None``.

    There is no id to build ``<user_id>/<team_id>`` from before ``TeamManager``
    mints one, and by rule 1 nothing routes on it. This half is also what
    catches the resolver ever starting to read ``team_id`` for the named and
    metadata layouts: they are resolved here with no id at all.
    """
    with_id = declared_workspaces(
        _three_layout_team(), user_id="alice", team_id=uuid.uuid4(), metadata=CaseMetadata()
    )
    without_id = declared_workspaces(
        _three_layout_team(), user_id="alice", team_id=None, metadata=CaseMetadata()
    )

    assert without_id.shared == with_id.shared
    assert without_id.own is None
    assert with_id.own is not None


def test_the_resolver_takes_no_process_and_no_store() -> None:
    """AC #1: pinned by signature, the way the route-side resolver's scope is.

    At dispatch there is no ``Process`` — choosing a worker before placement
    builds one is the point — so the shape makes that unrepresentable rather
    than merely avoided.
    """
    params = inspect.signature(declared_workspaces).parameters
    assert "process" not in params
    assert "store" not in params
    assert "event_store" not in params
    assert set(params) == {"team_card", "user_id", "team_id", "metadata"}
    # The owner is a keyword, named as the owner: pre-dispatch the caller and
    # the owner coincide, but the route-side argument rests on the distinction.
    assert params["user_id"].kind is inspect.Parameter.KEYWORD_ONLY


def test_a_team_declaring_no_workspace_resolves_to_the_empty_value() -> None:
    """The seeded catalog shape: plain ``BaseConfig`` members declare nothing."""
    team = TeamCard(
        name="Bare",
        entry_point=TeamCardMember(card=bare_card("Human")),
        members=[TeamCardMember(card=bare_card("Manager"))],
    )
    value = declared_workspaces(team, user_id="alice", team_id=None, metadata=None)
    assert value == DeclaredWorkspaces()
    assert value.routing_key() is None


# ---------------------------------------------------------------------------
# AC #2 — the pre-dispatch and route-side resolvers agree on the same team
# ---------------------------------------------------------------------------


def test_the_two_resolvers_agree_on_one_team_including_a_profile_only_card() -> None:
    """AC #2, load-bearing: one rule, two resolvers, one answer.

    The ``Process`` is built the way the route sees it — over the projection's
    cards, which are exactly what ``TeamManager.create_team`` persists — with the
    same owner, the same id and the same metadata. The profile-only ``drafts``
    card is the falsifier: ``TeamCard.agent_cards`` omits ``agent_profiles``, so
    a pre-dispatch resolver written over it drops a real tree and this goes red.
    """
    team = _three_layout_team(with_profile=True)
    team_id = uuid.uuid4()
    metadata = CaseMetadata()
    cards = derive_team_projection(team).cards
    process = process_with_cards(cards, team_id=team_id, user_id="alice", metadata=metadata)
    store = RecordingCardStore(cards)

    route = declared_workspace_paths(process=process, store=store)
    pre = declared_workspaces(team, user_id="alice", team_id=team_id, metadata=metadata)

    assert pre.own is not None
    assert pre.shared | {pre.own} == set(route.values())
    # The falsifier is really in both answers, not absent from both.
    assert PurePosixPath("alice/drafts") in pre.shared
    assert PurePosixPath("alice/drafts") in route.values()


def test_the_two_resolvers_agree_when_the_id_is_minted_later() -> None:
    """AC #2, ``team_id=None`` pre-dispatch: everything but the own tree agrees."""
    team = _three_layout_team(with_profile=True)
    team_id = uuid.uuid4()
    metadata = CaseMetadata()
    cards = derive_team_projection(team).cards
    process = process_with_cards(cards, team_id=team_id, user_id="alice", metadata=metadata)
    store = RecordingCardStore(cards)

    route = declared_workspace_paths(process=process, store=store)
    pre = declared_workspaces(team, user_id="alice", team_id=None, metadata=metadata)

    own_leaf = str(team_id)
    assert pre.shared == {path for path in route.values() if path.name != own_leaf}
    assert pre.own is None
    # The route-side map really holds the own tree the pre-dispatch value lacks.
    assert own_leaf in route


def test_declared_layouts_reads_one_card_the_way_the_route_side_loop_did() -> None:
    """The shared card-shape read: the ``AgentConfig`` skip and the type check."""
    assert list(declared_layouts(bare_card("Manager"))) == []
    assert list(declared_layouts(tool_card("Manager"))) == []
    card = tool_card(
        "Writer",
        WorkspaceTool(workspace_id="notes"),
        WorkspaceTool(workspace_metadata_keys=["case_id"]),
        WorkspaceTool(),
    )
    assert list(declared_layouts(card)) == [("notes", []), (None, ["case_id"]), (None, [])]


# ---------------------------------------------------------------------------
# AC #3 — two cards, one leaf, two scopes, both carried
# ---------------------------------------------------------------------------


def test_two_cards_resolving_to_one_leaf_on_two_scopes_are_both_carried() -> None:
    """AC #3: a set of paths cannot collapse two scopes into one leaf.

    Backlog row 30 is the route-side counterpart: ``declared_workspace_paths``
    keys by ``path.name`` and keeps whichever card the store returned last. On
    this seam that would let a user-named path shadow a ``_meta/`` path and
    silently break rule 2, so the value is a set and this story deliberately
    does not touch the route-side dict.

    The owner is ``Alice`` so that the user-named path sorts **before** the
    ``_meta/`` one (``A`` is 0x41, ``_`` is 0x5F): the key assertion then really
    tests precedence, where with a lowercase owner a "smallest of all" router
    would return the ``_meta/`` tree by accident.
    """
    team = TeamCard(
        name="One leaf, two scopes",
        entry_point=TeamCardMember(
            card=tool_card("Human", WorkspaceTool(workspace_id="case_id-42"))
        ),
        members=[
            TeamCardMember(
                card=tool_card("Analyst", WorkspaceTool(workspace_metadata_keys=["case_id"]))
            )
        ],
    )

    value = declared_workspaces(
        team, user_id="Alice", team_id=None, metadata=CaseMetadata(case_id="42")
    )

    assert value.shared == {
        PurePosixPath("Alice/case_id-42"),
        PurePosixPath(METADATA_SCOPE) / "case_id-42",
    }
    assert value.routing_key() == PurePosixPath(METADATA_SCOPE) / "case_id-42"


# ---------------------------------------------------------------------------
# The resolver's ValueError becomes a 422, here and only here
# ---------------------------------------------------------------------------


def test_an_unsatisfiable_metadata_card_is_a_typed_422_carrying_the_resolvers_message() -> None:
    """A card the request's metadata cannot satisfy is refused before placement.

    Today the same failure raises inside team building, after actors started,
    and surfaces as a 503 with ``Retry-After`` — a retry that cannot succeed.
    """
    team = TeamCard(
        name="Metadata card, no metadata",
        entry_point=TeamCardMember(
            card=tool_card("Human", WorkspaceTool(workspace_metadata_keys=["case_id"]))
        ),
    )
    with pytest.raises(WorkspaceDeclarationError) as excinfo:
        declared_workspaces(team, user_id="alice", team_id=None, metadata=None)
    err = excinfo.value
    assert isinstance(err, ServerError)
    assert err.status_code == 422
    assert err.code == "workspace_unresolvable"
    assert "carries no metadata" in err.detail
    assert "case_id" in err.detail
    assert isinstance(err.__cause__, ValueError)


def test_an_unusable_owner_id_is_the_same_422() -> None:
    """Every resolver ``ValueError`` maps to one status pre-dispatch (Open Question #1)."""
    team = TeamCard(
        name="Named card, empty owner",
        entry_point=TeamCardMember(card=tool_card("Human", WorkspaceTool(workspace_id="notes"))),
    )
    with pytest.raises(WorkspaceDeclarationError, match="not usable as a workspace directory"):
        declared_workspaces(team, user_id="", team_id=None, metadata=None)


def test_the_route_side_resolver_still_raises_a_bare_value_error() -> None:
    """AC #10: the 422 wrap is pre-dispatch only; the route-side contract is unchanged."""
    card = tool_card("Writer", WorkspaceTool(workspace_metadata_keys=["case_id"]))
    process = process_with_cards([card], metadata=None)
    store = RecordingCardStore([card])

    with pytest.raises(ValueError, match="carries no metadata") as excinfo:
        declared_workspace_paths(process=process, store=store)
    assert not isinstance(excinfo.value, WorkspaceDeclarationError)
