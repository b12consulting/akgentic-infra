"""Tests for the per-team resource-ownership gates.

``require_team_access`` / ``require_workspace_access`` (ADR-034 §Layered authz,
ADR-035 Decision 8) are ``async`` and delegate the allow/deny rule to the wired
``TeamAccessPolicy`` while keeping the infra-owned team lookup + 404-over-403
no-existence-leak machinery. Direct unit calls pass ``policy=`` explicitly
(``Depends`` defaults are only resolved by FastAPI at request time).
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from akgentic.core.agent_card import AgentCard
from akgentic.team.models import Process
from akgentic.tool.workspace import WorkspaceTool
from fastapi import HTTPException
from starlette.datastructures import State

from akgentic.infra.adapters.shared.owner_or_admin_policy import OwnerOrAdminPolicy
from akgentic.infra.protocols.authz import TeamAccessContext, TeamAccessPolicy
from akgentic.infra.server.auth import RequestUser
from akgentic.infra.server.routes._team_access import (
    require_team_access,
    require_workspace_access,
)
from akgentic.infra.server.routes._workspace_resolution import stashed_workspace_paths

from ._workspace_cards import (
    CaseMetadata,
    RecordingCardStore,
    exec_only_workspace,
    process_with_cards,
    tool_card,
)


class _FakeProcess:
    """Minimal team-owner stand-in carrying just the ownership field.

    ``agent_cards`` is empty, which is the honest shape for a team that
    declares no workspace: since ADR-048 Decision 7 such a team's cards name
    nothing, so every ``?workspace_id=`` against it is refused.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.team_id = uuid.uuid4()
        self.agent_cards: list[Any] = []
        self.metadata = None


class _FakeTeamService:
    """Team-access seam stub returning a fixed Process (or None for missing)."""

    def __init__(self, process: _FakeProcess | Process | None) -> None:
        self._process = process

    def get_team(self, team_id: uuid.UUID) -> _FakeProcess | Process | None:
        return self._process


class _FakeRequest:
    """The one thing the gate needs of a request: a ``state`` to stash into."""

    def __init__(self) -> None:
        self.state = State()


class _FixedPolicy:
    """Fake ``TeamAccessPolicy`` returning a fixed verdict and recording calls."""

    def __init__(self, verdict: bool) -> None:
        self._verdict = verdict
        self.calls: list[TeamAccessContext] = []

    async def is_allowed(self, *, ctx: TeamAccessContext, user: RequestUser) -> bool:
        self.calls.append(ctx)
        return self._verdict


class _RaisingPolicy:
    """Fake policy that fails the test if consulted (missing-team assertion)."""

    async def is_allowed(self, *, ctx: TeamAccessContext, user: RequestUser) -> bool:
        raise AssertionError("policy must not be consulted on the pass-through path")


async def _call(
    user: RequestUser,
    owner: str | None,
    policy: TeamAccessPolicy | None = None,
    *,
    request: _FakeRequest | None = None,
) -> RequestUser:
    """Invoke require_team_access with a stubbed seam; returns the user on allow."""
    process = None if owner is None else _FakeProcess(owner)
    service = _FakeTeamService(process)
    resolved = OwnerOrAdminPolicy() if policy is None else policy
    return await require_team_access(
        request=request or _FakeRequest(),  # type: ignore[arg-type]
        team_id=uuid.uuid4(),
        user=user,
        service=service,  # type: ignore[arg-type]
        policy=resolved,
    )


async def _call_workspace(
    user: RequestUser,
    workspace_id: str | None,
    owner: str | None,
    policy: TeamAccessPolicy | None = None,
    *,
    process: Process | None = None,
    store: RecordingCardStore | None = None,
    request: _FakeRequest | None = None,
) -> RequestUser:
    """Invoke require_workspace_access with a stubbed seam.

    ``owner`` builds the card-less stand-in used by the foreign-team branch;
    ``process`` + ``store`` supply a team that really declares workspaces.
    """
    resolved_process = (
        process if process is not None else (None if owner is None else _FakeProcess(owner))
    )
    service = _FakeTeamService(resolved_process)
    resolved = OwnerOrAdminPolicy() if policy is None else policy
    return await require_workspace_access(
        request=request or _FakeRequest(),  # type: ignore[arg-type]
        team_id=uuid.uuid4(),
        workspace_id=workspace_id,
        user=user,
        service=service,  # type: ignore[arg-type]
        policy=resolved,
        store=store or RecordingCardStore([]),  # type: ignore[arg-type]
    )


def _declaring_team(
    *tools: WorkspaceTool,
    owner: str = "alice",
    metadata: CaseMetadata | None = None,
) -> tuple[Process, RecordingCardStore]:
    """A team whose single card declares *tools*, plus the store resolving it."""
    cards: list[AgentCard] = [tool_card("Writer", *tools)]
    process = process_with_cards(cards, user_id=owner, metadata=metadata)
    return process, RecordingCardStore(cards)


# --- require_team_access: default (owner-or-admin) policy ---------------------


async def test_owner_is_allowed() -> None:
    """The team owner is allowed and the principal is returned."""
    user = RequestUser(user_id="alice")
    assert await _call(user, owner="alice") is user


async def test_admin_on_non_owned_team_is_allowed() -> None:
    """An admin bypasses ownership on a team they do not own."""
    user = RequestUser(user_id="bob", roles=["admin"])
    assert await _call(user, owner="alice") is user


async def test_non_owner_non_admin_is_404() -> None:
    """A non-owner non-admin is rejected with 404 (no existence leak)."""
    user = RequestUser(user_id="mallory")
    with pytest.raises(HTTPException) as excinfo:
        await _call(user, owner="alice")
    assert excinfo.value.status_code == 404


async def test_missing_team_is_404() -> None:
    """A missing team is rejected with 404 (same shape as a non-owned team)."""
    user = RequestUser(user_id="alice")
    with pytest.raises(HTTPException) as excinfo:
        await _call(user, owner=None)
    assert excinfo.value.status_code == 404


async def test_missing_team_does_not_consult_policy() -> None:
    """AC #4: a missing team is 404 raised BEFORE the policy is awaited."""
    user = RequestUser(user_id="alice")
    with pytest.raises(HTTPException) as excinfo:
        await _call(user, owner=None, policy=_RaisingPolicy())
    assert excinfo.value.status_code == 404


# --- require_team_access: injected non-default policy (AC #8) -----------------


async def test_injected_true_policy_lets_non_owner_through() -> None:
    """AC #8: a fake returning True lets a non-owner pass (returns the user)."""
    user = RequestUser(user_id="mallory")
    policy = _FixedPolicy(True)
    assert await _call(user, owner="alice", policy=policy) is user
    assert len(policy.calls) == 1


async def test_injected_false_policy_gives_owner_404() -> None:
    """AC #8: a fake returning False makes even the owner get 404."""
    user = RequestUser(user_id="alice")
    policy = _FixedPolicy(False)
    with pytest.raises(HTTPException) as excinfo:
        await _call(user, owner="alice", policy=policy)
    assert excinfo.value.status_code == 404
    assert len(policy.calls) == 1


# --- require_workspace_access: the omitted param is the only pass-through -----


async def test_workspace_none_passes_through() -> None:
    """An omitted workspace_id passes through without a policy call.

    It selects the team's own tree, which ``require_team_access`` has already
    authorized — the one pass-through ADR-048 Decision 7 leaves standing.
    """
    user = RequestUser(user_id="mallory")
    assert (
        await _call_workspace(user, workspace_id=None, owner=None, policy=_RaisingPolicy()) is user
    )


# --- require_workspace_access: the two removed pass-throughs (AC #2) ----------
#
# Both `return user` arms are gone. A non-UUID value and a UUID naming no team
# used to be served as directory names on the caller's say-so; each is now
# checked against the workspaces the authorized team's own cards declare.


async def test_workspace_non_uuid_undeclared_is_404() -> None:
    """AC #2: the first removed pass-through — a non-UUID value is no longer served."""
    user = RequestUser(user_id="mallory")
    with pytest.raises(HTTPException) as excinfo:
        await _call_workspace(user, workspace_id="agent-segment", owner="mallory")
    assert excinfo.value.status_code == 404


async def test_workspace_uuid_naming_no_team_undeclared_is_404() -> None:
    """AC #2: the second removed pass-through — a UUID naming no team is refused."""
    user = RequestUser(user_id="mallory")
    stray = str(uuid.uuid4())
    with pytest.raises(HTTPException) as excinfo:
        await _call_workspace(user, workspace_id=stray, owner="mallory")
    assert excinfo.value.status_code == 404


async def test_undeclared_refusal_is_404_not_400_or_403() -> None:
    """AC #2: the refusal is 404-over-403 with no existence leak, and never a 400."""
    user = RequestUser(user_id="alice")
    process, store = _declaring_team(WorkspaceTool(workspace_id="notes"))
    with pytest.raises(HTTPException) as excinfo:
        await _call_workspace(
            user, workspace_id="not-declared", owner=None, process=process, store=store
        )
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "Team not found"


async def test_malformed_workspace_id_is_400_before_any_card_read() -> None:
    """AC #7: the segment guard still answers 400, and the store is never touched."""
    user = RequestUser(user_id="alice")
    process, store = _declaring_team(WorkspaceTool(workspace_id="notes"))
    with pytest.raises(HTTPException) as excinfo:
        await _call_workspace(user, workspace_id="../x", owner=None, process=process, store=store)
    assert excinfo.value.status_code == 400
    assert store.calls == []


# --- require_workspace_access: what the team declares is what passes (AC #1) ---


async def test_declared_workspace_passes_and_stashes_its_path() -> None:
    """A declared id passes, and its resolved three-segment path reaches the route."""
    user = RequestUser(user_id="alice")
    process, store = _declaring_team(WorkspaceTool(workspace_id="notes"))
    request = _FakeRequest()

    result = await _call_workspace(
        user, workspace_id="notes", owner=None, process=process, store=store, request=request
    )

    assert result is user
    stashed = stashed_workspace_paths(request)  # type: ignore[arg-type]
    assert stashed is not None
    assert str(stashed["notes"]) == "alice/_id/notes"


async def test_declared_exec_only_workspace_passes() -> None:
    """A shell-only card's id is in the allowed set — it is a real declaration."""
    user = RequestUser(user_id="alice")
    process, store = _declaring_team(exec_only_workspace("shell"))
    assert (
        await _call_workspace(user, workspace_id="shell", owner=None, process=process, store=store)
        is user
    )


async def test_declared_metadata_workspace_resolves_under_meta() -> None:
    """A metadata card's joined leaf passes, under the owner with ``_meta`` as its kind."""
    user = RequestUser(user_id="alice")
    process, store = _declaring_team(
        WorkspaceTool(workspace_metadata_keys=["customer_id", "case_id"]),
        metadata=CaseMetadata(),
    )
    request = _FakeRequest()
    leaf = "customer_id-ACME__case_id-42"

    await _call_workspace(
        user, workspace_id=leaf, owner=None, process=process, store=store, request=request
    )

    stashed = stashed_workspace_paths(request)  # type: ignore[arg-type]
    assert stashed is not None
    assert str(stashed[leaf]) == f"alice/_meta/{leaf}"


# The metadata leaf is the team's own metadata, encoded (Story 67.2). A metadata
# leaf is admitted iff it is byte-equal to the one ``process.metadata`` produces
# through the card's declared keys, in declaration order; the gate never parses
# a leaf or compares pairs, so a foreign leaf is *absent* from the map rather
# than present and refused. Each negative below is paired with the positive in
# the same fixture, positive first — a foreign leaf is absent from an empty map
# too, so an unpaired 404 says nothing about the gate.

_ACME_LEAF = "customer_id-ACME__case_id-42"
_FOREIGN_META_LEAVES = [
    "customer_id-CONTOSO__case_id-42",  # another customer, the same key set
    "customer_id-ACME",  # a key set the card does not declare
    "case_id-42__customer_id-ACME",  # the same keys, the other order
]


def _acme_case_team() -> tuple[Process, RecordingCardStore]:
    """A team on case ACME/42 whose one card declares ``["customer_id", "case_id"]``."""
    return _declaring_team(
        WorkspaceTool(workspace_metadata_keys=["customer_id", "case_id"]),
        metadata=CaseMetadata(customer_id="ACME", case_id="42"),
    )


async def test_the_admitted_metadata_leaf_is_the_only_key_in_the_stashed_map() -> None:
    """Story 67.2, the positive: ACME/42 reaches its own leaf, and the map holds only it.

    Exactly one key, so the refusals beside this can be read as *absent from
    the map* rather than present-and-refused — nothing compares pairs, because
    the leaf is derived from the metadata rather than matched against it.
    """
    user = RequestUser(user_id="alice")
    process, store = _acme_case_team()
    request = _FakeRequest()

    result = await _call_workspace(
        user, workspace_id=_ACME_LEAF, owner=None, process=process, store=store, request=request
    )

    assert result is user
    stashed = stashed_workspace_paths(request)  # type: ignore[arg-type]
    assert stashed is not None
    assert {leaf: str(path) for leaf, path in stashed.items()} == {
        _ACME_LEAF: f"alice/_meta/{_ACME_LEAF}"
    }
    assert len(store.calls) == 1


@pytest.mark.parametrize("leaf", _FOREIGN_META_LEAVES)
async def test_a_leaf_the_teams_metadata_cannot_produce_is_404_and_stashes_nothing(
    leaf: str,
) -> None:
    """Story 67.2, the refusals: another case's values, an undeclared key set, the other order.

    The positive is re-asserted first on the same declaration, so the 404 that
    follows is a membership decision on a non-empty map. The refusal carries
    the body a missing team gets, stashes nothing, and is decided from the
    team's own cards read once — not from a second lookup keyed on the leaf.
    The reversed-order leaf names a different workspace under the sequence
    model, so its refusal is real rather than a leftover of the correction
    that moved the tool to declaration order.
    """
    user = RequestUser(user_id="alice")
    admitted_process, admitted_store = _acme_case_team()
    assert (
        await _call_workspace(
            user,
            workspace_id=_ACME_LEAF,
            owner=None,
            process=admitted_process,
            store=admitted_store,
        )
        is user
    )

    process, store = _acme_case_team()
    request = _FakeRequest()
    with pytest.raises(HTTPException) as excinfo:
        await _call_workspace(
            user, workspace_id=leaf, owner=None, process=process, store=store, request=request
        )

    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "Team not found"
    assert stashed_workspace_paths(request) is None  # type: ignore[arg-type]
    assert len(store.calls) == 1


async def test_card_set_is_read_once_per_request() -> None:
    """AC #8: the gate makes exactly one ``load_agent_cards`` call."""
    user = RequestUser(user_id="alice")
    process, store = _declaring_team(
        WorkspaceTool(workspace_id="notes"), exec_only_workspace("shell")
    )
    await _call_workspace(user, workspace_id="notes", owner=None, process=process, store=store)
    assert len(store.calls) == 1


async def test_unresolvable_card_is_500_not_a_quiet_404() -> None:
    """A hash the store cannot resolve fails loudly rather than shrinking the set."""
    user = RequestUser(user_id="alice")
    cards = [tool_card("Writer", WorkspaceTool(workspace_id="notes"))]
    process = process_with_cards(cards)
    store = RecordingCardStore(cards, missing=True)
    with pytest.raises(HTTPException) as excinfo:
        await _call_workspace(user, workspace_id="notes", owner=None, process=process, store=store)
    assert excinfo.value.status_code == 500


async def test_unusable_owner_id_is_500() -> None:
    """ADR-048 Decision 4, read-path row: an unusable owner id is a server defect.

    It is the **team's** stored ``user_id`` that has to be a directory name, not
    the caller's — the caller's never reaches the resolver. A caller who cannot
    be a directory name is still perfectly able to read a team they own.
    """
    user = RequestUser(user_id="alice")
    process, store = _declaring_team(WorkspaceTool(workspace_id="notes"), owner="")
    with pytest.raises(HTTPException) as excinfo:
        await _call_workspace(user, workspace_id="notes", owner=None, process=process, store=store)
    assert excinfo.value.status_code == 500


async def test_an_odd_caller_id_does_not_affect_resolution() -> None:
    """The caller's principal is not an input to the path at all.

    An empty caller id would have been a 500 under caller-scoping; the team
    resolves under its owner, so it is simply irrelevant here.
    """
    process, store = _declaring_team(WorkspaceTool(workspace_id="notes"), owner="alice")
    request = _FakeRequest()

    await _call_workspace(
        RequestUser(user_id=""),
        workspace_id="notes",
        owner=None,
        process=process,
        store=store,
        request=request,
        policy=_FixedPolicy(True),
    )

    stashed = stashed_workspace_paths(request)  # type: ignore[arg-type]
    assert stashed is not None
    assert str(stashed["notes"]) == "alice/_id/notes"


async def test_missing_authorized_team_is_404() -> None:
    """A present workspace_id against a team that is gone is 404, never a fallback."""
    user = RequestUser(user_id="alice")
    with pytest.raises(HTTPException) as excinfo:
        await _call_workspace(user, workspace_id="notes", owner=None)
    assert excinfo.value.status_code == 404


# --- require_workspace_access: existing-team branch (AC #5, #8) ---------------


async def test_workspace_foreign_team_default_policy_is_404() -> None:
    """A foreign team's workspace_id is 404 under the default policy."""
    user = RequestUser(user_id="mallory")
    with pytest.raises(HTTPException) as excinfo:
        await _call_workspace(user, workspace_id=str(uuid.uuid4()), owner="alice")
    assert excinfo.value.status_code == 404


async def test_workspace_owner_passing_the_policy_is_still_undeclared_404() -> None:
    """Passing the foreign-team branch is necessary, not sufficient (ADR-048 D7).

    The owner clears the policy, then meets the declared-workspace check — and
    a team that declares no workspace names none, whoever is asking.
    """
    user = RequestUser(user_id="alice")
    with pytest.raises(HTTPException) as excinfo:
        await _call_workspace(user, workspace_id=str(uuid.uuid4()), owner="alice")
    assert excinfo.value.status_code == 404


async def test_workspace_foreign_team_denied_is_404() -> None:
    """AC #8: an injected fake returning False on a foreign team → 404."""
    user = RequestUser(user_id="mallory")
    policy = _FixedPolicy(False)
    with pytest.raises(HTTPException) as excinfo:
        await _call_workspace(user, workspace_id=str(uuid.uuid4()), owner="alice", policy=policy)
    assert excinfo.value.status_code == 404
    assert len(policy.calls) == 1


async def test_workspace_foreign_team_allowed_still_meets_the_declared_check() -> None:
    """AC #8: an injected fake returning True no longer short-circuits to a pass."""
    user = RequestUser(user_id="mallory")
    policy = _FixedPolicy(True)
    with pytest.raises(HTTPException) as excinfo:
        await _call_workspace(user, workspace_id=str(uuid.uuid4()), owner="alice", policy=policy)
    assert excinfo.value.status_code == 404
    assert len(policy.calls) == 1
