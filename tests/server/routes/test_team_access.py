"""Tests for the per-team resource-ownership gates.

``require_team_access`` / ``require_workspace_access`` (ADR-034 §Layered authz,
ADR-035 Decision 8) are ``async`` and delegate the allow/deny rule to the wired
``TeamAccessPolicy`` while keeping the infra-owned team lookup + 404-over-403
no-existence-leak machinery. Direct unit calls pass ``policy=`` explicitly
(``Depends`` defaults are only resolved by FastAPI at request time).
"""

from __future__ import annotations

import uuid
from pathlib import PurePosixPath
from typing import Any

import pytest
from akgentic.core.agent_card import AgentCard
from akgentic.team.models import Process
from akgentic.tool.workspace import SHARED_SCOPE, WorkspaceTool
from fastapi import HTTPException
from starlette.datastructures import State

from akgentic.infra.adapters.shared.owner_or_admin_policy import OwnerOrAdminPolicy
from akgentic.infra.errors import SharedWorkspaceRefusedError
from akgentic.infra.protocols.authz import TeamAccessContext, TeamAccessPolicy
from akgentic.infra.server.auth import RequestUser
from akgentic.infra.server.routes._team_access import (
    check_workspace_scope,
    require_team_access,
    require_workspace_access,
)
from akgentic.infra.server.routes._workspace_resolution import (
    declared_workspace_paths,
    stashed_workspace_path,
)

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


# --- require_workspace_access: the omitted param is no longer a pass-through ---
#
# Story 70.2 reverses this by decision. The omitted branch used to return before
# anything was resolved, and the route resolved the team's own tree with no check
# of its own. The gate is now the one convergence point: it selects and checks
# the path on both branches.


async def test_an_omitted_workspace_id_is_checked_like_a_named_one() -> None:
    """The gate itself refuses a stranger the team's own tree, with no team gate in front.

    The team gate is not called here, so a gate that still passed the omitted
    branch through would return the user. The scope check reads ``alice`` off
    the selected path and the real policy refuses ``mallory``.
    """
    with pytest.raises(HTTPException) as excinfo:
        await _call_workspace(RequestUser(user_id="mallory"), workspace_id=None, owner="alice")
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "Team not found"


async def test_an_omitted_workspace_id_stashes_the_owners_team_tree() -> None:
    """With no default-layout card, the omitted branch selects the per-principal default.

    The policy is asked about that path's scope, with the authorized team.
    """
    process, store = _declaring_team(WorkspaceTool(workspace_id="notes"))
    policy = _FixedPolicy(True)
    request = _FakeRequest()

    await _call_workspace(
        RequestUser(user_id="alice"),
        workspace_id=None,
        owner=None,
        policy=policy,
        process=process,
        store=store,
        request=request,
    )

    stashed = stashed_workspace_path(request)  # type: ignore[arg-type]
    assert str(stashed) == f"alice/_team/{process.team_id}"
    assert [ctx.owner_user_id for ctx in policy.calls] == ["alice"]
    assert len(store.calls) == 1


async def test_an_omitted_workspace_id_on_a_sharable_default_card_is_refused() -> None:
    """The omitted branch reads the default card's flag, so its shared tree is refused."""
    process, store = _declaring_team(WorkspaceTool(workspace_sharable=True))
    request = _FakeRequest()

    with pytest.raises(SharedWorkspaceRefusedError):
        await _call_workspace(
            RequestUser(user_id="alice"),
            workspace_id=None,
            owner=None,
            policy=_RaisingPolicy(),
            process=process,
            store=store,
            request=request,
        )
    assert stashed_workspace_path(request) is None  # type: ignore[arg-type]


async def test_default_cards_that_disagree_on_sharing_are_500() -> None:
    """Two default-layout cards with different flags name no single tree."""
    process, store = _declaring_team(WorkspaceTool(), WorkspaceTool(workspace_sharable=True))
    with pytest.raises(HTTPException) as excinfo:
        await _call_workspace(
            RequestUser(user_id="alice"),
            workspace_id=None,
            owner=None,
            process=process,
            store=store,
        )
    assert excinfo.value.status_code == 500


async def test_a_named_shared_workspace_is_refused_to_an_admin() -> None:
    """The named branch reaches the same check, and ``_shared`` has no admin exception."""
    process, store = _declaring_team(WorkspaceTool(workspace_id="notes", workspace_sharable=True))
    request = _FakeRequest()

    with pytest.raises(SharedWorkspaceRefusedError):
        await _call_workspace(
            RequestUser(user_id="root", roles=["admin"]),
            workspace_id="notes",
            owner=None,
            process=process,
            store=store,
            request=request,
        )
    assert stashed_workspace_path(request) is None  # type: ignore[arg-type]


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
    stashed = stashed_workspace_path(request)  # type: ignore[arg-type]
    assert str(stashed) == "alice/_id/notes"


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

    stashed = stashed_workspace_path(request)  # type: ignore[arg-type]
    assert str(stashed) == f"alice/_meta/{leaf}"


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


async def test_the_admitted_metadata_leaf_is_the_only_leaf_the_team_declares() -> None:
    """Story 67.2, the positive: ACME/42 reaches its own leaf, and the map holds only it.

    Exactly one key, so the refusals beside this can be read as *absent from
    the map* rather than present-and-refused — nothing compares pairs, because
    the leaf is derived from the metadata rather than matched against it.

    Since Story 70.2 the gate stashes the one path it authorized rather than the
    map, so the map is read from the same resolution function, on a fresh store
    so the gate's one-read count stays its own.
    """
    user = RequestUser(user_id="alice")
    process, store = _acme_case_team()
    request = _FakeRequest()

    result = await _call_workspace(
        user, workspace_id=_ACME_LEAF, owner=None, process=process, store=store, request=request
    )

    assert result is user
    assert str(stashed_workspace_path(request)) == f"alice/_meta/{_ACME_LEAF}"  # type: ignore[arg-type]
    _, fresh_store = _acme_case_team()
    declared = declared_workspace_paths(process=process, store=fresh_store)  # type: ignore[arg-type]
    assert {leaf: str(path) for leaf, path in declared.items()} == {
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
    assert stashed_workspace_path(request) is None  # type: ignore[arg-type]
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

    assert str(stashed_workspace_path(request)) == "alice/_id/notes"  # type: ignore[arg-type]


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


# --- check_workspace_scope: the scope segment is the whole input (Story 70.2) --
#
# The check alone, with no HTTP and the real ``OwnerOrAdminPolicy``. A user
# scope is put to the policy with the scope as owner; ``_shared`` is refused for
# every caller without consulting the policy at all.

_OWNER = RequestUser(user_id="alice")
_ADMIN = RequestUser(user_id="root", roles=["admin"])
_STRANGER = RequestUser(user_id="mallory")
_SHARED_CODE = "shared_workspace_entitlement_undecided"


async def _check(
    path: str,
    user: RequestUser,
    policy: TeamAccessPolicy | None = None,
    team_id: uuid.UUID | None = None,
) -> None:
    await check_workspace_scope(
        PurePosixPath(path),
        team_id=team_id or uuid.uuid4(),
        user=user,
        policy=OwnerOrAdminPolicy() if policy is None else policy,
    )


@pytest.mark.parametrize("path", ["alice/_team/t", "alice/_id/notes", "alice/_meta/case_id-42"])
@pytest.mark.parametrize("user", [_OWNER, _ADMIN], ids=["owner", "admin"])
async def test_a_user_scope_serves_its_principal_and_an_admin(path: str, user: RequestUser) -> None:
    """The owner of the scope is allowed, and so is an admin, through the policy."""
    await _check(path, user)


@pytest.mark.parametrize("path", ["alice/_team/t", "alice/_id/notes", "alice/_meta/case_id-42"])
async def test_a_user_scope_refuses_a_stranger_with_404(path: str) -> None:
    """Another principal is refused with the gates' 404, never a 403."""
    with pytest.raises(HTTPException) as excinfo:
        await _check(path, _STRANGER)
    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "Team not found"


async def test_the_policy_is_asked_about_the_scope_and_the_authorized_team() -> None:
    """The context carries the scope segment as owner and the authorized team, nothing else."""
    policy = _FixedPolicy(True)
    team_id = uuid.uuid4()
    await _check("mallory/_id/alice", _OWNER, policy=policy, team_id=team_id)
    assert policy.calls == [TeamAccessContext(team_id=team_id, owner_user_id="mallory")]


async def test_only_the_scope_is_read_never_the_leaf() -> None:
    """Pairs that differ only in where the caller's id sits: the scope decides, the leaf never."""
    await _check("alice/_id/mallory", _OWNER)
    with pytest.raises(HTTPException) as excinfo:
        await _check("mallory/_id/alice", _OWNER)
    assert excinfo.value.status_code == 404


@pytest.mark.parametrize("user", [_OWNER, _ADMIN, _STRANGER], ids=["owner", "admin", "stranger"])
@pytest.mark.parametrize(
    "path",
    [
        f"{SHARED_SCOPE}/_team/t",
        f"{SHARED_SCOPE}/_id/notes",
        f"{SHARED_SCOPE}/_meta/case_id-42",
        f"{SHARED_SCOPE}/_id/alice",
        f"{SHARED_SCOPE.upper()}/_id/notes",
    ],
)
async def test_a_shared_scope_is_refused_to_every_caller_without_asking_the_policy(
    path: str, user: RequestUser
) -> None:
    """``_shared`` is 403 with the entitlement code, for every caller and every kind.

    ``_RaisingPolicy`` fails the spec if consulted, so an arm that fell through
    to the policy cannot pass by the policy happening to refuse. The leaf equal
    to the caller's id and the upper-cased scope are refused all the same.
    """
    with pytest.raises(SharedWorkspaceRefusedError) as excinfo:
        await _check(path, user, policy=_RaisingPolicy())
    assert excinfo.value.status_code == 403
    assert excinfo.value.code == _SHARED_CODE
