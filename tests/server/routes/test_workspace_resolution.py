"""Tests for the workspace-resolution seam (Story 67.1, ADR-048 Decisions 1, 5, 7).

The seam does two things and neither may be re-implemented anywhere else in
``akgentic-infra``: it guards the ``?workspace_id=`` value the caller sent, and
it enumerates the workspaces the authorized team's own cards declare — through
the single tool-side ``resolve_workspace_path``.
"""

from __future__ import annotations

import inspect
import re
import uuid
from pathlib import Path, PurePosixPath

import pytest
from akgentic.team.models import Process
from akgentic.team.ports import AgentCardNotFoundError
from akgentic.team.projection import hash_agent_card
from akgentic.tool import MetadataTool
from akgentic.tool.workspace import (
    ID_KIND,
    METADATA_KIND,
    SHARED_SCOPE,
    WorkspaceTool,
)
from fastapi import HTTPException
from starlette.datastructures import State

import akgentic.infra
from akgentic.infra.server.routes._workspace_resolution import (
    declared_workspace_paths,
    default_workspace_path,
    select_declared_path,
    stash_workspace_path,
    stashed_workspace_path,
    validate_workspace_id,
)

from ._workspace_cards import (
    CaseMetadata,
    RecordingCardStore,
    bare_card,
    exec_only_workspace,
    process_with_cards,
    tool_card,
)
from ._workspace_ids import PATH_SAFE_UNDECLARED_IDS, REJECTED_WORKSPACE_IDS

# ---------------------------------------------------------------------------
# validate_workspace_id — the tool's leaf_segment, as a traversal guard (Story 70.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", REJECTED_WORKSPACE_IDS)
def test_validate_workspace_id_rejects_non_segments(bad_value: str) -> None:
    """The request-boundary guard answers 400 for anything ``leaf_segment`` refuses.

    The detail is fixed. The tool's message reflects the caller's value and
    names internal directories, so it stays out of the response.
    """
    with pytest.raises(HTTPException) as excinfo:
        validate_workspace_id(bad_value)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "Invalid workspace_id"


@pytest.mark.parametrize(
    "good",
    [
        "notes",
        "a" * 128,
        "a.b_c-d",
        str(uuid.uuid4()),
        # ``%2F`` is a literal three-character name: nothing decodes the
        # selector again after the query layer. The literal ``/`` stays 400.
        "case_id-42%2F",
        *PATH_SAFE_UNDECLARED_IDS,
    ],
)
def test_validate_workspace_id_returns_valid_segments(good: str) -> None:
    """A single leaf comes back unchanged, whatever its length and whether it carries ``%``."""
    assert validate_workspace_id(good) == good


# ---------------------------------------------------------------------------
# declared_workspace_paths — the three layouts (AC #1, #8)
#
# Since story 71.2 the reader returns a **list of whole paths**, de-duplicated
# on the path rather than keyed on the leaf. Every spec below pins the property
# it always pinned; only the container moved.
# ---------------------------------------------------------------------------


def _declared(process: Process, store: RecordingCardStore) -> list[str]:
    """The declared paths as strings, in the order the reader returns them."""
    return [str(path) for path in declared_workspace_paths(process=process, store=store)]


def test_named_workspace_resolves_under_the_caller() -> None:
    """A named card resolves to ``<owner user_id>/_id/<workspace_id>``."""
    card = tool_card("Writer", WorkspaceTool(workspace_id="notes"))
    process = process_with_cards([card])
    store = RecordingCardStore([card])

    assert _declared(process, store) == ["alice/_id/notes"]


def test_bare_workspace_card_resolves_to_the_team_leaf() -> None:
    """A bare ``WorkspaceTool()`` declares the team's own tree as its leaf."""
    team_id = uuid.uuid4()
    card = tool_card("Writer", WorkspaceTool())
    process = process_with_cards([card], team_id=team_id)
    store = RecordingCardStore([card])

    assert _declared(process, store) == [f"alice/_team/{team_id}"]


def test_metadata_card_resolves_per_principal_under_the_meta_kind() -> None:
    """A metadata card resolves under the owner, with ``_meta`` as its kind.

    ``_meta`` says how the leaf was derived and nothing about who may reach the
    tree: a metadata tree is per-principal by default, like the other two kinds,
    and is shared only when its card declares ``workspace_sharable``.
    """
    card = tool_card("Writer", WorkspaceTool(workspace_metadata_keys=["customer_id", "case_id"]))
    process = process_with_cards([card], metadata=CaseMetadata())
    store = RecordingCardStore([card])

    leaf = "customer_id-ACME__case_id-42"
    assert _declared(process, store) == [f"alice/_meta/{leaf}"]


def test_scope_is_the_team_owner_never_the_caller() -> None:
    """The ``<scope>`` is ``process.user_id`` — the calling principal is not an input.

    The agent writes under the owner, so a caller-scoped route would send an
    admin who has already passed ``require_team_access`` to a different, empty
    directory. There is no caller argument here at all, which is what makes that
    unrepresentable rather than merely avoided.
    """
    card = tool_card("Writer", WorkspaceTool(workspace_id="notes"))
    store = RecordingCardStore([card])

    for owner in ("alice", "bob"):
        process = process_with_cards([card], user_id=owner)
        assert _declared(process, store) == [f"{owner}/_id/notes"]

    assert "user_id" not in inspect.signature(declared_workspace_paths).parameters


def test_one_workspace_id_on_two_owners_teams_is_two_trees() -> None:
    """The isolation the layout buys: one string, two owners, two paths.

    Bob declaring ``workspace_id="notes"`` on a team of his own resolves under
    *his* principal, so it cannot reach Alice's tree. Reaching hers needs a team
    she owns, which ``require_team_access`` refuses him.
    """
    card = tool_card("Writer", WorkspaceTool(workspace_id="notes"))
    store = RecordingCardStore([card])

    alice = _declared(process_with_cards([card], user_id="alice"), store)
    bob = _declared(process_with_cards([card], user_id="bob"), store)

    assert alice == ["alice/_id/notes"]
    assert bob == ["bob/_id/notes"]


# ---------------------------------------------------------------------------
# workspace_sharable is read off the card, never supplied as a literal
# ---------------------------------------------------------------------------

_CASE_LEAF = "customer_id-ACME__case_id-42"


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        (WorkspaceTool(workspace_id="notes", workspace_sharable=True), "_shared/_id/notes"),
        (
            WorkspaceTool(
                workspace_metadata_keys=["customer_id", "case_id"], workspace_sharable=True
            ),
            f"_shared/_meta/{_CASE_LEAF}",
        ),
    ],
    ids=["id", "meta"],
)
def test_a_sharable_card_resolves_under_the_shared_scope(
    tool: WorkspaceTool, expected: str
) -> None:
    """A card declaring ``workspace_sharable=True`` resolves under ``_shared/``.

    Nothing else in the server resolves a ``_shared`` path, so if this call site
    passed a literal the shared scope would be unreachable from any request — a
    refusal written for it later would be correct and never fire.
    """
    card = tool_card("Writer", tool)
    process = process_with_cards([card], metadata=CaseMetadata())
    store = RecordingCardStore([card])

    assert _declared(process, store) == [expected]


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        (WorkspaceTool(workspace_id="notes", workspace_sharable=False), "alice/_id/notes"),
        (
            WorkspaceTool(
                workspace_metadata_keys=["customer_id", "case_id"], workspace_sharable=False
            ),
            f"alice/_meta/{_CASE_LEAF}",
        ),
    ],
    ids=["id", "meta"],
)
def test_a_non_sharable_twin_resolves_under_the_owner(tool: WorkspaceTool, expected: str) -> None:
    """The same card without ``workspace_sharable`` stays under the team owner."""
    card = tool_card("Writer", tool)
    process = process_with_cards([card], metadata=CaseMetadata())
    store = RecordingCardStore([card])

    assert _declared(process, store) == [expected]


def test_sharable_is_read_per_card_not_per_team() -> None:
    """A sharable card and a non-sharable card on one team each keep their own scope."""
    shared = tool_card("Librarian", WorkspaceTool(workspace_id="library", workspace_sharable=True))
    own = tool_card("Writer", WorkspaceTool(workspace_id="notes"))
    process = process_with_cards([shared, own])
    store = RecordingCardStore([shared, own])

    assert _declared(process, store) == ["_shared/_id/library", "alice/_id/notes"]


# ---------------------------------------------------------------------------
# No declared workspace is lost to a leaf collision (Story 71.2, AC #1)
#
# A ``<leaf>`` is unique only within a scope AND a kind, so the shipped
# ``paths[path.name] = path`` keying dropped one of every colliding pair,
# silently and on the strength of card order. Both axes are pinned, because
# the two are reached by different card shapes and a fix for one need not fix
# the other.
#
# These need no environment: the platform sharing gate is read at card *bind*,
# inside the tool's ``observer()``, while this module calls the resolver
# directly with ``workspace_sharable`` as a plain argument.
# ---------------------------------------------------------------------------

_COLLIDING_LEAF = "customer_id-ACME"


def test_two_kinds_sharing_a_leaf_are_both_declared() -> None:
    """``<owner>/_id/x`` beside ``<owner>/_meta/x``: two trees, one leaf, both kept.

    The metadata card's leaf is derived from the team's own metadata and the
    named card declares the same string, so the two paths differ in their
    ``<kind>`` alone. Under the shipped leaf keying the second overwrote the
    first and the tree that lost was unreachable with nothing logged.
    """
    named = tool_card("Writer", WorkspaceTool(workspace_id=_COLLIDING_LEAF))
    keyed = tool_card("Filer", WorkspaceTool(workspace_metadata_keys=["customer_id"]))
    process = process_with_cards([named, keyed], metadata=CaseMetadata())
    store = RecordingCardStore([named, keyed])

    paths = declared_workspace_paths(process=process, store=store)

    assert [str(path) for path in paths] == [
        f"alice/{ID_KIND}/{_COLLIDING_LEAF}",
        f"alice/{METADATA_KIND}/{_COLLIDING_LEAF}",
    ]
    assert len(paths) == 2


def test_two_scopes_sharing_a_leaf_are_both_declared() -> None:
    """``<owner>/_id/notes`` beside ``_shared/_id/notes``: the likelier axis, and the worse.

    ``check_workspace_scope`` refuses every ``_shared`` path with 403, so when
    the shared entry won the collision a caller was told their **own** tree was
    a shared workspace they may not reach; when the owner entry won, a shared
    tree the gate exists to refuse was quietly served as a private one. Which a
    deployment got was decided by card order.
    """
    own = tool_card("Writer", WorkspaceTool(workspace_id="notes"))
    shared = tool_card("Librarian", WorkspaceTool(workspace_id="notes", workspace_sharable=True))
    process = process_with_cards([own, shared])
    store = RecordingCardStore([own, shared])

    paths = declared_workspace_paths(process=process, store=store)

    assert [str(path) for path in paths] == [
        f"alice/{ID_KIND}/notes",
        f"{SHARED_SCOPE}/{ID_KIND}/notes",
    ]
    assert len(paths) == 2


def test_the_same_path_declared_twice_is_one_entry() -> None:
    """De-duplication is on the whole path, so two cards naming one tree yield one entry.

    The rule that keeps a collision is the same rule that must still collapse a
    genuine duplicate — otherwise the list would grow a copy per card and the
    gate would report every leaf as ambiguous.
    """
    first = tool_card("Writer", WorkspaceTool(workspace_id="notes"))
    second = tool_card("Editor", WorkspaceTool(workspace_id="notes"))
    process = process_with_cards([first, second])
    store = RecordingCardStore([first, second])

    assert _declared(process, store) == ["alice/_id/notes"]


# ---------------------------------------------------------------------------
# select_declared_path — one leaf, three answers (Story 71.2, AC #3, #4)
# ---------------------------------------------------------------------------

_ALICE_NOTES = PurePosixPath("alice", ID_KIND, "notes")
_SHARED_NOTES = PurePosixPath(SHARED_SCOPE, ID_KIND, "notes")
_ALICE_SHELL = PurePosixPath("alice", ID_KIND, "shell")


def test_an_unambiguous_leaf_selects_its_own_path() -> None:
    """The ordinary case: exactly what the leaf-keyed lookup returned."""
    selected = select_declared_path(paths=[_ALICE_NOTES, _ALICE_SHELL], leaf="notes")
    assert selected == _ALICE_NOTES


def test_a_leaf_no_declared_path_carries_is_none() -> None:
    """``None``, which the gate turns into its membership 404 — never an exception."""
    assert select_declared_path(paths=[_ALICE_NOTES], leaf="not-declared") is None
    assert select_declared_path(paths=[], leaf="notes") is None


def test_an_ambiguous_leaf_raises_naming_every_colliding_path() -> None:
    """Refused, not picked — and the message carries the leaf and **both** paths.

    That message is what the gate logs, so it is the only thing telling an
    operator which two cards to repair. A refusal naming just the leaf would
    say a workspace is broken without saying which trees.
    """
    with pytest.raises(ValueError) as excinfo:
        select_declared_path(paths=[_ALICE_NOTES, _SHARED_NOTES, _ALICE_SHELL], leaf="notes")

    message = str(excinfo.value)
    assert "notes" in message
    assert str(_ALICE_NOTES) in message
    assert str(_SHARED_NOTES) in message
    assert str(_ALICE_SHELL) not in message, "only the colliding paths belong in the record"


def test_the_refusal_is_a_value_error_so_it_takes_the_existing_resolution_arm() -> None:
    """A ``ValueError``, deliberately — the gate's 500 arm already catches one.

    A ``LookupError`` would take the card-read arm instead, which answers a
    different 500 detail and says the cards could not be read, which is false.
    """
    with pytest.raises(ValueError) as excinfo:
        select_declared_path(paths=[_ALICE_NOTES, _SHARED_NOTES], leaf="notes")
    assert not isinstance(excinfo.value, LookupError)


# ---------------------------------------------------------------------------
# Sandboxed execution is a WorkspaceTool capability, not a second card shape
# ---------------------------------------------------------------------------


def test_exec_only_workspace_contributes_a_leaf() -> None:
    """A shell-only card is a real declaration site — omitting it would 404 a live id."""
    card = tool_card("Runner", exec_only_workspace("shell"))
    process = process_with_cards([card])
    store = RecordingCardStore([card])

    assert _declared(process, store) == ["alice/_id/shell"]


def test_exec_only_and_file_workspaces_on_one_team() -> None:
    """A shell-only card and a file card resolve side by side, through one shape."""
    workspace = tool_card("Writer", WorkspaceTool(workspace_id="notes"))
    runner = tool_card("Runner", exec_only_workspace("shell"))
    process = process_with_cards([workspace, runner])
    store = RecordingCardStore([workspace, runner])

    assert _declared(process, store) == ["alice/_id/notes", "alice/_id/shell"]


def test_exec_only_metadata_workspace_resolves_under_meta() -> None:
    """A metadata-keyed shell is reachable — the reason the standalone card went."""
    tool = exec_only_workspace(workspace_metadata_keys=["customer_id", "case_id"])
    card = tool_card("Runner", tool)
    process = process_with_cards([card], metadata=CaseMetadata())
    store = RecordingCardStore([card])

    leaf = "customer_id-ACME__case_id-42"
    assert _declared(process, store) == [f"alice/_meta/{leaf}"]


_EXEC_TOOL_REFERENCE = re.compile(r"\bExecTool\b|akgentic\.tool\.sandbox\b")


def test_no_infra_module_imports_the_retired_exec_card() -> None:
    """Nothing under ``akgentic.infra`` names the retired standalone exec card.

    ``akgentic-tool`` removed it and makes the name raise ``ImportError`` at
    import time, so one surviving reference in any server module fails the
    whole suite at collection. The guard scans the source rather than importing,
    because an import-time failure is exactly what it must catch early.
    """
    src_root = Path(akgentic.infra.__file__).parent
    offenders = [
        str(path.relative_to(src_root))
        for path in sorted(src_root.rglob("*.py"))
        if _EXEC_TOOL_REFERENCE.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_bare_base_config_card_declares_nothing() -> None:
    """A card carrying a plain ``BaseConfig`` has no ``tools`` and must not raise."""
    card = bare_card("Manager")
    process = process_with_cards([card])
    store = RecordingCardStore([card])

    assert declared_workspace_paths(process=process, store=store) == []


def test_a_non_workspace_tool_declares_nothing() -> None:
    """A team whose only card declares no workspace names no workspace."""
    card = tool_card("Manager")  # AgentConfig with an empty tools list
    process = process_with_cards([card])
    store = RecordingCardStore([card])

    assert declared_workspace_paths(process=process, store=store) == []


# ---------------------------------------------------------------------------
# One round trip, and a loud failure (AC #8)
# ---------------------------------------------------------------------------


def test_whole_card_set_is_resolved_in_one_call() -> None:
    """AC #8: one ``load_agent_cards`` call whatever the number of roles.

    A per-hash loop returns the identical result, so only this call-count
    assertion catches the N+1 the card store exists to prevent.
    """
    cards = [tool_card(f"Role{n}", WorkspaceTool(workspace_id=f"ws-{n}")) for n in range(5)]
    process = process_with_cards(cards)
    store = RecordingCardStore(cards)

    paths = declared_workspace_paths(process=process, store=store)

    assert len(paths) == 5
    assert len(store.calls) == 1
    assert len(store.calls[0]) == 5


def test_unresolvable_hash_fails_loudly() -> None:
    """A card the store cannot resolve raises — it is never dropped.

    Dropping it would silently shrink the allowed set, so a caller with a real
    workspace gets a 404 and nothing says why.
    """
    card = tool_card("Writer", WorkspaceTool(workspace_id="notes"))
    process = process_with_cards([card])
    store = RecordingCardStore([card], missing=True)

    with pytest.raises(AgentCardNotFoundError) as excinfo:
        declared_workspace_paths(process=process, store=store)
    message = str(excinfo.value)
    assert "Writer" in message
    assert hash_agent_card(card) in message


def test_agent_card_not_found_is_a_lookup_error_not_a_value_error() -> None:
    """The type distinction the corrupted-document handlers depend on."""
    assert issubclass(AgentCardNotFoundError, LookupError)
    assert not issubclass(AgentCardNotFoundError, ValueError)


# ---------------------------------------------------------------------------
# What cannot be a directory name raises rather than resolving
# ---------------------------------------------------------------------------


def test_unusable_owner_id_raises_value_error() -> None:
    """An empty owner id — reachable from a token carrying no ``sub`` — raises."""
    card = tool_card("Writer", WorkspaceTool(workspace_id="notes"))
    process = process_with_cards([card], user_id="")
    store = RecordingCardStore([card])

    with pytest.raises(ValueError, match="not usable as a workspace directory name"):
        declared_workspace_paths(process=process, store=store)


def test_metadata_card_without_metadata_raises() -> None:
    """A shared workspace never silently un-shares itself into a user path."""
    card = tool_card("Writer", WorkspaceTool(workspace_metadata_keys=["case_id"]))
    process = process_with_cards([card], metadata=None)
    store = RecordingCardStore([card])

    with pytest.raises(ValueError, match="carries no metadata"):
        declared_workspace_paths(process=process, store=store)


def test_a_declared_tool_that_is_not_a_workspace_card_is_skipped() -> None:
    """A card can carry tools that declare no workspace; they contribute nothing."""
    card = tool_card("Reader", MetadataTool())
    process = process_with_cards([card])
    store = RecordingCardStore([card])

    assert declared_workspace_paths(process=process, store=store) == []


def test_stashed_path_is_none_before_the_gate_runs() -> None:
    """No stash means the gate never authorized a path — never a licence to resolve."""

    class _Conn:
        def __init__(self) -> None:
            self.state = State()

    conn = _Conn()
    assert stashed_workspace_path(conn) is None  # type: ignore[arg-type]
    stash_workspace_path(conn, PurePosixPath("bob/_id/notes"))  # type: ignore[arg-type]
    assert str(stashed_workspace_path(conn)) == "bob/_id/notes"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# default_workspace_path — the tree an omitted selector serves (Story 70.2)
# ---------------------------------------------------------------------------


def _default_of(*tools: WorkspaceTool, owner: str = "alice") -> tuple[str, RecordingCardStore]:
    """Resolve the default path of a team whose single card declares *tools*."""
    card = tool_card("Writer", *tools)
    process = process_with_cards([card], user_id=owner)
    store = RecordingCardStore([card])
    path = default_workspace_path(process=process, store=store)  # type: ignore[arg-type]
    assert path.name == str(process.team_id)
    return str(path.parent), store


def test_no_default_card_is_the_per_principal_default() -> None:
    """A team with no default-layout card gets ``<owner>/_team``, as it always has.

    A sharable *named* card and a sharable *metadata* card are not default-layout
    cards, so neither moves it.
    """
    parent, store = _default_of(
        WorkspaceTool(workspace_id="notes", workspace_sharable=True),
        WorkspaceTool(workspace_metadata_keys=["case_id"], workspace_sharable=True),
    )
    assert parent == "alice/_team"
    assert len(store.calls) == 1


def test_a_non_sharable_default_card_is_per_principal() -> None:
    parent, _ = _default_of(WorkspaceTool())
    assert parent == "alice/_team"


def test_a_sharable_default_card_is_under_the_shared_scope() -> None:
    """The card's own flag picks the scope, as it does for the agent at bind."""
    parent, _ = _default_of(WorkspaceTool(workspace_sharable=True))
    assert parent == "_shared/_team"


def test_default_cards_that_agree_are_one_tree() -> None:
    """Two default cards with the same flag name the same tree: nothing to refuse."""
    assert _default_of(WorkspaceTool(), exec_only_workspace())[0] == "alice/_team"
    shared = _default_of(
        WorkspaceTool(workspace_sharable=True), WorkspaceTool(workspace_sharable=True)
    )
    assert shared[0] == "_shared/_team"


def test_default_cards_that_disagree_raise_value_error() -> None:
    """A ``ValueError``, so the gate's resolution arm turns it into a logged 500."""
    with pytest.raises(ValueError, match="disagree on workspace_sharable"):
        _default_of(WorkspaceTool(), WorkspaceTool(workspace_sharable=True))


def test_default_path_on_an_unresolvable_hash_is_a_lookup_error() -> None:
    """The omitted branch cannot know the scope without every card, so it raises."""
    card = tool_card("Writer", WorkspaceTool())
    process = process_with_cards([card])
    store = RecordingCardStore([card], missing=True)
    with pytest.raises(AgentCardNotFoundError):
        default_workspace_path(process=process, store=store)  # type: ignore[arg-type]
