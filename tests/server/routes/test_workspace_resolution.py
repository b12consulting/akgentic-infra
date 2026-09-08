"""Tests for the workspace-resolution seam (Story 67.1, ADR-048 Decisions 1, 5, 7).

The seam does two things and neither may be re-implemented anywhere else in
``akgentic-infra``: it guards the ``?workspace_id=`` value the caller sent, and
it enumerates the workspaces the authorized team's own cards declare — through
the single tool-side ``resolve_workspace_path``.
"""

from __future__ import annotations

import inspect
import uuid
from pathlib import PurePosixPath

import pytest
from akgentic.team.ports import AgentCardNotFoundError
from akgentic.team.projection import hash_agent_card
from akgentic.tool import MetadataTool
from akgentic.tool.sandbox import ExecTool
from akgentic.tool.workspace import WorkspaceTool
from fastapi import HTTPException
from starlette.datastructures import State

from akgentic.infra.server.routes._workspace_resolution import (
    declared_workspace_paths,
    stash_workspace_paths,
    stashed_workspace_paths,
    validate_workspace_id,
)

from ._workspace_cards import (
    CaseMetadata,
    RecordingCardStore,
    bare_card,
    process_with_cards,
    tool_card,
)

# ---------------------------------------------------------------------------
# validate_workspace_id — unchanged by ADR-048 (AC #7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    ["../x", "a/b", "a\\b", "/abs", "..", ".", "", "a" * 129, "case_id-42%2F"],
)
def test_validate_workspace_id_rejects_non_segments(bad_value: str) -> None:
    """AC #7: the request-boundary guard still answers 400 for a non-segment."""
    with pytest.raises(HTTPException) as excinfo:
        validate_workspace_id(bad_value)
    assert excinfo.value.status_code == 400


@pytest.mark.parametrize("good", ["notes", "a" * 128, "a.b_c-d", str(uuid.uuid4())])
def test_validate_workspace_id_returns_valid_segments(good: str) -> None:
    """A single safe segment comes back unchanged."""
    assert validate_workspace_id(good) == good


# ---------------------------------------------------------------------------
# declared_workspace_paths — the three layouts (AC #1, #8)
# ---------------------------------------------------------------------------


def test_named_workspace_resolves_under_the_caller() -> None:
    """A named card resolves to ``<caller user_id>/<workspace_id>``."""
    card = tool_card("Writer", WorkspaceTool(workspace_id="notes"))
    process = process_with_cards([card])
    store = RecordingCardStore([card])

    paths = declared_workspace_paths(process=process, store=store)

    assert {k: str(v) for k, v in paths.items()} == {"notes": "alice/notes"}


def test_bare_workspace_card_resolves_to_the_team_leaf() -> None:
    """A bare ``WorkspaceTool()`` declares the team's own tree as its leaf."""
    team_id = uuid.uuid4()
    card = tool_card("Writer", WorkspaceTool())
    process = process_with_cards([card], team_id=team_id)
    store = RecordingCardStore([card])

    paths = declared_workspace_paths(process=process, store=store)

    assert {k: str(v) for k, v in paths.items()} == {str(team_id): f"alice/{team_id}"}


def test_metadata_card_resolves_under_the_reserved_meta_scope() -> None:
    """AC #1: a metadata card resolves under ``_meta/``, not under the caller."""
    card = tool_card("Writer", WorkspaceTool(workspace_metadata_keys=["customer_id", "case_id"]))
    process = process_with_cards([card], metadata=CaseMetadata())
    store = RecordingCardStore([card])

    paths = declared_workspace_paths(process=process, store=store)

    leaf = "case_id-42__customer_id-ACME"
    assert {k: str(v) for k, v in paths.items()} == {leaf: f"_meta/{leaf}"}


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
        paths = declared_workspace_paths(process=process, store=store)
        assert str(paths["notes"]) == f"{owner}/notes"

    assert "user_id" not in inspect.signature(declared_workspace_paths).parameters


def test_one_workspace_id_on_two_owners_teams_is_two_trees() -> None:
    """The isolation the layout buys: one string, two owners, two paths.

    Bob declaring ``workspace_id="notes"`` on a team of his own resolves under
    *his* principal, so it cannot reach Alice's tree. Reaching hers needs a team
    she owns, which ``require_team_access`` refuses him.
    """
    card = tool_card("Writer", WorkspaceTool(workspace_id="notes"))
    store = RecordingCardStore([card])

    alice = declared_workspace_paths(
        process=process_with_cards([card], user_id="alice"), store=store
    )
    bob = declared_workspace_paths(process=process_with_cards([card], user_id="bob"), store=store)

    assert str(alice["notes"]) == "alice/notes"
    assert str(bob["notes"]) == "bob/notes"


# ---------------------------------------------------------------------------
# The two card shapes are not one shape
# ---------------------------------------------------------------------------


def test_exec_tool_contributes_a_leaf() -> None:
    """``ExecTool`` is a real declaration site — omitting it would 404 a live id."""
    card = tool_card("Runner", ExecTool(workspace_id="shell"))
    process = process_with_cards([card])
    store = RecordingCardStore([card])

    paths = declared_workspace_paths(process=process, store=store)

    assert {k: str(v) for k, v in paths.items()} == {"shell": "alice/shell"}


def test_exec_tool_and_workspace_tool_on_one_team() -> None:
    """Both shapes resolve side by side; ``ExecTool``'s missing field is exercised.

    ``ExecTool`` carries no ``workspace_metadata_keys`` and never will, so a
    team holding both cards is what proves the enumeration reads the field off
    ``WorkspaceTool`` only rather than assuming one shape.
    """
    workspace = tool_card("Writer", WorkspaceTool(workspace_id="notes"))
    runner = tool_card("Runner", ExecTool(workspace_id="shell"))
    process = process_with_cards([workspace, runner])
    store = RecordingCardStore([workspace, runner])

    paths = declared_workspace_paths(process=process, store=store)

    assert {k: str(v) for k, v in paths.items()} == {
        "notes": "alice/notes",
        "shell": "alice/shell",
    }


def test_exec_tool_has_no_metadata_keys_field() -> None:
    """The field really is absent — the reason the two shapes are written apart."""
    assert "workspace_metadata_keys" not in ExecTool.model_fields
    assert "workspace_metadata_keys" in WorkspaceTool.model_fields


def test_bare_base_config_card_declares_nothing() -> None:
    """A card carrying a plain ``BaseConfig`` has no ``tools`` and must not raise."""
    card = bare_card("Manager")
    process = process_with_cards([card])
    store = RecordingCardStore([card])

    assert declared_workspace_paths(process=process, store=store) == {}


def test_a_non_workspace_tool_declares_nothing() -> None:
    """A team whose only card declares no workspace names no workspace."""
    card = tool_card("Manager")  # AgentConfig with an empty tools list
    process = process_with_cards([card])
    store = RecordingCardStore([card])

    assert declared_workspace_paths(process=process, store=store) == {}


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

    assert declared_workspace_paths(process=process, store=store) == {}


def test_stashed_paths_is_none_before_the_gate_runs() -> None:
    """No stash means the gate never authorized an id — never a licence to resolve."""

    class _Conn:
        def __init__(self) -> None:
            self.state = State()

    conn = _Conn()
    assert stashed_workspace_paths(conn) is None  # type: ignore[arg-type]
    stash_workspace_paths(conn, {"notes": PurePosixPath("bob/notes")})  # type: ignore[arg-type]
    stashed = stashed_workspace_paths(conn)  # type: ignore[arg-type]
    assert stashed is not None
    assert str(stashed["notes"]) == "bob/notes"
