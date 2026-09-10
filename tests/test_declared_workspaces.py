"""The value the placement seam carries, and the affinity rule it answers (Story 68.2).

Every spec here proves that the rule was **transcribed** correctly: the sibling
tiers route on exactly the answer ``routing_key()`` gives, so a transcription
error is a wrong pin on every tier. None of them proves affinity — two teams
declaring one tree landing on one worker needs two worker processes and a
registry, neither of which exists in this package, and no spec below claims it.
"""

from __future__ import annotations

from pathlib import PurePosixPath

import pytest
from akgentic.tool.workspace import METADATA_SCOPE
from pydantic import ValidationError

from akgentic.infra.errors import ServerError
from akgentic.infra.protocols.placement import (
    DeclaredWorkspaces,
    PlacementError,
    UnroutableWorkspacesError,
)

_NOTES = PurePosixPath("alice/notes")
_DRAFTS = PurePosixPath("alice/drafts")
_OWN = PurePosixPath("alice/0b1e6b8a-5b7b-4b6e-9c6b-1a2b3c4d5e6f")
_META_CASE = PurePosixPath(METADATA_SCOPE) / "customer_id-ACME__case_id-42"
_META_OTHER = PurePosixPath(METADATA_SCOPE) / "case_id-42"
# A principal that sorts BEFORE the metadata scope. ``_`` is 0x5F and ``a`` is
# 0x61, so every lowercase principal sorts after ``_meta/`` and a "smallest of
# all" router would pick the ``_meta/`` tree by accident; an Azure AD ``sub`` is
# base64url and routinely starts with an uppercase letter or a digit.
_UPPER_NOTES = PurePosixPath("Alice/notes")


# ---------------------------------------------------------------------------
# Rule 2 — a _meta/ tree wins, a user-named tree otherwise, freely when none
# ---------------------------------------------------------------------------


def test_a_metadata_tree_wins_over_a_user_named_one() -> None:
    """Rule 2: one ``_meta/`` tree routes the team, whatever else it declares.

    Transcription check. Falsified by a router that hashed the smallest tree
    regardless of scope — which is why the principal here is ``Alice``, not
    ``alice``: ``Alice/notes`` sorts before ``_meta/…``, so that mutation returns
    the user tree, whereas with a lowercase principal it would return the
    ``_meta/`` tree for the wrong reason and this spec would be decoration.
    """
    value = DeclaredWorkspaces(shared={_UPPER_NOTES, _META_CASE})
    assert str(_UPPER_NOTES) < str(_META_CASE)
    assert value.routing_key() == _META_CASE


def test_a_user_named_tree_routes_when_there_is_no_metadata_tree() -> None:
    """Rule 2, second clause: no ``_meta/`` tree, so the user-named tree is the key."""
    value = DeclaredWorkspaces(shared={_NOTES})
    assert value.routing_key() == _NOTES


def test_a_team_with_nothing_shareable_places_freely_even_when_it_has_an_own_tree() -> None:
    """Rule 2, third clause — and the herding guard.

    The value is built **with** ``own`` populated on purpose: a router that
    hashed the own tree (or ``None``) would herd every default team onto one
    worker, and with ``own=None`` that mutation would have nothing to hash and
    this spec would be decoration. ``<user_id>/<team_id>`` is reachable by one
    team by construction, so it never routes on path.
    """
    value = DeclaredWorkspaces(shared=frozenset(), own=_OWN)
    assert value.own == _OWN
    assert value.routing_key() is None


def test_the_empty_value_places_freely() -> None:
    """A team declaring no workspace at all — the seeded catalog team — has no key."""
    assert DeclaredWorkspaces().routing_key() is None


def test_several_user_named_trees_route_on_the_lexicographically_smallest() -> None:
    """Rule 2 as amended: several user-named trees and no ``_meta/`` → the smallest.

    Deterministic and memoryless, so every replica agrees with no shared state.
    The result is identical whichever order the set was built in, which is what
    makes it a rule rather than an accident of iteration.
    """
    forward = DeclaredWorkspaces(shared=[_NOTES, _DRAFTS])
    backward = DeclaredWorkspaces(shared=[_DRAFTS, _NOTES])
    assert forward.routing_key() == _DRAFTS
    assert backward.routing_key() == _DRAFTS


# ---------------------------------------------------------------------------
# Rule 3 — two _meta/ trees on one team is the unsatisfiable case
# ---------------------------------------------------------------------------


def test_two_metadata_trees_raise_the_typed_refusal_naming_both() -> None:
    """Rule 3: refused loudly, with the status a tier that routes must answer.

    409 and not the placement default of 503: the condition is a property of the
    team's definition, not of the cluster's capacity, so a ``Retry-After`` would
    be a lie. Both trees are named so the admin sees which two cards disagree.
    """
    value = DeclaredWorkspaces(shared={_NOTES, _META_CASE, _META_OTHER})
    with pytest.raises(UnroutableWorkspacesError) as excinfo:
        value.routing_key()
    err = excinfo.value
    assert err.status_code == 409
    assert err.code == "workspace_affinity_unsatisfiable"
    assert err.headers is None
    assert str(_META_CASE) in err.detail
    assert str(_META_OTHER) in err.detail
    # Refusal, not retry: nothing in the message invites the client back.
    assert "Retry-After" not in (err.headers or {})


def test_the_refusal_is_a_placement_error_and_a_server_error() -> None:
    """The single ``ServerError`` handler answers it; ``except PlacementError`` keeps holding."""
    assert issubclass(UnroutableWorkspacesError, PlacementError)
    assert issubclass(UnroutableWorkspacesError, ServerError)
    assert issubclass(UnroutableWorkspacesError, RuntimeError)


# ---------------------------------------------------------------------------
# The value itself — frozen, hashable, order-independent
# ---------------------------------------------------------------------------


def test_the_value_is_frozen() -> None:
    """A value crossing the seam cannot be edited by a tier on its way through."""
    value = DeclaredWorkspaces(shared={_NOTES})
    with pytest.raises(ValidationError):
        value.shared = frozenset()  # type: ignore[misc]
    with pytest.raises(ValidationError):
        value.own = _OWN  # type: ignore[misc]


def test_two_values_built_in_different_orders_compare_equal_and_hash_equal() -> None:
    """A set, not a list: the store's card order must not change the key."""
    forward = DeclaredWorkspaces(shared=[_NOTES, _META_CASE], own=_OWN)
    backward = DeclaredWorkspaces(shared=[_META_CASE, _NOTES], own=_OWN)
    assert forward == backward
    assert hash(forward) == hash(backward)
    assert isinstance(forward.shared, frozenset)


def test_the_fields_are_paths_not_strings() -> None:
    """The seam carries the resolver's own type; a string would invite re-parsing."""
    value = DeclaredWorkspaces(shared={_NOTES}, own=_OWN)
    assert all(isinstance(path, PurePosixPath) for path in value.shared)
    assert isinstance(value.own, PurePosixPath)
