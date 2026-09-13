"""The one place the server turns a team's cards into workspace directories.

**Nothing here composes a path.** Every directory the server opens or removes is
produced by :func:`akgentic.tool.workspace.resolve_workspace_path`, the single
statement of the rule (ADR-048 Decision 5). This module resolves the team's
declared cards through it and hands back what it produced.

The layout it produces is three segments, ``<scope>/<kind>/<leaf>`` (ADR-052
Decision 1). ``<scope>`` answers who may reach the tree, ``<kind>`` how its leaf
was derived, and ``<leaf>`` which one:

==============================================  ================================================
Card                                            Path
==============================================  ================================================
``WorkspaceTool()``                             ``<user_id>/_team/<team_id>``
``WorkspaceTool(workspace_id="notes")``         ``<user_id>/_id/notes``
``WorkspaceTool(workspace_metadata_keys=…)``    ``<user_id>/_meta/customer_id-ACME__case_id-42``
``… workspace_sharable=True`` (each of above)   ``_shared/<kind>/<leaf>``
==============================================  ================================================

Per-principal is the default for every kind, metadata included; a card's
``workspace_sharable`` swaps the principal for the reserved shared scope and
changes nothing else. This module reads that field off the card and hands it to
the resolver. It supplies the answer itself in one case only: a team with no
default-layout card, where no card exists to answer (see
:func:`default_workspace_path`).

``<user_id>`` above is the **team owner's** ``Process.user_id``, never the
calling principal's. The caller's identity governs authorization; the team's
owner governs path resolution. See :func:`declared_workspace_paths`.

**Why this lives under ``services/`` and not under ``routes/``.** Both the
workspace routes and ``TeamService.delete_team`` need these readers. The
established direction of dependency is ``routes -> services``; nothing under
``services/``, ``protocols/`` or ``adapters/`` imports from ``routes/``, and
reversing that for one import would be the worse of the two available mistakes —
the other being a second copy of the card read inside the service, which is the
shape of the defect epic 71 exists to remove. ``routes/_workspace_resolution.py``
re-exports :func:`declared_workspace_paths` and :func:`default_workspace_path`,
so every route and every existing test import keeps resolving unchanged. Nothing
here imports FastAPI; the request-boundary plumbing stays on the other side of
that seam.

Nothing here authorizes. The access gate selects one path from here, reads its
``<scope>`` and decides from that alone.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath

from akgentic.agent.config import AgentConfig
from akgentic.team import resolve_agent_cards
from akgentic.team.models import Process
from akgentic.team.ports import EventStore
from akgentic.tool import ToolCard
from akgentic.tool.workspace import WorkspaceTool, resolve_workspace_path

logger = logging.getLogger(__name__)

__all__ = [
    "declared_workspace_paths",
    "default_workspace_path",
    "deletion_candidate_paths",
]

# One card's workspace declaration: ``(workspace_id, workspace_metadata_keys,
# workspace_sharable)``. Named once so the three readers below agree on it.
_Layout = tuple[str | None, list[str], bool]


def _declared_layout(tool: ToolCard) -> _Layout | None:
    """The ``(workspace_id, workspace_metadata_keys, workspace_sharable)`` a card declares.

    ``None`` for a card that declares no workspace. ``WorkspaceTool`` is the
    **only** card that declares one. Sandboxed execution is one of its
    capabilities (``workspace_exec=...``), not a card of its own: a shell-only
    agent is a ``WorkspaceTool`` with every file capability off and
    ``workspace_exec`` on, and it declares its directory through the same three
    fields as any other. There is no second shape to read, and a blanket
    ``getattr(tool, "workspace_metadata_keys", [])`` would only swallow a
    ``WorkspaceTool`` that lost the field — so the type is checked and the fields
    are read directly.

    ``workspace_sharable`` travels with the other two because it is as much the
    card's declaration as they are: it picks the ``<scope>``, and the resolver
    takes it as a required argument precisely so no caller can answer it for the
    card.
    """
    if isinstance(tool, WorkspaceTool):
        return tool.workspace_id, tool.workspace_metadata_keys, tool.workspace_sharable
    return None


def _team_layouts(process: Process, store: EventStore) -> list[_Layout]:
    """Every workspace layout the team's cards declare, from one card-store read.

    All three public readers below are built on this, and a caller calls exactly
    one of them, so a request — or a deletion — reads the card store once on
    every branch.
    """
    layouts: list[_Layout] = []
    for card in resolve_agent_cards(process.agent_cards, store):
        config = card.config
        # ``AgentCard.config`` is typed ``BaseConfig`` in core and ``tools``
        # lives on ``AgentConfig``; a card carrying the bare base declares no
        # tools and therefore no workspace.
        if not isinstance(config, AgentConfig):
            continue
        for tool in config.tools:
            layout = _declared_layout(tool)
            if layout is not None:
                layouts.append(layout)
    return layouts


def _resolve_layout(layout: _Layout, process: Process) -> PurePosixPath:
    """One declared layout, through the single resolver."""
    workspace_id, metadata_keys, sharable = layout
    return resolve_workspace_path(
        workspace_id=workspace_id,
        workspace_metadata_keys=metadata_keys,
        team_id=str(process.team_id),
        user_id=process.user_id,
        metadata=process.metadata,
        workspace_sharable=sharable,
    )


def _default_from_layouts(layouts: list[_Layout], process: Process) -> PurePosixPath:
    """The team's own ``_team`` tree, with the sharing axis read off its default card.

    Split out of :func:`default_workspace_path` so
    :func:`deletion_candidate_paths` can reach it on the layouts it has already
    read, rather than paying for a second card-store read.

    Raises:
        ValueError: If the team's default-layout cards disagree on
            ``workspace_sharable``, or from the resolver for an owner id that
            cannot be a directory name.
    """
    sharable = {
        card_sharable
        for workspace_id, metadata_keys, card_sharable in layouts
        if workspace_id is None and not metadata_keys
    }
    if len(sharable) > 1:
        raise ValueError(
            "the team's default-layout workspace cards disagree on workspace_sharable, "
            "so no single tree is the team's own"
        )
    return resolve_workspace_path(
        workspace_id=None,
        workspace_metadata_keys=[],
        team_id=str(process.team_id),
        user_id=process.user_id,
        metadata=process.metadata,
        # With no default-layout card, nothing declares this tree. It is the
        # team's per-principal default, which is what these routes have always
        # served for a team without one (the seeded catalog team is such a
        # team). ``False`` is honest only on that branch, because no card
        # exists to say otherwise. With a card, the card's own value is used.
        workspace_sharable=sharable.pop() if sharable else False,
    )


def declared_workspace_paths(*, process: Process, store: EventStore) -> dict[str, PurePosixPath]:
    """Every workspace the team declares, keyed by the leaf a client may name.

    ADR-048 Decision 7 in one sentence: resolve every card of the authorized
    team through the same resolver, and the query's ``workspace_id`` must equal
    one of the identifiers that produces. The matching card supplies the kind
    and the scope: a metadata workspace resolves under ``<owner>/_meta/`` and a
    named one under ``<owner>/_id/``, where ``<owner>`` is the **team owner's**
    principal, or ``_shared`` when the card declares ``workspace_sharable``. The
    route never infers any of it from the string, and never falls back to an
    unscoped path when a directory is absent, which is the hole being closed.

    **A principal scope is ``process.user_id``, and the calling principal is not
    an input here at all.** The caller's identity governs *authorization*; the
    team's owner governs *path resolution*. The agent writes under
    ``observer.user_id``, which is propagated from ``Process.user_id``, so
    resolving the caller's scope instead would send an admin who has already
    passed ``require_team_access`` to a different, empty directory — worse than
    a refusal, because nothing signals it and the caller concludes the agent
    wrote nothing.

    That is safe against the obvious attack without consulting the caller: Bob
    cannot reach ``<alice>/_id/notes`` by declaring ``workspace_id="notes"`` on
    his own team, because his team resolves under *his* ``process.user_id``.
    Reaching Alice's tree needs a team Alice owns, which ``require_team_access``
    refuses him. That argument holds for a per-principal tree only. A
    ``_shared`` tree has no owner, so this map holds it for every team that
    declares the same kind and leaf. **This map is not the authorization.** The
    gate reads the scope segment of the one path it selects from here and
    refuses a ``_shared`` path outright (``check_workspace_scope`` in
    ``_team_access``).

    A metadata card is the same rule with no second clause: its key is the leaf
    ``process.metadata`` produces through the declared keys, in declaration
    order, so a ``?workspace_id=`` naming a metadata workspace is admitted iff
    it is byte-equal to that leaf. Nothing here parses a leaf or compares
    key-value pairs — the leaf is derived from the metadata, not matched
    against it — which is why another case's values, another key set, or the
    reversed order are absent from the map rather than present and refused.
    Pinned in pairs by ``tests/server/routes/test_workspace_routes.py`` and
    ``tests/server/routes/test_team_access.py``.

    The cards are resolved through ``akgentic.team.resolve_agent_cards`` — the
    one place a hash becomes a card — which makes a **single** batch
    ``load_agent_cards`` call whatever the number of roles. A per-hash loop
    returns the identical result, so only a call-count assertion catches one
    coming back. The card store is the authority rather than the catalog:
    ``catalog_namespace`` is nullable and a catalog entry can be edited after a
    team is created, so it would answer an authorization question with a value
    that may no longer describe the team, whereas a blob at a hash is the bytes
    that hash names, forever.

    Args:
        process: The **authorized** team — the one named in the route path. Its
            ``user_id`` is the ``<scope>`` of every card that does not declare
            ``workspace_sharable``.
        store: The card store to resolve the team's ``agent_cards`` against.

    Returns:
        Leaf -> resolved three-segment path, for every workspace any of the
        team's cards declares. Empty when the team declares no workspace at all, which
        is a team no ``?workspace_id=`` can name.

    Raises:
        AgentCardNotFoundError: If a ``card_hash`` does not resolve. A
            ``LookupError``, never a ``ValueError`` — the backends' corrupted
            document handlers catch ``ValueError``, so it would be swallowed on
            the very path it exists to fail loudly on. Dropping the card instead
            would silently shrink the allowed set, and a caller with a real
            workspace would get a 404 with nothing in the logs saying why.
        ValueError: Propagated from the resolver for a principal or a leaf that
            cannot be a directory name, or for a metadata card the team's
            metadata cannot satisfy.
    """
    paths: dict[str, PurePosixPath] = {}
    for layout in _team_layouts(process, store):
        path = _resolve_layout(layout, process)
        paths[path.name] = path
    return paths


def default_workspace_path(*, process: Process, store: EventStore) -> PurePosixPath:
    """The tree an omitted ``?workspace_id=`` serves: the one the team's default card binds to.

    A default-layout card is a ``WorkspaceTool`` that names no workspace and
    declares no metadata keys, so its agents write to the ``_team`` kind with
    the team id as the leaf. Which **scope** that tree sits under is the card's
    ``workspace_sharable``, so this reads it off the card exactly as the agent
    side does at bind. Serving the owner's tree while the agents write to the
    shared one would answer 200 over an empty directory.

    Args:
        process: The **authorized** team.
        store: The card store to resolve the team's ``agent_cards`` against.

    Returns:
        The team's own three-segment path, ``<owner>/_team/<team_id>`` or
        ``_shared/_team/<team_id>``.

    Raises:
        AgentCardNotFoundError: If a ``card_hash`` does not resolve. Without
            every card the scope cannot be known, so this fails rather than
            guess.
        ValueError: If the team's default-layout cards disagree on
            ``workspace_sharable``. That is a configuration defect with no
            right answer, and picking one would depend on the order the store
            returns cards. Also propagated from the resolver for an owner id
            that cannot be a directory name.
    """
    return _default_from_layouts(_team_layouts(process, store), process)


def deletion_candidate_paths(*, process: Process, store: EventStore) -> list[PurePosixPath]:
    """Every tree a team's deletion may *consider*, and the only set it may consider.

    The candidates are the team's own declared workspaces plus its own default
    ``_team`` tree — nothing else can enter the set. That is the first of the
    two limits the deletion path enforces rather than documents: a
    :class:`~akgentic.infra.protocols.workspace_deletion.WorkspaceDeletionPolicy`
    is asked about these paths and answers delete-or-keep, so no policy — not
    even a hostile one — can introduce a target of its own. Whether a candidate
    is then *removed* is the policy's answer plus the caller's containment check;
    membership here is necessary, never sufficient.

    **De-duplicated on the whole path, never on the leaf.**
    :func:`declared_workspace_paths` keys its map on ``path.name``, which stopped
    being unique the moment a path gained a scope and a kind: a team declaring
    both ``<owner>/_id/x`` and ``<owner>/_meta/x`` loses one of them there. That
    map is not reused here for exactly that reason. (Re-keying it is story 71-2's
    scope and deliberately not done here.)

    Order is declaration order with the default tree last, and duplicates keep
    their first position — so a team declaring its own default tree explicitly
    yields one candidate, not two.

    Args:
        process: The team being deleted.
        store: The card store to resolve the team's ``agent_cards`` against.

    Returns:
        The de-duplicated resolved three-segment paths, relative to the
        workspaces root. Never empty: the default tree is always a candidate,
        for a team that declares no workspace card at all as much as for one
        that does.

    Raises:
        AgentCardNotFoundError: If a ``card_hash`` does not resolve. The sharing
            axis is unknowable without every card, so this fails rather than
            guess — and guessing would silently reinstate the per-principal
            default, which is the defect this module's readers exist to remove.
        ValueError: From the resolver for an owner id, a leaf or a metadata
            declaration that cannot yield a path, or when the team's
            default-layout cards disagree on ``workspace_sharable``.
    """
    layouts = _team_layouts(process, store)
    candidates = [_resolve_layout(layout, process) for layout in layouts]
    candidates.append(_default_from_layouts(layouts, process))
    # ``dict.fromkeys`` rather than ``set``: it de-duplicates on the whole path
    # while keeping first-seen order, so the candidate list a log record names is
    # stable run to run.
    return list(dict.fromkeys(candidates))
