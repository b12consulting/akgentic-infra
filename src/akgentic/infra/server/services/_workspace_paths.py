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
re-exports :func:`declared_workspace_paths`, :func:`default_workspace_path` and
:func:`select_declared_path`, so every route and every existing test import keeps
resolving unchanged. Nothing here imports FastAPI; the request-boundary plumbing
stays on the other side of that seam.

**Every reader keys on the whole path.** A ``<leaf>`` is unique only within a
scope and a kind, so a set of declared workspaces keyed on ``path.name`` loses
one of ``<owner>/_id/x`` and ``<owner>/_meta/x`` — and one of ``<owner>/_id/x``
and ``_shared/_id/x`` — to whichever the card order put last. Both readers
de-duplicate on the whole resolved path through :func:`_declared_from_layouts`,
which is the single statement of that rule. The leaf survives only as the
*selector* a client sends, and :func:`select_declared_path` refuses a leaf two
declared paths share rather than picking one.

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
    "select_declared_path",
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


def _declared_from_layouts(layouts: list[_Layout], process: Process) -> list[PurePosixPath]:
    """Every declared layout resolved, de-duplicated on the **whole path**.

    The one statement of the keying rule both public readers use. ``path.name``
    is the ``<leaf>``, which stopped being unique the moment a path gained a
    scope and a kind, so keying on it drops one of ``<owner>/_id/x`` and
    ``<owner>/_meta/x`` — and one of ``<owner>/_id/x`` and ``_shared/_id/x`` —
    on the strength of the order the card store happened to return rows in.

    ``dict.fromkeys`` rather than ``set``: it de-duplicates on the whole path
    while keeping first-seen declaration order, so the list a log record names
    is stable run to run.
    """
    return list(dict.fromkeys(_resolve_layout(layout, process) for layout in layouts))


def _default_sharable(layouts: list[_Layout]) -> set[bool]:
    """Every ``workspace_sharable`` the team's **default-layout** cards declare.

    Empty when no card declares the default layout; a set of two when the
    team's default cards disagree, which is a configuration defect with no
    right answer. The two readers differ only in what they do about that.
    """
    return {
        card_sharable
        for workspace_id, metadata_keys, card_sharable in layouts
        if workspace_id is None and not metadata_keys
    }


def _resolve_default(process: Process, *, sharable: bool) -> PurePosixPath:
    """The team's own ``_team`` tree under the given sharing axis.

    *sharable* is the value a default-layout card declared. ``False`` is only
    honest for a team with no such card, where nothing declares this tree and
    the per-principal default is what these routes have always served (the
    seeded catalog team is such a team).
    """
    return resolve_workspace_path(
        workspace_id=None,
        workspace_metadata_keys=[],
        team_id=str(process.team_id),
        user_id=process.user_id,
        metadata=process.metadata,
        workspace_sharable=sharable,
    )


def select_declared_path(*, paths: list[PurePosixPath], leaf: str) -> PurePosixPath | None:
    """The one declared path a client's ``?workspace_id=`` leaf names.

    The wire carries a ``<leaf>`` and nothing else (ADR-048 Decision 8), so this
    is the lookup that replaced the leaf-keyed map. A leaf is unique within a
    scope and a kind, never across them, which leaves three answers rather than
    two:

    - **one match**: that path, exactly as the map's lookup returned it;
    - **no match**: ``None`` — the team declares no such workspace, and the gate
      answers its membership 404;
    - **more than one match**: a ``ValueError`` naming the leaf and *every*
      colliding path.

    **The ambiguous leaf is refused, never disambiguated.** With a leaf-only
    wire there is no input that separates two declared paths sharing a leaf, so
    the only choices are to pick one or to refuse — and picking one is the
    defect this replaced, whatever tie-break dresses it up. "Prefer the user
    scope over ``_shared``" is the tempting one and the wrong one: it bakes an
    authorization rule into path resolution and stops being right the moment an
    entitlement policy makes ``_shared`` reachable.

    A ``ValueError`` rather than a type of its own because the plumbing exists:
    the gate's resolution arm already turns one into a logged 500, for the
    sibling configuration defect of default cards that disagree on
    ``workspace_sharable``. Both are unanswerable questions about the team's
    cards rather than anything the caller can restate.

    Args:
        paths: The team's declared paths, from :func:`declared_workspace_paths`.
        leaf: The value the client sent, already through the tool's
            ``leaf_segment`` guard.

    Returns:
        The single declared path whose ``<leaf>`` is *leaf*, or ``None``.

    Raises:
        ValueError: If two or more declared paths share *leaf*. The message
            carries both the leaf and every colliding path, so the log record
            the gate writes tells an operator which cards to fix.
    """
    matches = [path for path in paths if path.name == leaf]
    if len(matches) > 1:
        collisions = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"the workspace leaf {leaf!r} is declared by more than one of the team's "
            f"cards, so no single path is the one requested: {collisions}"
        )
    return matches[0] if matches else None


def declared_workspace_paths(*, process: Process, store: EventStore) -> list[PurePosixPath]:
    """Every workspace the team declares, de-duplicated on the whole resolved path.

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
    ``_shared`` tree has no owner, so this list holds it for every team that
    declares the same kind and leaf. **This list is not the authorization.** The
    gate reads the scope segment of the one path it selects from here and
    refuses a ``_shared`` path outright (``check_workspace_scope`` in
    ``_team_access``).

    A metadata card is the same rule with no second clause: its leaf is the one
    ``process.metadata`` produces through the declared keys, in declaration
    order, so a ``?workspace_id=`` naming a metadata workspace is admitted iff
    it is byte-equal to that leaf. Nothing here parses a leaf or compares
    key-value pairs — the leaf is derived from the metadata, not matched
    against it — which is why another case's values, another key set, or the
    reversed order are absent from the list rather than present and refused.
    Pinned in pairs by ``tests/server/routes/test_workspace_routes.py`` and
    ``tests/server/routes/test_team_access.py``.

    **The de-duplication is on the whole path, so nothing is lost to a leaf
    collision.** A team declaring both ``<owner>/_id/x`` and ``<owner>/_meta/x``
    gets both, and so does one declaring ``<owner>/_id/notes`` beside
    ``_shared/_id/notes``. Which of the two a client's leaf then names is
    :func:`select_declared_path`'s question, and it refuses rather than picks.

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
        The resolved three-segment paths, one per workspace any of the team's
        cards declares, de-duplicated on the whole path and in declaration
        order. Empty when the team declares no workspace at all, which is a team
        no ``?workspace_id=`` can name.

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
    return _declared_from_layouts(_team_layouts(process, store), process)


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
    sharable = _default_sharable(_team_layouts(process, store))
    if len(sharable) > 1:
        raise ValueError(
            "the team's default-layout workspace cards disagree on workspace_sharable, "
            "so no single tree is the team's own"
        )
    return _resolve_default(process, sharable=sharable.pop() if sharable else False)


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

    **De-duplicated on the whole path, never on the leaf** — through the same
    :func:`_declared_from_layouts` the client-facing reader uses, so the rule is
    stated once. The two readers stay separate because they answer different
    questions: that one is "what may a client name", this one is "what may this
    team's deletion consider", and only this one includes the team's own default
    ``_team`` tree, which exists whether or not any card declares it. Folding
    them would mean one caller filtering that member back out.

    Order is declaration order with the default tree last, and duplicates keep
    their first position — so a team declaring its own default tree explicitly
    yields one candidate, not two.

    **Default cards that disagree on ``workspace_sharable`` do not empty this
    set.** ``default_workspace_path`` refuses that team, because a route serving
    one of two trees would be guessing. Deletion is the opposite case: both
    trees are ``_team/<this team id>``, both were resolved from this team's own
    cards, and once the team is gone neither can be addressed again. Discarding
    the whole list — which is what letting the refusal propagate did — left both
    trees and both ``.index`` sidecars behind for ever, in the one configuration
    where the code had already computed both correct targets. Both are declared
    layouts, so both are already in the list; the disagreement is recorded at
    WARNING because it is still a card defect an operator should repair.

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
            declaration that cannot yield a path.
    """
    layouts = _team_layouts(process, store)
    candidates = _declared_from_layouts(layouts, process)
    sharable = _default_sharable(layouts)
    if len(sharable) > 1:
        logger.warning(
            "the team's default-layout workspace cards disagree on workspace_sharable; "
            "every tree they declare stays a deletion candidate — team_id=%s",
            process.team_id,
        )
    else:
        candidates.append(_resolve_default(process, sharable=sharable.pop() if sharable else False))
    # ``dict.fromkeys`` rather than ``set``: it de-duplicates on the whole path
    # while keeping first-seen order, so the candidate list a log record names is
    # stable run to run.
    return list(dict.fromkeys(candidates))
