"""The one seam through which the server learns a workspace's directory.

Two things live here, and they are here together on purpose: the
request-boundary segment guard for the ``?workspace_id=`` query parameter, and
the enumeration of the workspaces a team actually declares. Both are needed by
``routes/workspace.py`` (which opens the directory) *and* by
``routes/_team_access.py`` (which refuses the ones the team never declared), and
``workspace.py`` already imports ``_team_access``. A second copy in either
module is the duplication ADR-048 exists to remove, one scale down.

**Nothing here composes a path.** Every directory the server opens is produced
by :func:`akgentic.tool.workspace.resolve_workspace_path`, the single statement
of the rule (ADR-048 Decision 5). This module resolves the team's declared cards
through it and hands back what it produced.

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

Nothing here authorizes. The access gate selects one path from here, reads its
``<scope>`` and decides from that alone.

Only the ``<leaf>`` is ever on the wire (ADR-048 Decision 8): the scope and the
kind are not the client's to choose, and the server recomputes both from the
matching card.

``<user_id>`` above is the **team owner's** ``Process.user_id``, never the
calling principal's. The caller's identity governs authorization; the team's
owner governs path resolution. See :func:`declared_workspace_paths`.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath

from fastapi import HTTPException
from starlette.requests import HTTPConnection

from akgentic.agent.config import AgentConfig
from akgentic.team import resolve_agent_cards
from akgentic.team.models import Process
from akgentic.team.ports import EventStore
from akgentic.tool import ToolCard
from akgentic.tool.workspace import WorkspaceTool, leaf_segment, resolve_workspace_path

logger = logging.getLogger(__name__)

__all__ = [
    "declared_workspace_paths",
    "default_workspace_path",
    "stash_team_process",
    "stash_workspace_path",
    "stashed_team_process",
    "stashed_workspace_path",
    "validate_workspace_id",
]

# Per-request slot holding the ONE resolved path the access gate authorized, on
# either branch. The route that opens the directory opens this path and nothing
# else, so a path the gate's scope check never saw cannot be opened. Named once,
# here, because both sides of the seam address it.
_AUTHORIZED_PATH_SLOT = "akgentic_authorized_workspace_path"

# Per-request slot holding the ``Process`` the team-access gate already loaded.
# ``get_team`` is ``EventStore.load_team`` on the department and enterprise
# tiers — a database read, not an in-process lookup — so the gate, the workspace
# gate and the route reading the same team three times is three queries where
# the request only ever concerns one team. Same idea as the authorized path
# stashed above: resolve once, read back.
_TEAM_PROCESS_SLOT = "akgentic_authorized_team_process"


def validate_workspace_id(workspace_id: str) -> str:
    """Reject any workspace_id that cannot be a workspace leaf, with HTTP 400.

    ``?workspace_id=`` is a **leaf selector**, so the guard is the tool's own
    :func:`akgentic.tool.workspace.leaf_segment` (ADR-052 Decision 6), applied
    to the value the caller sent. It refuses the empty string, a leading dot
    (``.`` and ``..`` included), ``/``, ``\\``, NUL, the three kind names and
    the sidecar suffixes, in any letter case, raising **before** any card is
    read or any ``Filesystem`` is constructed. It caps no length and admits
    ``%``, so every leaf a card can declare, a percent-encoded metadata leaf
    included, gets through.

    This is a traversal guard, not an access policy. Authorization is
    membership: the selector is served only when it is byte-equal to a leaf in
    :func:`declared_workspace_paths`. A path-safe value no card declares is
    refused there with 404.

    The detail is fixed: the tool's message reflects the caller's value and
    names internal directories, so it stays out of the response.

    Returns:
        The value unchanged when it is a valid leaf.
    """
    try:
        return leaf_segment(workspace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid workspace_id") from None


def _declared_layout(tool: ToolCard) -> tuple[str | None, list[str], bool] | None:
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


def _team_layouts(process: Process, store: EventStore) -> list[tuple[str | None, list[str], bool]]:
    """Every workspace layout the team's cards declare, from one card-store read.

    Both public readers below are built on this, and a request calls exactly one
    of them, so a request reads the card store once on either branch.
    """
    layouts: list[tuple[str | None, list[str], bool]] = []
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
    for workspace_id, metadata_keys, sharable in _team_layouts(process, store):
        path = resolve_workspace_path(
            workspace_id=workspace_id,
            workspace_metadata_keys=metadata_keys,
            team_id=str(process.team_id),
            user_id=process.user_id,
            metadata=process.metadata,
            workspace_sharable=sharable,
        )
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
    sharable = {
        card_sharable
        for workspace_id, metadata_keys, card_sharable in _team_layouts(process, store)
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


def stash_workspace_path(conn: HTTPConnection, path: PurePosixPath) -> None:
    """Record the one path the gate authorized, for the route that opens it."""
    setattr(conn.state, _AUTHORIZED_PATH_SLOT, path)


def stashed_workspace_path(conn: HTTPConnection) -> PurePosixPath | None:
    """The path the gate authorized for this request, or ``None`` if it never ran.

    ``None`` is not a licence to resolve the path some other way, on either
    branch: the route answers 404. A request that reaches the directory without
    the gate having resolved and checked the path is the hole the single
    convergence point closes.
    """
    stashed = getattr(conn.state, _AUTHORIZED_PATH_SLOT, None)
    if isinstance(stashed, PurePosixPath):
        return stashed
    return None


def stash_team_process(conn: HTTPConnection, process: Process) -> None:
    """Record the team ``require_team_access`` authorized, for the rest of the request.

    Only that gate writes this slot, and it writes the team named in the route
    path — never the foreign team ``_deny_foreign_named_team`` looks up, which
    is a different team and must not be mistaken for the authorized one.
    """
    setattr(conn.state, _TEAM_PROCESS_SLOT, process)


def stashed_team_process(conn: HTTPConnection) -> Process | None:
    """The authorized team the gate loaded, or ``None`` if it never ran.

    ``None`` is safe to fall back on here, unlike
    :func:`stashed_workspace_path`: re-reading the team is a redundant query,
    not a skipped authorization. The gate answered that question already, and
    the value it stashed is the same team the fallback would fetch.
    """
    stashed = getattr(conn.state, _TEAM_PROCESS_SLOT, None)
    if isinstance(stashed, Process):
        return stashed
    return None
