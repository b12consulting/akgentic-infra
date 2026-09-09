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

The layout it produces is two segments, ``<scope>/<leaf>`` (ADR-048 Decision 1):

===========================================  ===================================
Card                                         Path
===========================================  ===================================
``WorkspaceTool()``                          ``<user_id>/<team_id>``
``WorkspaceTool(workspace_id="notes")``      ``<user_id>/notes``
``WorkspaceTool(workspace_metadata_keys=…)``  ``_meta/customer_id-ACME__case_id-42``
===========================================  ===================================

Only the ``<leaf>`` is ever on the wire (ADR-048 Decision 8): the scope is not
the client's to choose, and the server recomputes it from the matching card.

``<user_id>`` above is the **team owner's** ``Process.user_id``, never the
calling principal's. The caller's identity governs authorization; the team's
owner governs path resolution. See :func:`declared_workspace_paths`.
"""

from __future__ import annotations

import logging
import re
from pathlib import PurePosixPath

from fastapi import HTTPException
from starlette.requests import HTTPConnection

from akgentic.agent.config import AgentConfig
from akgentic.team import resolve_agent_cards
from akgentic.team.models import Process
from akgentic.team.ports import EventStore
from akgentic.tool import ToolCard
from akgentic.tool.sandbox import ExecTool
from akgentic.tool.workspace import WorkspaceTool, resolve_workspace_path

logger = logging.getLogger(__name__)

__all__ = [
    "declared_workspace_paths",
    "stash_team_process",
    "stash_workspace_paths",
    "stashed_team_process",
    "stashed_workspace_paths",
    "validate_workspace_id",
]

# A workspace_id is a single safe path segment: alphanumerics plus dot, dash, and
# underscore, 1-128 chars. This is a route-boundary traversal guard (a correctness
# invariant) applying to the **logical** id the caller sent, NOT an access-policy
# check — the allow/deny answer is the declared-workspace check below.
_WORKSPACE_ID_RE = re.compile(r"\A[A-Za-z0-9._-]{1,128}\Z")

# Per-request slot holding the leaf -> resolved-path map the access gate built,
# so the route that opens the directory reads what the gate authorized instead
# of resolving the team's cards a second time. Named once, here, because both
# sides of the seam address it.
_DECLARED_PATHS_SLOT = "akgentic_declared_workspace_paths"

# Per-request slot holding the ``Process`` the team-access gate already loaded.
# ``get_team`` is ``EventStore.load_team`` on the department and enterprise
# tiers — a database read, not an in-process lookup — so the gate, the workspace
# gate and the route reading the same team three times is three queries where
# the request only ever concerns one team. Same reason the declared map is
# stashed above: resolve once, read back.
_TEAM_PROCESS_SLOT = "akgentic_authorized_team_process"


def validate_workspace_id(workspace_id: str) -> str:
    """Reject any workspace_id that is not a single safe path segment.

    Mandatory traversal guard (ADR-029 §2): rejects ``""``, ``"."``, ``".."``,
    any value containing a path separator, absolute paths, and over-length
    values with HTTP 400, raising **before** any ``Filesystem`` is constructed.
    Returns the value unchanged when it is a valid single segment.

    It guards what the *caller* sent, which is why it survives ADR-048
    untouched: :func:`akgentic.tool.workspace.leaf_segment` guards what the
    *resolver* emits, and the composed two-segment path is built server-side
    from values the caller cannot supply, so it never passes through here.
    """
    if workspace_id in ("", ".", "..") or not _WORKSPACE_ID_RE.fullmatch(workspace_id):
        raise HTTPException(status_code=400, detail="Invalid workspace_id")
    return workspace_id


def _declared_layout(tool: ToolCard) -> tuple[str | None, list[str]] | None:
    """The ``(workspace_id, workspace_metadata_keys)`` a card declares, or ``None``.

    **The two cards do not share a shape, and reading them as if they did would
    hide a regression.** ``WorkspaceTool`` carries both fields;
    ``ExecTool`` — the deprecated card epic 41 removes — carries
    ``workspace_id`` only and will never gain ``workspace_metadata_keys``
    (ADR-048 Decision 2), a metadata-scoped shell being already reachable
    through ``WorkspaceTool(workspace_exec=...)``. A blanket
    ``getattr(tool, "workspace_metadata_keys", [])`` across both would also
    swallow a ``WorkspaceTool`` that lost the field.

    ``ExecTool`` is included rather than skipped because it is a real
    declaration site: it resolves its own directory during ``observer()``, so
    omitting it would 404 a ``workspace_id`` an agent really is configured with.
    """
    if isinstance(tool, WorkspaceTool):
        return tool.workspace_id, tool.workspace_metadata_keys
    if isinstance(tool, ExecTool):
        return tool.workspace_id, []
    return None


def declared_workspace_paths(*, process: Process, store: EventStore) -> dict[str, PurePosixPath]:
    """Every workspace the team declares, keyed by the leaf a client may name.

    ADR-048 Decision 7 in one sentence: resolve every card of the authorized
    team through the same resolver, and the query's ``workspace_id`` must equal
    one of the identifiers that produces. The matching card's layout supplies
    the scope, so a metadata workspace resolves under ``_meta/`` and a named one
    under the **team owner's** principal — the route never infers which from the
    string, and never falls back to an unscoped path when a directory is absent,
    which is the hole being closed.

    **The scope is ``process.user_id``, and the calling principal is not an
    input here at all.** The caller's identity governs *authorization*; the
    team's owner governs *path resolution*. The agent writes under
    ``observer.user_id``, which is propagated from ``Process.user_id``, so
    resolving the caller's scope instead would send an admin who has already
    passed ``require_team_access`` to a different, empty directory — worse than
    a refusal, because nothing signals it and the caller concludes the agent
    wrote nothing.

    That is safe against the obvious attack without consulting the caller: Bob
    cannot reach ``<alice>/notes`` by declaring ``workspace_id="notes"`` on his
    own team, because his team resolves under *his* ``process.user_id``.
    Reaching Alice's tree needs a team Alice owns, which ``require_team_access``
    refuses him.

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
            ``user_id`` is the ``<scope>`` of every per-user layout.
        store: The card store to resolve the team's ``agent_cards`` against.

    Returns:
        Leaf -> resolved two-segment path, for every workspace any of the team's
        cards declares. Empty when the team declares no workspace at all, which
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
    for card in resolve_agent_cards(process.agent_cards, store):
        config = card.config
        # ``AgentCard.config`` is typed ``BaseConfig`` in core and ``tools``
        # lives on ``AgentConfig``; a card carrying the bare base declares no
        # tools and therefore no workspace.
        if not isinstance(config, AgentConfig):
            continue
        for tool in config.tools:
            layout = _declared_layout(tool)
            if layout is None:
                continue
            workspace_id, metadata_keys = layout
            path = resolve_workspace_path(
                workspace_id=workspace_id,
                workspace_metadata_keys=metadata_keys,
                team_id=str(process.team_id),
                user_id=process.user_id,
                metadata=process.metadata,
            )
            paths[path.name] = path
    return paths


def stash_workspace_paths(conn: HTTPConnection, paths: dict[str, PurePosixPath]) -> None:
    """Record the gate's resolved map for the route that opens the directory."""
    setattr(conn.state, _DECLARED_PATHS_SLOT, paths)


def stashed_workspace_paths(conn: HTTPConnection) -> dict[str, PurePosixPath] | None:
    """The map the gate recorded for this request, or ``None`` if it never ran.

    ``None`` is not a licence to resolve the id some other way: the route
    answers 404, because a request that reached the directory without the gate
    having authorized the id is exactly the fail-open this story removes.
    """
    stashed = getattr(conn.state, _DECLARED_PATHS_SLOT, None)
    if isinstance(stashed, dict):
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
    :func:`stashed_workspace_paths`: re-reading the team is a redundant query,
    not a skipped authorization. The gate answered that question already, and
    the value it stashed is the same team the fallback would fetch.
    """
    stashed = getattr(conn.state, _TEAM_PROCESS_SLOT, None)
    if isinstance(stashed, Process):
        return stashed
    return None
