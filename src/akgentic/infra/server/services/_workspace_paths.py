"""The pre-dispatch workspace resolver — the seam's value, from the request's own inputs.

**Why this is not ``routes/_workspace_resolution.declared_workspace_paths``.**
That function takes a persisted ``Process`` and an ``EventStore`` and resolves
*hired* cards through ``resolve_agent_cards``. At dispatch there is no
``Process`` — choosing a worker before ``placement.create_team`` builds one is
the whole point. The inputs that do exist are the four ``TeamService.create_team``
already holds: the team card, the owner's ``user_id``, the optional
caller-supplied ``team_id`` and the validated metadata. Every one of them is
exactly what the route-side resolver reads off the ``Process`` later, which is
what lets the two agree.

**Why the card list is the projection's.** ``derive_team_projection(card).cards``
is the deduplicated list ``TeamManager.create_team`` saves to the store and the
refs ``resolve_agent_cards`` loads back — ``agent_profiles`` included, and a
profile card overriding a tree card of the same role. ``TeamCard.agent_cards``
walks the tree only and omits the profiles, and a hireable profile's
``WorkspaceTool`` binds on the team's worker when hired, so it needs affinity
too. Using the projection makes the two resolvers agree on *which* cards by
construction rather than by a second walk of the card tree.

**What is shared with the route side, exactly.** The card-shape read — which
tools declare a workspace, and with which two fields — is :func:`declared_layouts`,
one generator both resolvers iterate. Their loops legitimately differ in what
they collect (a leaf-keyed map that must include the team's own tree, versus a
set of shareable trees plus the own tree when it exists), so a single shared
loop would need a switch, and a switch is where the two would drift.

**Why ``own`` may be ``None``.** A default ``WorkspaceTool()`` resolves to
``<user_id>/<team_id>``, and when the caller supplied no ``team_id`` the
identifier does not exist until ``TeamManager`` mints one. That tree is
reachable by exactly one team by construction, so by rule 1 of the affinity
rule it never routes on path; the missing key is structurally absent from the
only field ``routing_key()`` consults rather than a ``None`` a router could hash.

**Why the result is a set.** The route-side map keys by leaf and keeps whichever
card the store returned last (backlog row 30). Carried on the seam that would
let a user-named path shadow a ``_meta/`` path and silently break rule 2; a set
of paths cannot collapse two scopes into one leaf.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from akgentic.agent.config import AgentConfig
from akgentic.infra.errors import WorkspaceDeclarationError
from akgentic.infra.protocols.placement import DeclaredWorkspaces
from akgentic.team import derive_team_projection
from akgentic.tool.workspace import WorkspaceTool, resolve_workspace_path

if TYPE_CHECKING:
    from akgentic.core.agent_card import AgentCard
    from akgentic.core.utils.serializer import SerializableBaseModel
    from akgentic.team.models import TeamCard

__all__ = ["declared_layouts", "declared_workspaces"]


def declared_layouts(card: AgentCard) -> Iterator[tuple[str | None, list[str]]]:
    """``(workspace_id, workspace_metadata_keys)`` for every workspace-declaring tool on *card*.

    ``WorkspaceTool`` is the **only** card that declares a workspace. Sandboxed
    execution is one of its capabilities (``workspace_exec=...``), not a card of
    its own: a shell-only agent is a ``WorkspaceTool`` with every file
    capability off and ``workspace_exec`` on, and it declares its directory
    through the same two fields as any other. There is no second shape to read,
    and a blanket ``getattr(tool, "workspace_metadata_keys", [])`` would only
    swallow a ``WorkspaceTool`` that lost the field — so the type is checked and
    the fields are read directly.

    ``AgentCard.config`` is typed ``BaseConfig`` in core and ``tools`` lives on
    ``AgentConfig``; a card carrying the bare base declares no tools and
    therefore no workspace. The two fields are mutually exclusive at card
    construction, so a layout never carries both and needs no precedence.
    """
    config = card.config
    if not isinstance(config, AgentConfig):
        return
    for tool in config.tools:
        if isinstance(tool, WorkspaceTool):
            yield tool.workspace_id, tool.workspace_metadata_keys


def declared_workspaces(
    team_card: TeamCard,
    *,
    user_id: str,
    team_id: uuid.UUID | None,
    metadata: SerializableBaseModel | None,
) -> DeclaredWorkspaces:
    """Resolve every workspace *team_card* declares, from the request's inputs alone.

    Args:
        team_card: The team about to be created, as resolved from the catalog.
        user_id: The **owner's** principal — the ``<scope>`` of every per-user
            layout. Pre-dispatch the caller and the owner coincide, but the
            route-side resolver's whole correctness argument rests on the
            distinction, so it is named for what it is.
        team_id: The caller-supplied team identifier, or ``None`` when
            ``TeamManager`` will mint one. Read only for the default layout.
        metadata: The team's metadata, already validated against the card's
            ``metadata_type``. Consulted only by a metadata-keyed layout.

    Returns:
        The shareable trees in ``shared`` and, when *team_id* was supplied and a
        default card exists, the team's own tree in ``own``.

    Raises:
        WorkspaceDeclarationError: For any ``ValueError`` the tool-side resolver
            raises — a metadata card the metadata cannot satisfy, an unusable
            owner id, an unsafe leaf. Raised before placement runs, so nothing
            is created; the resolver's message is carried verbatim.
    """
    shared: set[PurePosixPath] = set()
    own: PurePosixPath | None = None
    for card in derive_team_projection(team_card).cards:
        for workspace_id, metadata_keys in declared_layouts(card):
            is_default = workspace_id is None and not metadata_keys
            if is_default and team_id is None:
                # A default card has no team_id before TeamManager mints one, so
                # it has no path before dispatch — and by rule 1 it needs none,
                # because <user_id>/<team_id> is reachable by exactly one team.
                # Skipped rather than keyed on a placeholder: a router that
                # hashed a missing key would herd every default team together.
                continue
            try:
                path = resolve_workspace_path(
                    workspace_id=workspace_id,
                    workspace_metadata_keys=metadata_keys,
                    # The resolver reads team_id only for the default layout,
                    # which this branch never asks for without a real id.
                    team_id="" if team_id is None else str(team_id),
                    user_id=user_id,
                    metadata=metadata,
                )
            except ValueError as exc:
                raise WorkspaceDeclarationError(str(exc)) from exc
            if is_default:
                own = path
            else:
                shared.add(path)
    return DeclaredWorkspaces(shared=frozenset(shared), own=own)
