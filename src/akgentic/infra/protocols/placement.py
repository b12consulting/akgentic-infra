"""PlacementStrategy protocol — creates teams on worker instances.

Also home to the placement error hierarchy (the protocol's error contract).
The placement errors subclass both ``ServerError`` (so the single infra handler
maps them to HTTP statuses) and ``RuntimeError`` (so the documented contract and
any existing ``except RuntimeError`` keep holding). See ADR-031 §Decision 2.

And home to ``DeclaredWorkspaces``, the value the seam carries so that a
multi-worker tier can pin a team to the worker its shared tree hashes to
(core ADR-022 §Decision 10). The rule lives on the value, as ``routing_key()``,
so a reader of the seam finds it beside the data rather than in a decision
record, and every tier that routes refuses the one unsatisfiable shape with the
same type, the same status and the same message without importing a second
module.
"""

from __future__ import annotations

import uuid
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated, Protocol, runtime_checkable

from pydantic import AfterValidator, BaseModel, ConfigDict

from akgentic.infra.errors import ServerError
from akgentic.tool.workspace import METADATA_SCOPE

if TYPE_CHECKING:
    from akgentic.core.utils.serializer import SerializableBaseModel
    from akgentic.infra.protocols.team_handle import TeamHandle
    from akgentic.team.models import TeamCard


def _resolved_tree(path: PurePosixPath) -> PurePosixPath:
    """Admit only the resolver's shape: a relative ``<scope>/<leaf>`` path.

    ``routing_key()`` classifies a tree by ``parts[0]``, so a path with no
    parts would raise a bare ``IndexError`` there and an absolute one would be
    classed user-named and routed on. The tool resolver never produces either;
    a value built by hand in a sibling tier is refused at construction instead
    of misrouted later.
    """
    if path.is_absolute() or len(path.parts) != 2:
        msg = f"a workspace tree is a relative <scope>/<leaf> path, not {str(path)!r}"
        raise ValueError(msg)
    return path


_Tree = Annotated[PurePosixPath, AfterValidator(_resolved_tree)]


class PlacementError(ServerError, RuntimeError):
    """A team could not be placed on any worker. Defaults to 503."""

    status_code = 503
    code = "placement_failed"


class NoCapacityError(PlacementError):
    """No eligible worker had free capacity — transient, retryable.

    Attaches a default ``Retry-After`` header when the caller supplies none, so
    the 503 response tells the client to retry later.
    """

    code = "no_worker_capacity"

    def __init__(
        self,
        detail: str,
        *,
        status_code: int | None = None,
        headers: dict[str, str] | None = None,
        code: str | None = None,
    ) -> None:
        if headers is None:
            headers = {"Retry-After": "30"}
        super().__init__(detail, status_code=status_code, headers=headers, code=code)


class NoSandboxCapacityError(NoCapacityError):
    """A sandbox-requiring team found no sandbox-capable worker."""

    code = "no_sandbox_capacity"


class WorkerRejectedError(PlacementError):
    """The selected worker returned a non-2xx from team creation — upstream fault."""

    status_code = 502
    code = "worker_rejected"


class UnroutableWorkspacesError(PlacementError):
    """A team declares more than one ``_meta/`` tree, so no worker can hold all of it.

    Rule 3 of the affinity rule: two shared-across-users trees on one team is the
    genuinely unsatisfiable case — a team on both P and Q while other teams sit
    on each — and no assignment keeps every tree under one writer. Refusing
    loudly beats breaking the invariant silently.

    **409, not the placement default of 503.** The condition is a property of the
    team's definition, not of the cluster's capacity, so a ``Retry-After`` would
    be a lie; 409 is what this package already answers for a stored definition
    the tier cannot honour. No header is attached.

    Raised by :meth:`DeclaredWorkspaces.routing_key`, which only a tier that
    routes calls. The community tier has one worker and no routing, so it never
    raises this.
    """

    status_code = 409
    code = "workspace_affinity_unsatisfiable"


class DeclaredWorkspaces(BaseModel):
    """The trees a team declares, resolved before dispatch, and the rule that pins them.

    ``shared`` holds every tree more than one team can reach — a
    ``<user_id>/<workspace_id>`` tree shared across one user's teams, and a
    ``_meta/…`` tree shared across teams and users. ``own`` holds the team's
    private ``<user_id>/<team_id>`` tree when the caller supplied a ``team_id``,
    and is ``None`` when it did not: there is no id to build the path from
    before ``TeamManager`` mints one, and nothing routes on it either way.

    A pure function of the request's own inputs, so any replica computes the
    same value for the same team with no shared memory, no sticky session and
    nothing in ``app.state``. Frozen, so a tier cannot edit it on its way
    through; a ``frozenset`` rather than a leaf-keyed map, so two cards resolving
    to one leaf on two scopes cannot collapse into one entry; every path checked
    at construction to be a relative two-segment tree, the only shape the
    resolver produces and the only one ``routing_key()`` can classify.
    """

    model_config = ConfigDict(frozen=True)

    shared: frozenset[_Tree] = frozenset()
    own: _Tree | None = None

    def routing_key(self) -> PurePosixPath | None:
        """The tree a multi-worker router pins this team to, or ``None`` to place freely.

        Core ADR-022 §Decision 10, as amended, in code:

        1. Only a tree more than one team can reach needs affinity, so only
           ``shared`` is consulted. ``own`` is reachable by exactly one team by
           construction and is never a key — a router that hashed a missing key
           would herd every default team onto one worker.
        2. A ``_meta/`` tree wins, whatever else the team declares. With no
           ``_meta/`` tree the team routes on its user-named tree, and when it
           declares several, on the lexicographically smallest of them, so that
           every replica agrees with no state. With nothing shareable, ``None``.
        3. More than one ``_meta/`` tree is refused.

        Raises:
            UnroutableWorkspacesError: When more than one ``_meta/`` tree is
                declared. The community tier never calls this method.
        """
        metadata_trees = sorted(
            (path for path in self.shared if path.parts[0] == METADATA_SCOPE), key=str
        )
        if len(metadata_trees) > 1:
            named = ", ".join(str(path) for path in metadata_trees)
            msg = (
                f"a team may declare at most one {METADATA_SCOPE}/ workspace to be placed on "
                f"one worker; this one declares {len(metadata_trees)}: {named}"
            )
            raise UnroutableWorkspacesError(msg)
        if metadata_trees:
            return metadata_trees[0]
        user_named = [path for path in self.shared if path.parts[0] != METADATA_SCOPE]
        if user_named:
            return min(user_named, key=str)
        return None


@runtime_checkable
class PlacementStrategy(Protocol):
    """Creates a team on a selected worker instance and returns a handle.

    Encapsulates worker selection and team creation so that ``TeamService``
    never needs to know about ``TeamManager`` or actor internals.

    Worker selection semantics vary by tier:

    - **Community** (``LocalPlacement``): single-process — always places on the
      local ``TeamManager`` instance. No network involved.
    - **Department** (``HttpPlacement``): selects the least-loaded eligible worker
      (lowest ``active_teams / max_teams`` ratio) via the service registry, honours
      the ``sandbox`` label, then creates the team with an HTTP ``POST /teams`` to
      the chosen worker.
    - **Enterprise** (``DaprPlacement``): runs a ``LabelMatchPlacement`` →
      ``WeightedPlacement`` → ``ZoneAwarePlacement`` pipeline (filter → rank →
      prefer) to select a worker, then creates the team via Dapr service invocation
      to the worker's ``POST /teams`` endpoint.

    Affinity contract (core ADR-022 §Decision 10):
        ``workspaces`` carries the team's declared workspace trees, resolved by
        ``TeamService`` from the request's own inputs before any worker is
        contacted. A multi-worker tier calls ``workspaces.routing_key()`` and
        pins the team to the worker that tree hashes to, so that one tree has
        one host and one ``#Workspace`` across the cluster; the refusal it
        raises, ``UnroutableWorkspacesError``, must propagate unchanged — it is
        already a ``PlacementError`` the single handler maps to 409. The
        community tier accepts the value for signature parity and ignores it:
        one process is one worker, so affinity holds by construction and it
        refuses nothing. This package proves the resolution and the rule; it
        cannot prove two teams landing on one worker, which needs a deployment.

    Error contract:
        Raises ``PlacementError`` (a ``ServerError``, and — for backward
        compatibility — a ``RuntimeError``) if no healthy worker is available
        or team creation fails on the selected worker. Callers must not retry
        automatically — surface the error to the user.
    """

    def create_team(
        self,
        team_card: TeamCard,
        user_id: str,
        user_email: str = "",
        team_id: uuid.UUID | None = None,
        catalog_namespace: str | None = None,
        metadata: SerializableBaseModel | None = None,
        workspaces: DeclaredWorkspaces | None = None,
    ) -> TeamHandle:
        """Create a team on a worker instance and return a handle.

        Args:
            team_card: Team configuration card.
            user_id: ID of the user creating the team.
            user_email: Email of the user creating the team.
            team_id: Optional caller-supplied team identifier. When omitted, the
                underlying TeamManager generates a fresh UUID.
            catalog_namespace: Opaque tag identifying the catalog namespace
                the team was instantiated from. Forwarded through to
                ``TeamManager.create_team`` (community tier) or the remote
                worker (department / enterprise tiers). ``None`` for teams
                not sourced from a v2 catalog namespace.
            metadata: Business metadata for the team, already validated by the
                caller against ``team_card.metadata_type``. Forwarded so it lands
                on the persisted ``Process.metadata``; the derived index is
                computed once inside ``akgentic-team``, never here. Optional and
                defaulting to ``None``, so existing callers are unaffected.
            workspaces: The team's declared workspace trees, resolved by the
                caller before dispatch. A tier that routes reads
                ``routing_key()`` off it; the community tier ignores it. Optional
                and defaulting to ``None`` so existing callers are unaffected;
                ``None`` means "the caller did not resolve", which a memoryless
                router treats as "place freely" and which ``TeamService`` never
                sends.

        Returns:
            A TeamHandle for interacting with the newly created team.

        Raises:
            PlacementError: If no worker is available or team creation fails.
                A ``ServerError``, and — for backward compatibility — a
                ``RuntimeError``. ``UnroutableWorkspacesError`` is the member a
                routing tier raises for a team no single worker can hold.
        """
        ...
