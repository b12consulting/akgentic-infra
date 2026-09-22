"""PlacementStrategy protocol — creates teams on worker instances.

Also home to the placement error hierarchy (the protocol's error contract).
The placement errors subclass both ``ServerError`` (so the single infra handler
maps them to HTTP statuses) and ``RuntimeError`` (so the documented contract and
any existing ``except RuntimeError`` keep holding). See ADR-031 §Decision 2.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from akgentic.infra.errors import ServerError

if TYPE_CHECKING:
    from akgentic.core.utils.serializer import SerializableBaseModel
    from akgentic.infra.protocols.team_handle import TeamHandle
    from akgentic.team.models import TeamCard


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
    ) -> TeamHandle:
        """Create a team on a worker instance and return a handle.

        **``team_id`` is a creation key, never an address.** It exists so that
        concurrent creations of one logical team — two deliveries racing to start
        the same conversation's team, a client retrying a ``POST /teams`` — yield
        one team instead of several. It can never be used to reach a team that
        already exists. Every implementation MUST honour this, however it
        achieves it (community parks duplicates in-process; a distributed tier
        needs an atomic claim in a shared store):

        - **key unknown** → create the team under that id, owned by ``user_id``;
        - **key being created right now, for the same** ``user_id`` → wait for
          that creation and return a handle to the same team; if it fails, fail
          the same way;
        - **key being created for a different user, or naming a team that
          already exists** → raise ``PlacementError`` with ``status_code=409``
          and ``code="team_id_conflict"``. Never return that team and never
          recreate it.

        Why the refusals are not optional:

        - Returning an existing team would let any caller who can name an id —
          including an unauthenticated channel payload — attach itself to a team
          it does not own.
        - Recreating it overwrites a live team: ``TeamManager.create_team``
          performs no duplicate check of its own.
        - Collapsing into another user's in-flight creation would hand one
          user's team to another.

        A late duplicate — one arriving after the first creation has finished —
        is therefore refused, not served; a caller that wants idempotency past
        completion must resolve the existing team itself, as the owner.

        Args:
            team_card: Team configuration card.
            user_id: ID of the user creating the team.
            user_email: Email of the user creating the team.
            team_id: Optional **creation key** — see below. When omitted, a
                fresh UUID is generated and nothing is collapsed.
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

        Returns:
            A TeamHandle for interacting with the newly created team.

        Raises:
            PlacementError: 409 ``team_id_conflict`` per the creation-key rules
                above; otherwise if no worker is available or creation fails.
                A ``ServerError``, and — for backward compatibility — a
                ``RuntimeError``.
        """
        ...
