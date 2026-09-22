"""LocalPlacement — community-tier placement that creates teams in the current process."""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import Future
from typing import TYPE_CHECKING

from akgentic.infra.adapters.community.local_team_handle import LocalTeamHandle
from akgentic.infra.protocols.placement import PlacementError
from akgentic.team.manager import TeamManager
from akgentic.team.ports import ServiceRegistry

if TYPE_CHECKING:
    from akgentic.core.utils.serializer import SerializableBaseModel
    from akgentic.infra.protocols.team_handle import TeamHandle
    from akgentic.team.models import TeamCard


logger = logging.getLogger(__name__)

# How long a duplicate waits for the claim it is parked on — a creation or a
# resume alike. Both spawn actors and take about a second; one still unfinished
# after this is stuck, and parking on it forever would stack blocked threads
# behind it.
_PARKED_CREATION_TIMEOUT_S = 120.0


def _team_id_conflict(detail: str) -> PlacementError:
    """A 409 for a creation key that cannot be honoured."""
    return PlacementError(detail, status_code=409, code="team_id_conflict")


class LocalPlacement:
    """Places teams in the current process instance — new ones and returning ones.

    Satisfies the PlacementStrategy protocol via structural subtyping.
    Delegates team creation and resumption to ``TeamManager`` and wraps the
    result in a ``LocalTeamHandle``.

    A caller-supplied ``team_id`` is a **creation key, never an address**, and
    this is where community enforces that contract — every creation passes
    through here, channel initiations and ``POST /teams`` alike:

    - unknown key → created, and recorded as *in flight* while it runs;
    - key in flight **for the same user** → the duplicate is *parked* on the
      first creation's future and receives the very same handle, or the same
      exception if it fails. Concurrent initiations of one conversation thus
      yield one team;
    - key in flight for a **different** user, or naming a team that **already
      exists** → 409. Returning it would let a payload bind a chat to any team
      it can name; recreating it would overwrite a live team, because
      ``TeamManager.create_team`` performs no duplicate check of its own.

    **One table serves both operations.** A resume claims the same ``team_id``
    key the same way, so two concurrent resumes of one stopped team start one
    runtime and share a handle (ADR-045 §D4). There is deliberately no second
    table and no second park helper — see the invariant recorded at
    ``_in_flight`` for why creations and resumes cannot collide over a key.

    The in-flight table is guarded by a ``threading.Lock``, not an asyncio one:
    claims arrive on threadpool threads (``POST /teams`` runs through
    ``asyncio.to_thread``), so the guard must hold across threads. It lives in
    the placement rather than the server, which must keep no cross-request
    state — community is a single process, and the table is exactly as
    process-local as the ``TeamManager`` it fronts.
    """

    def __init__(
        self,
        team_manager: TeamManager,
        service_registry: ServiceRegistry,
    ) -> None:
        self._instance_id = uuid.uuid4()
        self._team_manager = team_manager
        self._service_registry = service_registry
        # team_id → (owner user_id, future of the handle). Present only while a
        # claim on that key runs; removed the moment it resolves either way.
        #
        # Creations and resumes share this one table, and a create and a resume
        # can never contend for the same key — by construction, not by luck:
        # ``create_team`` refuses outright any key naming a team that already
        # exists, and a resume only ever reaches this placement for a team that
        # *does* exist and is stopped (``TeamService.restore_team`` reads the
        # persisted status first, and ``LocalRuntimeCache.warm()`` resumes teams
        # the event store just returned). So the create↔resume branch is
        # unreachable, which is what makes one table safe. If a later change
        # makes it reachable — a create that no longer refuses an existing team,
        # or a resume that can reach placement for a team that does not exist —
        # this analysis has to be redone before the table can stay shared.
        self._in_flight: dict[uuid.UUID, tuple[str, Future[TeamHandle]]] = {}
        self._in_flight_lock = threading.Lock()

    @property
    def instance_id(self) -> uuid.UUID:
        """The worker instance ID representing this process."""
        return self._instance_id

    def create_team(
        self,
        team_card: TeamCard,
        user_id: str,
        user_email: str = "",
        team_id: uuid.UUID | None = None,
        catalog_namespace: str | None = None,
        metadata: SerializableBaseModel | None = None,
    ) -> TeamHandle:
        """Create a team in the local process, honouring ``team_id`` as a creation key.

        See the class docstring for the key's contract. Without a ``team_id``
        there is nothing to collapse, and the team is created straight away.

        Raises:
            PlacementError: 409 ``team_id_conflict`` when ``team_id`` names an
                existing team or one being created for another user; 503 when a
                parked creation outlives ``_PARKED_CREATION_TIMEOUT_S``; the
                underlying failure, re-raised to every parked duplicate, when the
                creation they waited on fails.
        """
        if team_id is None:
            return self._create(
                team_card, user_id, user_email, team_id, catalog_namespace, metadata
            )

        with self._in_flight_lock:
            # Checked under the lock, and before the in-flight lookup: a
            # creation finishing between this check and the insert below would
            # otherwise be recreated, and a key naming an existing team must be
            # refused whatever else holds it — including an in-flight resume of
            # that very team, which a same-owner create would otherwise park on
            # and be handed the resumed team's handle. A creation in flight is
            # invisible here, because ``TeamManager.create_team`` persists the
            # ``Process`` only once the build has succeeded; duplicates still
            # park for the whole of it.
            if self._team_manager.get_team(team_id) is not None:
                raise _team_id_conflict(f"Team {team_id} already exists")
            pending = self._in_flight.get(team_id)
            if pending is None:
                future: Future[TeamHandle] = Future()
                self._in_flight[team_id] = (user_id, future)
                is_creator = True
            else:
                is_creator = False

        if not is_creator:
            assert pending is not None
            return self._park(team_id, user_id, *pending)

        try:
            handle = self._create(
                team_card, user_id, user_email, team_id, catalog_namespace, metadata
            )
        except BaseException as exc:
            future.set_exception(exc)
            raise
        else:
            future.set_result(handle)
            return handle
        finally:
            with self._in_flight_lock:
                self._in_flight.pop(team_id, None)

    def resume_team(self, team_id: uuid.UUID) -> TeamHandle:
        """Resume a stopped team in the local process, collapsing concurrent attempts.

        Placement decides where a returning team runs, exactly as it does for a
        new one (ADR-045 §D3); community always answers "here". Concurrent
        resumes of one team park on the same in-flight table creations use, so
        one runtime starts and both callers get the same handle.

        A resume has no *requesting* user, so the entry records the team's own
        persisted owner: the resumer and every duplicate parked behind it then
        present the same value, ``_park`` stays a single helper, and creation's
        refusal of a key in flight for a different user is untouched.

        Failures are **not** translated into ``PlacementError``. ``TeamService``
        and the worker route classify a resume failure by its message — "not
        found" / "deleted" → 404, anything else → 409 — and a ``PlacementError``
        is a ``ServerError`` the single infra handler would answer 503 to
        instead. The move is a relocation, not a redesign of the error path.

        Raises:
            ValueError: Propagated unchanged from ``TeamManager.resume_team``
                when the team is unknown, already running, or deleted.
            PlacementError: 503 only, when a parked duplicate outlives
                ``_PARKED_CREATION_TIMEOUT_S``.
        """
        process = self._team_manager.get_team(team_id)
        owner = process.user_id if process is not None else ""

        with self._in_flight_lock:
            pending = self._in_flight.get(team_id)
            if pending is None:
                future: Future[TeamHandle] = Future()
                self._in_flight[team_id] = (owner, future)
                is_resumer = True
            else:
                is_resumer = False

        if not is_resumer:
            assert pending is not None
            return self._park(team_id, owner, *pending)

        logger.debug("LocalPlacement resuming team: team_id=%s", team_id)
        try:
            runtime = self._team_manager.resume_team(team_id)
        except BaseException as exc:
            future.set_exception(exc)
            raise
        else:
            handle = LocalTeamHandle(runtime)
            future.set_result(handle)
            return handle
        finally:
            with self._in_flight_lock:
                self._in_flight.pop(team_id, None)

    def _park(
        self, team_id: uuid.UUID, user_id: str, owner: str, future: Future[TeamHandle]
    ) -> TeamHandle:
        """Wait for the in-flight claim on ``team_id`` and return its handle.

        Refused outright when the claim belongs to another user: collapsing into
        it would hand one user's team to another.
        """
        if owner != user_id:
            raise _team_id_conflict(f"Team {team_id} is being created for another user")
        logger.debug("Parking duplicate claim on team %s", team_id)
        try:
            return future.result(timeout=_PARKED_CREATION_TIMEOUT_S)
        except TimeoutError as exc:
            msg = (
                f"Creation of team {team_id} did not finish within "
                f"{_PARKED_CREATION_TIMEOUT_S:.0f}s"
            )
            raise PlacementError(msg) from exc

    def _create(
        self,
        team_card: TeamCard,
        user_id: str,
        user_email: str = "",
        team_id: uuid.UUID | None = None,
        catalog_namespace: str | None = None,
        metadata: SerializableBaseModel | None = None,
    ) -> TeamHandle:
        """Create the team — no key handling; ``create_team`` does that.

        Args:
            team_card: Team configuration card.
            user_id: ID of the user creating the team.
            user_email: Email of the user creating the team.
            team_id: Optional caller-supplied team identifier; TeamManager
                auto-generates a UUID when None.
            catalog_namespace: Opaque tag identifying the catalog namespace
                the team was instantiated from. Forwarded verbatim to
                ``TeamManager.create_team`` so the persisted ``Process``
                records it. ``None`` for teams not sourced from a catalog.
            metadata: Pre-validated business metadata, forwarded verbatim to
                ``TeamManager.create_team`` so it lands on the persisted
                ``Process.metadata`` alongside its derived index.

        Returns:
            A LocalTeamHandle for interacting with the newly created team.
        """
        logger.debug(
            "LocalPlacement creating team: user_id=%s, catalog_namespace=%s, team_id=%s",
            user_id,
            catalog_namespace,
            team_id,
        )
        try:
            runtime = self._team_manager.create_team(
                team_card,
                user_id,
                user_email=user_email,
                team_id=team_id,
                catalog_namespace=catalog_namespace,
                metadata=metadata,
            )
        except PlacementError:
            # Already typed — re-raise so the create-path surfaces the carried
            # HTTP mapping unchanged.
            raise
        except Exception as exc:
            # Community is single-process: the create-path failure is
            # TeamManager.create_team raising, not capacity exhaustion. Surface
            # it as a PlacementError (a ServerError) so the single handler maps
            # it instead of leaking a bare exception (ADR-031 §Decision 4).
            msg = f"Local team creation failed: {exc}"
            raise PlacementError(msg) from exc
        logger.debug("Team created locally: team_id=%s", runtime.id)
        return LocalTeamHandle(runtime)
