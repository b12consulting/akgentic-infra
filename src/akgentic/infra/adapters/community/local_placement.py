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

# How long a duplicate creation waits for the one it is parked on. Team creation
# spawns actors and takes about a second; a creation still unfinished after this
# is stuck, and parking on it forever would stack blocked threads behind it.
_PARKED_CREATION_TIMEOUT_S = 120.0


def _team_id_conflict(detail: str) -> PlacementError:
    """A 409 for a creation key that cannot be honoured."""
    return PlacementError(detail, status_code=409, code="team_id_conflict")


class LocalPlacement:
    """Creates teams in the current process instance.

    Satisfies the PlacementStrategy protocol via structural subtyping.
    Delegates team creation to ``TeamManager`` and wraps the result
    in a ``LocalTeamHandle``.

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

    The in-flight table is guarded by a ``threading.Lock``, not an asyncio one:
    creations arrive on threadpool threads (``POST /teams`` runs through
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
        # team_id → (owner user_id, future of the handle). Present only while
        # the creation runs; removed the moment it resolves either way.
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
            pending = self._in_flight.get(team_id)
            if pending is None:
                # Checked under the lock: a creation finishing between this
                # check and the insert below would otherwise be recreated.
                if self._team_manager.get_team(team_id) is not None:
                    raise _team_id_conflict(f"Team {team_id} already exists")
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

    def _park(
        self, team_id: uuid.UUID, user_id: str, owner: str, future: Future[TeamHandle]
    ) -> TeamHandle:
        """Wait for the in-flight creation of ``team_id`` and return its handle.

        Refused outright when the creation belongs to another user: collapsing
        into it would hand one user's team to another.
        """
        if owner != user_id:
            raise _team_id_conflict(f"Team {team_id} is being created for another user")
        logger.debug("Parking duplicate creation of team %s", team_id)
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
