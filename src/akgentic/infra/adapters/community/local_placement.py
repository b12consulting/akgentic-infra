"""LocalPlacement — community-tier placement that creates teams in the current process."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from akgentic.infra.adapters.community.local_team_handle import LocalTeamHandle
from akgentic.infra.protocols.placement import PlacementError
from akgentic.team.manager import TeamManager
from akgentic.team.ports import ServiceRegistry

if TYPE_CHECKING:
    from akgentic.core.utils.serializer import SerializableBaseModel
    from akgentic.infra.protocols.placement import DeclaredWorkspaces
    from akgentic.infra.protocols.team_handle import TeamHandle
    from akgentic.team.models import TeamCard


logger = logging.getLogger(__name__)


class LocalPlacement:
    """Creates teams in the current process instance.

    Satisfies the PlacementStrategy protocol via structural subtyping.
    Delegates team creation to ``TeamManager`` and wraps the result
    in a ``LocalTeamHandle``.

    One process is one worker, so the affinity the protocol's ``workspaces``
    value exists for holds by construction here: the value is accepted for
    signature parity and consulted for nothing, ``routing_key()`` is never
    called, and the community tier refuses nothing — a team a multi-worker
    tier would refuse for declaring two ``_meta/`` trees is simply created.
    """

    def __init__(
        self,
        team_manager: TeamManager,
        service_registry: ServiceRegistry,
    ) -> None:
        self._instance_id = uuid.uuid4()
        self._team_manager = team_manager
        self._service_registry = service_registry

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
        workspaces: DeclaredWorkspaces | None = None,
    ) -> TeamHandle:
        """Create a team in the local process and return a handle.

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
            workspaces: The team's resolved workspace trees. Accepted so this
                adapter's signature is the protocol's, and **not forwarded**:
                ``TeamManager.create_team`` has no such parameter, and with one
                worker there is nothing to honour. Logged at DEBUG only — the
                paths carry user ids.

        Returns:
            A LocalTeamHandle for interacting with the newly created team.
        """
        logger.debug(
            "LocalPlacement creating team: user_id=%s, catalog_namespace=%s, team_id=%s, "
            "workspaces=%s",
            user_id,
            catalog_namespace,
            team_id,
            workspaces,
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
