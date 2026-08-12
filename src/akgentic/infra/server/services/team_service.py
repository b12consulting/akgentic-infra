"""TeamService — orchestrates catalog resolution and team lifecycle via protocols."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.core.messages.orchestrator import SentMessage
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.infra.errors import PlacementConsistencyError
from akgentic.infra.protocols.event_stream import EventStream
from akgentic.infra.protocols.runtime_cache import RuntimeCache
from akgentic.infra.protocols.team_handle import TeamHandle
from akgentic.infra.server.deps import TierServices
from akgentic.infra.server.services._metadata_payload import validate_metadata
from akgentic.team.models import AgentStateSnapshot, PersistedEvent, Process, TeamStatus

if TYPE_CHECKING:
    from akgentic.core.messages.message import Message

logger = logging.getLogger(__name__)

# Maximum page size for GET /teams; the default is 250 (ADR-032 §Decision 1).
MAX_PAGE_SIZE = 500


def _remove_workspace_dir(workspaces_root: Path, team_id: uuid.UUID) -> None:
    """Best-effort removal of a team's workspace directory.

    Removes ``{workspaces_root}/{team_id}`` recursively. A missing directory
    is a silent no-op (ephemeral teams that never invoked a ``Filesystem``
    write have no directory to clean). Any ``shutil.rmtree`` failure is logged
    at WARNING and suppressed so team deletion still completes in the system
    of record — a later janitor pass can sweep orphans.

    Generalized from akgentic-infra-enterprise's
    ``routes/enterprise_server_teams.py`` per Epic 24 (Tier-Alignment Fixes
    from Department + Enterprise); see ADR-022 §D7 for the original
    best-effort, log-not-raise rationale.
    """
    target = workspaces_root / str(team_id)
    if not target.exists():
        return
    try:
        shutil.rmtree(target)
    except Exception as exc:  # noqa: BLE001 — log-not-raise; cleanup is best-effort
        logger.warning(
            "Workspace cleanup failed — team_id=%s error=%s",
            team_id,
            exc,
        )


class TeamService:
    """Service layer bridging catalog resolution with team lifecycle management.

    Resolves catalog entry IDs to TeamCards, delegates lifecycle operations
    through PlacementStrategy and WorkerHandle protocols, and queries
    EventStore for listing. Delegates runtime interaction through
    RuntimeCache/TeamHandle protocols.
    """

    def __init__(self, services: TierServices, *, workspaces_root: Path) -> None:
        """Construct a TeamService.

        Args:
            services: Pre-wired tier services container.
            workspaces_root: Server-side root directory under which each
                team's workspace lives at ``{workspaces_root}/{team_id}/``.
                Used by ``delete_team`` for best-effort FS cleanup.
        """
        self._services = services
        self._cache: RuntimeCache = services.runtime_cache
        self._workspaces_root = workspaces_root

    def create_team(
        self,
        catalog_namespace: str,
        user_id: str,
        user_email: str = "",
        team_id: uuid.UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Process:
        """Resolve a catalog namespace to a TeamCard and create a running team.

        Loads the team definition via the v2 unified ``Catalog.load_team``
        API and forwards the namespace tag through placement so that the
        persisted ``Process.catalog_namespace`` records the binding.

        Args:
            catalog_namespace: v2 catalog namespace holding exactly one
                ``kind="team"`` entry.
            user_id: Identifier of the user creating the team.
            user_email: Email of the user creating the team.
            team_id: Optional caller-supplied team identifier; the placement
                layer auto-generates a UUID when None.
            metadata: Optional plain-JSON business metadata. Validated against
                the ``metadata_type`` the resolved card declares — the client
                never names the type — and forwarded down the create path so it
                lands on the persisted ``Process.metadata``. The derived index
                is computed once, inside ``akgentic-team``, never here.

        Returns:
            The persisted ``Process`` for the newly created team.

        Raises:
            EntryNotFoundError: If ``catalog_namespace`` has no team entry.
                ``Catalog.load_team`` surfaces the condition as
                ``CatalogValidationError``; this layer translates it so
                the existing teams router's ``EntryNotFoundError → 404``
                handler applies unchanged.
            MetadataValidationError: If ``metadata`` carries a ``__model__`` key,
                is supplied for a card declaring no contract, or fails the
                declared schema. Raised before placement runs, so a rejected
                body never leaves a half-created team behind.
        """
        logger.debug("Resolving team for catalog namespace: %s", catalog_namespace)
        try:
            team_card = self._services.catalog.load_team(catalog_namespace)
        except CatalogValidationError as exc:
            # Translate v2's validation error into the existing 404-mapped
            # exception so the teams router's error-handling stays a no-op
            # for this story (Story 18.3 consolidates error handling).
            raise EntryNotFoundError(catalog_namespace) from exc
        # Before placement, never after: nothing is created when this rejects.
        validated_metadata = validate_metadata(team_card.metadata_type, metadata)
        handle = self._services.placement.create_team(
            team_card,
            user_id,
            user_email=user_email,
            team_id=team_id,
            catalog_namespace=catalog_namespace,
            metadata=validated_metadata,
        )
        self._cache.store(handle.team_id, handle)
        # Consistency invariant: create_team() writes to event store, so
        # get_team() must find it immediately. If this fires, there is a bug
        # in placement or event store — not a transient race condition.
        process = self._services.worker_handle.get_team(handle.team_id)
        if process is None:  # pragma: no cover
            msg = f"Team {handle.team_id} was created but not found in event store"
            raise PlacementConsistencyError(msg)
        logger.info(
            "Team created: team_id=%s, catalog_namespace=%s",
            process.team_id,
            catalog_namespace,
        )
        return process

    def list_teams(
        self,
        *,
        user_id: str,
        status: TeamStatus | None = None,
        metadata: dict[str, str] | None = None,
        page: int = 1,
        size: int = 250,
    ) -> tuple[list[Process], int]:
        """Return one numbered page of the user's teams plus the filtered count.

        Phase 1: the store returns the matching set, sorted ``created_at DESC,
        team_id DESC`` and sliced here (ADR-032 §Decision 2). Stateless — a pure
        function of ``user_id`` + ``status`` + ``metadata`` + ``page`` + ``size``
        + store contents. An out-of-range page yields an empty list with the
        correct total.

        Every filter pushes into the EventStore rather than loading the user's
        teams into Python and filtering here, so per-request cost scales with
        the answer rather than with the archive — and so the total, counted from
        what the store returned, is the FILTERED count on every page rather than
        a count of the set the filter was drawn from. Nothing filters after the
        slice; that ordering is what keeps pages contiguous.

        ``status=None`` and ``metadata=None`` each mean *no such filter*, so a
        caller passing neither gets exactly the result set it got before. Both
        are forwarded unconditionally: a branch that omits a kwarg when it is
        ``None`` is how a filter later gets silently dropped. ``metadata``
        values travel verbatim — index derivation and ``|`` escaping happen once,
        inside ``akgentic-team`` (ADR-24 §D4).

        Neither filter replaces the owner filter: they only narrow *within* the
        user's teams. Metadata is caller-supplied and non-secret, so allowing it
        to widen the set — or the count — would make this a cross-tenant
        enumeration primitive. See team-package ADR-16 (owner), ADR-23
        (lifecycle state) and ADR-24 (metadata) for the Protocol changes.
        """
        rows = self._services.event_store.list_teams(
            user_id=user_id, status=status, metadata=metadata
        )
        rows.sort(key=lambda p: (p.created_at, p.team_id), reverse=True)
        total = len(rows)
        size = max(1, min(size, MAX_PAGE_SIZE))
        page = max(1, page)
        start = (page - 1) * size
        return rows[start : start + size], total

    def get_team(self, team_id: uuid.UUID) -> Process | None:
        """Get a single team by ID."""
        return self._services.worker_handle.get_team(team_id)

    def update_team_metadata(
        self, team_id: uuid.UUID, raw: dict[str, Any]
    ) -> SerializableBaseModel | None:
        """Replace a team's business metadata with a complete new document.

        Validation happens here, against the ``metadata_type`` the team's card
        declared **at creation** and carries on the persisted ``Process`` — not
        against the catalog entry as it stands now, which may have been edited
        since. The type cannot change for a live team (ADR-24 §D7), so
        re-resolving it would let a catalog edit silently change what an
        existing team accepts.

        The write itself belongs to ``akgentic-team``: validate → one database
        write of the value and its re-derived index → best-effort push to a live
        orchestrator. This layer adds nothing around it — no cache write, no
        event publish, no re-read to "confirm", and no inspection of the push
        outcome. The database is the system of record, so a failed push is not
        an error: the index stays truthful and the actor repopulates from the
        ``Process`` on its next resume.

        Args:
            team_id: The team whose metadata is being replaced.
            raw: The complete plain-JSON document. An empty dict clears the
                team's metadata.

        Returns:
            The metadata carried on the ``Process`` the write path returned —
            what was persisted, not what was sent.

        Raises:
            ValueError: If the team is unknown or has been deleted. The message
                carries ``not found`` for the unknown case, which the router's
                ``_raise_action_error`` maps to 404.
            MetadataValidationError: If the body carries a ``__model__`` key at
                any depth, if the team declares no metadata contract, or if the
                body fails the declared schema. Raised before the write path is
                reached, so a rejected body changes nothing.
        """
        process = self._services.worker_handle.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        validated = validate_metadata(process.team_card.metadata_type, raw)
        updated = self._services.worker_handle.update_team_metadata(team_id, validated)
        logger.info("Team metadata updated: team_id=%s", team_id)
        return updated.metadata

    def delete_team(self, team_id: uuid.UUID) -> None:
        """Stop (if running) and delete a team.

        After the team is removed from the system of record, the team's
        workspace directory (``{workspaces_root}/{team_id}/``) is removed on a
        best-effort basis — a missing directory or an ``rmtree`` failure does
        not prevent deletion from completing.

        Raises:
            ValueError: If team not found or already deleted. Raised before
                any filesystem work, so a missing team never triggers FS
                cleanup.
        """
        process = self._services.worker_handle.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        if process.status == TeamStatus.RUNNING:
            self._services.worker_handle.stop_team(team_id)
        self._cache.remove(team_id)
        # Safety net: remove ephemeral stream if not already removed on stop
        try:
            self._services.event_stream.remove(team_id)
        except Exception:
            logger.debug("event_stream.remove() on delete — stream may already be removed")
        self._services.worker_handle.delete_team(team_id)
        # FS cleanup runs LAST — after the worker-side delete — so a worker
        # delete failure does not leave behind a removed workspace dir.
        _remove_workspace_dir(self._workspaces_root, team_id)
        logger.info("Team deleted: team_id=%s", team_id)

    def emit_message(self, team_id: uuid.UUID, message: Message) -> None:
        """Publish a pre-formed message into a running team's event record.

        Resolves the running handle and delegates to ``handle.emitMessage``
        — same shape as ``send_message``. The message reaches the team's
        subscribers (durable store + live stream) with no agent processing
        and no outbound channel dispatch (ADR-22).

        Raises:
            ValueError: If team not found or not running.
        """
        handle = self._get_running_handle(team_id)
        handle.emitMessage(message)
        logger.debug("Message emitted to team %s", team_id)

    def send_message(self, team_id: uuid.UUID, content: str | Message) -> None:
        """Send a message to a running team.

        Raises:
            ValueError: If team not found or not running.
        """
        handle = self._get_running_handle(team_id)
        handle.send(content)
        logger.debug("Message sent to team %s", team_id)

    def send_message_to(self, team_id: uuid.UUID, agent_name: str, content: str | Message) -> None:
        """Send a message to a specific agent in a running team.

        Raises:
            ValueError: If team not found, not running, or agent not found.
        """
        handle = self._get_running_handle(team_id)
        handle.send_to(agent_name, content)
        logger.debug("Message sent to agent '%s' in team %s", agent_name, team_id)

    def send_message_from_to(
        self, team_id: uuid.UUID, sender_name: str, recipient_name: str, content: str | Message
    ) -> None:
        """Send a message from a specific agent to another agent in a running team.

        Raises:
            ValueError: If team not found, not running, sender not found, or recipient not found.
        """
        handle = self._get_running_handle(team_id)
        handle.send_from_to(sender_name, recipient_name, content)
        logger.debug(
            "Message sent from '%s' to '%s' in team %s", sender_name, recipient_name, team_id
        )

    def process_human_input(
        self,
        team_id: uuid.UUID,
        content: str,
        message_id: str,
    ) -> None:
        """Route human input to HumanProxy for a specific message.

        Raises:
            ValueError: If team not found, not running, or message not found.
        """
        handle = self._get_running_handle(team_id)
        # _find_message resolves by inner id and returns only SentMessage, so
        # event.message is the inner Message to route (ADR-027 §Decision 1).
        event = self._find_message(team_id, message_id)
        handle.process_human_input(content, event.message)
        logger.debug("Human input routed to team %s, message_id=%s", team_id, message_id)

    def stop_team(self, team_id: uuid.UUID) -> None:
        """Stop a running team without deleting persisted data.

        Raises:
            ValueError: If team not found or not in a stoppable state.
        """
        process = self._services.worker_handle.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        if process.status == TeamStatus.STOPPED:
            msg = f"Team {team_id} is already stopped"
            raise ValueError(msg)
        if process.status == TeamStatus.DELETED:
            msg = f"Team {team_id} has been deleted"
            raise ValueError(msg)
        self._services.worker_handle.stop_team(team_id)
        self._cache.remove(team_id)
        try:
            self._services.event_stream.remove(team_id)
        except Exception:
            logger.debug("event_stream.remove() on stop — stream may already be removed")
        logger.info("Team stopped: team_id=%s", team_id)

    def restore_team(self, team_id: uuid.UUID) -> Process:
        """Restore a stopped team.

        Raises:
            ValueError: If team not found or not in a restorable state.
        """
        process = self._services.worker_handle.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        if process.status == TeamStatus.RUNNING:
            msg = f"Team {team_id} is already running"
            raise ValueError(msg)
        if process.status == TeamStatus.DELETED:
            msg = f"Team {team_id} has been deleted"
            raise ValueError(msg)
        handle = self._services.worker_handle.resume_team(team_id)
        self._cache.store(handle.team_id, handle)
        updated = self._services.worker_handle.get_team(team_id)
        if updated is None:  # pragma: no cover
            msg = f"Team {team_id} was restored but not found in event store"
            raise RuntimeError(msg)
        logger.info("Team restored: team_id=%s", team_id)
        return updated

    def get_events(
        self, team_id: uuid.UUID, after_event_id: uuid.UUID | None = None
    ) -> list[PersistedEvent]:
        """Get persisted events for a team, ordered by sequence ASC.

        Args:
            team_id: Team whose events to load.
            after_event_id: If provided, return only events after the matching
                event — anchor excluded. If None, return the full log.

        Raises:
            ValueError: If team not found.
            EventNotFoundError: Propagated from the store when after_event_id
                does not resolve to an event of this team.
        """
        process = self._services.worker_handle.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        logger.debug("Loading events for team %s (after_event_id=%s)", team_id, after_event_id)
        return self._services.event_store.load_events(team_id, after_event_id=after_event_id)

    def get_agent_states(self, team_id: uuid.UUID) -> list[AgentStateSnapshot]:
        """Get all persisted agent-state snapshots for a team.

        A thin, faithful read of the snapshot store: returns every snapshot as
        persisted, with no liveness filtering and no name->UUID resolution. The
        team-exists guard mirrors ``get_events`` — ``get_team`` returns the
        persisted process for a stopped team too, so this fires only for a
        genuinely unknown team.

        Raises:
            ValueError: If team not found.
        """
        process = self._services.worker_handle.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        logger.debug("Loading agent states for team %s", team_id)
        return self._services.event_store.load_agent_states(team_id)

    def get_event_stream(self) -> EventStream:
        """Return the tier's EventStream for cursor-based replay and fan-out."""
        return self._services.event_stream

    def get_handle(self, team_id: uuid.UUID) -> TeamHandle | None:
        """Return the cached TeamHandle for a team, or None if not cached.

        Args:
            team_id: Team UUID.

        Returns:
            TeamHandle if cached, else None.
        """
        return self._cache.get(team_id)

    def _get_running_handle(self, team_id: uuid.UUID) -> TeamHandle:
        """Look up a cached handle, verifying the team is running.

        Raises:
            ValueError: If team not found, not running, or handle not cached.
        """
        process = self._services.worker_handle.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        if process.status != TeamStatus.RUNNING:
            msg = f"Team {team_id} is not running"
            raise ValueError(msg)
        logger.debug("Resolving running handle for team %s", team_id)
        handle = self._cache.get(team_id)
        if handle is None:
            msg = f"Team {team_id} handle not cached"
            raise ValueError(msg)
        return handle

    def _find_message(self, team_id: uuid.UUID, message_id: str) -> SentMessage:
        """Find a SentMessage by its inner ``message.id`` in persisted events.

        Mirrors the worker route's ``_find_message``: resolution is by the
        **inner** ``SentMessage.message.id`` — the id every distributed tier
        puts on the wire — not the outer envelope ``SentMessage.id``
        (ADR-027 §Decision 1).

        Raises:
            ValueError: If no matching SentMessage is found. The ``not found``
                substring is load-bearing for the 404 mapping.
        """
        events = self._services.event_store.load_events(team_id)
        for ev in events:
            if isinstance(ev.event, SentMessage) and str(ev.event.message.id) == message_id:
                return ev.event
        msg = f"Message {message_id} not found"
        raise ValueError(msg)
