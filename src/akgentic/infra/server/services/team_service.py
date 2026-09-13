"""TeamService — orchestrates catalog resolution and team lifecycle via protocols."""

from __future__ import annotations

import logging
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.core.messages.orchestrator import SentMessage
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.infra.errors import (
    PlacementConsistencyError,
    TeamNotFoundError,
    TeamStateConflictError,
)
from akgentic.infra.protocols.event_stream import EventStream
from akgentic.infra.protocols.runtime_cache import RuntimeCache
from akgentic.infra.protocols.team_handle import TeamHandle
from akgentic.infra.protocols.workspace_deletion import (
    WorkspaceDeletionContext,
    WorkspaceDeletionPolicy,
)
from akgentic.infra.server.services._metadata_payload import validate_metadata
from akgentic.infra.server.services._workspace_paths import deletion_candidate_paths
from akgentic.team.models import AgentStateSnapshot, PersistedEvent, Process, TeamStatus
from akgentic.team.ports import AgentCardNotFoundError
from akgentic.tool.workspace import git_dir_for, meta_dir_for

if TYPE_CHECKING:
    from akgentic.core.messages.message import Message
    from akgentic.infra.server.deps import TierServices

logger = logging.getLogger(__name__)

# Maximum page size for GET /teams; the default is 250 (ADR-032 §Decision 1).
MAX_PAGE_SIZE = 500

# The number of segments a resolved workspace path has, by construction
# (ADR-052 Decision 1: ``<scope>/<kind>/<leaf>``). Named because the containment
# guard below tests it: depth is what makes no workspace path a proper prefix of
# another, and a candidate of any other depth is a path this code does not
# recognise and must not remove.
_WORKSPACE_PATH_SEGMENTS = 3


class CatalogTeamEntryMissingError(EntryNotFoundError):
    """A catalog namespace exists, but holds no ``kind="team"`` entry.

    Distinct from a plain ``EntryNotFoundError`` (the namespace holds nothing at
    all) so the two 404s can say different things: "the namespace is not there"
    and "the namespace is there but has no team in it" call for different
    repairs. ``Catalog.load_team`` reports both conditions — and the invalid one
    — as ``CatalogValidationError``, so they are split by probing the namespace,
    never by matching the message.

    A subclass of ``EntryNotFoundError`` on purpose: every existing
    ``except EntryNotFoundError``, including the catalog package's app-level 404
    handler that serves the webhook ingestion path, keeps catching it, so the
    split is additive. It lives here rather than in ``akgentic.infra.errors``
    because that module imports nothing but ``__future__`` — a guard
    ``tests/test_errors.py::TestModuleHygiene`` asserts exactly — and this type
    extends the catalog's hierarchy, not the ``ServerError`` one.
    """


def _contained_target(
    workspaces_root: Path, candidate: PurePosixPath, team_id: uuid.UUID
) -> Path | None:
    """The absolute tree for *candidate*, or ``None`` when it may not be touched.

    The second of the two limits the deletion path enforces rather than
    documents (the first being that candidates come only from the team's own
    cards). A candidate that is not exactly three segments, or that does not
    resolve **strictly inside** ``workspaces_root`` three segments deep, is
    refused and logged — whatever any policy answered about it. The containment
    half is checked on the *resolved* path, so a symlinked tree pointing out of
    the root is refused too.

    **Depth is measured twice, on the literal candidate and on what it
    resolves to, because the two can disagree.** ``a/../b`` is three parts and
    lands inside the root, yet names ``<root>/b`` — one segment deep, and the
    *parent* of every tree beneath it. Measuring only the literal would approve
    it and hand back a target containing trees the team never bound, which is
    exactly the proper-prefix hazard ADR-052's fixed depth exists to remove.
    Checking the literal as well is what keeps the caller's
    ``scope, kind, leaf`` unpack total.

    Returns:
        The resolved absolute tree, or ``None`` when the candidate is refused.
    """
    root = workspaces_root.resolve()
    target = (workspaces_root / candidate).resolve()
    if (
        len(candidate.parts) != _WORKSPACE_PATH_SEGMENTS
        or not target.is_relative_to(root)
        or len(target.relative_to(root).parts) != _WORKSPACE_PATH_SEGMENTS
    ):
        logger.warning(
            "Workspace cleanup refused, candidate is not a contained workspace path — "
            "team_id=%s candidate=%s root=%s",
            team_id,
            candidate,
            root,
        )
        return None
    return target


def _rmtree_best_effort(target: Path, team_id: uuid.UUID) -> None:
    """Remove *target* recursively, logging and swallowing whatever goes wrong.

    A missing directory is a **silent** no-op: an ephemeral team that never
    invoked a ``Filesystem`` write has no tree to clean, and neither sidecar
    exists until something creates it. Any other failure is logged at WARNING
    and suppressed so team deletion still completes in the system of record — a
    later janitor pass can sweep orphans. Callers invoke this once per target
    precisely so one failure cannot skip the next: a failed ``<tree>.git`` must
    not take ``<tree>.index`` with it, which is the retention half.

    See ADR-022 §D7 for the original best-effort, log-not-raise rationale;
    generalized here from akgentic-infra-enterprise's
    ``routes/enterprise_server_teams.py`` per Epic 24.
    """
    if not target.exists():
        return
    try:
        shutil.rmtree(target)
    except Exception as exc:  # noqa: BLE001 — log-not-raise; cleanup is best-effort
        logger.warning(
            "Workspace cleanup failed — team_id=%s target=%s error=%s",
            team_id,
            target,
            exc,
        )


def _remove_workspace_trees(
    workspaces_root: Path,
    team_id: uuid.UUID,
    process: Process,
    candidates: list[PurePosixPath],
    policy: WorkspaceDeletionPolicy,
) -> None:
    """Best-effort removal of the trees *policy* approves, and both of their sidecars.

    *candidates* are the paths the team's **own** cards resolved to, plus its own
    default tree — never a free-form path, and never a path this function
    composed. The sharing axis therefore comes from the card that declared the
    tree, which is the whole of the defect this replaced: that code asked the
    resolver for a target but answered ``workspace_sharable=False`` itself, so a
    team whose card declared sharing wrote to ``_shared/_team/<team_id>`` while
    deletion looked under ``<owner>/_team/<team_id>``. ``exists()`` was false,
    the function returned, and the tree survived for ever. ADR-048 §The rule is
    written nine times calls this the most dangerous of its ten sites precisely
    because it raises nothing, logs nothing and fails no test.

    For each approved candidate, three targets are removed independently: the
    tree, its journal (``<tree>.git``), and its metadata directory
    (``<tree>.index``). Both siblings are located through the tool's own
    ``git_dir_for`` / ``meta_dir_for`` — never by appending a suffix here, which
    would be one more copy of the placement rule and the same shape of defect
    one scale down. ``<tree>.index/rag/*.yaml`` holds the extracted text of
    every document the tree indexed, so removing it is a retention fix rather
    than tidying.

    **The two siblings anchor differently, deliberately.** The tree and its
    ``.git`` are contained against the injected *workspaces_root*
    (:func:`_contained_target`). The ``.index`` is anchored to whatever parent
    ``meta_dir_for`` resolved, because an operator may legitimately relocate the
    metadata root with ``AKGENTIC_WORKSPACE_META_ROOT`` — and refusing to delete
    it there would re-open the retention leak in precisely the deployment that
    configured a separate root.
    """
    for candidate in candidates:
        target = _contained_target(workspaces_root, candidate, team_id)
        if target is None:
            continue
        scope, kind, leaf = candidate.parts
        ctx = WorkspaceDeletionContext(
            team_id=team_id,
            owner_user_id=process.user_id,
            path=candidate,
            scope=scope,
            kind=kind,
            leaf=leaf,
        )
        if not policy.may_delete(ctx=ctx):
            # On the record, because a deletion nobody can explain afterwards is
            # worse than one that did not happen.
            logger.info(
                "Workspace kept — team_id=%s candidate=%s policy=%s",
                team_id,
                candidate,
                type(policy).__name__,
            )
            continue
        _rmtree_best_effort(target, team_id)
        _rmtree_best_effort(git_dir_for(target), team_id)
        _rmtree_best_effort(meta_dir_for(str(candidate)), team_id)


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
            workspaces_root: Server-side root directory under which every
                workspace tree lives, at ``<scope>/<kind>/<leaf>`` (ADR-052).
                Used by ``delete_team`` for best-effort FS cleanup, and as the
                containment anchor no deletion candidate may resolve outside of.
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
            EntryNotFoundError: If ``catalog_namespace`` holds no entry at all
                — the namespace does not exist.
            CatalogTeamEntryMissingError: If the namespace exists but holds no
                ``kind="team"`` entry. A subclass of ``EntryNotFoundError``, so
                a caller that does not care about the distinction still catches
                it; the router uses the distinction to say which of the two it
                is.
            CatalogValidationError: If the namespace exists and its stored
                entries are invalid. Propagated verbatim, message and all: the
                catalog's own diagnosis is the only thing that tells an operator
                what to repair, and the app-level catalog handler answers 409
                with it — the same body ``GET /admin/catalog/team/{ns}/resolve``
                returns for the same condition.
            MetadataValidationError: If ``metadata`` carries a ``__model__`` key,
                is supplied for a card declaring no contract, or fails the
                declared schema. Raised before placement runs, so a rejected
                body never leaves a half-created team behind.
        """
        logger.debug("Resolving team for catalog namespace: %s", catalog_namespace)
        try:
            team_card = self._services.catalog.load_team(catalog_namespace)
        except CatalogValidationError as exc:
            # load_team reports an absent namespace, a namespace holding no team
            # entry, and a namespace whose stored entries are invalid all three
            # as CatalogValidationError, so the type alone cannot tell them
            # apart. Probe to decide — inside the except block, so a successful
            # create never pays for the extra query.
            entries = self._services.catalog.list_by_namespace(catalog_namespace)
            if not entries:
                raise EntryNotFoundError(catalog_namespace) from exc
            if not any(entry.kind == "team" for entry in entries):
                msg = f"Catalog namespace '{catalog_namespace}' has no team entry"
                raise CatalogTeamEntryMissingError(msg) from exc
            # Nothing is missing — the stored catalog is broken. Re-raise the
            # original so its message and traceback survive to the client.
            raise
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
        metadata: Mapping[str, list[str]] | None = None,
        catalog_namespace: str | None = None,
        page: int = 1,
        size: int = 250,
    ) -> tuple[list[Process], int]:
        """Return one numbered page of the user's teams plus the filtered count.

        Phase 1: the store returns the matching set, sorted ``created_at DESC,
        team_id DESC`` and sliced here (ADR-032 §Decision 2). Stateless — a pure
        function of ``user_id`` + ``status`` + ``metadata`` + ``catalog_namespace``
        + ``page`` + ``size`` + store contents. An out-of-range page yields an
        empty list with the correct total.

        ``user_id``, ``status`` and ``metadata`` push into the EventStore rather
        than loading the user's teams into Python and filtering here, so
        per-request cost scales with the answer rather than with the archive.
        ``catalog_namespace`` is the one exception, and deliberately so:
        ``EventStore.list_teams`` has no such parameter, and adding one is an
        ``akgentic-team`` Protocol change. It is applied here instead — **before
        the sort and the slice**, so the total is still the FILTERED count on
        every page and pages stay contiguous. The cost is honest: on that path
        the store returns the ``status``+``metadata``-narrowed set and the
        namespace narrows it in Python. That is a performance property, not a
        correctness one. Nothing filters after the slice.

        ``status=None``, ``metadata=None`` and ``catalog_namespace=None`` each
        mean *no such filter*, so a caller passing none of them gets exactly the
        result set it got before. The three the store understands are forwarded
        unconditionally: a branch that omits a kwarg when it is ``None`` is how a
        filter later gets silently dropped.

        ``metadata`` maps an indexed field name to a list of **prefix terms** —
        terms within one key OR-combine, distinct keys AND-combine, and matching
        is an anchored, case-insensitive prefix rather than equality (ADR-28).
        The terms travel verbatim: index derivation, and escaping for whichever
        dialect the store speaks, happen once inside ``akgentic-team`` (ADR-24
        §D4). Escaping here would compose with that and match nothing.

        No filter replaces the owner filter: they only narrow *within* the
        user's teams, ``catalog_namespace`` included. Metadata is caller-supplied
        and non-secret, so allowing it to widen the set — or the count — would
        make this a cross-tenant enumeration primitive. See team-package ADR-16
        (owner), ADR-23 (lifecycle state) and ADR-24 (metadata) for the Protocol
        changes.
        """
        rows = self._services.event_store.list_teams(
            user_id=user_id, status=status, metadata=metadata
        )
        # Truthiness, not ``is not None``: a blank ?catalog_namespace= is an
        # empty form field, not a request to filter on the literal empty string.
        if catalog_namespace:
            rows = [row for row in rows if row.catalog_namespace == catalog_namespace]
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
            TeamNotFoundError: If the team is unknown.
            ValueError: Propagated from the write path for a deleted team.
            MetadataValidationError: If the body carries a ``__model__`` key at
                any depth, if the team declares no metadata contract, or if the
                body fails the declared schema. Raised before the write path is
                reached, so a rejected body changes nothing.
        """
        process = self._services.worker_handle.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise TeamNotFoundError(msg)
        validated = validate_metadata(process.metadata_type, raw)
        updated = self._services.worker_handle.update_team_metadata(team_id, validated)
        logger.info("Team metadata updated: team_id=%s", team_id)
        return updated.metadata

    def _deletion_candidates(self, process: Process) -> list[PurePosixPath]:
        """The trees this team's deletion may consider, or an empty list and a WARNING.

        **No team may become undeletable.** Two kinds of failure can stop the
        candidate set being resolved, and neither may propagate: a declaration
        the resolver cannot turn into a path — an owner id that cannot be a
        directory name (ADR-048 Decision 4's delete-path row), a leaf that
        cannot either, or a metadata card whose keys the team's metadata does
        not satisfy — and a ``card_hash`` the store cannot resolve. Each is
        logged at WARNING and skips workspace cleanup for that team; the record
        deletion still succeeds. Letting either through would trade an orphaned
        directory for a stuck record.

        Default-layout cards that disagree on ``workspace_sharable`` used to be
        a third, and are not one any more: both of the trees they name are this
        team's own ``_team`` tree, so ``deletion_candidate_paths`` keeps both as
        candidates and records the disagreement itself. Skipping there left both
        trees and both ``.index`` sidecars behind for ever.
        """
        team_id = process.team_id
        try:
            return deletion_candidate_paths(process=process, store=self._services.event_store)
        except ValueError as exc:
            # Some declaration of this team cannot be turned into a path: the
            # owner id, a leaf, or a metadata card's keys against the team's
            # metadata. ``owner`` is logged because it is the commonest of the
            # three, not because it is the only one — ``error`` names the
            # actual cause.
            logger.warning(
                "Workspace cleanup skipped, no candidate resolved — team_id=%s owner=%r error=%s",
                team_id,
                process.user_id,
                exc,
            )
        except AgentCardNotFoundError as exc:
            # A card blob the team references is gone. The sharing axis is
            # unknowable without every card, and the wrong guess — the
            # per-principal default — is precisely the defect epic 71 removes,
            # so nothing is removed rather than the wrong thing.
            logger.warning(
                "Workspace cleanup skipped, a team card is unresolvable — team_id=%s error=%s",
                team_id,
                exc,
            )
        return []

    def delete_team(self, team_id: uuid.UUID) -> None:
        """Stop (if running) and delete a team.

        After the team is removed from the system of record, the trees the team
        **owns** — each with its ``<tree>.git`` journal and its ``<tree>.index``
        metadata directory — are removed on a best-effort basis. A missing
        directory, an unusable owner id, an unresolvable card or an ``rmtree``
        failure does not prevent deletion from completing.

        Which trees those are is not a constant: the candidates come from the
        team's own cards (so the sharing axis is the card's answer, never this
        method's), and ``TierServices.workspace_deletion_policy`` decides which
        of them go. Its default approves the team's own ``_team`` tree in either
        scope and refuses every other kind.

        **The candidate set is resolved early — right after the team is loaded
        and stopped — while the filesystem work still runs last.** The ordering
        of the removal is load-bearing and unchanged: it follows the worker-side
        delete so a worker failure does not leave a removed workspace behind.
        But resolving the candidates depends on a card-store read, and doing
        that read on the far side of the worker delete would make this code
        depend on card blobs outliving the team. Card blobs are content-
        addressed rather than team-keyed, so they should — but if a tier ever
        purged them the read would come back empty, the sharing axis would
        silently revert to the per-principal default, and this story's defect
        would return with nothing failing. Resolution moves; deletion does not.

        Raises:
            TeamNotFoundError: If the team is unknown. Raised before any
                filesystem work, so a missing team never triggers FS cleanup.
            ValueError: Propagated from the worker for a team already deleted.
        """
        process = self._services.worker_handle.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise TeamNotFoundError(msg)
        if process.status == TeamStatus.RUNNING:
            self._services.worker_handle.stop_team(team_id)
        candidates = self._deletion_candidates(process)
        self._cache.remove(team_id)
        # Safety net: remove ephemeral stream if not already removed on stop
        try:
            self._services.event_stream.remove(team_id)
        except Exception:
            logger.debug("event_stream.remove() on delete — stream may already be removed")
        self._services.worker_handle.delete_team(team_id)
        # FS cleanup runs LAST — after the worker-side delete — so a worker
        # delete failure does not leave behind a removed workspace dir.
        _remove_workspace_trees(
            self._workspaces_root,
            team_id,
            process,
            candidates,
            self._services.workspace_deletion_policy,
        )
        logger.info("Team deleted: team_id=%s", team_id)

    def emit_message(self, team_id: uuid.UUID, message: Message) -> None:
        """Publish a pre-formed message into a running team's event record.

        Resolves the running handle and delegates to ``handle.emitMessage``
        — same shape as ``send_message``. The message reaches the team's
        subscribers (durable store + live stream) with no agent processing
        and no outbound channel dispatch (ADR-22).

        Raises:
            TeamNotFoundError: If the team is unknown.
            TeamStateConflictError: If the team exists but is not running.
        """
        handle = self._get_running_handle(team_id)
        handle.emitMessage(message)
        logger.debug("Message emitted to team %s", team_id)

    def send_message(self, team_id: uuid.UUID, content: str | Message) -> None:
        """Send a message to a running team.

        Raises:
            TeamNotFoundError: If the team is unknown.
            TeamStateConflictError: If the team exists but is not running.
        """
        handle = self._get_running_handle(team_id)
        handle.send(content)
        logger.debug("Message sent to team %s", team_id)

    def send_message_to(self, team_id: uuid.UUID, agent_name: str, content: str | Message) -> None:
        """Send a message to a specific agent in a running team.

        Raises:
            TeamNotFoundError: If the team is unknown.
            TeamStateConflictError: If the team exists but is not running.
            ValueError: If the agent is not found — raised inside the team
                package, so it arrives unclassified.
        """
        handle = self._get_running_handle(team_id)
        handle.send_to(agent_name, content)
        logger.debug("Message sent to agent '%s' in team %s", agent_name, team_id)

    def send_message_from_to(
        self, team_id: uuid.UUID, sender_name: str, recipient_name: str, content: str | Message
    ) -> None:
        """Send a message from a specific agent to another agent in a running team.

        Raises:
            TeamNotFoundError: If the team is unknown.
            TeamStateConflictError: If the team exists but is not running.
            ValueError: If the sender or the recipient is not found — raised
                inside the team package, so it arrives unclassified.
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
            TeamNotFoundError: If the team, or the message, is unknown.
            TeamStateConflictError: If the team exists but is not running.
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
            TeamNotFoundError: If the team is unknown.
            TeamStateConflictError: If the team is already stopped.
            ValueError: If the team has been deleted — deliberately left
                unclassified so it keeps answering 404 rather than 409. Whether
                a deleted team is "gone" or "in a conflicting state" is a
                client-visible contract question beyond this fix.
        """
        process = self._services.worker_handle.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise TeamNotFoundError(msg)
        if process.status == TeamStatus.STOPPED:
            msg = f"Team {team_id} is already stopped"
            raise TeamStateConflictError(msg)
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
            TeamNotFoundError: If the team is unknown.
            TeamStateConflictError: If the team is already running.
            ValueError: If the team has been deleted — left unclassified for
                the same reason as ``stop_team``.
        """
        process = self._services.worker_handle.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise TeamNotFoundError(msg)
        if process.status == TeamStatus.RUNNING:
            msg = f"Team {team_id} is already running"
            raise TeamStateConflictError(msg)
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
            TeamNotFoundError: If the team is unknown.
            EventNotFoundError: Propagated from the store when after_event_id
                does not resolve to an event of this team.
        """
        process = self._services.worker_handle.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise TeamNotFoundError(msg)
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
            TeamNotFoundError: If the team is unknown.
        """
        process = self._services.worker_handle.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise TeamNotFoundError(msg)
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
            TeamNotFoundError: If the team is unknown.
            TeamStateConflictError: If the team exists but is not running.
            ValueError: If the team is running but no handle is cached — a
                server-side inconsistency rather than a state the caller can
                reason about, so it stays unclassified.
        """
        process = self._services.worker_handle.get_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise TeamNotFoundError(msg)
        if process.status != TeamStatus.RUNNING:
            msg = f"Team {team_id} is not running"
            raise TeamStateConflictError(msg)
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
            TeamNotFoundError: If no matching SentMessage is found. A missing
                resource, so it maps to 404 like an unknown team; the type now
                carries that, and the ``not found`` substring keeps the routes
                still mapping by message on today's answer.
        """
        events = self._services.event_store.load_events(team_id)
        for ev in events:
            if isinstance(ev.event, SentMessage) and str(ev.event.message.id) == message_id:
                return ev.event
        msg = f"Message {message_id} not found"
        raise TeamNotFoundError(msg)
