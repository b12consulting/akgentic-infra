"""A real workspace bind, through ``wire_community``, observed on the team's stream.

The team comes from the catalog through the wired ``TeamService.create_team``, so the
bind runs the production path end to end: ``LocalPlacement``, ``TeamManager``,
``TeamFactory``, a real ``BaseAgent.on_start``, the card and the orchestrator's forward.
No model is called: ``on_start`` constructs the model client and never uses it, and the
suite's autouse dummy key lets it construct.

**There is no resource host in this process, and that is the point.** The workspace is a
tree on disk, not an actor, so nothing in wiring creates a host and nothing here looks one
up. What replaced the old registry probe is the disk: the card writes, and the write is
either there or it is not.

**What a failed bind looks like from here, which is why these specs read the stream.** A
card that raises does so inside the member's ``on_start``; pykka catches it and core turns
it into an ``ErrorMessage`` on the team's stream. The member keeps running,
``create_team`` raises nothing and the ``Process`` is ``RUNNING`` — a team whose only
workspace card failed to bind is indistinguishable from a healthy one by status. So the
guard is the stream: exactly one ``WorkspaceAttached`` and no ``ErrorMessage``.

**The flush is load-bearing.** ``TeamFactory.build`` returns once the members are
*started*, not once their ``on_start`` finished. pykka runs ``on_start`` before the
actor's mailbox, so a proxy read on the member resolves only after its bind completed.
Without that read, every "exactly one" and every "zero" below races the bind — a race the
bind usually wins, because the test thread still has work to do after ``create_team``.
So one spec delays the bind on purpose: with the flush gone, that spec goes red instead
of the suite staying green by luck.

**The workspaces root is set as an environment variable, and that is deliberate.** The
tool reads ``AKGENTIC_WORKSPACES_ROOT`` itself, separately from
``CommunitySettings.workspaces_root``, and nothing in wiring passes the setting on. The
fixture sets the variable before wiring and the first spec asserts the tree landed under
the test's root; without the variable the card would write ``./workspaces`` relative to
the working directory.

What is out of reach here: a second replica, and two workers over one tree. These specs
hold one process and say nothing about another.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import NamedTuple

import pytest
from akgentic.core import ActorAddress, ActorRegistry, ActorSystem, Akgent, Orchestrator
from akgentic.core.messages import ErrorMessage, EventMessage, StartMessage
from akgentic.core.messages.message import Message
from akgentic.team.models import Process, TeamStatus
from akgentic.tool import ActorToolObserver
from akgentic.tool.workspace import Resource, WorkspaceAttached, WorkspaceTool

from akgentic.infra.server.deps import CommunityServices, TierServices
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community
from tests.fixtures.team_metadata import seed_metadata_namespace

TIMEOUT = 10.0
LATE_BIND_S = 0.5
"""How long the late-bind spec holds the card's bind back.

Far longer than the test thread's own work between ``create_team`` and the stream read,
so without the flush that read lands first. It only widens a window: a flushed read
waits for the bind however long it takes, so the delay can never turn the spec red.
"""

NOTES_NS = "acme-notes"
"""A team whose Manager declares one named workspace."""
BARE_NS = "acme-bare"
"""The same team shape with no tools: the Manager is a plain ``Akgent``."""

USER_ID = "alice"
WORKSPACE_PATH = "alice/notes"
"""Where ``WorkspaceTool(workspace_id="notes")`` resolves for ``alice``."""

SEEDED_FILE = "seeded.md"
SEEDED_TEXT = "# acme notes\n"
"""A file the card writes into its tree at bind time, through the card's own write path.

``WorkspaceTool.resources`` is a declared field of the card, so this needs no harness:
binding the card is what writes the file. It is the disk-side proof that the card works
with no actor to count — a tree that exists but is empty says only that something called
``mkdir``.
"""


class _Team(NamedTuple):
    """A created team and the addresses a spec reads it through."""

    process: Process
    orchestrator: ActorAddress
    orch: Orchestrator
    manager: ActorAddress


##
## Helpers
##
def _orchestrator_of(services: CommunityServices, team_id: uuid.UUID) -> ActorAddress:
    """The team's orchestrator, by exact-type lookup, confirmed by its ``team_id``."""
    matches = [
        address
        for address in ActorSystem.find_by_class(Orchestrator)
        if services.actor_system.proxy_ask(address, Orchestrator, timeout=TIMEOUT).team_id
        == team_id
    ]
    assert len(matches) == 1, f"expected one orchestrator for team {team_id}, got {matches}"
    return matches[0]


def _flush_member(services: CommunityServices, orch: Orchestrator, name: str) -> ActorAddress:
    """Return member *name* once its ``on_start`` — and therefore its bind — has run.

    pykka runs ``on_start`` on the actor's thread before it reads its mailbox, so this
    proxy read resolves only after ``on_start`` returned or raised. An ``on_start`` that
    raised leaves the actor alive and answering, so the read resolves under a failed
    bind too, and the stream then carries the error.
    """
    member = orch.get_team_member(name)
    assert member is not None, f"{name} is not a member of the team"
    team_id = services.actor_system.proxy_ask(member, Akgent, timeout=TIMEOUT).team_id
    assert team_id == orch.team_id
    return member


def _create(services: CommunityServices, namespace: str) -> _Team:
    """Create a team from *namespace* through the wired ``TeamService`` and flush it."""
    assert services.team_service is not None
    process = services.team_service.create_team(namespace, user_id=USER_ID)
    orchestrator = _orchestrator_of(services, process.team_id)
    orch = services.actor_system.proxy_ask(orchestrator, Orchestrator, timeout=TIMEOUT)
    manager = _flush_member(services, orch, "@Manager")
    return _Team(process=process, orchestrator=orchestrator, orch=orch, manager=manager)


def _stream(team: _Team) -> list[Message]:
    """The team's stream as emitted, copied so a later emission cannot change it."""
    return list(team.orch.get_messages(None, None))


def _attached(stream: list[Message]) -> list[EventMessage]:
    """Every ``EventMessage`` whose payload is a ``WorkspaceAttached``."""
    return [
        m for m in stream if isinstance(m, EventMessage) and isinstance(m.event, WorkspaceAttached)
    ]


def _errors(stream: list[Message]) -> list[ErrorMessage]:
    """Every ``ErrorMessage`` on the stream."""
    return [m for m in stream if isinstance(m, ErrorMessage)]


def _render(errors: list[ErrorMessage]) -> str:
    """The errors as ``content_type: content`` lines, so a failure prints core's text."""
    return "\n".join(f"{e.content_type}: {e.content}" for e in errors)


def _started_by(stream: list[Message], member: ActorAddress) -> list[StartMessage]:
    """The ``StartMessage``s *member* sent: proof the stream read is live."""
    return [
        m
        for m in stream
        if isinstance(m, StartMessage)
        and m.sender is not None
        and m.sender.agent_id == member.agent_id
    ]


##
## Fixture
##
@pytest.fixture()
def wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[CommunityServices]:
    """A wired community whose catalog holds a workspace team and a bare one.

    The environment variable is set **before** wiring: the tool reads it, not the
    settings.

    ``ActorRegistry.stop_all`` after ``shutdown`` is a safety net, not the mechanism:
    a failed teardown must still let the process reach the next test with an empty
    registry.
    """
    workspaces_root = tmp_path / "workspaces"
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(workspaces_root))
    settings = CommunitySettings(
        workspaces_root=workspaces_root,
        event_store_path=tmp_path / "event_store",
        catalog_path=tmp_path / "catalog",
    )
    seed_metadata_namespace(
        settings.catalog_path,
        NOTES_NS,
        with_type=False,
        tools=[
            WorkspaceTool(
                workspace_id="notes",
                resources=[Resource(file_name=SEEDED_FILE, content=SEEDED_TEXT)],
            )
        ],
    )
    seed_metadata_namespace(settings.catalog_path, BARE_NS, with_type=False)

    services = wire_community(settings)
    try:
        yield services
    finally:
        try:
            services.actor_system.shutdown()
        finally:
            ActorRegistry.stop_all()


##
## Specs
##
class TestTheBindRunsWithNoHostInTheProcess:
    """A team whose card declares a workspace binds it, and the stream says so."""

    def test_a_declared_workspace_lands_on_disk_under_the_tests_root(
        self, wired: CommunityServices, tmp_path: Path
    ) -> None:
        """The card creates its tree and writes into it, with no host anywhere."""
        team = _create(wired, NOTES_NS)
        # Holds under a failed bind too: create_team raises nothing when a member's
        # on_start does. The lines below are the guard; this one is context.
        assert team.process.status is TeamStatus.RUNNING

        tree = tmp_path / "workspaces" / "alice" / "notes"
        assert tree.is_dir(), (
            f"the card created no tree; the team's stream carries:\n"
            f"{_render(_errors(_stream(team)))}"
        )

        # The mutation, not merely the mkdir: a tree that exists but is empty says only
        # that something called mkdir. This file is on disk because the card wrote it.
        seeded = tree / SEEDED_FILE
        assert seeded.is_file(), (
            f"the card created its tree but wrote nothing into it; the team's stream "
            f"carries:\n{_render(_errors(_stream(team)))}"
        )
        assert seeded.read_text() == SEEDED_TEXT

    def test_the_streams_one_attached_event_names_the_manager_and_no_error(
        self, wired: CommunityServices
    ) -> None:
        """Exactly one ``WorkspaceAttached``, naming the Manager, in memory and persisted."""
        team = _create(wired, NOTES_NS)
        stream = _stream(team)

        errors = _errors(stream)
        assert errors == [], (
            f"the team's stream carries errors (process {team.process.status.value}, "
            f"{len(_attached(stream))} WorkspaceAttached):\n"
            f"{_render(errors)}"
        )
        assert _started_by(stream, team.manager), "no StartMessage from the Manager"

        attached = _attached(stream)
        assert len(attached) == 1
        envelope = attached[0]
        assert envelope.event == WorkspaceAttached(
            agent_id=team.manager.agent_id, workspace_path=WORKSPACE_PATH
        )
        assert isinstance(envelope.event.agent_id, uuid.UUID)
        assert envelope.sender is not None
        # The sender is the BINDING MEMBER, not the orchestrator. Tool story 52-5 moved the
        # emission off the orchestrator's getResourceOrCreate onto the card's own
        # observer.notify_event, which emits from self; the payload did not move.
        #
        # That asymmetry is why this line is worth pinning rather than dropping. The
        # frontend's two consumers key off the payload's agent_id and never off the
        # envelope sender (workspace-registry.selector.ts:70,
        # workspace-invalidation.selector.ts:283-294), so they survived the move untouched.
        # Demanding sender == orchestrator here would pin the one thing they deliberately
        # refuse to depend on. Resolved b12consulting/akgentic-infra#453.
        assert envelope.sender.agent_id == team.manager.agent_id
        assert envelope.sender.agent_id != team.orchestrator.agent_id
        assert envelope.team_id == team.process.team_id

        persisted = [
            p
            for p in wired.event_store.load_events(team.process.team_id)
            if p.event.id == envelope.id
        ]
        assert len(persisted) == 1
        assert isinstance(persisted[0].event, EventMessage)
        # Observed, not assumed: the YAML store rehydrates the nested dataclass through its
        # ``__model__`` tag, so a Python reader of the persisted log gets the typed payload.
        assert isinstance(persisted[0].event.event, WorkspaceAttached)
        assert persisted[0].event.event == envelope.event

    def test_the_reads_wait_for_a_bind_that_lands_late(
        self, wired: CommunityServices, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bind slower than the test thread is still read after it lands.

        The other specs pass with the flush deleted, because the Manager's bind normally
        finishes before the test thread reads the stream. Holding the card's ``observer``
        back, which is where the bind runs, reverses that race. So this spec is red
        without the flush and green with it, and the flush is shown to be the ordering.
        """
        original = WorkspaceTool.observer

        def _late(self: WorkspaceTool, observer: ActorToolObserver) -> WorkspaceTool:
            time.sleep(LATE_BIND_S)
            return original(self, observer)

        monkeypatch.setattr(WorkspaceTool, "observer", _late)

        team = _create(wired, NOTES_NS)
        stream = _stream(team)

        errors = _errors(stream)
        assert errors == [], f"the team's stream carries errors:\n{_render(errors)}"
        attached = _attached(stream)
        assert len(attached) == 1, "read before the bind"
        assert attached[0].event == WorkspaceAttached(
            agent_id=team.manager.agent_id, workspace_path=WORKSPACE_PATH
        )

    def test_a_team_declaring_no_workspace_binds_nothing(
        self, wired: CommunityServices, tmp_path: Path
    ) -> None:
        """The negative beside the positive: no card, no tree, no event, no error."""
        team = _create(wired, BARE_NS)
        stream = _stream(team)

        assert team.process.status is TeamStatus.RUNNING
        assert _started_by(stream, team.manager), "no StartMessage from the Manager"
        assert _attached(stream) == []
        # The disk half of the negative, mirroring the positive spec's tree assertion.
        # Without it this spec reads only the stream, and a card that wrote a tree while
        # emitting nothing would pass — the same blind spot the retired actor-count left.
        assert not (tmp_path / "workspaces" / USER_ID).exists(), (
            "a team declaring no workspace still created a tree under the test's root"
        )
        errors = _errors(stream)
        assert errors == [], f"the team's stream carries errors:\n{_render(errors)}"


class TestTheContainerCarriesNoHost:
    """The retired ``resource_host`` slot is on neither container."""

    def test_resource_host_is_on_neither_services_container(self) -> None:
        """Successor to the two field assertions the retired host suite carried.

        ``CommunityServices`` inherits ``TierServices``, so a reintroduction on the base
        would satisfy a positive assertion on the subclass. Both are named here for that
        reason: the slot is gone from the tier this story changed *and* from the one it
        never had it on.
        """
        assert "resource_host" not in CommunityServices.model_fields
        assert "resource_host" not in TierServices.model_fields
