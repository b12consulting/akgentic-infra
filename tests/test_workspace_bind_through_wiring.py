"""A real workspace bind, through ``wire_community``, observed on the team's stream.

The first specs in this package that bind a ``WorkspaceTool`` card on a live team. The
team comes from the catalog through the wired ``TeamService.create_team``, so the bind
runs the production path end to end: ``LocalPlacement``, ``TeamManager``,
``TeamFactory``, a real ``BaseAgent.on_start``, the card, the orchestrator's forward and
the ``WorkspaceHost`` wiring created. No model is called: ``on_start`` constructs the
model client and never uses it, and the suite's autouse dummy key lets it construct.

**What a failed bind looks like from here, which is why these specs read the stream.**
The forward refuses with a ``RuntimeError`` when no host of the exact class is running.
That error is raised inside the member's ``on_start``; pykka catches it and core turns it
into an ``ErrorMessage`` on the team's stream. The member keeps running, ``create_team``
raises nothing and the ``Process`` is ``RUNNING`` — a team whose only workspace card
failed to bind is indistinguishable from a healthy one by status. So the guard is the
stream: one ``WorkspaceAttached`` and no ``ErrorMessage``, beside the hosted actor itself.

**The flush is load-bearing.** ``TeamFactory.build`` returns once the members are
*started*, not once their ``on_start`` finished. pykka runs ``on_start`` before the
actor's mailbox, so a proxy read on the member resolves only after its bind completed.
Without that read, every "exactly one" and every "zero" below races the bind.

**The workspaces root is set as an environment variable, and that is deliberate.** The
tool reads ``AKGENTIC_WORKSPACES_ROOT`` itself, separately from
``CommunitySettings.workspaces_root``, and nothing in wiring passes the setting on. The
fixture sets the variable before wiring and the first spec asserts the tree landed under
the test's root; without the variable the actors would write ``./workspaces`` relative
to the working directory.

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
from akgentic.core import (
    ActorAddress,
    ActorRegistry,
    ActorSystem,
    Akgent,
    Orchestrator,
    ResourceHost,
)
from akgentic.core.messages import ErrorMessage, EventMessage, StartMessage
from akgentic.core.messages.message import Message
from akgentic.team.models import Process, TeamStatus
from akgentic.tool.workspace import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    WorkspaceAttached,
    WorkspaceConfig,
    WorkspaceHost,
    WorkspaceTool,
    workspace_actor_name,
)

from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community
from tests.fixtures.team_metadata import seed_metadata_namespace

TIMEOUT = 10.0
TEARDOWN_GRACE = 5.0

NOTES_NS = "acme-notes"
"""A team whose Manager declares one named workspace."""
BARE_NS = "acme-bare"
"""The same team shape with no tools: the Manager is a plain ``Akgent``."""

USER_ID = "alice"
WORKSPACE_PATH = "alice/notes"
"""Where ``WorkspaceTool(workspace_id="notes")`` resolves for ``alice``."""


class _Team(NamedTuple):
    """A created team and the addresses a spec reads it through."""

    process: Process
    orchestrator: ActorAddress
    orch: Orchestrator
    manager: ActorAddress


##
## Helpers
##
def _assert_registry_empty() -> None:
    """No host of either class and no hosted workspace is live in the process."""
    assert ActorSystem.find_by_class(WorkspaceHost) == []
    assert ActorSystem.find_by_class(ResourceHost) == []
    assert ActorSystem.find_by_class(WorkspaceActor) == []


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


def _wait_for_no_host() -> None:
    """Allow the hosts the hand-off between stop and deregistration, then assert none."""
    deadline = time.monotonic() + TEARDOWN_GRACE
    while (
        ActorSystem.find_by_class(WorkspaceHost) or ActorSystem.find_by_class(ResourceHost)
    ) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ActorSystem.find_by_class(WorkspaceHost) == []
    assert ActorSystem.find_by_class(ResourceHost) == []


##
## Fixture
##
@pytest.fixture()
def wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[CommunityServices]:
    """A wired community whose catalog holds a workspace team and a bare one.

    The environment variable is set **before** wiring: the tool reads it, not the
    settings. Teardown is the registry's, not the host's — ``shutdown`` stops the
    orchestrators, then ``ActorRegistry.stop_all`` reaches the host and the hosted
    workspace, gracefully, so its executor drains and the interpreter can exit.
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
        tools=[WorkspaceTool(workspace_id="notes")],
    )
    seed_metadata_namespace(settings.catalog_path, BARE_NS, with_type=False)

    _assert_registry_empty()
    services = wire_community(settings)
    try:
        yield services
    finally:
        services.actor_system.shutdown()
        ActorRegistry.stop_all()
    assert ActorSystem.find_by_class(WorkspaceActor) == []
    _wait_for_no_host()


##
## Specs
##
class TestTheBindRunsThroughTheWiredHost:
    """A team whose card declares a workspace binds it, and the stream says so."""

    def test_a_declared_workspace_binds_through_the_wired_host(
        self, wired: CommunityServices, tmp_path: Path
    ) -> None:
        """One hosted workspace, in the wired host's own registry, under the test's root."""
        team = _create(wired, NOTES_NS)
        # Holds under a failed bind too: create_team raises nothing when a member's
        # on_start does. The lines below are the guard; this one is context.
        assert team.process.status is TeamStatus.RUNNING

        workspaces = ActorSystem.find_by_class(WorkspaceActor)
        assert len(workspaces) == 1, (
            f"expected one hosted workspace, found {len(workspaces)}; the team's stream "
            f"carries:\n{_render(_errors(_stream(team)))}"
        )
        [workspace] = workspaces
        assert workspace.name == workspace_actor_name(WORKSPACE_PATH)
        assert (tmp_path / "workspaces" / "alice" / "notes").is_dir()

        # The registry probe. Had the card bound through any other host, this ask is a
        # miss: it answers a second actor and the count below becomes two.
        [host] = ActorSystem.find_by_class(WorkspaceHost)
        again = wired.actor_system.proxy_ask(
            host, WorkspaceHost, timeout=TIMEOUT
        ).getResourceOrCreate(
            WorkspaceActor,
            WorkspaceConfig(
                name=workspace_actor_name(WORKSPACE_PATH),
                role=WORKSPACE_ACTOR_ROLE,
                workspace_path=WORKSPACE_PATH,
            ),
        )
        assert again.agent_id == workspace.agent_id
        assert len(ActorSystem.find_by_class(WorkspaceActor)) == 1
        assert ActorSystem.find_by_class(ResourceHost) == []

    def test_the_streams_one_attached_event_names_the_manager_and_no_error(
        self, wired: CommunityServices
    ) -> None:
        """Exactly one ``WorkspaceAttached``, naming the Manager, in memory and persisted."""
        team = _create(wired, NOTES_NS)
        stream = _stream(team)

        errors = _errors(stream)
        assert errors == [], (
            f"the team's stream carries errors (process {team.process.status.value}, "
            f"{len(_attached(stream))} WorkspaceAttached, "
            f"{len(ActorSystem.find_by_class(WorkspaceActor))} WorkspaceActor):\n"
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
        assert envelope.sender.agent_id == team.orchestrator.agent_id
        assert envelope.sender.agent_id != team.manager.agent_id
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

    def test_a_team_declaring_no_workspace_binds_nothing(self, wired: CommunityServices) -> None:
        """The negative beside the positive: no card, no actor, no event, no error."""
        team = _create(wired, BARE_NS)
        stream = _stream(team)

        assert team.process.status is TeamStatus.RUNNING
        assert _started_by(stream, team.manager), "no StartMessage from the Manager"
        assert ActorSystem.find_by_class(WorkspaceActor) == []
        assert _attached(stream) == []
        errors = _errors(stream)
        assert errors == [], f"the team's stream carries errors:\n{_render(errors)}"
