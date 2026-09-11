"""Team deletion removes the tree a real agent bound, proven on disk.

``TeamService.delete_team`` cleans up the team's workspace on a best-effort
basis, and a missing directory is a silent no-op by contract. A deletion target
that has drifted from the layout the agent writes therefore raises nothing,
logs nothing, and fails no spec that seeds its own directory at the path it
expects: deletion simply stops removing anything. The only evidence it works is
a directory that is gone, so that is what this spec asserts.

**A real card chooses the path, not the spec.** The team comes from the catalog
through the wired ``TeamService.create_team``, and its Manager is a
``BaseAgent`` carrying a default ``WorkspaceTool()``. Its bind creates the tree.
The spec asserts that tree at the literal ``<root>/<owner>/_team/<team_id>``
*before* deleting. That precondition is what makes the postcondition mean
anything: it proves the tree the deletion removes is the one the agent wrote.

**One root for both readers.** The tool reads ``AKGENTIC_WORKSPACES_ROOT`` when
a card binds; deletion reads ``CommunitySettings.workspaces_root``. Nothing links
the two, so the fixture sets the variable to the settings root **before**
wiring. Without it the agent writes ``./workspaces`` relative to the working
directory, and the precondition below goes red.

**Stop, then delete.** ``delete_team`` stops a RUNNING team itself, but the stop
path's subscribers can still be flushing event-store writes while the team's
records are removed; three specs elsewhere in this suite are skipped for that
race. This spec stops the team first and deletes a STOPPED one.

**Flushing the bind.** ``create_team`` returns once the members are *started*,
not once their ``on_start`` finished, and the bind runs in ``on_start``. pykka
runs ``on_start`` before the actor reads its mailbox, so a proxy read on the
Manager resolves only after its bind completed.

No model is called: ``on_start`` constructs the model client and never uses it,
and the suite's autouse dummy key lets it construct.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from akgentic.core import ActorSystem, Akgent, Orchestrator
from akgentic.team.models import Process
from akgentic.tool.workspace import WorkspaceTool

from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community
from tests.fixtures.team_metadata import seed_metadata_namespace

TIMEOUT = 10.0

DESK_NS = "acme-desk"
"""A team whose Manager carries a default ``WorkspaceTool()``."""

OWNER = "alice"
"""The principal the team is created for — the ``<scope>`` of its tree."""


@pytest.fixture()
def bound_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[CommunityServices, None, None]:
    """A wired community whose agents and whose deletion read one workspaces root.

    The working directory moves to an empty scratch directory that is **not**
    the settings root. Should the variable ever go missing, the tool's
    ``./workspaces`` fallback then lands there — harmlessly, and visibly, as a
    red precondition — rather than in whatever checkout the suite runs from.
    """
    workspaces_root = tmp_path / "workspaces"
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(workspaces_root))
    settings = CommunitySettings(
        workspaces_root=workspaces_root,
        event_store_path=tmp_path / "event_store",
        catalog_path=tmp_path / "catalog",
    )
    seed_metadata_namespace(
        settings.catalog_path, DESK_NS, with_type=False, tools=[WorkspaceTool()]
    )
    services = wire_community(settings)
    yield services
    services.actor_system.shutdown()


def _orchestrator_of(services: CommunityServices, team_id: uuid.UUID) -> Orchestrator:
    """The team's orchestrator, found by exact type and confirmed by its ``team_id``."""
    matches = [
        proxy
        for proxy in (
            services.actor_system.proxy_ask(address, Orchestrator, timeout=TIMEOUT)
            for address in ActorSystem.find_by_class(Orchestrator)
        )
        if proxy.team_id == team_id
    ]
    assert len(matches) == 1, f"expected one orchestrator for team {team_id}"
    return matches[0]


def _create_bound_team(services: CommunityServices) -> Process:
    """Create the desk team through the wired service, and return once its bind ran."""
    assert services.team_service is not None
    process = services.team_service.create_team(catalog_namespace=DESK_NS, user_id=OWNER)
    manager = _orchestrator_of(services, process.team_id).get_team_member("@Manager")
    assert manager is not None, "the desk team has no @Manager"
    assert services.actor_system.proxy_ask(manager, Akgent, timeout=TIMEOUT).team_id == (
        process.team_id
    )
    return process


class TestDeletingATeamRemovesTheTreeItsAgentBound:
    """AC #3, Guard 1: the tree a real default card bound is gone after deletion."""

    def test_the_bound_tree_exists_at_the_three_part_location_and_deletion_removes_it(
        self, bound_services: CommunityServices, tmp_path: Path
    ) -> None:
        service = bound_services.team_service
        assert service is not None
        process = _create_bound_team(bound_services)
        tree = tmp_path / "workspaces" / OWNER / "_team" / str(process.team_id)
        assert tree.is_dir(), f"the Manager's bind wrote no tree at {tree}"

        service.stop_team(process.team_id)
        service.delete_team(process.team_id)

        assert not tree.exists()
