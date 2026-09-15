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
records are removed; four specs elsewhere in this suite are skipped for that
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
from pathlib import Path, PurePosixPath

import pytest
from akgentic.core import ActorAddressImpl, ActorRegistry, Akgent, Orchestrator
from akgentic.team.models import Process
from akgentic.tool.workspace import (
    SHARED_SCOPE,
    TEAM_KIND,
    WorkspaceTool,
    git_dir_for,
    meta_dir_for,
)

from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community
from tests.fixtures.team_metadata import seed_metadata_namespace

TIMEOUT = 10.0

DESK_NS = "acme-desk"
"""A team whose Manager carries a default ``WorkspaceTool()``."""

OWNER = "alice"
"""The principal the team is created for — the ``<scope>`` of its tree."""

SHARED_KINDS_ENV = "AKGENTIC_WORKSPACE_SHARED_KINDS"
"""The platform's permission for shared trees, named as a literal on purpose.

``workspace_sharable=True`` on a card is a *request*; this variable is the
*permission*, read at bind by ``WorkspaceTool.observer``. Unset means no kind
may be shared, and a card asking for an unpermitted kind **raises at bind** — so
a spec that only set the card field would fail at team creation with a message
about sharing, never reaching its assertion.

The tool defines this name but does not re-export it from
``akgentic.tool.workspace`` at the version this package runs against, so there
is no constant to import; the literal is the only way to name it here.
"""


def _wire_community_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *tools: WorkspaceTool
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
    seed_metadata_namespace(settings.catalog_path, DESK_NS, with_type=False, tools=list(tools))
    services = wire_community(settings)
    yield services
    services.actor_system.shutdown()


@pytest.fixture()
def bound_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[CommunityServices, None, None]:
    """A community whose Manager carries a default, per-principal ``WorkspaceTool()``."""
    yield from _wire_community_with(tmp_path, monkeypatch, WorkspaceTool())


@pytest.fixture()
def sharing_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[CommunityServices, None, None]:
    """A community whose Manager declares ``workspace_sharable=True``, and may.

    The permission is set **before** wiring, for the reason
    :data:`SHARED_KINDS_ENV` gives: the bind refuses an unpermitted shared kind
    rather than quietly handing back a per-principal tree.
    """
    monkeypatch.setenv(SHARED_KINDS_ENV, TEAM_KIND.removeprefix("_"))
    yield from _wire_community_with(tmp_path, monkeypatch, WorkspaceTool(workspace_sharable=True))


def _orchestrator_of(services: CommunityServices, team_id: uuid.UUID) -> Orchestrator:
    """The team's orchestrator, found in the actor registry and confirmed by its ``team_id``.

    The candidates come from core's public registry export, wrapped as
    addresses, so the spec runs on the core that ships. ``get_by_class`` also
    returns subclasses, unlike the exact-class lookup this replaced; the
    ``team_id`` filter and the single-match assertion are what make the answer
    unambiguous.
    """
    candidates = [
        ActorAddressImpl(ref) for ref in ActorRegistry.get_by_class(Orchestrator) if ref.is_alive()
    ]
    matches = [
        proxy
        for proxy in (
            services.actor_system.proxy_ask(address, Orchestrator, timeout=TIMEOUT)
            for address in candidates
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


class TestDeletingASharingTeamRemovesTheSharedTreeAndBothSiblings:
    """AC #1 + AC #2: the sharing axis comes from the card, and both siblings go too.

    The defect this reverses answered the sharing question at the deletion site
    with a hard-coded ``workspace_sharable=False``. A team whose card declares
    sharing therefore wrote its tree to ``_shared/_team/<team_id>`` while
    deletion looked under ``<owner>/_team/<team_id>``, which never existed:
    ``exists()`` was false, the function returned, the tree survived for ever.

    The precondition below is what makes the postcondition mean anything — it
    proves the tree deletion removes is the one a *real card* put there. Both
    sidecar directories are located the way production locates them
    (``git_dir_for`` / ``meta_dir_for``), never by appending a suffix here: a
    hand-built copy of the placement rule is the shape of the original defect.

    The ``.index`` seeding is not decoration. ``<tree>.index/rag/*.yaml`` holds
    the extracted text of every document the tree indexed, so leaving it behind
    is a retention leak, and a spec asserting only that the tree is gone would
    pass over a half-fix.
    """

    def test_the_shared_tree_and_both_sidecars_are_gone_and_the_owner_scope_was_never_used(
        self, sharing_services: CommunityServices, tmp_path: Path
    ) -> None:
        service = sharing_services.team_service
        assert service is not None
        process = _create_bound_team(sharing_services)
        workspaces_root = tmp_path / "workspaces"
        relative = PurePosixPath(SHARED_SCOPE) / TEAM_KIND / str(process.team_id)
        tree = workspaces_root / relative
        assert tree.is_dir(), f"the Manager's bind wrote no shared tree at {tree}"
        per_principal = workspaces_root / OWNER / TEAM_KIND / str(process.team_id)
        assert not per_principal.exists(), "the card asked for sharing; nothing is under the owner"

        journal = git_dir_for(tree)
        journal.mkdir(parents=True, exist_ok=True)
        (journal / "HEAD").write_text("ref: refs/heads/main\n")
        index = meta_dir_for(str(relative))
        (index / "rag").mkdir(parents=True, exist_ok=True)
        (index / "rag" / "doc.yaml").write_text("text: the extracted text of a document\n")

        service.stop_team(process.team_id)
        service.delete_team(process.team_id)

        assert not tree.exists()
        assert not journal.exists()
        assert not index.exists()
