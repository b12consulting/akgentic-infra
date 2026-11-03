"""The two halves of a team deletion, each in both modes, on a real community stack.

``team_delete_mode`` decides the record (purge or retire) and
``workspace_delete_mode`` decides the trees (remove the team's own, or keep
every one). The two are independent, so every combination is driven here
through ``TeamService.delete_team`` against a tree a real ``WorkspaceTool``
bound — the same stack, and the same reasons for its shape, as
``test_team_deletion_removes_the_bound_tree``.

The modes are set through the environment, before wiring, because that is how
an operator sets them: ``CommunitySettings`` reads both variables.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path, PurePosixPath

import pytest
from akgentic.team.models import Process, TeamStatus
from akgentic.tool.workspace import TEAM_KIND, WorkspaceTool, git_dir_for, meta_dir_for

from akgentic.infra.errors import TeamNotFoundError
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.services.team_service import TeamService
from tests.server.services.test_team_deletion_removes_the_bound_tree import (
    OWNER,
    _create_bound_team,
    _wire_community_with,
)


def _services_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, team: str, workspace: str
) -> Generator[CommunityServices, None, None]:
    monkeypatch.setenv("AKGENTIC_TEAM_DELETE_MODE", team)
    monkeypatch.setenv("AKGENTIC_WORKSPACE_DELETE_MODE", workspace)
    yield from _wire_community_with(tmp_path, monkeypatch, WorkspaceTool())


@pytest.fixture()
def soft_hard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[CommunityServices, None, None]:
    """Soft record, hard workspace."""
    yield from _services_with(tmp_path, monkeypatch, team="soft", workspace="hard")


@pytest.fixture()
def soft_soft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[CommunityServices, None, None]:
    """Soft record, soft workspace."""
    yield from _services_with(tmp_path, monkeypatch, team="soft", workspace="soft")


@pytest.fixture()
def hard_soft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[CommunityServices, None, None]:
    """Hard record, soft workspace."""
    yield from _services_with(tmp_path, monkeypatch, team="hard", workspace="soft")


@pytest.fixture()
def hard_hard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[CommunityServices, None, None]:
    """The default: both hard, set explicitly."""
    yield from _services_with(tmp_path, monkeypatch, team="hard", workspace="hard")


def _service(services: CommunityServices) -> TeamService:
    assert services.team_service is not None
    return services.team_service


def _bound_tree_with_sidecars(services: CommunityServices, tmp_path: Path) -> tuple[Process, Path]:
    """Create the desk team, check its bound tree, and seed both sidecars."""
    process = _create_bound_team(services)
    relative = PurePosixPath(OWNER) / TEAM_KIND / str(process.team_id)
    tree = tmp_path / "workspaces" / relative
    assert tree.is_dir(), f"the Manager's bind wrote no tree at {tree}"
    journal = git_dir_for(tree)
    journal.mkdir(parents=True, exist_ok=True)
    (journal / "HEAD").write_text("ref: refs/heads/main\n")
    index = meta_dir_for(str(relative))
    (index / "rag").mkdir(parents=True, exist_ok=True)
    (index / "rag" / "doc.yaml").write_text("text: extracted\n")
    return process, tree


def _stop_and_delete(services: CommunityServices, process: Process) -> None:
    service = _service(services)
    service.stop_team(process.team_id)
    service.delete_team(process.team_id)


class TestSoftRecordHardWorkspace:
    """The record can be recovered, but its files are gone."""

    def test_the_record_is_retired_and_the_tree_with_both_sidecars_is_gone(
        self, soft_hard: CommunityServices, tmp_path: Path
    ) -> None:
        process, tree = _bound_tree_with_sidecars(soft_hard, tmp_path)
        relative = PurePosixPath(OWNER) / TEAM_KIND / str(process.team_id)

        _stop_and_delete(soft_hard, process)

        retired = soft_hard.event_store.load_team(process.team_id)
        assert retired is not None
        assert retired.status == TeamStatus.DELETED
        assert soft_hard.event_store.load_events(process.team_id)
        assert not tree.exists()
        assert not git_dir_for(tree).exists()
        assert not meta_dir_for(str(relative)).exists()


class TestSoftRecordSoftWorkspace:
    """Nothing is removed: the record is retired and the tree stays."""

    def test_the_record_is_retired_and_the_tree_is_still_on_disk(
        self, soft_soft: CommunityServices, tmp_path: Path
    ) -> None:
        process, tree = _bound_tree_with_sidecars(soft_soft, tmp_path)

        _stop_and_delete(soft_soft, process)

        retired = soft_soft.event_store.load_team(process.team_id)
        assert retired is not None
        assert retired.status == TeamStatus.DELETED
        assert tree.is_dir()
        assert git_dir_for(tree).is_dir()


class TestHardRecordSoftWorkspace:
    """The trees stay on disk with no record pointing at them."""

    def test_the_record_is_purged_and_the_tree_is_still_on_disk(
        self, hard_soft: CommunityServices, tmp_path: Path
    ) -> None:
        process, tree = _bound_tree_with_sidecars(hard_soft, tmp_path)

        _stop_and_delete(hard_soft, process)

        assert hard_soft.event_store.load_team(process.team_id) is None
        assert tree.is_dir()


class TestHardRecordHardWorkspace:
    """The default, unchanged: the record is purged and the team's own tree removed."""

    def test_the_record_is_purged_and_the_tree_is_gone(
        self, hard_hard: CommunityServices, tmp_path: Path
    ) -> None:
        process, tree = _bound_tree_with_sidecars(hard_hard, tmp_path)

        _stop_and_delete(hard_hard, process)

        assert hard_hard.event_store.load_team(process.team_id) is None
        assert not tree.exists()
        assert not git_dir_for(tree).exists()


class TestARetiredTeamIsNotFound:
    """After a soft delete the service treats the team as gone — except to an audit listing."""

    def test_get_team_listing_and_a_second_delete(
        self, soft_hard: CommunityServices, tmp_path: Path
    ) -> None:
        service = _service(soft_hard)
        process, _ = _bound_tree_with_sidecars(soft_hard, tmp_path)
        _stop_and_delete(soft_hard, process)

        assert service.get_team(process.team_id) is None
        listed, total = service.list_teams(user_id=OWNER)
        assert process.team_id not in {row.team_id for row in listed}
        assert total == len(listed)
        retired, _ = service.list_teams(user_id=OWNER, status=TeamStatus.DELETED)
        assert [row.team_id for row in retired] == [process.team_id]
        with pytest.raises(TeamNotFoundError, match="not found"):
            service.delete_team(process.team_id)
