"""Behaviour of the workspace-directory reaper.

This is the one reaper whose deletions cannot be undone, so most of what is
asserted here is what it refuses to touch.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from akgentic.infra.maintenance.models import ResourceKind, ResourceRef
from akgentic.infra.maintenance.reapers import WorkspaceReaper, default_workspace_root


def _workspace(root: Path, name: str, *, files: int = 1, journal: bool = False) -> Path:
    """Create one workspace directory under *root*, optionally with a journal."""
    path = root / name
    path.mkdir(parents=True)
    for index in range(files):
        (path / f"note-{index}.txt").write_text("content")
    if journal:
        journal_dir = root / f"{name}.git"
        journal_dir.mkdir()
        (journal_dir / "HEAD").write_text("ref: refs/heads/main")
    return path


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def test_a_uuid_named_directory_is_a_candidate(tmp_path: Path) -> None:
    """The default workspace name is the team id, and that is the normal case."""
    team_id = str(uuid.uuid4())
    _workspace(tmp_path, team_id, files=3)

    refs = WorkspaceReaper(tmp_path).scan()

    assert len(refs) == 1
    assert refs[0].kind is ResourceKind.WORKSPACE
    assert refs[0].team_id == team_id
    assert refs[0].detail == str(tmp_path / team_id)
    assert refs[0].size_hint == 3


def test_a_named_shared_workspace_is_never_a_candidate(tmp_path: Path) -> None:
    """``WorkspaceTool(workspace_id="shared")`` is supported and operator-owned.

    A named tree is addressed by no team id, so no team's deletion can ever
    make it an orphan. It must not appear in a plan at all.
    """
    _workspace(tmp_path, "shared")
    _workspace(tmp_path, "team-archive")

    assert WorkspaceReaper(tmp_path).scan() == []


def test_the_git_journal_is_not_scanned_on_its_own(tmp_path: Path) -> None:
    """It is reaped with its tree; as a candidate it would be an orphan twice."""
    team_id = str(uuid.uuid4())
    _workspace(tmp_path, team_id, journal=True)

    refs = WorkspaceReaper(tmp_path).scan()

    assert [ref.team_id for ref in refs] == [team_id]


def test_symlinks_are_skipped(tmp_path: Path) -> None:
    """A link points at something whose ownership this reaper cannot reason about."""
    target = tmp_path / "elsewhere"
    target.mkdir()
    root = tmp_path / "workspaces"
    root.mkdir()
    (root / str(uuid.uuid4())).symlink_to(target, target_is_directory=True)

    assert WorkspaceReaper(root).scan() == []


def test_files_in_the_root_are_skipped(tmp_path: Path) -> None:
    """A stray file is not a workspace, whatever it is called."""
    (tmp_path / f"{uuid.uuid4()}").write_text("not a directory")

    assert WorkspaceReaper(tmp_path).scan() == []


def test_a_missing_root_scans_empty_rather_than_failing(tmp_path: Path) -> None:
    """A deployment whose agents never wrote a file has nothing here."""
    assert WorkspaceReaper(tmp_path / "never-created").scan() == []


def test_the_scan_carries_the_directory_age(tmp_path: Path) -> None:
    """mtime drives the grace period, so an active workspace reads as young."""
    _workspace(tmp_path, str(uuid.uuid4()))

    age = WorkspaceReaper(tmp_path).scan()[0].age_seconds

    assert age is not None
    assert age < 60


def test_an_old_directory_reads_as_old(tmp_path: Path) -> None:
    """Otherwise nothing would ever pass the grace period and be reaped."""
    path = _workspace(tmp_path, str(uuid.uuid4()))
    os.utime(path, (1_600_000_000, 1_600_000_000))

    age = WorkspaceReaper(tmp_path).scan()[0].age_seconds

    assert age is not None
    assert age > 86_400


# ---------------------------------------------------------------------------
# Purging
# ---------------------------------------------------------------------------


def test_purge_removes_the_tree_and_its_journal(tmp_path: Path) -> None:
    """A journal outliving its tree is a leak whose name resolves to nothing."""
    team_id = str(uuid.uuid4())
    path = _workspace(tmp_path, team_id, journal=True)
    reaper = WorkspaceReaper(tmp_path)

    removed = reaper.purge(reaper.scan()[0])

    assert removed == 2
    assert not path.exists()
    assert not (tmp_path / f"{team_id}.git").exists()


def test_purge_without_a_journal_removes_only_the_tree(tmp_path: Path) -> None:
    """Most workspaces have no journal, and its absence is not an error."""
    path = _workspace(tmp_path, str(uuid.uuid4()))
    reaper = WorkspaceReaper(tmp_path)

    assert reaper.purge(reaper.scan()[0]) == 1
    assert not path.exists()


def test_purge_refuses_a_path_outside_the_root(tmp_path: Path) -> None:
    """The last check before an ``rmtree``: the reference must be one of ours."""
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "workspaces"
    root.mkdir()
    ref = ResourceRef(
        kind=ResourceKind.WORKSPACE,
        team_id="x",
        detail=str(outside),
        label="outside",
    )

    with pytest.raises(OSError, match="not a direct child"):
        WorkspaceReaper(root).purge(ref)
    assert outside.exists()


def test_purge_refuses_a_nested_path(tmp_path: Path) -> None:
    """A grandchild is a file inside somebody's workspace, not a workspace."""
    team_id = str(uuid.uuid4())
    nested = _workspace(tmp_path, team_id) / "subdir"
    nested.mkdir()
    ref = ResourceRef(
        kind=ResourceKind.WORKSPACE,
        team_id=team_id,
        detail=str(nested),
        label="nested",
    )

    with pytest.raises(OSError, match="not a direct child"):
        WorkspaceReaper(tmp_path).purge(ref)
    assert nested.exists()


# ---------------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------------


def test_the_root_defaults_to_the_configured_workspaces_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep must look where the writers write."""
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", "/srv/akgentic/workspaces")

    assert default_workspace_root() == Path("/srv/akgentic/workspaces")


def test_the_root_falls_back_to_the_same_default_the_writers_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``get_workspace`` defaults to ``./workspaces``; so must this."""
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)

    assert default_workspace_root() == Path("./workspaces")
