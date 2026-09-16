"""Behaviour of the workspace-tree reaper.

This is the one reaper whose deletions cannot be undone, so most of what is
asserted here is what it refuses to touch.

Every kind name, scope name and sidecar suffix is spelled through the tool's
exported constants and never as a literal — a spec written that way stays
correct when the tool renames one, where a literal would go red, or silently
stop testing the reserved name, on whichever spelling it did not match.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import pytest
from akgentic.tool.workspace import (
    GIT_DIR_SUFFIX,
    ID_KIND,
    META_DIR_SUFFIX,
    METADATA_KIND,
    SHARED_SCOPE,
    TEAM_KIND,
)

from akgentic.infra.maintenance.models import ResourceKind, ResourceRef
from akgentic.infra.maintenance.reapers import WorkspaceReaper, default_workspace_root
from akgentic.infra.protocols.workspace_deletion import WorkspaceDeletionContext

OWNER = "alice"


class _RefusingPolicy:
    """A ``WorkspaceDeletionPolicy`` that keeps every tree it is asked about.

    The default ``TeamTreeOnlyPolicy`` approves exactly the trees this reaper's
    candidate rule already produces, so under it the policy filter cannot be
    observed at all. A refusing policy is the only way to assert the filter is
    wired, which is what keeps it live until a tier ships one of its own.
    """

    def __init__(self) -> None:
        self.asked: list[WorkspaceDeletionContext] = []

    def may_delete(self, *, ctx: WorkspaceDeletionContext) -> bool:
        """Record the question and refuse it."""
        self.asked.append(ctx)
        return False


class _ApprovingPolicy:
    """A ``WorkspaceDeletionPolicy`` that approves every tree it is asked about.

    Needed because the reaper has **two** independent guards over the same
    ground, and the default policy hides one of them: ``TeamTreeOnlyPolicy``
    refuses any kind that is not ``_team``, so under it the reaper's own kind
    filter cannot be observed and a spec that dropped it would still pass. A
    permissive policy is what makes the structural rule — the kind segment, not
    a delegated verdict — assertable on its own.
    """

    def may_delete(self, *, ctx: WorkspaceDeletionContext) -> bool:
        """Approve unconditionally."""
        return True


def _tree(
    root: Path,
    leaf: str,
    *,
    scope: str = OWNER,
    kind: str = TEAM_KIND,
    files: int = 1,
    journal: bool = False,
    index: bool = False,
) -> Path:
    """Create one ``<root>/<scope>/<kind>/<leaf>`` tree and its requested sidecars."""
    path = root / scope / kind / leaf
    path.mkdir(parents=True)
    for number in range(files):
        (path / f"note-{number}.txt").write_text("content")
    if journal:
        journal_dir = path.parent / f"{leaf}{GIT_DIR_SUFFIX}"
        journal_dir.mkdir()
        (journal_dir / "HEAD").write_text("ref: refs/heads/main")
    if index:
        rag = path.parent / f"{leaf}{META_DIR_SUFFIX}" / "rag"
        rag.mkdir(parents=True)
        (rag / "note-0.yaml").write_text("text: the extracted document\n")
    return path


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def test_a_team_tree_is_a_candidate(tmp_path: Path) -> None:
    """``<scope>/_team/<team_id>`` is the tree a team's own deletion would take."""
    team_id = str(uuid.uuid4())
    path = _tree(tmp_path, team_id, files=3)

    refs = WorkspaceReaper(tmp_path).scan()

    assert len(refs) == 1
    assert refs[0].kind is ResourceKind.WORKSPACE
    assert refs[0].team_id == team_id
    assert refs[0].detail == str(path)
    assert refs[0].size_hint == 3


def test_the_reference_is_keyed_and_labelled_on_the_whole_path(tmp_path: Path) -> None:
    """A leaf is unique only within a scope and a kind, so neither may be dropped.

    The driver diffs ``claim_key`` against paths a live team claims, and an
    operator reads ``label`` to see which scope an orphan is in.
    """
    team_id = str(uuid.uuid4())
    _tree(tmp_path, team_id)

    ref = WorkspaceReaper(tmp_path).scan()[0]

    assert ref.claim_key == f"{OWNER}/{TEAM_KIND}/{team_id}"
    assert ref.label == f"{OWNER}/{TEAM_KIND}/{team_id}"


def test_a_shared_team_tree_is_a_candidate_too(tmp_path: Path) -> None:
    """``_shared/_team/<team_id>`` is still exactly one team's own tree.

    The leaf is a team id the resolver derived from the binding team, so no
    other team's card can produce it — which is why the sharing axis does not
    change whose tree it is.
    """
    team_id = str(uuid.uuid4())
    _tree(tmp_path, team_id, scope=SHARED_SCOPE)

    refs = WorkspaceReaper(tmp_path).scan()

    assert [ref.claim_key for ref in refs] == [f"{SHARED_SCOPE}/{TEAM_KIND}/{team_id}"]


def test_an_id_tree_and_a_meta_tree_are_not_in_the_plan_at_all(tmp_path: Path) -> None:
    """No team id addresses them, so no team's deletion can orphan them.

    They must be absent from the plan, not merely left unpurged: a named tree is
    reachable by every team of the principal and a metadata tree by every team
    carrying those values, so this reaper cannot know when either is spent.
    """
    team_id = str(uuid.uuid4())
    _tree(tmp_path, team_id)
    _tree(tmp_path, "notes", kind=ID_KIND)
    _tree(tmp_path, "customer_id-ACME", kind=METADATA_KIND)

    refs = WorkspaceReaper(tmp_path).scan()

    assert [ref.team_id for ref in refs] == [team_id]


def test_an_unreserved_kind_is_skipped(tmp_path: Path) -> None:
    """A kind this layout does not know belongs to no team and is nobody's orphan."""
    _tree(tmp_path, str(uuid.uuid4()), kind="_experimental")

    assert WorkspaceReaper(tmp_path).scan() == []


def test_only_the_team_kind_is_a_candidate_even_under_a_permissive_policy(
    tmp_path: Path,
) -> None:
    """The kind segment is the reaper's own rule, not one it delegates to a policy.

    Every tree here has a UUID-shaped leaf, which an operator can produce with
    ``WorkspaceTool(workspace_id="<a uuid>")``, so the leaf check does not
    exclude them — and the policy approves everything, so it does not either.
    What is left is the kind filter, and it is the only thing that can make this
    scan empty. Under the default ``TeamTreeOnlyPolicy`` the same scan is empty
    whether the filter is there or not, which is precisely why this spec exists.
    """
    for kind in (ID_KIND, METADATA_KIND, "_experimental"):
        _tree(tmp_path, str(uuid.uuid4()), kind=kind)

    assert WorkspaceReaper(tmp_path, _ApprovingPolicy()).scan() == []


def test_the_team_kind_is_still_a_candidate_under_that_same_policy(tmp_path: Path) -> None:
    """Otherwise the spec above would pass on a reaper that scanned nothing at all."""
    team_id = str(uuid.uuid4())
    _tree(tmp_path, team_id)

    refs = WorkspaceReaper(tmp_path, _ApprovingPolicy()).scan()

    assert [ref.team_id for ref in refs] == [team_id]


def test_a_leaf_that_is_not_a_team_id_is_not_a_candidate(tmp_path: Path) -> None:
    """The kind says the leaf should be a team id; a leaf that is not, is not one."""
    _tree(tmp_path, "archive")

    assert WorkspaceReaper(tmp_path).scan() == []


def test_a_tree_at_the_root_is_not_a_candidate(tmp_path: Path) -> None:
    """The two-segment layout is gone; a bare UUID at the root names no tree."""
    bare = tmp_path / str(uuid.uuid4())
    bare.mkdir()
    (bare / "note.txt").write_text("content")

    assert WorkspaceReaper(tmp_path).scan() == []


def test_the_git_journal_is_not_a_candidate_on_its_own(tmp_path: Path) -> None:
    """It is reaped with its tree; as a candidate it would be an orphan twice."""
    team_id = str(uuid.uuid4())
    _tree(tmp_path, team_id, journal=True)

    refs = WorkspaceReaper(tmp_path).scan()

    assert [ref.team_id for ref in refs] == [team_id]


def test_the_index_directory_is_not_a_candidate_on_its_own(tmp_path: Path) -> None:
    """Same rule as the journal, and the sidecar that holds the document text."""
    team_id = str(uuid.uuid4())
    _tree(tmp_path, team_id, index=True)

    refs = WorkspaceReaper(tmp_path).scan()

    assert [ref.team_id for ref in refs] == [team_id]


def test_a_symlinked_scope_is_skipped(tmp_path: Path) -> None:
    """A link points at something whose ownership this reaper cannot reason about."""
    elsewhere = tmp_path / "elsewhere"
    _tree(elsewhere, str(uuid.uuid4()), scope="bob")
    root = tmp_path / "workspaces"
    root.mkdir()
    (root / OWNER).symlink_to(elsewhere / "bob", target_is_directory=True)

    assert WorkspaceReaper(root).scan() == []
    assert (elsewhere / "bob" / TEAM_KIND).is_dir()


def test_a_symlinked_kind_is_skipped(tmp_path: Path) -> None:
    """A kind directory is a directory of other teams' trees; a link to one is not ours."""
    elsewhere = tmp_path / "elsewhere"
    _tree(elsewhere, str(uuid.uuid4()), scope="bob")
    root = tmp_path / "workspaces"
    (root / OWNER).mkdir(parents=True)
    (root / OWNER / TEAM_KIND).symlink_to(elsewhere / "bob" / TEAM_KIND, target_is_directory=True)

    assert WorkspaceReaper(root).scan() == []
    assert list((elsewhere / "bob" / TEAM_KIND).iterdir()) != []


def test_a_symlinked_leaf_is_skipped(tmp_path: Path) -> None:
    """The link's target is somebody else's tree, and it must survive untouched."""
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "note.txt").write_text("content")
    root = tmp_path / "workspaces"
    (root / OWNER / TEAM_KIND).mkdir(parents=True)
    (root / OWNER / TEAM_KIND / str(uuid.uuid4())).symlink_to(target, target_is_directory=True)

    assert WorkspaceReaper(root).scan() == []
    assert (target / "note.txt").exists()


def test_files_at_every_level_are_skipped(tmp_path: Path) -> None:
    """A stray file is not a scope, a kind or a tree, whatever it is called."""
    (tmp_path / OWNER).mkdir()
    (tmp_path / "stray-scope").write_text("not a directory")
    (tmp_path / OWNER / TEAM_KIND).mkdir()
    (tmp_path / OWNER / "stray-kind").write_text("not a directory")
    (tmp_path / OWNER / TEAM_KIND / str(uuid.uuid4())).write_text("not a directory")

    assert WorkspaceReaper(tmp_path).scan() == []


def test_a_missing_root_scans_empty_rather_than_failing(tmp_path: Path) -> None:
    """A deployment whose agents never wrote a file has nothing here."""
    assert WorkspaceReaper(tmp_path / "never-created").scan() == []


def test_the_scan_is_sorted_so_a_plan_is_stable(tmp_path: Path) -> None:
    """An operator who reads a dry run twice must see the same plan twice."""
    leaves = sorted(str(uuid.uuid4()) for _ in range(3))
    for scope in ("bob", OWNER):
        for leaf in reversed(leaves):
            _tree(tmp_path, leaf, scope=scope)

    keys = [ref.claim_key for ref in WorkspaceReaper(tmp_path).scan()]

    assert keys == [f"{scope}/{TEAM_KIND}/{leaf}" for scope in ("alice", "bob") for leaf in leaves]


def test_the_scan_carries_the_directory_age(tmp_path: Path) -> None:
    """mtime drives the grace period, so an active workspace reads as young."""
    _tree(tmp_path, str(uuid.uuid4()))

    age = WorkspaceReaper(tmp_path).scan()[0].age_seconds

    assert age is not None
    assert age < 60


def test_an_old_directory_reads_as_old(tmp_path: Path) -> None:
    """Otherwise nothing would ever pass the grace period and be reaped."""
    path = _tree(tmp_path, str(uuid.uuid4()))
    os.utime(path, (1_600_000_000, 1_600_000_000))

    age = WorkspaceReaper(tmp_path).scan()[0].age_seconds

    assert age is not None
    assert age > 86_400


# ---------------------------------------------------------------------------
# The deletion policy
# ---------------------------------------------------------------------------


def test_a_refusing_policy_condemns_nothing(tmp_path: Path) -> None:
    """The reaper must never condemn what the delete path's policy would keep.

    A refused tree is absent from the plan rather than present and unpurged: it
    then also drops out of ``scanned``, lowering the denominator of the
    orphan-fraction guard, which is the conservative direction.
    """
    _tree(tmp_path, str(uuid.uuid4()))
    policy = _RefusingPolicy()

    assert WorkspaceReaper(tmp_path, policy).scan() == []
    assert len(policy.asked) == 1


def test_the_policy_is_asked_about_the_split_path(tmp_path: Path) -> None:
    """A policy decides on kind and leaf far more often than on the joined string."""
    team_id = str(uuid.uuid4())
    _tree(tmp_path, team_id, scope=SHARED_SCOPE)
    policy = _RefusingPolicy()

    WorkspaceReaper(tmp_path, policy).scan()

    ctx = policy.asked[0]
    assert (ctx.scope, ctx.kind, ctx.leaf) == (SHARED_SCOPE, TEAM_KIND, team_id)
    assert str(ctx.path) == f"{SHARED_SCOPE}/{TEAM_KIND}/{team_id}"
    assert str(ctx.team_id) == team_id


def test_the_owner_of_a_principal_tree_is_the_scope_segment(tmp_path: Path) -> None:
    """The team is gone, so its ``Process.user_id`` cannot be read.

    ``user_segment`` is identity — no encoding, no digest — so the scope segment
    *is* the owner's user id for a principal scope, and the reserved
    ``_shared`` where the scope names no principal. A tier policy keying on the
    owner therefore cannot match a shared tree and refuses it, which is the safe
    direction for an unrecoverable delete.
    """
    _tree(tmp_path, str(uuid.uuid4()))
    _tree(tmp_path, str(uuid.uuid4()), scope=SHARED_SCOPE)
    policy = _RefusingPolicy()

    WorkspaceReaper(tmp_path, policy).scan()

    assert sorted(ctx.owner_user_id for ctx in policy.asked) == [SHARED_SCOPE, OWNER]


def test_the_default_policy_refuses_a_leaf_that_is_not_its_canonical_team_id(
    tmp_path: Path,
) -> None:
    """The resolver spells a team id one way, so a tree spelled another way is not its own.

    An upper-case UUID parses, but nothing the resolver writes produces one — so
    the default policy refuses it rather than condemning a directory no delete
    path would have created.
    """
    _tree(tmp_path, str(uuid.uuid4()).upper())

    assert WorkspaceReaper(tmp_path).scan() == []


# ---------------------------------------------------------------------------
# Purging
# ---------------------------------------------------------------------------


def test_purge_removes_the_tree_and_both_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A journal or an index outliving its tree is a leak whose name resolves to nothing.

    ``AKGENTIC_WORKSPACES_ROOT`` is set because ``meta_dir_for`` reads the
    tool's own root resolution: a test that only injects the reaper's root has
    ``.index`` resolve somewhere else entirely.
    """
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(tmp_path))
    team_id = str(uuid.uuid4())
    path = _tree(tmp_path, team_id, journal=True, index=True)
    reaper = WorkspaceReaper(tmp_path)

    removed = reaper.purge(reaper.scan()[0])

    assert removed == 3
    assert not path.exists()
    assert not (path.parent / f"{team_id}{GIT_DIR_SUFFIX}").exists()
    assert not (path.parent / f"{team_id}{META_DIR_SUFFIX}").exists()


def test_purge_without_sidecars_removes_only_the_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither sidecar exists until something creates it, and its absence is not an error."""
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(tmp_path))
    path = _tree(tmp_path, str(uuid.uuid4()))
    reaper = WorkspaceReaper(tmp_path)

    assert reaper.purge(reaper.scan()[0]) == 1
    assert not path.exists()


def test_purge_refuses_a_kind_directory_holding_other_teams_trees(tmp_path: Path) -> None:
    """The most dangerous single reference this reaper could be handed.

    A path two segments below the root is a ``<scope>/<kind>`` directory holding
    every team's tree of that kind — an ``rmtree`` of one would take all of
    them, so depth is checked as well as containment.
    """
    first, second = _tree(tmp_path, str(uuid.uuid4())), _tree(tmp_path, str(uuid.uuid4()))
    ref = ResourceRef(
        kind=ResourceKind.WORKSPACE,
        team_id="x",
        detail=str(tmp_path / OWNER / TEAM_KIND),
        label=f"{OWNER}/{TEAM_KIND}",
    )

    with pytest.raises(OSError, match="not a workspace tree"):
        WorkspaceReaper(tmp_path).purge(ref)
    assert first.exists()
    assert second.exists()


def test_purge_refuses_a_three_part_reference_that_resolves_one_segment_deep(
    tmp_path: Path,
) -> None:
    """``a/../b`` is three parts and names the *parent* of every tree beneath it.

    Depth measured on the literal reference alone would approve this, which is
    why it is measured twice.
    """
    tree = _tree(tmp_path, str(uuid.uuid4()))
    ref = ResourceRef(
        kind=ResourceKind.WORKSPACE,
        team_id="x",
        detail=str(tmp_path / OWNER / ".." / OWNER),
        label="traversal",
    )

    with pytest.raises(OSError, match="not a workspace tree"):
        WorkspaceReaper(tmp_path).purge(ref)
    assert tree.exists()


def test_purge_refuses_a_traversal_that_resolves_onto_a_real_tree(tmp_path: Path) -> None:
    """The other half of measuring depth twice: the *literal* reference.

    ``<scope>/<kind>/x/../<leaf>`` is five literal parts and resolves onto a
    tree exactly three deep, so the resolved measurement approves it on its own
    and only the literal one refuses. Without a case that separates the two, a
    reader can delete the literal measurement with the whole suite green — which
    is what the story's "measured twice" rule exists to prevent.
    """
    team_id = str(uuid.uuid4())
    tree = _tree(tmp_path, team_id)
    ref = ResourceRef(
        kind=ResourceKind.WORKSPACE,
        team_id=team_id,
        detail=str(tmp_path / OWNER / TEAM_KIND / "x" / ".." / team_id),
        label="traversal-onto-a-tree",
    )

    with pytest.raises(OSError, match="not a workspace tree"):
        WorkspaceReaper(tmp_path).purge(ref)
    assert tree.exists()


def test_purge_refuses_a_path_outside_the_root(tmp_path: Path) -> None:
    """The last check before an ``rmtree``: the reference must be one of ours."""
    outside = tmp_path / "outside" / "a" / "b"
    outside.mkdir(parents=True)
    root = tmp_path / "workspaces"
    root.mkdir()
    ref = ResourceRef(
        kind=ResourceKind.WORKSPACE,
        team_id="x",
        detail=str(outside),
        label="outside",
    )

    with pytest.raises(OSError, match="not a workspace tree"):
        WorkspaceReaper(root).purge(ref)
    assert outside.exists()


def test_purge_refuses_a_path_inside_a_tree(tmp_path: Path) -> None:
    """A fourth segment is a file inside somebody's workspace, not a workspace."""
    nested = _tree(tmp_path, str(uuid.uuid4())) / "subdir"
    nested.mkdir()
    ref = ResourceRef(
        kind=ResourceKind.WORKSPACE,
        team_id="x",
        detail=str(nested),
        label="nested",
    )

    with pytest.raises(OSError, match="not a workspace tree"):
        WorkspaceReaper(tmp_path).purge(ref)
    assert nested.exists()


def test_a_failed_sidecar_removal_does_not_skip_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``.git`` that will not go must not take ``.index`` with it.

    The retention half: ``<leaf>.index/rag/*.yaml`` holds the extracted text of
    every document the tree indexed, so it going is a retention fix rather than
    tidying.
    """
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(tmp_path))
    team_id = str(uuid.uuid4())
    path = _tree(tmp_path, team_id, journal=True, index=True)
    journal = path.parent / f"{team_id}{GIT_DIR_SUFFIX}"
    index = path.parent / f"{team_id}{META_DIR_SUFFIX}"
    reaper = WorkspaceReaper(tmp_path)
    ref = reaper.scan()[0]

    real_rmtree = shutil.rmtree

    def _rmtree(target: Path) -> None:
        if target == journal:
            msg = "device busy"
            raise OSError(msg)
        real_rmtree(target)

    monkeypatch.setattr(shutil, "rmtree", _rmtree)

    with pytest.raises(OSError, match="device busy"):
        reaper.purge(ref)
    assert not path.exists()
    assert journal.exists()
    assert not index.exists()


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
