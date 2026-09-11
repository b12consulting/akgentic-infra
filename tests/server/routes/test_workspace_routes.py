"""Tests for workspace file access endpoints.

Every directory these routes open is the three-segment ``<scope>/<kind>/<leaf>``
path the tool-side resolver produces, so every seed here sits under the **team
owner's** principal — ``anonymous`` for a team created by the unauthenticated
community client — and under the kind its card declares: ``_team`` for the
team's own tree, ``_id`` for a named one, ``_meta`` for a metadata-keyed one.
Each seed spells that path literally, so a spec pins the layout rather than
agreeing with whatever the resolver returns. The caller's
identity governs authorization; the owner's governs the path, which is why an
admin reading a team they do not own still reaches the owner's files.

A ``?workspace_id=`` is served only when one of the authorized team's own cards
declares it; the seeded catalog team carries plain ``BaseConfig`` members and
therefore declares nothing, so tests that need a named workspace declare it
explicitly through ``declare_workspaces``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
import pytest
from akgentic.team.models import AgentCardRef, Process
from akgentic.team.ports import EventStore
from akgentic.tool.workspace import (
    ID_KIND,
    METADATA_KIND,
    SHARED_SCOPE,
    WorkspaceTool,
    resolve_workspace_path,
)
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import State

from akgentic.infra.server.auth import RequestUser, get_request_user
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.routes import _workspace_resolution
from akgentic.infra.server.routes._workspace_resolution import (
    stash_team_process,
    stash_workspace_path,
)
from akgentic.infra.server.routes.workspace import _get_workspace, router
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import CommunitySettings, ServerSettings

from ._workspace_cards import (
    CaseMetadata,
    declare_workspaces,
    exec_only_workspace,
    process_with_cards,
    tool_card,
)
from ._workspace_ids import PATH_SAFE_UNDECLARED_IDS, REJECTED_WORKSPACE_IDS

ANONYMOUS = "anonymous"
"""The principal the community client carries — and so its workspace scope."""


def _declare(
    services: CommunityServices,
    team_id: uuid.UUID,
    *tools: WorkspaceTool,
    role: str = "WorkspaceHolder",
    metadata: CaseMetadata | None = None,
) -> None:
    """Give a live team a card declaring *tools*, through the real card store."""
    store: EventStore = services.event_store
    process = store.load_team(team_id)
    assert process is not None
    declare_workspaces(store, process, *tools, role=role, metadata=metadata)


@pytest.fixture()
def team_with_workspace(client: TestClient, seeded_settings: ServerSettings) -> uuid.UUID:
    """Create a team via REST and seed workspace files in the caller's scope."""
    resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert resp.status_code == 201
    team_id = uuid.UUID(resp.json()["team_id"])
    ws_root = seeded_settings.workspaces_root / ANONYMOUS / "_team" / str(team_id)
    ws_root.mkdir(parents=True, exist_ok=True)
    (ws_root / "output.txt").write_text("hello world")
    (ws_root / "subdir").mkdir()
    (ws_root / "subdir" / "data.json").write_text('{"key": "value"}')
    return team_id


# --- Tree listing tests ---


def test_workspace_tree(client: TestClient, team_with_workspace: uuid.UUID) -> None:
    """GET /workspace/{team_id}/tree returns file listing."""
    resp = client.get(f"/workspace/{team_with_workspace}/tree")
    assert resp.status_code == 200
    body = resp.json()
    assert body["team_id"] == str(team_with_workspace)
    names = [e["name"] for e in body["entries"]]
    assert "output.txt" in names
    assert "subdir" in names


def test_workspace_tree_not_found(client: TestClient) -> None:
    """GET /workspace/{team_id}/tree returns 404 for non-existent team."""
    fake_id = uuid.uuid4()
    resp = client.get(f"/workspace/{fake_id}/tree")
    assert resp.status_code == 404


def test_workspace_tree_traversal_attack(
    client: TestClient, team_with_workspace: uuid.UUID
) -> None:
    """GET /workspace/{team_id}/tree rejects in-root path traversal with 403."""
    resp = client.get(f"/workspace/{team_with_workspace}/tree", params={"path": "../../etc"})
    assert resp.status_code == 403


# --- File read tests ---


def test_workspace_file_read(client: TestClient, team_with_workspace: uuid.UUID) -> None:
    """GET /workspace/{team_id}/file returns file content."""
    resp = client.get(f"/workspace/{team_with_workspace}/file", params={"path": "output.txt"})
    assert resp.status_code == 200
    assert resp.content == b"hello world"
    assert "Content-Disposition" in resp.headers


def test_workspace_file_not_found(client: TestClient, team_with_workspace: uuid.UUID) -> None:
    """GET /workspace/{team_id}/file returns 404 for non-existent file."""
    resp = client.get(
        f"/workspace/{team_with_workspace}/file", params={"path": "does-not-exist.txt"}
    )
    assert resp.status_code == 404


def test_workspace_file_size_limit(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
) -> None:
    """GET /workspace/{team_id}/file returns 413 for files exceeding 10 MB."""
    ws_root = seeded_settings.workspaces_root / ANONYMOUS / "_team" / str(team_with_workspace)
    big_file = ws_root / "huge.bin"
    big_file.write_bytes(b"\x00" * (10_485_760 + 1))
    resp = client.get(f"/workspace/{team_with_workspace}/file", params={"path": "huge.bin"})
    assert resp.status_code == 413


def test_workspace_file_traversal_attack(
    client: TestClient, team_with_workspace: uuid.UUID
) -> None:
    """GET /workspace/{team_id}/file rejects path traversal attempts with 403."""
    resp = client.get(f"/workspace/{team_with_workspace}/file", params={"path": "../../etc/passwd"})
    assert resp.status_code == 403


# --- File upload tests ---


def test_workspace_file_upload(client: TestClient, team_with_workspace: uuid.UUID) -> None:
    """POST /workspace/{team_id}/file uploads and stores file."""
    resp = client.post(
        f"/workspace/{team_with_workspace}/file",
        data={"path": "uploaded.txt"},
        files={"file": ("uploaded.txt", b"upload content", "text/plain")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["path"] == "uploaded.txt"
    assert body["size"] == len(b"upload content")

    # Verify file can be read back
    read_resp = client.get(
        f"/workspace/{team_with_workspace}/file", params={"path": "uploaded.txt"}
    )
    assert read_resp.status_code == 200
    assert read_resp.content == b"upload content"


def test_workspace_file_upload_team_not_found(client: TestClient) -> None:
    """POST /workspace/{team_id}/file returns 404 for non-existent team."""
    fake_id = uuid.uuid4()
    resp = client.post(
        f"/workspace/{fake_id}/file",
        data={"path": "test.txt"},
        files={"file": ("test.txt", b"data", "text/plain")},
    )
    assert resp.status_code == 404


def test_workspace_file_upload_size_limit(
    client: TestClient, team_with_workspace: uuid.UUID
) -> None:
    """POST /workspace/{team_id}/file returns 413 for uploads exceeding 10 MB."""
    big_data = b"\x00" * (10_485_760 + 1)
    resp = client.post(
        f"/workspace/{team_with_workspace}/file",
        data={"path": "huge-upload.bin"},
        files={"file": ("huge-upload.bin", big_data, "application/octet-stream")},
    )
    assert resp.status_code == 413


def test_workspace_file_upload_traversal_attack(
    client: TestClient, team_with_workspace: uuid.UUID
) -> None:
    """POST /workspace/{team_id}/file rejects path traversal with 403."""
    resp = client.post(
        f"/workspace/{team_with_workspace}/file",
        data={"path": "../../../etc/evil"},
        files={"file": ("evil.txt", b"malicious", "text/plain")},
    )
    assert resp.status_code == 403


# --- workspace_id selector tests (Story 33.1) ---

# The values ``validate_workspace_id`` refuses with HTTP 400 are the ones the
# tool's ``leaf_segment`` refuses: see ``REJECTED_WORKSPACE_IDS``, shared with
# the gate and unit specs. There is no length bound. A path-safe value the team
# does not declare reaches the membership check and gets its 404.
_INVALID_WORKSPACE_ID = "Invalid workspace_id"


def test_workspace_tree_honours_selector(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
) -> None:
    """GET .../tree with a declared ?workspace_id= lists the caller's own alt-ws."""
    _declare(community_services, team_with_workspace, WorkspaceTool(workspace_id="alt-ws"))
    alt_root = seeded_settings.workspaces_root / ANONYMOUS / "_id" / "alt-ws"
    alt_root.mkdir(parents=True, exist_ok=True)
    (alt_root / "alt-only.txt").write_text("alt content")

    resp = client.get(f"/workspace/{team_with_workspace}/tree", params={"workspace_id": "alt-ws"})
    assert resp.status_code == 200
    names = [e["name"] for e in resp.json()["entries"]]
    assert "alt-only.txt" in names
    # Isolation: the team directory's seeded files are NOT visible under alt-ws.
    assert "output.txt" not in names
    assert "subdir" not in names


def test_workspace_file_read_honours_selector(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
) -> None:
    """GET .../file with a declared ?workspace_id= reads <root>/<caller>/_id/alt-ws."""
    _declare(community_services, team_with_workspace, WorkspaceTool(workspace_id="alt-ws"))
    alt_root = seeded_settings.workspaces_root / ANONYMOUS / "_id" / "alt-ws"
    alt_root.mkdir(parents=True, exist_ok=True)
    (alt_root / "alt.txt").write_text("from alt")

    resp = client.get(
        f"/workspace/{team_with_workspace}/file",
        params={"path": "alt.txt", "workspace_id": "alt-ws"},
    )
    assert resp.status_code == 200
    assert resp.content == b"from alt"

    # Isolation: the same filename does NOT resolve in the team directory.
    team_resp = client.get(f"/workspace/{team_with_workspace}/file", params={"path": "alt.txt"})
    assert team_resp.status_code == 404


def test_workspace_file_upload_honours_selector(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
) -> None:
    """AC #4: POST with a declared ?workspace_id= writes into the caller's scope."""
    _declare(community_services, team_with_workspace, WorkspaceTool(workspace_id="alt-ws"))
    resp = client.post(
        f"/workspace/{team_with_workspace}/file",
        params={"workspace_id": "alt-ws"},
        data={"path": "uploaded-alt.txt"},
        files={"file": ("uploaded-alt.txt", b"alt upload", "text/plain")},
    )
    assert resp.status_code == 201
    assert resp.json()["path"] == "uploaded-alt.txt"

    # The write landed under <root>/<caller>/_id/alt-ws ...
    alt_file = seeded_settings.workspaces_root / ANONYMOUS / "_id" / "alt-ws" / "uploaded-alt.txt"
    assert alt_file.exists()
    assert alt_file.read_bytes() == b"alt upload"

    # ... never at the unscoped root, which is the layout this story removes.
    assert not (seeded_settings.workspaces_root / "alt-ws").exists()

    # ... and NOT under the team directory (isolation both ways).
    team_file = (
        seeded_settings.workspaces_root
        / ANONYMOUS
        / "_team"
        / str(team_with_workspace)
        / "uploaded-alt.txt"
    )
    assert not team_file.exists()
    read_back = client.get(
        f"/workspace/{team_with_workspace}/file", params={"path": "uploaded-alt.txt"}
    )
    assert read_back.status_code == 404


@pytest.mark.parametrize("bad_value", REJECTED_WORKSPACE_IDS)
def test_workspace_tree_rejects_bad_selector(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    bad_value: str,
) -> None:
    """GET .../tree returns 400 for any malformed workspace_id and creates no stray dir."""
    before = _directories(seeded_settings.workspaces_root)
    resp = client.get(f"/workspace/{team_with_workspace}/tree", params={"workspace_id": bad_value})
    assert resp.status_code == 400
    assert resp.json()["detail"] == _INVALID_WORKSPACE_ID
    # No directory was created or read outside the existing workspace roots.
    assert _directories(seeded_settings.workspaces_root) == before


@pytest.mark.parametrize("bad_value", REJECTED_WORKSPACE_IDS)
def test_workspace_file_read_rejects_bad_selector(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    bad_value: str,
) -> None:
    """GET .../file returns 400 for any malformed workspace_id and creates no stray dir."""
    before = _directories(seeded_settings.workspaces_root)
    resp = client.get(
        f"/workspace/{team_with_workspace}/file",
        params={"path": "output.txt", "workspace_id": bad_value},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == _INVALID_WORKSPACE_ID
    assert _directories(seeded_settings.workspaces_root) == before


@pytest.mark.parametrize("bad_value", REJECTED_WORKSPACE_IDS)
def test_workspace_file_upload_rejects_bad_selector(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    bad_value: str,
) -> None:
    """POST .../file returns 400 for any malformed workspace_id and creates no stray dir."""
    before = _directories(seeded_settings.workspaces_root)
    resp = client.post(
        f"/workspace/{team_with_workspace}/file",
        params={"workspace_id": bad_value},
        data={"path": "evil.txt"},
        files={"file": ("evil.txt", b"data", "text/plain")},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == _INVALID_WORKSPACE_ID
    assert _directories(seeded_settings.workspaces_root) == before


# --- Route-level authorization: path team_id (ADR-034 §Layered authz, AC1-AC5) ---


def _identity(app: FastAPI, user: RequestUser) -> TestClient:
    """A TestClient whose request-user seam resolves to ``user``."""
    app.dependency_overrides[get_request_user] = lambda: user
    return TestClient(app)


def _owned_team_with_file(
    owner_client: TestClient, ws_root_parent: Path, owner_user_id: str
) -> uuid.UUID:
    """Create a team via REST under the owner's identity and seed ``output.txt``.

    The seed sits at ``<root>/<owner>/_team/<team_id>`` — the three-segment
    layout — so it is reachable by the owner and by nobody else's principal.
    """
    resp = owner_client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert resp.status_code == 201
    team_id = uuid.UUID(resp.json()["team_id"])
    ws_root = ws_root_parent / owner_user_id / "_team" / str(team_id)
    ws_root.mkdir(parents=True, exist_ok=True)
    (ws_root / "output.txt").write_text("hello world")
    return team_id


def test_workspace_routes_deny_non_owner_404(app: FastAPI, seeded_settings: ServerSettings) -> None:
    """A non-owner non-admin gets 404 on tree/file/upload (no existence leak) — AC3."""
    owner = _identity(app, RequestUser(user_id="alice"))
    team_id = _owned_team_with_file(owner, seeded_settings.workspaces_root, "alice")
    # The owner reaches the route (sanity).
    assert owner.get(f"/workspace/{team_id}/tree").status_code == 200

    intruder = _identity(app, RequestUser(user_id="bob"))
    assert intruder.get(f"/workspace/{team_id}/tree").status_code == 404
    assert (
        intruder.get(f"/workspace/{team_id}/file", params={"path": "output.txt"}).status_code == 404
    )
    upload = intruder.post(
        f"/workspace/{team_id}/file",
        data={"path": "x.txt"},
        files={"file": ("x.txt", b"data", "text/plain")},
    )
    assert upload.status_code == 404
    app.dependency_overrides.clear()


def test_workspace_routes_admin_non_owner_allowed(
    app: FastAPI, seeded_settings: ServerSettings
) -> None:
    """An ``admin`` bypasses ownership and reads the OWNER's tree — AC4.

    **This is the regression the caller-vs-owner finding was about.** The admin
    clears ``require_team_access``, and the directory they then open must be the
    one the team's agents write to — ``<owner>/_team/<team_id>``, with the
    owner's file in it. Scoping on the caller instead would send an
    already-authorized admin to their own empty ``<admin>/_team/<team_id>``,
    which is worse than a refusal:
    nothing signals it, and the admin concludes the agent wrote nothing.
    """
    owner = _identity(app, RequestUser(user_id="alice"))
    team_id = _owned_team_with_file(owner, seeded_settings.workspaces_root, "alice")

    admin = _identity(app, RequestUser(user_id="root", roles=["admin"]))
    tree = admin.get(f"/workspace/{team_id}/tree")
    assert tree.status_code == 200
    assert "output.txt" in [e["name"] for e in tree.json()["entries"]]

    read = admin.get(f"/workspace/{team_id}/file", params={"path": "output.txt"})
    assert read.status_code == 200
    assert read.content == b"hello world"

    upload = admin.post(
        f"/workspace/{team_id}/file",
        data={"path": "by-admin.txt"},
        files={"file": ("by-admin.txt", b"admin data", "text/plain")},
    )
    assert upload.status_code == 201
    # The admin's write landed in the OWNER's tree — the one the agent shares.
    assert (
        seeded_settings.workspaces_root / "alice" / "_team" / str(team_id) / "by-admin.txt"
    ).read_bytes() == b"admin data"
    # Nothing was created under the admin's own principal.
    assert not (seeded_settings.workspaces_root / "root").exists()
    app.dependency_overrides.clear()


def test_admin_reads_the_owners_named_workspace_too(
    app: FastAPI,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
) -> None:
    """The same rule on the ``?workspace_id=`` path, not just the default tree."""
    root = seeded_settings.workspaces_root
    owner = _identity(app, RequestUser(user_id="alice"))
    team_id = _owned_team_with_file(owner, root, "alice")
    _declare(community_services, team_id, WorkspaceTool(workspace_id="notes"))
    notes = root / "alice" / "_id" / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "alice-note.txt").write_text("owned by alice")

    admin = _identity(app, RequestUser(user_id="root", roles=["admin"]))
    resp = admin.get(f"/workspace/{team_id}/tree", params={"workspace_id": "notes"})

    assert resp.status_code == 200
    assert [e["name"] for e in resp.json()["entries"]] == ["alice-note.txt"]
    app.dependency_overrides.clear()


def test_workspace_file_read_missing_team_404(client: TestClient) -> None:
    """GET .../file 404s for a non-existent team (now via the gate) — AC5."""
    fake_id = uuid.uuid4()
    resp = client.get(f"/workspace/{fake_id}/file", params={"path": "output.txt"})
    assert resp.status_code == 404


# --- ?workspace_id= refusal and isolation (ADR-048 Decision 7, AC #2, #3, #6) ---


def test_workspace_id_foreign_team_is_404(app: FastAPI, seeded_settings: ServerSettings) -> None:
    """A ?workspace_id= naming a foreign team's id is rejected with 404."""
    alice = _identity(app, RequestUser(user_id="alice"))
    alice_team = _owned_team_with_file(alice, seeded_settings.workspaces_root, "alice")

    bob = _identity(app, RequestUser(user_id="bob"))
    bob_team = _owned_team_with_file(bob, seeded_settings.workspaces_root, "bob")
    # bob owns bob_team (path passes) but points workspace_id at alice's team id.
    resp = bob.get(f"/workspace/{bob_team}/tree", params={"workspace_id": str(alice_team)})
    assert resp.status_code == 404
    app.dependency_overrides.clear()


def test_workspace_id_unknown_uuid_is_404(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
) -> None:
    """AC #2: a UUID naming no team is no longer passed through as a shared segment.

    This is the second removed ``return user``: the directory exists and used to
    be served on the strength of the caller naming it.
    """
    stray = uuid.uuid4()
    alt_root = seeded_settings.workspaces_root / ANONYMOUS / "_id" / str(stray)
    alt_root.mkdir(parents=True, exist_ok=True)
    (alt_root / "shared.txt").write_text("shared content")

    resp = client.get(f"/workspace/{team_with_workspace}/tree", params={"workspace_id": str(stray)})
    assert resp.status_code == 404


def test_workspace_id_undeclared_name_is_404_not_400_or_403(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    community_services: CommunityServices,
) -> None:
    """AC #2: a well-formed name no card declares is 404 — not 400, not 403."""
    _declare(community_services, team_with_workspace, WorkspaceTool(workspace_id="declared"))
    resp = client.get(
        f"/workspace/{team_with_workspace}/tree", params={"workspace_id": "undeclared"}
    )
    assert resp.status_code == 404
    assert resp.status_code not in (400, 403)


def test_workspace_id_own_team_is_allowed_when_declared(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    community_services: CommunityServices,
) -> None:
    """A bare ``WorkspaceTool()`` declares the team id, so naming it is allowed."""
    _declare(community_services, team_with_workspace, WorkspaceTool())
    resp = client.get(
        f"/workspace/{team_with_workspace}/tree",
        params={"workspace_id": str(team_with_workspace)},
    )
    assert resp.status_code == 200
    names = [e["name"] for e in resp.json()["entries"]]
    assert "output.txt" in names


def test_exec_only_declared_workspace_is_not_404(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
) -> None:
    """A shell-only card must not 404 on the id it is configured with.

    Sandboxed execution is a ``WorkspaceTool`` capability, so a shell-only
    agent declares its directory through the same fields as a file agent and
    the route serves both side by side.
    """
    _declare(
        community_services,
        team_with_workspace,
        exec_only_workspace("shell"),
        WorkspaceTool(workspace_id="notes"),
    )
    shell_root = seeded_settings.workspaces_root / ANONYMOUS / "_id" / "shell"
    shell_root.mkdir(parents=True, exist_ok=True)
    (shell_root / "run.log").write_text("ran")

    resp = client.get(f"/workspace/{team_with_workspace}/tree", params={"workspace_id": "shell"})
    assert resp.status_code == 200
    assert "run.log" in [e["name"] for e in resp.json()["entries"]]

    notes = client.get(f"/workspace/{team_with_workspace}/tree", params={"workspace_id": "notes"})
    assert notes.status_code == 200


def test_metadata_card_resolves_per_principal_under_the_meta_kind(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
) -> None:
    """A metadata card serves ``<root>/<owner>/_meta/<joined key>``.

    ``_meta`` is the card's kind, not a scope: a metadata tree sits under the
    team owner like the other two kinds, and is shared across principals only
    when its card declares ``workspace_sharable``. This one does not.
    """
    _declare(
        community_services,
        team_with_workspace,
        WorkspaceTool(workspace_metadata_keys=["customer_id", "case_id"]),
        metadata=CaseMetadata(),
    )
    leaf = "customer_id-ACME__case_id-42"
    root = seeded_settings.workspaces_root
    meta_root = root / ANONYMOUS / "_meta" / leaf
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / "case.txt").write_text("owned by the team's principal")

    resp = client.get(f"/workspace/{team_with_workspace}/tree", params={"workspace_id": leaf})
    assert resp.status_code == 200
    assert "case.txt" in [e["name"] for e in resp.json()["entries"]]
    # Neither the retired cross-user location nor the shared scope was touched.
    assert not (root / "_meta").exists()
    assert not (root / "_shared").exists()


def test_two_users_naming_one_workspace_id_stay_isolated(
    app: FastAPI,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
) -> None:
    """AC #3: the isolation regression — one string, two teams, two trees.

    Both teams declare the *same* ``workspace_id`` and each resolves under its
    own team's owner, so neither owner can reach the other's file. Before
    ADR-048 this string was a global key and both calls returned one tree.

    This is also why owner-scoping needs no extra check to be safe: Bob
    declaring ``notes`` on a team of his own lands in ``bob/_id/notes``, and
    reaching ``alice/_id/notes`` would require a team Alice owns — which
    ``require_team_access`` refuses him.

    ``_identity`` replaces one process-wide override, so each caller is
    re-established immediately before the request it makes.
    """
    root = seeded_settings.workspaces_root
    alice_team = _owned_team_with_file(_identity(app, RequestUser(user_id="alice")), root, "alice")
    bob_team = _owned_team_with_file(_identity(app, RequestUser(user_id="bob")), root, "bob")

    for team_id in (alice_team, bob_team):
        _declare(community_services, team_id, WorkspaceTool(workspace_id="notes"))
    for owner in ("alice", "bob"):
        notes = root / owner / "_id" / "notes"
        notes.mkdir(parents=True, exist_ok=True)
        (notes / f"{owner}.txt").write_text(f"{owner} only")

    alice = _identity(app, RequestUser(user_id="alice"))
    alice_tree = alice.get(f"/workspace/{alice_team}/tree", params={"workspace_id": "notes"})
    assert alice_tree.status_code == 200
    assert [e["name"] for e in alice_tree.json()["entries"]] == ["alice.txt"]
    # alice cannot reach bob's file through the shared name.
    assert (
        alice.get(
            f"/workspace/{alice_team}/file",
            params={"path": "bob.txt", "workspace_id": "notes"},
        ).status_code
        == 404
    )

    bob = _identity(app, RequestUser(user_id="bob"))
    bob_tree = bob.get(f"/workspace/{bob_team}/tree", params={"workspace_id": "notes"})
    assert bob_tree.status_code == 200
    assert [e["name"] for e in bob_tree.json()["entries"]] == ["bob.txt"]
    assert (
        bob.get(
            f"/workspace/{bob_team}/file",
            params={"path": "alice.txt", "workspace_id": "notes"},
        ).status_code
        == 404
    )
    app.dependency_overrides.clear()


def test_omitted_workspace_id_serves_the_teams_own_scoped_tree(
    app: FastAPI, seeded_settings: ServerSettings
) -> None:
    """AC #6: an omitted selector still yields the team's own tree, now scoped."""
    root = seeded_settings.workspaces_root
    alice = _identity(app, RequestUser(user_id="alice"))
    team_id = _owned_team_with_file(alice, root, "alice")

    resp = alice.get(f"/workspace/{team_id}/tree")
    assert resp.status_code == 200
    assert "output.txt" in [e["name"] for e in resp.json()["entries"]]
    # Nothing was created or read at the unscoped root.
    assert not (root / str(team_id)).exists()
    app.dependency_overrides.clear()


def test_unresolvable_card_hash_is_500_not_a_quiet_404(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    community_services: CommunityServices,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A ref whose hash the store cannot resolve fails the request loudly.

    Dropping the card would shrink the allowed set into a 404 with nothing in
    the logs saying why — the refusal path failing open into a refusal.
    """
    store: EventStore = community_services.event_store
    process = store.load_team(team_with_workspace)
    assert process is not None
    dangling = AgentCardRef(role="Ghost", card_hash="0" * 64)
    store.save_team(process.model_copy(update={"agent_cards": [*process.agent_cards, dangling]}))

    with caplog.at_level(logging.ERROR):
        resp = client.get(
            f"/workspace/{team_with_workspace}/tree", params={"workspace_id": "notes"}
        )

    assert resp.status_code == 500
    logged = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    assert str(team_with_workspace) in logged
    assert "Ghost" in logged
    assert "0" * 64 in logged


def test_card_store_is_read_once_per_request(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    community_services: CommunityServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #8: one ``load_agent_cards`` call per request, whatever the role count.

    A per-hash loop returns the identical result, so only this catches it.
    """
    _declare(
        community_services,
        team_with_workspace,
        WorkspaceTool(workspace_id="notes"),
        role="First",
    )
    _declare(
        community_services,
        team_with_workspace,
        WorkspaceTool(workspace_id="drafts"),
        role="Second",
    )
    store = community_services.event_store
    calls: list[list[str]] = []
    original = store.load_agent_cards

    def _counting(hashes: list[str]) -> dict[str, object]:
        calls.append(list(hashes))
        return original(hashes)

    monkeypatch.setattr(store, "load_agent_cards", _counting)

    resp = client.get(f"/workspace/{team_with_workspace}/tree", params={"workspace_id": "notes"})
    assert resp.status_code == 200
    assert len(calls) == 1
    assert len(calls[0]) >= 3

    # The omitted branch reads the cards too (Story 70.2): it resolves the tree
    # the team's default-layout card binds to. Once, in one batch, like the
    # named branch.
    calls.clear()
    omitted = client.get(f"/workspace/{team_with_workspace}/tree")
    assert omitted.status_code == 200
    assert len(calls) == 1
    assert len(calls[0]) >= 3


def test_team_is_read_once_per_request(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    team_service: TeamService,
    community_services: CommunityServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One ``get_team`` per request, on both the omitted and the named path.

    ``get_team`` is ``EventStore.load_team`` on the department and enterprise
    tiers — a database read. The access gate, the workspace gate and the route
    all want the same team, and a request only ever concerns one, so the gate
    stashes what it authorized. A re-read returns the identical ``Process``,
    which is why only a call-count assertion catches one coming back.
    """
    _declare(community_services, team_with_workspace, WorkspaceTool(workspace_id="notes"))
    calls: list[uuid.UUID] = []
    original = team_service.get_team

    def _counting(team_id: uuid.UUID) -> Process | None:
        calls.append(team_id)
        return original(team_id)

    monkeypatch.setattr(team_service, "get_team", _counting)

    assert client.get(f"/workspace/{team_with_workspace}/tree").status_code == 200
    assert calls == [team_with_workspace]

    calls.clear()
    named = client.get(f"/workspace/{team_with_workspace}/tree", params={"workspace_id": "notes"})
    assert named.status_code == 200
    assert calls == [team_with_workspace]


def test_unusable_owner_id_on_the_read_path_is_500(
    app: FastAPI, seeded_settings: ServerSettings, caplog: pytest.LogCaptureFixture
) -> None:
    """ADR-048 Decision 4, read-path row: an owner id that cannot be a directory.

    Reachable without anyone misbehaving — an OIDC token carrying no ``sub``
    yields ``""`` at team creation. It is a defect in the identity producer or
    in stored data, so it reads as 5xx rather than a 400 telling the caller to
    fix a field they never supplied or a 403 asserting a decision nobody made.
    """
    # The team is created by, and therefore owned by, a principal that is not a
    # usable directory name; the same identity reads it back so the ownership
    # gate still passes and the resolver is what fails.
    broken_owner = _identity(app, RequestUser(user_id=""))
    resp = broken_owner.post("/teams/", json={"catalog_namespace": "test-team"})
    assert resp.status_code == 201
    team_id = uuid.UUID(resp.json()["team_id"])

    reader = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR):
        tree = reader.get(f"/workspace/{team_id}/tree")

    assert tree.status_code == 500
    assert any(
        "workspace path resolution failed" in r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.ERROR
    )
    # Nothing was created at the root, which is what an empty scope would do.
    assert not (seeded_settings.workspaces_root / str(team_id)).exists()
    app.dependency_overrides.clear()


def test_get_workspace_fails_closed_without_the_gates_map() -> None:
    """A present workspace_id with no authorized path is 404, never an unscoped path.

    Calling ``_get_workspace`` directly is the only way to reach this arm — the
    routes always run the gate first — and it is worth reaching, because falling
    back here is precisely the bug ADR-048 closes.
    """

    class _Conn:
        def __init__(self) -> None:
            self.state = State()

    with pytest.raises(HTTPException) as excinfo:
        _get_workspace(
            uuid.uuid4(),
            CommunitySettings(),
            request=_Conn(),  # type: ignore[arg-type]
            workspace_id="notes",
        )
    assert excinfo.value.status_code == 404


# --- the metadata leaf is the team's own metadata, encoded (Story 67.2) ---
#
# A ``?workspace_id=`` naming a metadata workspace is served only when it is the
# leaf the authorized team's own ``process.metadata`` produces through the card's
# declared keys, in declaration order. Nothing parses the leaf or compares pairs:
# it is *derived* from the metadata, so another case's values, another key set,
# or the same keys in another order are absent from the declared map and refused
# with the 404 a missing team gets. Every negative below sits in the same fixture
# as a positive the same gate admits, and the positive is asserted first — a
# foreign leaf is also absent from an *empty* map, so an unpaired 404 is true of
# a gate that was deleted or a metadata card that was skipped.

_META_KEYS = ["customer_id", "case_id"]
_ACME_LEAF = "customer_id-ACME__case_id-42"
_CONTOSO_LEAF = "customer_id-CONTOSO__case_id-42"
# The three leaves ACME/42's metadata cannot produce through ``_META_KEYS``.
_FOREIGN_LEAVES = [
    _CONTOSO_LEAF,  # another customer, the same key set
    "customer_id-ACME",  # a key set the card does not declare
    "case_id-42__customer_id-ACME",  # the same keys, the other order
]
_TEAM_NOT_FOUND = "Team not found"
"""The gate's body: identical for a denied leaf and a missing team (404-over-403)."""


def _case_card() -> WorkspaceTool:
    """The one metadata card every team in this section declares."""
    return WorkspaceTool(workspace_metadata_keys=list(_META_KEYS))


def _seed_meta_tree(root: Path, leaf: str) -> str:
    """Seed ``<root>/anonymous/_meta/<leaf>/`` with one file named after the leaf.

    Every team in this section is created by the community client, so its
    metadata trees sit under that principal. Returns the seeded file's name.

    The foreign trees exist on disk so that a 404 on ``/tree`` can only be the
    gate's: ``Filesystem.__init__`` creates its root and ``list`` never 404s.
    """
    tree = root / ANONYMOUS / "_meta" / leaf
    tree.mkdir(parents=True, exist_ok=True)
    name = f"{leaf}.txt"
    (tree / name).write_text(f"seeded in {leaf}")
    return name


def _meta_listing(root: Path) -> dict[str, set[str]]:
    """Every ``anonymous/_meta/<leaf>`` directory and the names inside it."""
    meta = root / ANONYMOUS / "_meta"
    if not meta.exists():
        return {}
    return {d.name: {p.name for p in d.iterdir()} for d in meta.iterdir()}


def _directories(root: Path) -> set[Path]:
    return {p for p in root.rglob("*") if p.is_dir()}


def _tree(client: TestClient, team_id: uuid.UUID, leaf: str) -> httpx.Response:
    return client.get(f"/workspace/{team_id}/tree", params={"workspace_id": leaf})


_GATE_LOGGER = "akgentic.infra.server.routes._team_access"
_GATE_DENIED = "workspace-access gate denied"


def _denials(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """The gate's denial records — the audit line only ``require_workspace_access`` writes."""
    return [r for r in caplog.records if r.name == _GATE_LOGGER and r.getMessage() == _GATE_DENIED]


def _case_team(
    client: TestClient,
    community_services: CommunityServices,
    seeded_settings: ServerSettings,
    *,
    customer_id: str,
    leaf: str,
) -> uuid.UUID:
    """A team created by ``client`` on case ``<customer_id>/42``, its ``_meta/`` tree seeded."""
    resp = client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert resp.status_code == 201
    team_id = uuid.UUID(resp.json()["team_id"])
    _declare(
        community_services,
        team_id,
        _case_card(),
        metadata=CaseMetadata(customer_id=customer_id, case_id="42"),
    )
    _seed_meta_tree(seeded_settings.workspaces_root, leaf)
    return team_id


@pytest.fixture()
def acme_case_team(
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
) -> uuid.UUID:
    """``team_with_workspace`` on case ACME/42, with all four ``_meta/`` trees seeded first.

    Every value is encoding-free (``ACME``, ``CONTOSO``, ``42``), so no leaf
    carries a ``%`` and the segment guard's 400 cannot stand in for the gate's
    404. No named card is declared, so no leaf collapses onto a named one.
    """
    _declare(
        community_services,
        team_with_workspace,
        _case_card(),
        metadata=CaseMetadata(customer_id="ACME", case_id="42"),
    )
    for leaf in (_ACME_LEAF, *_FOREIGN_LEAVES):
        _seed_meta_tree(seeded_settings.workspaces_root, leaf)
    return team_with_workspace


def test_one_team_reaches_its_own_leaf_and_is_refused_the_three_foreign_ones(
    client: TestClient,
    acme_case_team: uuid.UUID,
    seeded_settings: ServerSettings,
) -> None:
    """AC #1: the served leaf is the team's own metadata, encoded — refused otherwise.

    The positive comes first, and it is what makes the refusals mean anything:
    it proves the map is non-empty and consulted, so the three 404s below are
    membership decisions rather than the answer an empty map gives.

    Each refusal is asserted ``== 404`` with the gate's body, never ``!= 200``:
    a 400 would be the segment guard and a 500 the resolver, and neither is a
    refusal. The reversed-order leaf is a real refusal of a *different*
    workspace — declaration order is part of the declaration — not a leftover
    of the correction that moved the tool to declaration order.
    """
    root = seeded_settings.workspaces_root
    before = _meta_listing(root)
    directories_before = _directories(root)

    own = _tree(client, acme_case_team, _ACME_LEAF)
    assert own.status_code == 200
    assert [e["name"] for e in own.json()["entries"]] == [f"{_ACME_LEAF}.txt"]

    for leaf in _FOREIGN_LEAVES:
        resp = _tree(client, acme_case_team, leaf)
        assert resp.status_code == 404, leaf
        assert resp.status_code not in (400, 403)
        assert resp.json()["detail"] == _TEAM_NOT_FOUND

    # The disk is exactly as seeded: nothing created, nothing moved. Compared
    # over every directory under the root, not at the one kind a fallback is
    # likeliest to pick: the caller's principal now holds three kinds, and a
    # sharable fallback would land under ``_shared/`` outside it altogether.
    assert _meta_listing(root) == before
    assert _directories(root) == directories_before


def test_a_foreign_leaf_is_refused_by_the_gate_not_only_by_the_routes_backstop(
    client: TestClient,
    acme_case_team: uuid.UUID,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The refusal is the gate's, and it leaves the gate's audit record.

    ``_get_workspace`` answers the same 404 on its own when a leaf is absent
    from the stashed map — the fail-closed backstop — so with the gate's
    declared check deleted, every other route-level refusal in this section
    stays green for the wrong reason. The one observable that tells the two
    apart is the denial record the gate writes, naming the leaf, the caller and
    the team's owner. Positive first: the team's own leaf leaves no such record,
    so the single record below is the foreign request's and not an accumulation.
    """
    with caplog.at_level(logging.INFO, logger=_GATE_LOGGER):
        own = _tree(client, acme_case_team, _ACME_LEAF)
        assert own.status_code == 200
        assert _denials(caplog) == []

        foreign = _tree(client, acme_case_team, _CONTOSO_LEAF)

    assert foreign.status_code == 404
    assert foreign.json()["detail"] == _TEAM_NOT_FOUND
    [denied] = _denials(caplog)
    assert denied.workspace_id == _CONTOSO_LEAF
    assert denied.user_id == ANONYMOUS
    assert denied.owner == ANONYMOUS


def test_two_teams_on_two_cases_each_reach_their_own_leaf_and_not_the_others(
    client: TestClient,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
) -> None:
    """AC #2: the same card on two teams yields two different admitted leaves.

    One caller, on purpose: the owner gate is pinned by story 67.1, and the
    variable under test here is the team's metadata. The leaf is a function of
    ``process.metadata``, not of the card — which is the sentence the decision
    maker asked for, as a test.
    """
    acme = _case_team(
        client, community_services, seeded_settings, customer_id="ACME", leaf=_ACME_LEAF
    )
    contoso = _case_team(
        client, community_services, seeded_settings, customer_id="CONTOSO", leaf=_CONTOSO_LEAF
    )

    acme_own = _tree(client, acme, _ACME_LEAF)
    assert acme_own.status_code == 200
    assert [e["name"] for e in acme_own.json()["entries"]] == [f"{_ACME_LEAF}.txt"]

    contoso_own = _tree(client, contoso, _CONTOSO_LEAF)
    assert contoso_own.status_code == 200
    assert [e["name"] for e in contoso_own.json()["entries"]] == [f"{_CONTOSO_LEAF}.txt"]

    acme_foreign = _tree(client, acme, _CONTOSO_LEAF)
    assert acme_foreign.status_code == 404
    assert acme_foreign.json()["detail"] == _TEAM_NOT_FOUND

    contoso_foreign = _tree(client, contoso, _ACME_LEAF)
    assert contoso_foreign.status_code == 404
    assert contoso_foreign.json()["detail"] == _TEAM_NOT_FOUND


def test_the_write_path_admits_the_own_leaf_and_refuses_the_foreign_one_writing_nothing(
    client: TestClient,
    acme_case_team: uuid.UUID,
    seeded_settings: ServerSettings,
) -> None:
    """AC #3: ``POST .../file`` refuses the foreign leaf the same way, and writes nothing.

    The positive first: the team's own leaf takes the upload, and it lands under
    ``anonymous/_meta/<own leaf>/`` and nowhere else — in particular not under
    another kind of the caller's principal, which is where a fallback to the
    named layout would put it. Then the foreign leaf: 404 with the gate's body,
    and the foreign tree byte-identical to its seed.
    """
    root = seeded_settings.workspaces_root
    contoso_tree = root / ANONYMOUS / "_meta" / _CONTOSO_LEAF
    contoso_before = {p.name: p.read_bytes() for p in contoso_tree.iterdir()}
    anonymous_before = set((root / ANONYMOUS).rglob("*"))

    own = client.post(
        f"/workspace/{acme_case_team}/file",
        params={"workspace_id": _ACME_LEAF},
        data={"path": "uploaded.txt"},
        files={"file": ("uploaded.txt", b"by the team", "text/plain")},
    )
    assert own.status_code == 201
    landed = root / ANONYMOUS / "_meta" / _ACME_LEAF / "uploaded.txt"
    assert landed.read_bytes() == b"by the team"
    assert list(root.rglob("uploaded.txt")) == [landed]

    foreign = client.post(
        f"/workspace/{acme_case_team}/file",
        params={"workspace_id": _CONTOSO_LEAF},
        data={"path": "intruded.txt"},
        files={"file": ("intruded.txt", b"from another case", "text/plain")},
    )
    assert foreign.status_code == 404
    assert foreign.json()["detail"] == _TEAM_NOT_FOUND
    assert {p.name: p.read_bytes() for p in contoso_tree.iterdir()} == contoso_before
    assert list(root.rglob("intruded.txt")) == []
    # The caller's own scope gained the one file the team's own leaf took, and
    # nothing on the foreign request.
    assert set((root / ANONYMOUS).rglob("*")) == anonymous_before | {landed}


def test_the_resolved_meta_path_is_400_even_for_the_team_that_owns_the_tree(
    client: TestClient,
    acme_case_team: uuid.UUID,
    seeded_settings: ServerSettings,
) -> None:
    """AC #5: ``workspace_id=<scope>/_meta/<own leaf>`` is 400 for the team whose tree it is.

    This is the exact ``workspace_path`` string a ``ResourceAttached`` event
    carries, sent back inbound. Only the leaf is ever on the wire; the scope is
    the server's to recompute from the matching card, never the client's to
    name. It is deliberately **not** a row in ``REJECTED_WORKSPACE_IDS``: those
    parametrised specs run against a team that declares nothing, so they cannot
    show that the refusal beats a legitimate declaration. Here the team *does*
    own the tree — the positive proves it — and the path form is refused anyway.
    """
    root = seeded_settings.workspaces_root
    before = _directories(root)

    assert _tree(client, acme_case_team, _ACME_LEAF).status_code == 200

    resp = _tree(client, acme_case_team, f"{ANONYMOUS}/_meta/{_ACME_LEAF}")
    assert resp.status_code == 400
    assert _directories(root) == before


def test_get_workspace_fails_closed_for_a_metadata_leaf_without_the_gates_map() -> None:
    """AC #6: the fail-closed arm holds for a leaf that *parses* as a metadata leaf.

    Beside ``test_get_workspace_fails_closed_without_the_gates_map`` on purpose:
    "if the map is missing and the leaf looks like metadata, build
    ``<scope>/_meta/<leaf>``" is the arm a decoupling refactor would most plausibly
    reach for, and this is the spec that goes red under it.
    """

    class _Conn:
        def __init__(self) -> None:
            self.state = State()

    with pytest.raises(HTTPException) as excinfo:
        _get_workspace(
            uuid.uuid4(),
            CommunitySettings(),
            request=_Conn(),  # type: ignore[arg-type]
            workspace_id=_ACME_LEAF,
        )
    assert excinfo.value.status_code == 404


# --- the check reads the scope segment (Story 70.2) ---
#
# Every workspace route converges on one gate. On both branches it selects the
# single path the route will open and decides who may reach it from that path's
# scope segment alone. A user scope goes to the wired policy with the scope as
# the owner; ``_shared`` is refused with 403 for every caller. The route opens
# only the path the gate stashed.
#
# Two rules keep these specs from passing for the wrong reason. Each refusal is
# asserted on something only the scope check produces: its ``code`` for
# ``_shared``, and its own denial record for a foreign scope. And the disk is
# compared over every path under the root, not at one folder.

_ROUTES = [
    ("GET", "/workspace/{team_id}/tree"),
    ("GET", "/workspace/{team_id}/file"),
    ("POST", "/workspace/{team_id}/file"),
]
"""Every route on ``workspace.router``. The completeness spec keeps this honest."""

_SHARED_REFUSED = "shared_workspace_entitlement_undecided"
# ``_shared`` spelled with U+017F, the long s: unchanged by ``.lower()``,
# ``_shared`` under ``.casefold()`` and on a case-insensitive filesystem.
_LONG_S_SHARED = "_ſhared"
_SCOPE_DENIED = "workspace-scope gate denied"
_TEAM_GATE_DENIED = "team-access gate denied"
_PROBE = "probe.txt"
_ALICE = RequestUser(user_id="alice")
_ADMIN = RequestUser(user_id="root", roles=["admin"])


def _served(method: str) -> int:
    return 201 if method == "POST" else 200


def _hit(
    client: TestClient,
    method: str,
    route: str,
    team_id: uuid.UUID,
    workspace_id: str | None = None,
) -> httpx.Response:
    """One request to *route*, naming *workspace_id* or, when ``None``, the team's own tree."""
    params = {} if workspace_id is None else {"workspace_id": workspace_id}
    url = route.format(team_id=team_id)
    if method == "POST":
        return client.post(
            url,
            params=params,
            data={"path": _PROBE},
            files={"file": (_PROBE, b"uploaded", "text/plain")},
        )
    if url.endswith("/file"):
        params["path"] = _PROBE
    return client.get(url, params=params)


def _seed_probe(tree: Path) -> Path:
    """Seed *tree* with ``probe.txt``, so a route that opened it would serve it."""
    tree.mkdir(parents=True, exist_ok=True)
    (tree / _PROBE).write_text(f"seeded in {tree.name}")
    return tree


def _disk(root: Path) -> dict[Path, bytes | None]:
    """Every path under *root*: a file maps to its bytes, a directory to ``None``."""
    return {p: (p.read_bytes() if p.is_file() else None) for p in root.rglob("*")}


def _scope_denials(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """The denial records only ``check_workspace_scope`` writes."""
    return [r for r in caplog.records if r.name == _GATE_LOGGER and r.getMessage() == _SCOPE_DENIED]


def _other_denials(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Denial records the scope check did not write: the team gate's and the declared set's."""
    return [
        r
        for r in caplog.records
        if r.name == _GATE_LOGGER and r.getMessage() in (_TEAM_GATE_DENIED, _GATE_DENIED)
    ]


@pytest.fixture()
def as_user(app: FastAPI) -> Iterator[Callable[[RequestUser], TestClient]]:
    """``_identity``, with the override cleared even when the spec fails."""
    yield lambda user: _identity(app, user)
    app.dependency_overrides.clear()


def _lie_about_the_scope(monkeypatch: pytest.MonkeyPatch, scope: str) -> None:
    """Make the gate's resolver put every tree under *scope*, with kind and leaf unchanged.

    The resolver scopes a principal tree on ``process.user_id``, so on a real
    request the path and the team gate always agree. Lying to the resolver is
    the one route-level way to hand the gate a path under another principal.
    """
    real = _workspace_resolution.resolve_workspace_path

    # ``**kwargs: Any``: a pass-through wrapper whose callers use the resolver's
    # own keyword set, forwarded unchanged.
    def _lying(**kwargs: Any) -> PurePosixPath:
        resolved = real(**kwargs)
        return PurePosixPath(scope, *resolved.parts[1:])

    monkeypatch.setattr(_workspace_resolution, "resolve_workspace_path", _lying)


def test_the_route_parametrization_is_every_route_on_the_router() -> None:
    """AC #4: a fourth workspace route turns this red until the specs below cover it.

    Every route counts, whatever its kind. A route with no HTTP methods (a
    websocket) is recorded under its class name rather than skipped.
    """
    on_router = {
        (method, getattr(route, "path", repr(route)))
        for route in router.routes
        for method in (getattr(route, "methods", None) or {type(route).__name__})
    }
    assert on_router == set(_ROUTES)


@pytest.mark.parametrize(("method", "route"), _ROUTES)
@pytest.mark.parametrize("named", [False, True], ids=["omitted", "named"])
def test_a_callers_own_scope_is_served_on_every_route(
    as_user: Callable[[RequestUser], TestClient],
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
    method: str,
    route: str,
    named: bool,
) -> None:
    """AC #2: the principal the path is scoped on is served, on both branches."""
    root = seeded_settings.workspaces_root
    alice = as_user(_ALICE)
    team_id = _owned_team_with_file(alice, root, "alice")
    if named:
        _declare(community_services, team_id, WorkspaceTool(workspace_id="notes"))
        tree = _seed_probe(root / "alice" / "_id" / "notes")
    else:
        tree = _seed_probe(root / "alice" / "_team" / str(team_id))

    resp = _hit(alice, method, route, team_id, "notes" if named else None)

    assert resp.status_code == _served(method)
    if method == "POST":
        assert (tree / _PROBE).read_bytes() == b"uploaded"


@pytest.mark.parametrize(("method", "route"), _ROUTES)
@pytest.mark.parametrize("named", [False, True], ids=["omitted", "named"])
def test_a_path_under_another_principals_scope_is_refused_on_every_route(
    as_user: Callable[[RequestUser], TestClient],
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    method: str,
    route: str,
    named: bool,
) -> None:
    """AC #2: the team's own owner is refused a path scoped on another principal.

    Alice owns the team and is not an admin, so ``require_team_access`` lets her
    through. Under the real ``OwnerOrAdminPolicy`` the only thing left that can
    refuse her is the scope check, reading ``mallory`` off the path.

    Positive first, on the same lying resolver: an admin is served, from
    mallory's tree. That proves the lie reaches the path the route opens, and
    that the check asks the policy rather than comparing the scope with the
    caller's id, which would refuse the admin too. The refusal is then pinned to
    the scope check by its own denial record, with neither of the other two
    denial records written.
    """
    root = seeded_settings.workspaces_root
    team_id = _owned_team_with_file(as_user(_ALICE), root, "alice")
    selector: str | None = None
    kind, leaf = "_team", str(team_id)
    if named:
        _declare(community_services, team_id, WorkspaceTool(workspace_id="notes"))
        selector, kind, leaf = "notes", "_id", "notes"
    foreign = _seed_probe(root / "mallory" / kind / leaf)
    _lie_about_the_scope(monkeypatch, "mallory")

    admin = _hit(as_user(_ADMIN), method, route, team_id, selector)
    assert admin.status_code == _served(method)
    if route.endswith("/tree"):
        assert [e["name"] for e in admin.json()["entries"]] == [_PROBE]
    elif method == "GET":
        assert admin.content == f"seeded in {leaf}".encode()
    else:
        assert (foreign / _PROBE).read_bytes() == b"uploaded"

    before = _disk(root)
    alice = as_user(_ALICE)
    with caplog.at_level(logging.INFO, logger=_GATE_LOGGER):
        refused = _hit(alice, method, route, team_id, selector)

    assert refused.status_code == 404
    assert refused.json()["detail"] == _TEAM_NOT_FOUND
    [denied] = _scope_denials(caplog)
    assert (denied.user_id, denied.owner) == ("alice", "mallory")
    assert _other_denials(caplog) == []
    assert _disk(root) == before


@pytest.mark.parametrize(("method", "route"), _ROUTES)
@pytest.mark.parametrize("caller", [_ALICE, _ADMIN], ids=["owner", "admin"])
@pytest.mark.parametrize("named", [False, True], ids=["omitted", "named"])
def test_a_shared_path_is_refused_with_its_code_for_the_owner_and_for_an_admin(
    as_user: Callable[[RequestUser], TestClient],
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
    method: str,
    route: str,
    caller: RequestUser,
    named: bool,
) -> None:
    """AC #3: a ``_shared`` path is 403 with the entitlement code, whoever asks.

    Named branch: a card ``WorkspaceTool(workspace_id="notes",
    workspace_sharable=True)``. Omitted branch: the team's default-layout card
    is ``WorkspaceTool(workspace_sharable=True)``, so its own tree is
    ``_shared/_team/<team_id>``.

    The ``_shared`` tree is seeded on disk, and so is the owner's
    ``alice/_team/<team_id>``, so a route whose check was deleted, or that
    resolved the omitted branch without the card's flag, would answer 200 rather
    than an incidental 404. Positive first: the same caller is served the same
    team's non-sharable twin, so the map is live. The refusal is asserted on
    its ``code``, which nothing but the scope check produces, and the disk is
    compared over every path.
    """
    root = seeded_settings.workspaces_root
    team_id = _owned_team_with_file(as_user(_ALICE), root, "alice")
    shared_card = (
        WorkspaceTool(workspace_id="notes", workspace_sharable=True)
        if named
        else WorkspaceTool(workspace_sharable=True)
    )
    _declare(community_services, team_id, shared_card, WorkspaceTool(workspace_id="drafts"))
    _seed_probe(root / "alice" / "_id" / "drafts")
    _seed_probe(root / SHARED_SCOPE / "_id" / "notes")
    _seed_probe(root / SHARED_SCOPE / "_team" / str(team_id))
    client = as_user(caller)

    assert _hit(client, method, route, team_id, "drafts").status_code == _served(method)

    before = _disk(root)
    refused = _hit(client, method, route, team_id, "notes" if named else None)

    assert refused.status_code == 403
    assert refused.json()["code"] == _SHARED_REFUSED
    assert _disk(root) == before


def test_a_shared_tree_named_after_the_caller_is_still_refused(
    as_user: Callable[[RequestUser], TestClient],
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
) -> None:
    """AC #3: the scope segment is the whole input, and the leaf is never parsed.

    ``_shared/_id/alice`` carries the caller's own id as its leaf. A check that
    read the leaf to recover whose tree it is would serve it. The scope says
    ``_shared``, so it is refused.
    """
    root = seeded_settings.workspaces_root
    alice = as_user(_ALICE)
    team_id = _owned_team_with_file(alice, root, "alice")
    _declare(
        community_services,
        team_id,
        WorkspaceTool(workspace_id="alice", workspace_sharable=True),
        WorkspaceTool(workspace_id="drafts"),
    )
    _seed_probe(root / SHARED_SCOPE / "_id" / "alice")

    assert _tree(alice, team_id, "drafts").status_code == 200
    before = _disk(root)
    resp = _tree(alice, team_id, "alice")

    assert resp.status_code == 403
    assert resp.json()["code"] == _SHARED_REFUSED
    assert _disk(root) == before


@pytest.mark.parametrize("named", [False, True], ids=["omitted", "named"])
def test_a_principal_whose_id_folds_to_shared_is_refused(
    as_user: Callable[[RequestUser], TestClient],
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
    caplog: pytest.LogCaptureFixture,
    named: bool,
) -> None:
    """An owner spelled ``_ſhared`` (long s) is an unusable owner: 500, and nothing is opened.

    A case-insensitive filesystem (APFS, NTFS) folds ``_ſhared/`` and opens
    ``_shared/``, so the owner of such a team must never pass the user-scope arm
    as herself: she would read and write the shared tree every sharable
    ``notes`` card binds to. The shared tree is seeded, so that outcome would
    answer 200.

    **This spec expected 403 until story 70-3, and the change is a decision
    applied, not a guard narrowed.** The tool's reserved-scope check now
    case-folds (b12consulting/akgentic-tool#383), so ``user_segment`` refuses
    this owner and the resolver never produces a path. An owner the resolver
    refuses is an unusable owner, and an unusable owner on the read path is 500,
    exactly as for ``""`` in ``test_unusable_owner_id_on_the_read_path_is_500``.
    The route's own case-folded ``_shared`` check stays as defence in depth, and
    ``test_team_access.py`` still covers it directly with this spelling.

    What matters is unchanged: nothing is served, nothing is created anywhere
    under the root (so no ``_shared`` tree is opened, since ``Filesystem``
    creates its root eagerly), and the ERROR names the owner the tool refused.
    """
    root = seeded_settings.workspaces_root
    folded = as_user(RequestUser(user_id=_LONG_S_SHARED))
    resp = folded.post("/teams/", json={"catalog_namespace": "test-team"})
    assert resp.status_code == 201
    team_id = uuid.UUID(resp.json()["team_id"])
    _declare(community_services, team_id, WorkspaceTool(workspace_id="notes"))
    _seed_probe(root / SHARED_SCOPE / "_id" / "notes")
    before = _disk(root)

    with caplog.at_level(logging.ERROR):
        refused = _hit(
            folded, "GET", "/workspace/{team_id}/tree", team_id, "notes" if named else None
        )

    assert refused.status_code == 500
    assert refused.json()["detail"] == "Workspace path could not be resolved"
    assert _PROBE not in refused.text
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(
        "workspace path resolution failed" in m and str(team_id) in m and repr(_LONG_S_SHARED) in m
        for m in errors
    ), errors
    assert _disk(root) == before


def test_a_refused_shared_upload_creates_nothing_anywhere(
    as_user: Callable[[RequestUser], TestClient],
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
) -> None:
    """AC #3: the refusal comes before any ``Filesystem``, so not even a directory appears.

    Nothing is seeded under ``_shared`` here: ``Filesystem.__init__`` creates its
    root eagerly, so a refusal that came after it would leave the directory
    behind even though the write never ran.
    """
    root = seeded_settings.workspaces_root
    alice = as_user(_ALICE)
    team_id = _owned_team_with_file(alice, root, "alice")
    _declare(community_services, team_id, WorkspaceTool(workspace_sharable=True))
    before = _disk(root)

    resp = _hit(alice, "POST", "/workspace/{team_id}/file", team_id)

    assert resp.status_code == 403
    assert resp.json()["code"] == _SHARED_REFUSED
    assert _disk(root) == before
    assert not (root / SHARED_SCOPE).exists()


def test_a_non_sharable_default_card_serves_the_owners_team_tree(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
) -> None:
    """AC #5: an omitted selector serves the tree the default card binds to, per-principal here."""
    _declare(community_services, team_with_workspace, WorkspaceTool())

    resp = client.get(f"/workspace/{team_with_workspace}/tree")

    assert resp.status_code == 200
    assert "output.txt" in [e["name"] for e in resp.json()["entries"]]
    assert not (seeded_settings.workspaces_root / SHARED_SCOPE).exists()


def test_a_sharable_named_card_does_not_move_the_teams_own_tree(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
) -> None:
    """AC #5: with no default-layout card, the omitted selector is the per-principal default.

    Only a default-layout card decides the team's own tree. A sharable *named*
    card declares a different tree and leaves this one where it was.
    """
    _declare(
        community_services,
        team_with_workspace,
        WorkspaceTool(workspace_id="notes", workspace_sharable=True),
    )

    resp = client.get(f"/workspace/{team_with_workspace}/tree")

    assert resp.status_code == 200
    assert "output.txt" in [e["name"] for e in resp.json()["entries"]]
    assert not (seeded_settings.workspaces_root / SHARED_SCOPE).exists()


def test_default_cards_that_disagree_on_sharing_are_500_with_an_error_naming_the_team(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC #5: two default-layout cards, one sharable and one not, have no right answer.

    Picking one would let the order the store returns cards decide whose tree a
    caller reaches. It is a configuration defect, so it is a 500 with an ERROR
    naming the team, and nothing is opened.
    """
    root = seeded_settings.workspaces_root
    _declare(community_services, team_with_workspace, WorkspaceTool(), role="Private")
    _declare(
        community_services, team_with_workspace, WorkspaceTool(workspace_sharable=True), role="Open"
    )
    before = _disk(root)

    with caplog.at_level(logging.ERROR):
        resp = client.get(f"/workspace/{team_with_workspace}/tree")

    assert resp.status_code == 500
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(
        str(team_with_workspace) in m and "disagree on workspace_sharable" in m for m in errors
    ), errors
    assert _disk(root) == before


def test_an_unresolvable_card_fails_the_omitted_branch_too(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC #5: the omitted branch reads the cards now, so it fails closed on one it cannot read.

    Without every card the default card's scope cannot be known. Serving the
    per-principal tree anyway would be a guess, and the wrong guess serves an
    empty directory while the agents write to the shared one.

    This is the one outcome the story moved from 200 to 500, so the operator
    must be able to tell why from the log alone: the ERROR names the team, the
    role and the hash that did not resolve, as the named branch's twin does.
    """
    store: EventStore = community_services.event_store
    process = store.load_team(team_with_workspace)
    assert process is not None
    dangling = AgentCardRef(role="Ghost", card_hash="0" * 64)
    store.save_team(process.model_copy(update={"agent_cards": [*process.agent_cards, dangling]}))
    before = _disk(seeded_settings.workspaces_root)

    with caplog.at_level(logging.ERROR):
        resp = client.get(f"/workspace/{team_with_workspace}/tree")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Workspace cards could not be read"
    logged = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    assert str(team_with_workspace) in logged
    assert "Ghost" in logged
    assert "0" * 64 in logged
    assert _disk(seeded_settings.workspaces_root) == before


class _BareConn:
    """The one thing ``_get_workspace`` needs of a request: a ``state``."""

    def __init__(self) -> None:
        self.state = State()


def test_get_workspace_fails_closed_on_the_omitted_branch_even_with_the_team_at_hand(
    tmp_path: Path,
) -> None:
    """AC #4: no stashed path is 404 for an omitted selector, as for a named one.

    The authorized team *is* stashed, so the fallback this story deleted, which
    resolved the team's own tree from it, would have everything it needs. Only
    the stashed path may be opened.

    The root is a temporary directory and must stay empty: ``Filesystem``
    creates its root eagerly, so a fallback that opened a tree before refusing
    leaves it behind, and with the settings' relative default it would leave
    it in whatever directory the suite happens to run from.
    """
    process = process_with_cards([tool_card("Writer", WorkspaceTool())])
    conn = _BareConn()
    stash_team_process(conn, process)  # type: ignore[arg-type]

    with pytest.raises(HTTPException) as excinfo:
        _get_workspace(
            process.team_id,
            CommunitySettings(workspaces_root=tmp_path),
            request=conn,  # type: ignore[arg-type]
        )

    assert excinfo.value.status_code == 404
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("workspace_id", [None, "drafts"], ids=["omitted", "named"])
def test_get_workspace_refuses_a_stashed_path_the_selector_does_not_name(
    tmp_path: Path, workspace_id: str | None
) -> None:
    """AC #4: the stash must be the path this request's selector names, or nothing opens."""
    conn = _BareConn()
    stash_workspace_path(conn, PurePosixPath("alice", "_id", "notes"))  # type: ignore[arg-type]

    with pytest.raises(HTTPException) as excinfo:
        _get_workspace(
            uuid.uuid4(),
            CommunitySettings(workspaces_root=tmp_path),
            request=conn,  # type: ignore[arg-type]
            workspace_id=workspace_id,
        )

    assert excinfo.value.status_code == 404
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("named", [False, True], ids=["omitted", "named"])
def test_get_workspace_opens_exactly_the_stashed_path(tmp_path: Path, named: bool) -> None:
    """The positive beside the refusals: the stashed path is the directory that opens."""
    team_id = uuid.uuid4()
    path = (
        PurePosixPath("alice", "_id", "notes")
        if named
        else PurePosixPath("alice", "_team", str(team_id))
    )
    conn = _BareConn()
    stash_workspace_path(conn, path)  # type: ignore[arg-type]

    _get_workspace(
        team_id,
        CommunitySettings(workspaces_root=tmp_path),
        request=conn,  # type: ignore[arg-type]
        workspace_id="notes" if named else None,
    )

    assert {p for p in tmp_path.rglob("*") if p.is_dir()} == {
        tmp_path.joinpath(*path.parts[:i]) for i in range(1, 4)
    }


# --- the route guards its selector with the tool's leaf rule (Story 70.3) ---
#
# ``?workspace_id=`` is a leaf selector. The guard in front of it is the tool's
# ``leaf_segment``, a traversal guard, and membership in the team's declared
# workspaces is the authorization. Every leaf a card can declare is therefore
# reachable, including a metadata leaf carrying ``%`` or running past 128 bytes,
# which the private regex this replaced refused with 400.
#
# Each leaf below comes from the tool's own resolver, never from infra's
# ``declared_workspace_paths``: the gate is built on that function, so a spec
# built on it would agree with whatever the gate does.

_LEAVES_THE_REGEX_REFUSED = [
    pytest.param(CaseMetadata(customer_id="Acme Corp", case_id="42"), "percent", id="space"),
    pytest.param(CaseMetadata(customer_id="Zürich", case_id="42"), "percent", id="non_ascii"),
    pytest.param(CaseMetadata(customer_id="A" * 150, case_id="42"), "length", id="over_128_bytes"),
]
"""Team metadata whose leaf the old regex refused: by its charset (``%``), or by its length."""


def _stash_ending_in(value: str) -> PurePosixPath:
    """A stashed path whose leaf is *value*, as the gate would stash a named workspace."""
    return PurePosixPath("alice", ID_KIND, value)


# The rows of ``REJECTED_WORKSPACE_IDS`` a stashed path can end in, derived from
# the shared table rather than listed again, so a row added there reaches this
# call site too. ``PurePosixPath`` does not keep ``"../x"``, ``"."``, ``"a/b"``,
# ``"/abs"`` or ``""`` as the last segment, so the stash could not name them.
_STASHABLE_REJECTED_IDS = [
    row
    for row in REJECTED_WORKSPACE_IDS
    if _stash_ending_in(str(row.values[0])).name == row.values[0]
]


@pytest.mark.parametrize(("method", "route"), _ROUTES)
@pytest.mark.parametrize(("metadata", "needed"), _LEAVES_THE_REGEX_REFUSED)
def test_a_declared_leaf_the_old_regex_refused_is_served_on_every_route(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
    method: str,
    route: str,
    metadata: CaseMetadata,
    needed: str,
) -> None:
    """AC #2: a metadata leaf with ``%`` or over 128 bytes is served when the team declares it.

    The preconditions come before any request. They prove each value really
    needed the change: two leaves carry ``%``, and the third is encoding-free
    but longer than the old 128-character bound.

    The leaf goes on the wire through ``params=``, as the frontend sends it with
    ``encodeURIComponent``. The ``%`` is encoded once by the client and decoded
    once by the server, so the selector the gate sees is byte-equal to the leaf.
    """
    root = seeded_settings.workspaces_root
    _declare(community_services, team_with_workspace, _case_card(), metadata=metadata)
    path = resolve_workspace_path(
        workspace_id=None,
        workspace_metadata_keys=list(_META_KEYS),
        team_id=str(team_with_workspace),
        user_id=ANONYMOUS,
        metadata=metadata,
        workspace_sharable=False,
    )
    assert path.parts[:2] == (ANONYMOUS, METADATA_KIND)
    if needed == "percent":
        assert "%" in path.name
    else:
        assert "%" not in path.name
        assert len(path.name.encode()) > 128
    tree = _seed_probe(root.joinpath(*path.parts))

    resp = _hit(client, method, route, team_with_workspace, workspace_id=path.name)

    assert resp.status_code == _served(method)
    if route.endswith("/tree"):
        assert [e["name"] for e in resp.json()["entries"]] == [_PROBE]
    elif method == "GET":
        assert resp.content == f"seeded in {path.name}".encode()
    else:
        assert (tree / _PROBE).read_bytes() == b"uploaded"
        assert list(root.rglob(_PROBE)) == [tree / _PROBE]


@pytest.mark.parametrize(("method", "route"), _ROUTES)
@pytest.mark.parametrize("value", PATH_SAFE_UNDECLARED_IDS)
def test_a_path_safe_value_no_card_declares_is_the_membership_404_on_every_route(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    caplog: pytest.LogCaptureFixture,
    method: str,
    route: str,
    value: str,
) -> None:
    """AC #4: a value the guard lets through, and no card declares, is 404 and never 400.

    A 404 alone could be the team gate's or the scope check's. The denial record
    only the membership check writes pins it, and there is exactly one.
    """
    root = seeded_settings.workspaces_root
    before = _disk(root)

    with caplog.at_level(logging.INFO, logger=_GATE_LOGGER):
        resp = _hit(client, method, route, team_with_workspace, workspace_id=value)

    assert resp.status_code == 404
    assert resp.json()["detail"] == _TEAM_NOT_FOUND
    [denied] = _denials(caplog)
    assert denied.workspace_id == value
    assert _disk(root) == before


def test_the_guard_is_the_tools_leaf_segment_and_not_a_copy_of_it(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    community_services: CommunityServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #1: the route consults the tool's function, so a stand-in for it decides.

    The team declares ``notes`` and is served it first. With ``leaf_segment``
    replaced by a stand-in that refuses ``notes`` only, the same request is 400.
    A guard hand-copied into infra would ignore the stand-in and still serve it.
    The stand-in's message is not echoed to the caller.
    """
    _declare(community_services, team_with_workspace, WorkspaceTool(workspace_id="notes"))
    assert _tree(client, team_with_workspace, "notes").status_code == 200

    def _refuses_notes(value: str) -> str:
        if value == "notes":
            raise ValueError("the stand-in refuses notes")
        return value

    monkeypatch.setattr(_workspace_resolution, "leaf_segment", _refuses_notes)
    resp = _tree(client, team_with_workspace, "notes")

    assert resp.status_code == 400
    assert resp.json()["detail"] == _INVALID_WORKSPACE_ID


@pytest.mark.parametrize("value", _STASHABLE_REJECTED_IDS)
def test_get_workspace_guards_the_selector_even_when_the_stash_names_it(
    tmp_path: Path, value: str
) -> None:
    """AC #3: the route's own guard refuses a non-leaf selector the stash agrees with.

    The stashed path ends in the selector, so the leaf comparison passes and
    only the guard can refuse. The gate runs the same guard first on every real
    request, so a direct call is the one way to see this call site on its own.
    The root must stay empty: ``Filesystem`` creates its root eagerly.
    """
    conn = _BareConn()
    stash_workspace_path(conn, _stash_ending_in(value))  # type: ignore[arg-type]

    with pytest.raises(HTTPException) as excinfo:
        _get_workspace(
            uuid.uuid4(),
            CommunitySettings(workspaces_root=tmp_path),
            request=conn,  # type: ignore[arg-type]
            workspace_id=value,
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == _INVALID_WORKSPACE_ID
    assert list(tmp_path.iterdir()) == []
