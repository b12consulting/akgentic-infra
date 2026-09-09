"""Tests for workspace file access endpoints.

Since ADR-048 (Story 67.1) every directory these routes open is the two-segment
``<scope>/<leaf>`` path the tool-side resolver produces, so every seed here sits
under the **team owner's** principal — ``anonymous`` for a team created by the
unauthenticated community client — rather than at the root. The caller's
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
from pathlib import Path
from typing import cast

import pytest
from akgentic.team.models import AgentCardRef, Process
from akgentic.team.ports import EventStore
from akgentic.tool.sandbox import ExecTool
from akgentic.tool.workspace import WorkspaceTool
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import State

from akgentic.infra.server.auth import RequestUser, get_request_user
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.routes.workspace import _get_workspace
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import CommunitySettings, ServerSettings

from ._workspace_cards import CaseMetadata, declare_workspaces

ANONYMOUS = "anonymous"
"""The principal the community client carries — and so its workspace scope."""


def _declare(
    services: CommunityServices,
    team_id: uuid.UUID,
    *tools: WorkspaceTool | ExecTool,
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
    ws_root = seeded_settings.workspaces_root / ANONYMOUS / str(team_id)
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
    ws_root = seeded_settings.workspaces_root / ANONYMOUS / str(team_with_workspace)
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

# Values that _validate_workspace_id must reject with HTTP 400: empty, the dot
# segments, anything containing a path separator, absolute paths, and an
# over-length (129-char) value.
_REJECTED_WORKSPACE_IDS = [
    "../x",
    "a/b",
    "a\\b",
    "/abs",
    "..",
    ".",
    "",
    "a" * 129,
]


def test_workspace_tree_honours_selector(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
) -> None:
    """GET .../tree with a declared ?workspace_id= lists the caller's own alt-ws."""
    _declare(community_services, team_with_workspace, WorkspaceTool(workspace_id="alt-ws"))
    alt_root = seeded_settings.workspaces_root / ANONYMOUS / "alt-ws"
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
    """GET .../file with a declared ?workspace_id= reads <root>/<caller>/alt-ws."""
    _declare(community_services, team_with_workspace, WorkspaceTool(workspace_id="alt-ws"))
    alt_root = seeded_settings.workspaces_root / ANONYMOUS / "alt-ws"
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

    # The write landed under <root>/<caller>/alt-ws ...
    alt_file = seeded_settings.workspaces_root / ANONYMOUS / "alt-ws" / "uploaded-alt.txt"
    assert alt_file.exists()
    assert alt_file.read_bytes() == b"alt upload"

    # ... never at the unscoped root, which is the layout this story removes.
    assert not (seeded_settings.workspaces_root / "alt-ws").exists()

    # ... and NOT under the team directory (isolation both ways).
    team_file = (
        seeded_settings.workspaces_root / ANONYMOUS / str(team_with_workspace) / "uploaded-alt.txt"
    )
    assert not team_file.exists()
    read_back = client.get(
        f"/workspace/{team_with_workspace}/file", params={"path": "uploaded-alt.txt"}
    )
    assert read_back.status_code == 404


@pytest.mark.parametrize("bad_value", _REJECTED_WORKSPACE_IDS)
def test_workspace_tree_rejects_bad_selector(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    bad_value: str,
) -> None:
    """GET .../tree returns 400 for any malformed workspace_id and creates no stray dir."""
    before = set(seeded_settings.workspaces_root.iterdir())
    resp = client.get(f"/workspace/{team_with_workspace}/tree", params={"workspace_id": bad_value})
    assert resp.status_code == 400
    # No directory was created or read outside the existing workspace roots.
    assert set(seeded_settings.workspaces_root.iterdir()) == before


@pytest.mark.parametrize("bad_value", _REJECTED_WORKSPACE_IDS)
def test_workspace_file_read_rejects_bad_selector(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    bad_value: str,
) -> None:
    """GET .../file returns 400 for any malformed workspace_id and creates no stray dir."""
    before = set(seeded_settings.workspaces_root.iterdir())
    resp = client.get(
        f"/workspace/{team_with_workspace}/file",
        params={"path": "output.txt", "workspace_id": bad_value},
    )
    assert resp.status_code == 400
    assert set(seeded_settings.workspaces_root.iterdir()) == before


@pytest.mark.parametrize("bad_value", _REJECTED_WORKSPACE_IDS)
def test_workspace_file_upload_rejects_bad_selector(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    bad_value: str,
) -> None:
    """POST .../file returns 400 for any malformed workspace_id and creates no stray dir."""
    before = set(seeded_settings.workspaces_root.iterdir())
    resp = client.post(
        f"/workspace/{team_with_workspace}/file",
        params={"workspace_id": bad_value},
        data={"path": "evil.txt"},
        files={"file": ("evil.txt", b"data", "text/plain")},
    )
    assert resp.status_code == 400
    assert set(seeded_settings.workspaces_root.iterdir()) == before


# --- Route-level authorization: path team_id (ADR-034 §Layered authz, AC1-AC5) ---


def _identity(app: FastAPI, user: RequestUser) -> TestClient:
    """A TestClient whose request-user seam resolves to ``user``."""
    app.dependency_overrides[get_request_user] = lambda: user
    return TestClient(app)


def _owned_team_with_file(
    owner_client: TestClient, ws_root_parent: Path, owner_user_id: str
) -> uuid.UUID:
    """Create a team via REST under the owner's identity and seed ``output.txt``.

    The seed sits at ``<root>/<owner>/<team_id>`` — the two-segment layout — so
    it is reachable by the owner and by nobody else's principal.
    """
    resp = owner_client.post("/teams/", json={"catalog_namespace": "test-team"})
    assert resp.status_code == 201
    team_id = uuid.UUID(resp.json()["team_id"])
    ws_root = ws_root_parent / owner_user_id / str(team_id)
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
    one the team's agents write to — ``<owner>/<team_id>``, with the owner's file
    in it. Scoping on the caller instead would send an already-authorized admin
    to their own empty ``<admin>/<team_id>``, which is worse than a refusal:
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
        seeded_settings.workspaces_root / "alice" / str(team_id) / "by-admin.txt"
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
    notes = root / "alice" / "notes"
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
    alt_root = seeded_settings.workspaces_root / ANONYMOUS / str(stray)
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


def test_exec_tool_declared_workspace_is_not_404(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
) -> None:
    """An ``ExecTool``-only team must not 404 on the id it is configured with.

    ``ExecTool`` carries no ``workspace_metadata_keys``, so this is also the
    route-level exercise of the missing attribute alongside a ``WorkspaceTool``.
    """
    _declare(
        community_services,
        team_with_workspace,
        ExecTool(workspace_id="shell"),
        WorkspaceTool(workspace_id="notes"),
    )
    shell_root = seeded_settings.workspaces_root / ANONYMOUS / "shell"
    shell_root.mkdir(parents=True, exist_ok=True)
    (shell_root / "run.log").write_text("ran")

    resp = client.get(f"/workspace/{team_with_workspace}/tree", params={"workspace_id": "shell"})
    assert resp.status_code == 200
    assert "run.log" in [e["name"] for e in resp.json()["entries"]]

    notes = client.get(f"/workspace/{team_with_workspace}/tree", params={"workspace_id": "notes"})
    assert notes.status_code == 200


def test_metadata_card_resolves_under_meta_scope(
    client: TestClient,
    team_with_workspace: uuid.UUID,
    seeded_settings: ServerSettings,
    community_services: CommunityServices,
) -> None:
    """AC #1, second half: a metadata card serves ``<root>/_meta/<joined key>``."""
    _declare(
        community_services,
        team_with_workspace,
        WorkspaceTool(workspace_metadata_keys=["customer_id", "case_id"]),
        metadata=CaseMetadata(),
    )
    leaf = "customer_id-ACME__case_id-42"
    meta_root = seeded_settings.workspaces_root / "_meta" / leaf
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / "case.txt").write_text("shared by declaration")

    resp = client.get(f"/workspace/{team_with_workspace}/tree", params={"workspace_id": leaf})
    assert resp.status_code == 200
    assert "case.txt" in [e["name"] for e in resp.json()["entries"]]
    # The shared tree is NOT under the caller's principal.
    assert not (seeded_settings.workspaces_root / ANONYMOUS / leaf).exists()


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
    declaring ``notes`` on a team of his own lands in ``bob/notes``, and
    reaching ``alice/notes`` would require a team Alice owns — which
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
        notes = root / owner / "notes"
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
    """A present workspace_id with no authorized map is 404, never an unscoped path.

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
            service=cast(TeamService, None),
            workspace_id="notes",
        )
    assert excinfo.value.status_code == 404
