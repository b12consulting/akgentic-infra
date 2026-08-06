"""Tests for the FastAPI application factory and CORS middleware."""

from __future__ import annotations

import logging

import pytest
from akgentic.catalog import ENV_VAR as CATALOG_PREFIXES_ENV_VAR
from akgentic.catalog import allowed_prefixes
from fastapi import FastAPI
from fastapi.testclient import TestClient

from akgentic.infra.server import app as app_module
from akgentic.infra.server.app import create_app
from akgentic.infra.server.deps import CommunityServices, TierServices
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import CommunitySettings, ServerSettings


def test_create_app_returns_fastapi(
    seeded_settings: CommunitySettings,
    community_services: CommunityServices,
) -> None:
    """create_app returns a FastAPI instance with routes mounted."""
    app = create_app(community_services, seeded_settings)
    assert app.title == "Akgentic Platform API"
    route_paths = [r.path for r in app.routes]  # type: ignore[union-attr]
    assert "/teams" in route_paths
    assert "/teams/{team_id}" in route_paths


def test_cors_headers_present(client: TestClient) -> None:
    """Responses include CORS headers for allowed origins."""
    resp = client.options(
        "/teams/",
        headers={
            "Origin": "http://localhost:4200",
            "Access-Control-Request-Method": "POST",
        },
    )
    # When allow_origins=["*"], Starlette reflects the request Origin
    assert "access-control-allow-origin" in resp.headers


def test_custom_cors_origins(
    seeded_settings: CommunitySettings,
    community_services: CommunityServices,
) -> None:
    """create_app respects custom cors_origins from settings."""
    settings = CommunitySettings(
        workspaces_root=seeded_settings.workspaces_root,
        cors_origins=["http://example.com"],
    )
    app = create_app(community_services, settings)
    test_client = TestClient(app)
    resp = test_client.options(
        "/teams/",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "http://example.com"


# ---------------------------------------------------------------------------
# Webhook wiring tests
# ---------------------------------------------------------------------------


def test_create_app_includes_webhook_routes(
    seeded_settings: CommunitySettings,
    community_services: CommunityServices,
) -> None:
    """create_app always includes /webhook routes (channel deps are auto-wired)."""
    app = create_app(community_services, seeded_settings)
    route_paths = [r.path for r in app.routes]  # type: ignore[union-attr]
    assert "/webhook/{channel}" in route_paths


# ---------------------------------------------------------------------------
# Catalog model_type allowlist application
#
# The catalog's prefix policy is a process-wide global; the suite-wide autouse
# fixture in conftest.py resets it and clears the env var around each test.
# ---------------------------------------------------------------------------


def test_explicit_settings_beat_ambient_environment(
    seeded_settings: CommunitySettings,
    community_services: CommunityServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly-constructed settings object wins over the environment.

    The catalog would otherwise resolve its own policy lazily from the env var;
    create_app calling set_allowed_prefixes makes the passed settings the single
    authority, so a stale ambient value cannot silently widen the allowlist.
    """
    monkeypatch.setenv(CATALOG_PREFIXES_ENV_VAR, "acme.")
    settings = CommunitySettings(
        workspaces_root=seeded_settings.workspaces_root,
        catalog_model_type_prefixes=["contoso."],
    )
    create_app(community_services, settings)
    assert allowed_prefixes() == ("akgentic.", "contoso.")


def test_empty_prefix_list_leaves_base_prefix_only(
    seeded_settings: CommunitySettings,
    community_services: CommunityServices,
) -> None:
    """An empty list is neither a narrowing nor a crash — akgentic. survives."""
    assert seeded_settings.catalog_model_type_prefixes == []
    create_app(community_services, seeded_settings)
    assert allowed_prefixes() == ("akgentic.",)


def test_boot_log_names_the_effective_prefix_tuple(
    seeded_settings: CommunitySettings,
    community_services: CommunityServices,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One INFO line names the effective tuple.

    Server and worker resolve this policy independently; the boot log is the
    only thing that makes a mismatch between them diagnosable from logs alone.
    """
    settings = CommunitySettings(
        workspaces_root=seeded_settings.workspaces_root,
        catalog_model_type_prefixes=["acme."],
    )
    app_logger = logging.getLogger("akgentic.infra.server.app")
    caplog.set_level(logging.INFO, logger=app_logger.name)
    # create_app calls configure_logging(), which replaces the ROOT logger's
    # handlers wholesale — dropping caplog's own handler mid-call. Attaching it
    # to the module logger instead keeps the boot line captured.
    app_logger.addHandler(caplog.handler)
    try:
        create_app(community_services, settings)
    finally:
        app_logger.removeHandler(caplog.handler)

    effective = allowed_prefixes()
    assert effective == ("akgentic.", "acme.")
    messages = [record.getMessage() for record in caplog.records if record.name == app_logger.name]
    assert any(str(effective) in message for message in messages), messages


def test_policy_is_applied_before_routes_are_mounted(
    seeded_settings: CommunitySettings,
    community_services: CommunityServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The policy is live by the time _build_app mounts the catalog routes.

    Asserting the post-conditions of create_app is not enough: moving
    set_allowed_prefixes below _build_app would leave every other test in this
    section green while re-opening the window this ordering exists to close —
    a route accepting an Entry under the pre-application policy.
    """
    settings = CommunitySettings(
        workspaces_root=seeded_settings.workspaces_root,
        catalog_model_type_prefixes=["acme."],
    )
    seen: list[tuple[str, ...]] = []
    real_build_app = app_module._build_app

    def _spy(
        services: TierServices,
        team_service: TeamService,
        build_settings: ServerSettings,
    ) -> FastAPI:
        seen.append(allowed_prefixes())
        return real_build_app(services, team_service, build_settings)

    monkeypatch.setattr(app_module, "_build_app", _spy)
    create_app(community_services, settings)

    assert seen == [("akgentic.", "acme.")]
