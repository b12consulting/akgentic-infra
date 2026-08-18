"""Tests for the FastAPI application factory and CORS middleware."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from types import ModuleType

import pytest
from akgentic.catalog import ENV_VAR as CATALOG_PREFIXES_ENV_VAR
from akgentic.catalog import allowed_prefixes
from fastapi import FastAPI
from fastapi.testclient import TestClient

from akgentic.infra.server import app as app_module
from akgentic.infra.server.app import create_app, create_server_app
from akgentic.infra.server.assembly import AppModule, build_manifest
from akgentic.infra.server.deps import CommunityServices, TierServices
from akgentic.infra.server.settings import CommunitySettings, ServerSettings
from akgentic.infra.server.state_keys import SERVICES, SETTINGS


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
# create_server_app — the uniform tier-bootstrap factory (Story 57.5, AC 5)
# ---------------------------------------------------------------------------


def test_create_server_app_wires_services_and_delegates_to_create_app(
    seeded_settings: CommunitySettings,
    community_services: CommunityServices,
) -> None:
    """create_server_app wires its own CommunityServices, then delegates.

    The composed app carries the passed settings and a freshly wired services
    container, and is manifest-identical to the pre-wired-services entry
    point's app — the assembly sequence exists once, in create_app.
    """
    app = create_server_app(seeded_settings)
    services = SERVICES.require(app)
    assert isinstance(services, CommunityServices)
    try:
        assert SETTINGS.require(app) is seeded_settings
        assert services is not community_services  # wired by the factory, not reused
        reference = create_app(community_services, seeded_settings)
        assert build_manifest(app) == build_manifest(reference)
    finally:
        services.actor_system.shutdown()


def test_create_server_app_constructs_default_settings_for_a_bare_factory_target(
    seeded_settings: CommunitySettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A no-argument call builds CommunitySettings itself (the --factory contract)."""
    monkeypatch.setattr(app_module, "CommunitySettings", lambda: seeded_settings)
    app = create_server_app()
    services = SERVICES.require(app)
    assert isinstance(services, CommunityServices)
    try:
        assert SETTINGS.require(app) is seeded_settings
    finally:
        services.actor_system.shutdown()


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
    # Attaching caplog's handler to the module logger keeps the boot line
    # captured independently of the ROOT logger's handler set — historically
    # configure_logging() replaced root handlers wholesale mid-call; since
    # story 57.7 it is additive, but this test stays decoupled either way.
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
    """The policy is live by the time build_app mounts the catalog routes.

    Asserting the post-conditions of create_app is not enough: moving
    set_allowed_prefixes below build_app would leave every other test in this
    section green while re-opening the window this ordering exists to close —
    a route accepting an Entry under the pre-application policy.
    """
    settings = CommunitySettings(
        workspaces_root=seeded_settings.workspaces_root,
        catalog_model_type_prefixes=["acme."],
    )
    seen: list[tuple[str, ...]] = []
    real_build_app = app_module.build_app

    def _spy(
        build_settings: ServerSettings,
        services: TierServices,
        modules: Sequence[AppModule],
    ) -> FastAPI:
        seen.append(allowed_prefixes())
        return real_build_app(build_settings, services, modules)

    monkeypatch.setattr(app_module, "build_app", _spy)
    create_app(community_services, settings)

    assert seen == [("akgentic.", "acme.")]


def test_create_app_imports_nothing_for_a_configured_prefix(
    seeded_settings: CommunitySettings,
    community_services: CommunityServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_app never imports a deployment's modules on a configured prefix's behalf.

    A boot-time preload of each configured prefix's module was prototyped for this
    epic and deliberately withdrawn. ``import_module`` executes the target module's
    top-level code, transitively, so a package that is broken or absent from the
    server image would take the whole *server* down — bought in exchange for
    populating a model-type dropdown. Neither validation nor resolution ever needed
    it: ``load_model_type`` imports on demand, and enumeration reports whatever the
    process has already imported. The setting authorises; it does not import.

    This test is the regression pin for that decision, so whoever reintroduces the
    preload lands here and reads the reasoning rather than finding an unexplained
    absence. The configured prefix names a module that does not exist, which covers
    both reintroduction shapes: a fail-fast preload raises out of create_app, and a
    tolerant one is caught by the recorded call name.
    """
    real_import_module = importlib.import_module
    imported: list[str] = []

    def _spy(name: str, package: str | None = None) -> ModuleType:
        imported.append(name)
        return real_import_module(name, package)

    # Two patch points. Patching the attribute on ``importlib`` covers both an
    # ``importlib.import_module(...)`` call and a function-local ``from importlib
    # import import_module``; patching app.py's own namespace covers a module-level
    # ``from importlib import import_module`` binding — a name that must not exist
    # on app.py today, hence raising=False. NOT covered: a preload built on the
    # ``__import__`` builtin (directly or via pkgutil) bypasses both patch points
    # and leaves this test green. That gap is accepted — every plausible
    # reintroduction goes through ``import_module`` — but it is a gap, not
    # indirect coverage, and a reviewer should not read it as one.
    monkeypatch.setattr(importlib, "import_module", _spy)
    monkeypatch.setattr(app_module, "import_module", _spy, raising=False)
    # The second patch point only bites while create_app genuinely executes in that
    # namespace. Re-exporting create_app from another module would retire the
    # module-level-binding shape's coverage while leaving this test green.
    assert create_app.__globals__ is vars(app_module)

    settings = CommunitySettings(
        workspaces_root=seeded_settings.workspaces_root,
        catalog_model_type_prefixes=["acme.models."],
    )
    create_app(community_services, settings)

    # The policy DID apply — this is a no-import assertion, not a no-effect one.
    assert allowed_prefixes() == ("akgentic.", "acme.models.")
    assert not [name for name in imported if name == "acme" or name.startswith("acme.")], imported
    # Self-check: a spy that silently stopped recording would make the assertion
    # above vacuously true, which is the failure mode a negative test invites.
    importlib.import_module("json")
    assert "json" in imported
