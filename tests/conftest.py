"""Shared test fixtures for akgentic-infra tests."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from akgentic.infra.server.app import create_app
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip tests marked with ``llm`` when OPENAI_API_KEY is not set."""
    if not os.environ.get("OPENAI_API_KEY"):
        skip_llm = pytest.mark.skip(reason="OPENAI_API_KEY not set")
        for item in items:
            if "llm" in item.keywords:
                item.add_marker(skip_llm)


def _write_yaml(path: Path, data: dict[str, object]) -> None:
    """Write a single YAML entry file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False))


def _seed_catalog(catalog_root: Path) -> None:
    """Create minimal YAML catalog entries for a test team.

    Seeds BOTH v1 (per-kind dir layout under ``teams/``, ``agents/``,
    ``tools/``, ``templates/``) AND v2 (per-namespace dir layout under
    ``{namespace}/{kind}/{id}.yaml``) so the community wiring can expose a
    working v1 four-catalog stack alongside the v2 ``Catalog`` that Story
    18.2 adds. Uses the namespace ``test-team`` so tests can post
    ``catalog_namespace="test-team"`` and hit the v2 code path, while the
    v1 entry id is also ``test-team`` so legacy tests that still use
    ``catalog_entry_id="test-team"`` resolve via the v2 API (the router
    now forwards ``catalog_namespace`` either way).
    """
    # --- v1 layout (kept in place through Story 18.2; removed in 18.3) -----
    _write_yaml(
        catalog_root / "agents" / "human-proxy.yaml",
        {
            "id": "human-proxy",
            "tool_ids": [],
            "card": {
                "role": "Human",
                "description": "Human user interface",
                "skills": [],
                "agent_class": "akgentic.agent.HumanProxy",
                "config": {"name": "@Human", "role": "Human"},
                "routes_to": ["@Manager"],
            },
        },
    )
    _write_yaml(
        catalog_root / "agents" / "manager.yaml",
        {
            "id": "manager",
            "tool_ids": [],
            "card": {
                "role": "Manager",
                "description": "Test manager agent",
                "skills": ["coordination"],
                "agent_class": "akgentic.agent.BaseAgent",
                "config": {"name": "@Manager", "role": "Manager"},
                "routes_to": [],
            },
        },
    )
    _write_yaml(
        catalog_root / "teams" / "test-team.yaml",
        {
            "id": "test-team",
            "name": "Test Team",
            "entry_point": "human-proxy",
            "message_types": ["akgentic.core.messages.UserMessage"],
            "members": [
                {"agent_id": "human-proxy"},
                {"agent_id": "manager"},
            ],
            "profiles": [],
        },
    )
    # Empty dirs for templates and tools (no entries needed for this team)
    (catalog_root / "templates").mkdir(parents=True, exist_ok=True)
    (catalog_root / "tools").mkdir(parents=True, exist_ok=True)

    # --- v2 unified-entry namespace bundle ----------------------------------
    # Layout: {catalog_root}/{namespace}/{kind}/{id}.yaml
    _seed_v2_namespace(catalog_root, namespace="test-team")


_TEAM_CARD_TYPE = "akgentic.team.models.TeamCard"


def _seed_v2_namespace(catalog_root: Path, namespace: str) -> None:
    """Write a minimal v2 team-namespace bundle into ``catalog_root``.

    Mirrors the v1 ``Test Team`` definition so the same fixture exercises
    both the v1 four-catalog pipeline and the v2 ``Catalog.load_team``
    pipeline. The ``TeamCard`` payload shape is taken from
    ``akgentic.team.models.TeamCard``; every agent_class / model_type
    string satisfies the v2 allowlist (``akgentic.*``).

    The member configs use plain ``akgentic.core.agent.Akgent`` (which
    expects ``BaseConfig``) because the v2 resolver hydrates
    ``AgentCard.config`` against the declared annotation (``BaseConfig``)
    and does not upgrade to the agent-class's specific ``ConfigType``
    subclass the way v1's ``AgentEntry.resolve_config`` validator does.
    Upgrading v2 to per-agent-class config types is out of scope for
    Story 18.2 (tracked as part of Epic 19's v1 removal). Tests that
    only need the agents to *exist* and route messages by name work
    with the plain base class.
    """
    team_payload = {
        "name": "Test Team",
        "description": "v2 test team for infra tests",
        "entry_point": {
            "card": {
                "role": "Human",
                "description": "Human user interface",
                "skills": [],
                "agent_class": "akgentic.core.agent.Akgent",
                "config": {"name": "@Human", "role": "Human"},
                "routes_to": ["@Manager"],
            },
            "headcount": 1,
            "members": [],
        },
        "members": [
            {
                "card": {
                    "role": "Manager",
                    "description": "Test manager agent",
                    "skills": ["coordination"],
                    "agent_class": "akgentic.core.agent.Akgent",
                    "config": {"name": "@Manager", "role": "Manager"},
                    "routes_to": [],
                },
                "headcount": 1,
                "members": [],
            },
        ],
        "message_types": [{"__type__": "akgentic.core.messages.UserMessage"}],
        "agent_profiles": [],
    }
    _write_yaml(
        catalog_root / namespace / "team" / "team.yaml",
        {
            "id": "team",
            "kind": "team",
            "namespace": namespace,
            "model_type": _TEAM_CARD_TYPE,
            "description": "v2 test team namespace bundle",
            "payload": team_payload,
        },
    )


@pytest.fixture(autouse=True)
def _ensure_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a dummy OPENAI_API_KEY so BaseAgent actors can initialise in unit tests."""
    if not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "test-dummy-key")


@pytest.fixture()
def server_settings(tmp_path: Path) -> CommunitySettings:
    """Server settings with tmp_path-based workspaces."""
    return CommunitySettings(
        workspaces_root=tmp_path / "workspaces",
        event_store_path=tmp_path / "event_store",
        catalog_path=tmp_path / "catalog",
    )


@pytest.fixture()
def seeded_settings(tmp_path: Path) -> CommunitySettings:
    """Server settings with pre-seeded catalog YAML files."""
    settings = CommunitySettings(
        workspaces_root=tmp_path / "workspaces",
        event_store_path=tmp_path / "event_store",
        catalog_path=tmp_path / "catalog",
    )
    _seed_catalog(settings.catalog_path)
    return settings


@pytest.fixture()
def community_services(
    seeded_settings: CommunitySettings,
) -> Generator[CommunityServices, None, None]:
    """Wired community services with seeded catalog data."""
    services = wire_community(seeded_settings)
    yield services
    services.actor_system.shutdown()


@pytest.fixture()
def team_service(community_services: CommunityServices) -> TeamService:
    """TeamService wired to community services."""
    return TeamService(services=community_services)


@pytest.fixture()
def app(
    seeded_settings: CommunitySettings,
    community_services: CommunityServices,
) -> Generator[FastAPI, None, None]:
    """FastAPI app via the two-arg factory."""
    application = create_app(community_services, seeded_settings)
    yield application
    application.state.services.actor_system.shutdown()


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    """Sync HTTP test client."""
    return TestClient(app)


@pytest.fixture()
def v1_seeded_settings(tmp_path: Path) -> CommunitySettings:
    """Server settings with V1 frontend adapter enabled and pre-seeded catalog."""
    settings = CommunitySettings(
        workspaces_root=tmp_path / "workspaces",
        event_store_path=tmp_path / "event_store",
        catalog_path=tmp_path / "catalog",
        frontend_adapter=(
            "akgentic.infra.server.routes.frontend_adapter.angular_v1.AngularV1Adapter"
        ),
    )
    _seed_catalog(settings.catalog_path)
    return settings


@pytest.fixture()
def v1_client(
    v1_seeded_settings: CommunitySettings,
) -> Generator[TestClient, None, None]:
    """Sync HTTP test client with V1 frontend adapter routes mounted."""
    services = wire_community(v1_seeded_settings)
    application = create_app(services, v1_seeded_settings)
    yield TestClient(application)
    services.actor_system.shutdown()
