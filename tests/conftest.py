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
from akgentic.infra.server.settings import ServerSettings
from akgentic.infra.wiring import wire_community


def _write_yaml(path: Path, data: dict[str, object]) -> None:
    """Write a single YAML entry file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False))


def _seed_catalog(catalog_root: Path) -> None:
    """Create minimal YAML catalog entries for a test team."""
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


@pytest.fixture(autouse=True)
def _ensure_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a dummy OPENAI_API_KEY so BaseAgent actors can initialise in unit tests."""
    if not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "test-dummy-key")


@pytest.fixture()
def server_settings(tmp_path: Path) -> ServerSettings:
    """Server settings with tmp_path-based workspaces."""
    return ServerSettings(workspaces_root=tmp_path / "workspaces")


@pytest.fixture()
def seeded_settings(tmp_path: Path) -> ServerSettings:
    """Server settings with pre-seeded catalog YAML files."""
    settings = ServerSettings(workspaces_root=tmp_path / "workspaces")
    _seed_catalog(settings.workspaces_root / "catalog")
    return settings


@pytest.fixture()
def community_services(
    seeded_settings: ServerSettings,
) -> Generator[CommunityServices, None, None]:
    """Wired community services with seeded catalog data."""
    services = wire_community(seeded_settings)
    yield services
    services.actor_system.shutdown()


@pytest.fixture()
def team_service(community_services: CommunityServices) -> TeamService:
    """TeamService wired to community services."""
    return TeamService(
        services=community_services,
        team_catalog=community_services.team_catalog,
        agent_catalog=community_services.agent_catalog,
        tool_catalog=community_services.tool_catalog,
        template_catalog=community_services.template_catalog,
    )


@pytest.fixture()
def app(seeded_settings: ServerSettings) -> Generator[FastAPI, None, None]:
    """FastAPI app via the single-arg factory."""
    application = create_app(seeded_settings)
    yield application
    application.state.services.actor_system.shutdown()


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    """Sync HTTP test client."""
    return TestClient(app)
