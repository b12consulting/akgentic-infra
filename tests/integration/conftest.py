"""Integration test fixtures — real app, real actors, real LLM (no mocks)."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.testclient import TestClient

from akgentic.infra.adapters.channel_parser_registry import ChannelParserRegistry
from akgentic.infra.adapters.local_ingestion import LocalIngestion
from akgentic.infra.adapters.yaml_channel_registry import YamlChannelRegistry
from akgentic.infra.server.app import create_app
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import ServerSettings
from akgentic.infra.wiring import wire_community

if TYPE_CHECKING:
    from .test_channels import StubChannelAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _write_yaml(path: Path, data: dict[str, object]) -> None:
    """Write a single YAML entry file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False))


def _seed_integration_catalog(catalog_root: Path) -> None:
    """Seed YAML catalog with an LLM-capable agent for integration testing.

    Uses gpt-4o-mini for fast, cheap LLM calls.
    """
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
                "description": "Integration test manager agent",
                "skills": ["coordination"],
                "agent_class": "akgentic.agent.BaseAgent",
                "config": {
                    "name": "@Manager",
                    "role": "Manager",
                    "prompt": {
                        "template": (
                            "You are a helpful assistant. "
                            "Reply concisely in one or two sentences."
                        ),
                    },
                    "model_cfg": {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "temperature": 0.0,
                    },
                    "usage_limits": {
                        "request_limit": 5,
                        "total_tokens_limit": 10000,
                    },
                },
                "routes_to": [],
            },
        },
    )
    _write_yaml(
        catalog_root / "teams" / "test-team.yaml",
        {
            "id": "test-team",
            "name": "Integration Test Team",
            "entry_point": "human-proxy",
            "message_types": ["akgentic.agent.AgentMessage"],
            "members": [
                {"agent_id": "human-proxy"},
                {"agent_id": "manager"},
            ],
            "profiles": [],
        },
    )
    (catalog_root / "templates").mkdir(parents=True, exist_ok=True)
    (catalog_root / "tools").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def openai_api_key() -> str:
    """Ensure OPENAI_API_KEY is available; fail fast if not."""
    load_dotenv(_PROJECT_ROOT / ".env")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.fail("OPENAI_API_KEY not set — required for integration tests")
    return key


@pytest.fixture()
def integration_settings(tmp_path: Path) -> ServerSettings:
    """ServerSettings backed by tmp_path with a seeded integration catalog."""
    settings = ServerSettings(workspaces_root=tmp_path / "workspaces")
    _seed_integration_catalog(settings.workspaces_root / "catalog")
    return settings


@pytest.fixture()
def integration_services(
    integration_settings: ServerSettings,
) -> Generator[CommunityServices, None, None]:
    """Wired community services with real actor system — shuts down on teardown."""
    services = wire_community(integration_settings)
    yield services
    services.actor_system.shutdown()


@pytest.fixture()
def integration_team_service(
    integration_services: CommunityServices,
) -> TeamService:
    """TeamService wired to integration services."""
    return TeamService(
        services=integration_services,
        team_catalog=integration_services.team_catalog,
        agent_catalog=integration_services.agent_catalog,
        tool_catalog=integration_services.tool_catalog,
        template_catalog=integration_services.template_catalog,
    )


@pytest.fixture()
def integration_app(
    integration_services: CommunityServices,
    integration_team_service: TeamService,
    integration_settings: ServerSettings,
) -> FastAPI:
    """FastAPI app backed by real actors and real LLM."""
    return create_app(
        integration_services, integration_team_service, settings=integration_settings,
    )


@pytest.fixture()
def integration_client(integration_app: FastAPI) -> TestClient:
    """Sync HTTP test client hitting a real FastAPI app."""
    return TestClient(integration_app)


# ---------------------------------------------------------------------------
# V1 Adapter Fixtures
# ---------------------------------------------------------------------------

V1_ADAPTER_FQDN = (
    "akgentic.infra.server.routes.frontend_adapter.angular_v1.AngularV1Adapter"
)


@pytest.fixture()
def v1_adapter_settings(tmp_path: Path) -> ServerSettings:
    """ServerSettings with V1 frontend adapter enabled."""
    settings = ServerSettings(
        workspaces_root=tmp_path / "workspaces",
        frontend_adapter=V1_ADAPTER_FQDN,
    )
    _seed_integration_catalog(settings.workspaces_root / "catalog")
    return settings


@pytest.fixture()
def v1_adapter_services(
    v1_adapter_settings: ServerSettings,
) -> Generator[CommunityServices, None, None]:
    """Wired community services for V1 adapter tests — shuts down on teardown."""
    services = wire_community(v1_adapter_settings)
    yield services
    services.actor_system.shutdown()


@pytest.fixture()
def v1_adapter_team_service(
    v1_adapter_services: CommunityServices,
) -> TeamService:
    """TeamService wired to V1 adapter services."""
    return TeamService(
        services=v1_adapter_services,
        team_catalog=v1_adapter_services.team_catalog,
        agent_catalog=v1_adapter_services.agent_catalog,
        tool_catalog=v1_adapter_services.tool_catalog,
        template_catalog=v1_adapter_services.template_catalog,
    )


@pytest.fixture()
def v1_adapter_app(
    v1_adapter_services: CommunityServices,
    v1_adapter_team_service: TeamService,
    v1_adapter_settings: ServerSettings,
) -> FastAPI:
    """FastAPI app with V1 frontend adapter loaded."""
    return create_app(
        v1_adapter_services, v1_adapter_team_service, settings=v1_adapter_settings,
    )


@pytest.fixture()
def v1_adapter_client(v1_adapter_app: FastAPI) -> TestClient:
    """Sync HTTP test client hitting a V1-adapter-enabled FastAPI app."""
    return TestClient(v1_adapter_app)


# ---------------------------------------------------------------------------
# Channel Integration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_adapter() -> StubChannelAdapter:
    """Shared StubChannelAdapter instance for inspecting delivered messages."""
    from .test_channels import StubChannelAdapter as _StubChannelAdapter

    return _StubChannelAdapter()


@pytest.fixture()
def channel_parser_registry(
    test_adapter: StubChannelAdapter,
) -> ChannelParserRegistry:
    """ChannelParserRegistry with test stubs manually injected."""
    from .test_channels import StubChannelParser

    registry = ChannelParserRegistry(channels_config={})
    parser = StubChannelParser()
    registry._parsers[parser.channel_name] = parser
    registry._adapters.append(test_adapter)
    return registry


@pytest.fixture()
def channel_registry_instance(tmp_path: Path) -> YamlChannelRegistry:
    """YamlChannelRegistry backed by tmp_path."""
    return YamlChannelRegistry(registry_path=tmp_path / "channel-registry.yaml")


@pytest.fixture()
def channel_ingestion(
    integration_team_service: TeamService,
) -> LocalIngestion:
    """LocalIngestion wired to the integration TeamService."""
    return LocalIngestion(team_service=integration_team_service)


@pytest.fixture()
def channel_app(
    integration_services: CommunityServices,
    integration_team_service: TeamService,
    integration_settings: ServerSettings,
    channel_parser_registry: ChannelParserRegistry,
    channel_registry_instance: YamlChannelRegistry,
    channel_ingestion: LocalIngestion,
) -> FastAPI:
    """FastAPI app with webhook wiring for channel integration tests."""
    return create_app(
        integration_services,
        integration_team_service,
        settings=integration_settings,
        channel_parser_registry=channel_parser_registry,
        channel_registry=channel_registry_instance,
        ingestion=channel_ingestion,
    )


@pytest.fixture()
def channel_client(channel_app: FastAPI) -> TestClient:
    """Sync HTTP test client hitting a channel-enabled FastAPI app."""
    return TestClient(channel_app)
