"""Integration test fixtures — real app, real actors, real LLM (no mocks)."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.testclient import TestClient

from akgentic.infra.adapters.channel_parser_registry import ChannelParserRegistry
from akgentic.infra.adapters.local_ingestion import LocalIngestion
from akgentic.infra.adapters.yaml_channel_registry import YamlChannelRegistry
from akgentic.infra.server.app import _build_app, create_app
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community

from ._helpers import seed_integration_catalog

if TYPE_CHECKING:
    from .test_channels import StubChannelAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


# Re-export for backward compat with existing test imports
_seed_integration_catalog = seed_integration_catalog


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
def integration_settings(tmp_path: Path) -> CommunitySettings:
    """CommunitySettings backed by tmp_path with a seeded integration catalog."""
    settings = CommunitySettings(workspaces_root=tmp_path / "workspaces")
    seed_integration_catalog(settings.workspaces_root / "catalog")
    return settings


@pytest.fixture()
def integration_services(
    integration_settings: CommunitySettings,
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
    return TeamService(services=integration_services)


@pytest.fixture()
def integration_app(
    integration_settings: CommunitySettings,
    integration_services: CommunityServices,
) -> Generator[FastAPI, None, None]:
    """FastAPI app backed by real actors and real LLM."""
    application = create_app(integration_services, integration_settings)
    yield application


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
def v1_adapter_settings(tmp_path: Path) -> CommunitySettings:
    """CommunitySettings with V1 frontend adapter enabled."""
    settings = CommunitySettings(
        workspaces_root=tmp_path / "workspaces",
        frontend_adapter=V1_ADAPTER_FQDN,
    )
    seed_integration_catalog(settings.workspaces_root / "catalog")
    return settings


@pytest.fixture()
def v1_adapter_app(
    v1_adapter_settings: CommunitySettings,
) -> Generator[FastAPI, None, None]:
    """FastAPI app with V1 frontend adapter loaded."""
    services = wire_community(v1_adapter_settings)
    application = create_app(services, v1_adapter_settings)
    yield application
    services.actor_system.shutdown()


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
    integration_settings: CommunitySettings,
    channel_parser_registry: ChannelParserRegistry,
    channel_registry_instance: YamlChannelRegistry,
    channel_ingestion: LocalIngestion,
) -> FastAPI:
    """FastAPI app with webhook wiring for channel integration tests.

    Uses _build_app with overridden channel deps on the services container.
    """
    integration_services.channel_parser_registry = channel_parser_registry
    integration_services.channel_registry = channel_registry_instance
    integration_services.ingestion = channel_ingestion
    return _build_app(
        integration_services,
        integration_team_service,
        integration_settings,
    )


@pytest.fixture()
def channel_client(channel_app: FastAPI) -> TestClient:
    """Sync HTTP test client hitting a channel-enabled FastAPI app."""
    return TestClient(channel_app)
