"""Tests for the FastAPI application factory and CORS middleware."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from akgentic.infra.server.app import create_app
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.services.team_service import TeamService


def test_create_app_returns_fastapi(
    community_services: CommunityServices,
    team_service: TeamService,
) -> None:
    """create_app returns a FastAPI instance with routes mounted."""
    app = create_app(community_services, team_service)
    assert app.title == "Akgentic Platform API"
    route_paths = [r.path for r in app.routes]  # type: ignore[union-attr]
    assert "/teams/" in route_paths
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
    community_services: CommunityServices,
    team_service: TeamService,
) -> None:
    """create_app respects custom cors_origins."""
    app = create_app(community_services, team_service, cors_origins=["http://example.com"])
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


def test_create_app_includes_webhook_routes_when_channel_deps_provided(
    community_services: CommunityServices,
    team_service: TeamService,
    tmp_path: Path,
) -> None:
    """create_app includes /webhook routes when channel deps are provided."""
    from akgentic.infra.adapters.channel_parser_registry import ChannelParserRegistry
    from akgentic.infra.adapters.yaml_channel_registry import YamlChannelRegistry

    parser_registry = ChannelParserRegistry.__new__(ChannelParserRegistry)
    parser_registry._parsers = {}  # noqa: SLF001
    parser_registry._adapters = []  # noqa: SLF001
    channel_registry = YamlChannelRegistry(tmp_path / "registry.yaml")
    ingestion = MagicMock()

    app = create_app(
        community_services,
        team_service,
        channel_parser_registry=parser_registry,
        channel_registry=channel_registry,
        ingestion=ingestion,
    )
    route_paths = [r.path for r in app.routes]  # type: ignore[union-attr]
    assert "/webhook/{channel}" in route_paths


def test_create_app_excludes_webhook_routes_by_default(
    community_services: CommunityServices,
    team_service: TeamService,
) -> None:
    """create_app excludes /webhook routes when channel deps are not provided."""
    app = create_app(community_services, team_service)
    route_paths = [r.path for r in app.routes]  # type: ignore[union-attr]
    assert "/webhook/{channel}" not in route_paths


def test_create_app_raises_when_channel_deps_incomplete(
    community_services: CommunityServices,
    team_service: TeamService,
) -> None:
    """create_app raises ValueError when channel_parser_registry provided without others."""
    from akgentic.infra.adapters.channel_parser_registry import ChannelParserRegistry

    parser_registry = ChannelParserRegistry.__new__(ChannelParserRegistry)
    parser_registry._parsers = {}  # noqa: SLF001
    parser_registry._adapters = []  # noqa: SLF001

    with pytest.raises(ValueError, match="channel_registry and ingestion are required"):
        create_app(
            community_services,
            team_service,
            channel_parser_registry=parser_registry,
        )
