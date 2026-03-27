"""Tests for the FastAPI application factory and CORS middleware."""

from __future__ import annotations

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
