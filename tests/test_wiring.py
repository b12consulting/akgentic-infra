"""Tests for wire_community() assembly function."""

from __future__ import annotations

from pathlib import Path

import pytest

from akgentic.infra.adapters.local_placement import LocalPlacement
from akgentic.infra.adapters.local_service_registry import LocalServiceRegistry
from akgentic.infra.adapters.no_auth import NoAuth
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.settings import ServerSettings
from akgentic.infra.wiring import wire_community
from akgentic.team.manager import TeamManager
from akgentic.team.repositories.yaml import YamlEventStore


class TestWireCommunity:
    """AC6: wire_community() assembles CommunityServices correctly."""

    @pytest.fixture()
    def services(self, tmp_path: Path) -> CommunityServices:
        """Wire community services with a temp workspace root."""
        settings = ServerSettings(workspaces_root=tmp_path)
        return wire_community(settings)

    def test_returns_community_services(self, services: CommunityServices) -> None:
        """wire_community returns a CommunityServices instance."""
        assert isinstance(services, CommunityServices)

    def test_placement_is_local(self, services: CommunityServices) -> None:
        """Placement strategy is LocalPlacement."""
        assert isinstance(services.placement, LocalPlacement)

    def test_auth_is_noauth(self, services: CommunityServices) -> None:
        """Auth strategy is NoAuth."""
        assert isinstance(services.auth, NoAuth)

    def test_service_registry_is_local(self, services: CommunityServices) -> None:
        """Service registry is LocalServiceRegistry."""
        assert isinstance(services.service_registry, LocalServiceRegistry)

    def test_event_store_is_yaml(self, services: CommunityServices) -> None:
        """Event store is YamlEventStore."""
        assert isinstance(services.event_store, YamlEventStore)

    def test_team_manager_is_present(self, services: CommunityServices) -> None:
        """TeamManager is present and correctly typed."""
        assert isinstance(services.team_manager, TeamManager)

    def test_uses_settings_workspaces_root(self, tmp_path: Path) -> None:
        """wire_community passes settings.workspaces_root to YamlEventStore."""
        settings = ServerSettings(workspaces_root=tmp_path)
        services = wire_community(settings)
        assert isinstance(services.event_store, YamlEventStore)
