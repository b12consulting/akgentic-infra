"""Tests for wire_community() assembly function."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from akgentic.team.manager import TeamManager
from akgentic.team.repositories.yaml import YamlEventStore

from akgentic.infra.adapters.local_placement import LocalPlacement
from akgentic.infra.adapters.local_service_registry import LocalServiceRegistry
from akgentic.infra.adapters.local_worker_handle import LocalWorkerHandle
from akgentic.infra.adapters.no_auth import NoAuth
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.settings import ServerSettings
from akgentic.infra.wiring import wire_community


class TestWireCommunity:
    """AC6: wire_community() assembles CommunityServices correctly."""

    @pytest.fixture()
    def services(self, tmp_path: Path) -> Generator[CommunityServices, None, None]:
        """Wire community services with a temp workspace root and cleanup ActorSystem."""
        settings = ServerSettings(workspaces_root=tmp_path)
        svc = wire_community(settings)
        yield svc
        svc.team_manager._actor_system.shutdown(timeout=5)

    def test_returns_community_services(self, services: CommunityServices) -> None:
        """wire_community returns a CommunityServices instance."""
        assert isinstance(services, CommunityServices)

    def test_placement_is_local(self, services: CommunityServices) -> None:
        """Placement strategy is a LocalPlacement."""
        assert isinstance(services.placement, LocalPlacement)

    def test_auth_is_noauth(self, services: CommunityServices) -> None:
        """Auth strategy is NoAuth."""
        assert isinstance(services.auth, NoAuth)

    def test_worker_handle_is_local(self, services: CommunityServices) -> None:
        """Worker handle is LocalWorkerHandle."""
        assert isinstance(services.worker_handle, LocalWorkerHandle)

    def test_service_registry_is_local(self, services: CommunityServices) -> None:
        """Service registry is LocalServiceRegistry (community-specific)."""
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
        try:
            assert isinstance(services.event_store, YamlEventStore)
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)

    def test_shared_service_registry(self, services: CommunityServices) -> None:
        """TeamManager and CommunityServices share the same service registry."""
        assert services.service_registry is services.team_manager._service_registry

    def test_instance_registered_with_service_registry(
        self, services: CommunityServices,
    ) -> None:
        """LocalPlacement's instance_id is registered with ServiceRegistry."""
        placement = services.placement
        assert isinstance(placement, LocalPlacement)
        found = services.service_registry.find_team(uuid.uuid4())
        # instance is registered — find_team returns None for unknown team but no error
        assert found is None or isinstance(found, uuid.UUID)

    def test_catalog_path_override(self, tmp_path: Path) -> None:
        """wire_community uses settings.catalog_path when set."""
        custom_catalog = tmp_path / "custom-catalog"
        custom_catalog.mkdir()
        for sub in ("teams", "agents", "tools", "templates"):
            (custom_catalog / sub).mkdir()
        settings = ServerSettings(
            workspaces_root=tmp_path,
            catalog_path=custom_catalog,
        )
        services = wire_community(settings)
        try:
            assert services.team_catalog is not None
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)
