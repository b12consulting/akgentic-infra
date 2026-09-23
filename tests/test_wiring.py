"""Tests for wire_community() assembly function."""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path

import pytest
from akgentic.team.manager import TeamManager
from akgentic.team.ports import NullServiceRegistry, ServiceRegistry
from akgentic.team.repositories.yaml import YamlEventStore

from akgentic.infra.adapters.community.local_event_stream import LocalEventStream
from akgentic.infra.adapters.community.local_placement import LocalPlacement
from akgentic.infra.adapters.community.local_worker_handle import LocalWorkerHandle
from akgentic.infra.adapters.community.no_auth import NoAuth
from akgentic.infra.adapters.community.yaml_channel_registry import YamlChannelRegistry
from akgentic.infra.adapters.channels.channel_dispatcher import InteractionChannelDispatcher
from akgentic.infra.adapters.channels.channel_parser_registry import ChannelConfig
from akgentic.infra.adapters.shared.event_stream_subscriber import EventStreamSubscriber
from akgentic.infra.adapters.shared.owner_or_admin_policy import OwnerOrAdminPolicy
from akgentic.infra.adapters.channels.telegram_adapter import TelegramChannelAdapter
from akgentic.infra.adapters.shared.telemetry_subscriber import TelemetrySubscriber
from akgentic.infra.protocols.channels import ChannelAddress
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import CommunitySettings
from akgentic.infra.wiring import wire_community
from akgentic.infra.worker.deps import WorkerServices


class TestWireCommunityLogging:
    """wire_community() emits expected log messages."""

    def test_emits_wiring_info_log(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """wire_community() emits 'Wiring community services' INFO log."""
        settings = CommunitySettings(
            workspaces_root=tmp_path / "workspaces",
            event_store_path=tmp_path / "event_store",
            catalog_path=tmp_path / "catalog",
        )
        with caplog.at_level(logging.INFO, logger="akgentic.infra.wiring"):
            services = wire_community(settings)
        try:
            assert any("Wiring community services" in r.message for r in caplog.records)
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)


class TestWireCommunity:
    """AC6: wire_community() assembles CommunityServices correctly."""

    @pytest.fixture()
    def services(self, tmp_path: Path) -> Generator[CommunityServices, None, None]:
        """Wire community services with a temp workspace root and cleanup ActorSystem."""
        settings = CommunitySettings(
            workspaces_root=tmp_path / "workspaces",
            event_store_path=tmp_path / "event_store",
            catalog_path=tmp_path / "catalog",
        )
        svc = wire_community(settings)
        yield svc
        svc.team_manager._actor_system.shutdown(timeout=5)

    def test_returns_community_services(self, services: CommunityServices) -> None:
        """wire_community returns a CommunityServices instance."""
        assert isinstance(services, CommunityServices)

    def test_uses_the_supplied_team_access_policy(self, tmp_path: Path) -> None:
        settings = CommunitySettings(
            workspaces_root=tmp_path / "workspaces",
            event_store_path=tmp_path / "event_store",
            catalog_path=tmp_path / "catalog",
        )
        policy = OwnerOrAdminPolicy()
        services = wire_community(settings, team_access_policy=policy)
        try:
            assert services.team_access_policy is policy
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)

    def test_placement_is_local(self, services: CommunityServices) -> None:
        """Placement strategy is a LocalPlacement."""
        assert isinstance(services.placement, LocalPlacement)

    def test_auth_is_noauth(self, services: CommunityServices) -> None:
        """Auth strategy is NoAuth."""
        assert isinstance(services.auth, NoAuth)

    def test_worker_handle_is_local(self, services: CommunityServices) -> None:
        """Worker handle is LocalWorkerHandle."""
        assert isinstance(services.worker_handle, LocalWorkerHandle)

    def test_service_registry_is_null(self, services: CommunityServices) -> None:
        """Service registry is NullServiceRegistry (write-only, community-tier)."""
        assert isinstance(services.service_registry, NullServiceRegistry)

    def test_event_store_is_yaml(self, services: CommunityServices) -> None:
        """Event store is YamlEventStore."""
        assert isinstance(services.event_store, YamlEventStore)

    def test_team_manager_is_present(self, services: CommunityServices) -> None:
        """TeamManager is present and correctly typed."""
        assert isinstance(services.team_manager, TeamManager)

    def test_uses_settings_event_store_path(self, tmp_path: Path) -> None:
        """wire_community passes settings.event_store_path to YamlEventStore."""
        settings = CommunitySettings(
            workspaces_root=tmp_path / "workspaces",
            event_store_path=tmp_path / "event_store",
            catalog_path=tmp_path / "catalog",
        )
        services = wire_community(settings)
        try:
            assert isinstance(services.event_store, YamlEventStore)
            assert services.event_store._data_dir == settings.event_store_path
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)

    def test_shared_service_registry(self, services: CommunityServices) -> None:
        """TeamManager and CommunityServices share the same service registry."""
        assert services.service_registry is services.team_manager._service_registry

    def test_service_registry_satisfies_protocol(
        self,
        services: CommunityServices,
    ) -> None:
        """Service registry satisfies the ServiceRegistry protocol."""
        assert isinstance(services.service_registry, ServiceRegistry)

    def test_catalog_path_override(self, tmp_path: Path) -> None:
        """wire_community uses settings.catalog_path when set."""
        custom_catalog = tmp_path / "custom-catalog"
        custom_catalog.mkdir()
        settings = CommunitySettings(
            workspaces_root=tmp_path / "workspaces",
            event_store_path=tmp_path / "event_store",
            catalog_path=custom_catalog,
        )
        services = wire_community(settings)
        try:
            assert services.catalog is not None
            assert services.catalog._repository._root == custom_catalog
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)

    async def test_default_channel_registry_is_disabled(
        self,
        services: CommunityServices,
    ) -> None:
        """When channel_registry_path is None (default), the YAML registry is disabled."""
        assert isinstance(services.channel_registry, YamlChannelRegistry)
        # Disabled: lookups return None (no file I/O).
        assert (
            await services.channel_registry.find_binding(
                ChannelAddress(channel="telegram", channel_user_id="user-1")
            )
            is None
        )

    def test_channel_registry_path_uses_yaml(self, tmp_path: Path) -> None:
        """When channel_registry_path is set, uses YamlChannelRegistry."""
        reg_path = tmp_path / "registry.yaml"
        settings = CommunitySettings(
            workspaces_root=tmp_path / "workspaces",
            event_store_path=tmp_path / "event_store",
            catalog_path=tmp_path / "catalog",
            channel_registry_path=reg_path,
        )
        services = wire_community(settings)
        try:
            assert isinstance(services.channel_registry, YamlChannelRegistry)
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)


class TestWireCommunityEventStream:
    """AC4: EventStreamSubscriber is wired as shared subscriber with LocalEventStream."""

    @pytest.fixture()
    def services(self, tmp_path: Path) -> Generator[CommunityServices, None, None]:
        settings = CommunitySettings(
            workspaces_root=tmp_path / "workspaces",
            event_store_path=tmp_path / "event_store",
            catalog_path=tmp_path / "catalog",
        )
        svc = wire_community(settings)
        yield svc
        svc.team_manager._actor_system.shutdown(timeout=5)

    def test_event_stream_is_local(self, services: CommunityServices) -> None:
        """AC4: CommunityServices.event_stream is a LocalEventStream."""
        assert isinstance(services.event_stream, LocalEventStream)

    def test_event_stream_subscriber_in_shared_subscribers(
        self, services: CommunityServices
    ) -> None:
        """AC4: EventStreamSubscriber is present in TeamManager shared_subscribers."""
        subscribers = services.team_manager._shared_subscribers
        has_event_stream_sub = any(isinstance(s, EventStreamSubscriber) for s in subscribers)
        assert has_event_stream_sub


class TestWireCommunityTeamService:
    """Story 57.6: ``TeamService`` is built and bound inside ``wire_community``."""

    @pytest.fixture()
    def settings(self, tmp_path: Path) -> CommunitySettings:
        return CommunitySettings(
            workspaces_root=tmp_path / "workspaces",
            event_store_path=tmp_path / "event_store",
            catalog_path=tmp_path / "catalog",
        )

    @pytest.fixture()
    def services(
        self, settings: CommunitySettings
    ) -> Generator[CommunityServices, None, None]:
        svc = wire_community(settings)
        yield svc
        svc.team_manager._actor_system.shutdown(timeout=5)

    def test_team_service_is_constructed(self, services: CommunityServices) -> None:
        """The container arrives with a real TeamService — no caller wiring left."""
        assert isinstance(services.team_service, TeamService)

    def test_workspaces_root_propagates_from_settings(
        self, services: CommunityServices, settings: CommunitySettings
    ) -> None:
        """wire_community reads settings.workspaces_root directly — no fallback."""
        assert services.team_service is not None
        assert services.team_service._workspaces_root == settings.workspaces_root


class TestWireCommunityDropsTelemetryField:
    """Story 28.1 AC #9: ``WorkerServices`` no longer carries ``telemetry_subscriber``."""

    def test_worker_services_model_lacks_telemetry_field(self) -> None:
        """Sanity check that the field was deleted from the model layer."""
        assert "telemetry_subscriber" not in WorkerServices.model_fields

    def test_wire_community_still_constructs_with_telemetry_subscriber_on_team_manager(
        self, tmp_path: Path
    ) -> None:
        """``wire_community`` succeeds and installs ``TelemetrySubscriber`` on TeamManager."""
        settings = CommunitySettings(
            workspaces_root=tmp_path / "workspaces",
            event_store_path=tmp_path / "event_store",
            catalog_path=tmp_path / "catalog",
        )
        services = wire_community(settings)
        try:
            assert any(
                isinstance(s, TelemetrySubscriber)
                for s in services.team_manager._shared_subscribers
            )
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)


class TestWireCommunityChannelDispatcher:
    """The outbound leg: one shared dispatcher, fed by the configured channels."""

    def test_dispatcher_is_a_shared_subscriber_exactly_once(self, tmp_path: Path) -> None:
        """Constructing it is not enough — it has to join ``shared_subscribers``.

        One instance, not one per team: the dispatcher reads the team off each
        message and off the lifecycle hooks.
        """
        settings = CommunitySettings(
            workspaces_root=tmp_path / "workspaces",
            event_store_path=tmp_path / "event_store",
            catalog_path=tmp_path / "catalog",
        )
        services = wire_community(settings)
        try:
            dispatchers = [
                s
                for s in services.team_manager._shared_subscribers
                if isinstance(s, InteractionChannelDispatcher)
            ]
            assert len(dispatchers) == 1
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)

    def test_no_channel_configured_means_no_adapters(self, tmp_path: Path) -> None:
        """The default is empty, so an unconfigured deployment behaves as before."""
        settings = CommunitySettings(
            workspaces_root=tmp_path / "workspaces",
            event_store_path=tmp_path / "event_store",
            catalog_path=tmp_path / "catalog",
        )
        services = wire_community(settings)
        try:
            dispatcher = next(
                s
                for s in services.team_manager._shared_subscribers
                if isinstance(s, InteractionChannelDispatcher)
            )
            assert dispatcher._adapters == []
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)

    def test_configured_channel_reaches_the_dispatchers_adapter_list(self, tmp_path: Path) -> None:
        """``settings.channels`` reaches the parser registry, and its adapters the
        dispatcher — this is ``get_adapters()``'s first caller in ``src/``."""
        settings = CommunitySettings(
            workspaces_root=tmp_path / "workspaces",
            event_store_path=tmp_path / "event_store",
            catalog_path=tmp_path / "catalog",
            channels={
                "telegram": ChannelConfig(
                    parser_fqcn=(
                        "akgentic.infra.adapters.channels.telegram_parser.TelegramChannelParser"
                    ),
                    adapter_fqcn=(
                        "akgentic.infra.adapters.channels.telegram_adapter.TelegramChannelAdapter"
                    ),
                    config={"bot_token": "test-token"},
                )
            },
        )
        services = wire_community(settings)
        try:
            dispatcher = next(
                s
                for s in services.team_manager._shared_subscribers
                if isinstance(s, InteractionChannelDispatcher)
            )
            assert len(dispatcher._adapters) == 1
            assert isinstance(dispatcher._adapters[0], TelegramChannelAdapter)
            assert dispatcher._adapters == services.channel_parser_registry.get_adapters()
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)

    def test_dispatcher_holds_the_containers_channel_registry(self, tmp_path: Path) -> None:
        """A second registry instance would answer from an index nothing writes to."""
        settings = CommunitySettings(
            workspaces_root=tmp_path / "workspaces",
            event_store_path=tmp_path / "event_store",
            catalog_path=tmp_path / "catalog",
        )
        services = wire_community(settings)
        try:
            dispatcher = next(
                s
                for s in services.team_manager._shared_subscribers
                if isinstance(s, InteractionChannelDispatcher)
            )
            assert dispatcher._registry is services.channel_registry
        finally:
            services.team_manager._actor_system.shutdown(timeout=5)
