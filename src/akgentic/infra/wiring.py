"""Community-tier service wiring — assembles all adapters for single-process deployment."""

from __future__ import annotations

from akgentic.catalog.repositories.yaml import (
    YamlAgentCatalogRepository,
    YamlTeamCatalogRepository,
    YamlTemplateCatalogRepository,
    YamlToolCatalogRepository,
)
from akgentic.catalog.services import (
    AgentCatalog,
    TeamCatalog,
    TemplateCatalog,
    ToolCatalog,
)
from akgentic.core import ActorSystem, EventSubscriber
from akgentic.infra.adapters.channel_parser_registry import ChannelParserRegistry
from akgentic.infra.adapters.local_ingestion import LocalIngestion
from akgentic.infra.adapters.local_placement import LocalPlacement
from akgentic.infra.adapters.local_runtime_cache import LocalRuntimeCache
from akgentic.infra.adapters.local_service_registry import LocalServiceRegistry
from akgentic.infra.adapters.local_worker_handle import LocalWorkerHandle
from akgentic.infra.adapters.no_auth import NoAuth
from akgentic.infra.adapters.telemetry_subscriber import TelemetrySubscriber
from akgentic.infra.adapters.yaml_channel_registry import YamlChannelRegistry
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.settings import ServerSettings
from akgentic.team.manager import TeamManager
from akgentic.team.repositories.yaml import YamlEventStore


def wire_community(settings: ServerSettings) -> CommunityServices:
    """Assemble community-tier services for single-process deployment.

    ``LocalIngestion`` is created with deferred ``team_service`` wiring —
    callers must set ``ingestion.team_service`` after constructing ``TeamService``.

    Args:
        settings: Server configuration

    Returns:
        Fully wired CommunityServices container
    """
    event_store = YamlEventStore(data_dir=settings.workspaces_root)
    service_registry = LocalServiceRegistry()
    actor_system, team_manager = _build_actor_layer(event_store, service_registry)
    catalogs = _build_catalogs(settings)

    local_placement = LocalPlacement(team_manager, service_registry)
    local_worker_handle = LocalWorkerHandle(team_manager, service_registry)
    service_registry.register_instance(local_placement.instance_id)

    return CommunityServices(
        placement=local_placement,
        worker_handle=local_worker_handle,
        service_registry=service_registry,
        auth=NoAuth(),
        event_store=event_store,
        runtime_cache=LocalRuntimeCache(),
        ingestion=LocalIngestion(),
        channel_registry=YamlChannelRegistry(
            registry_path=settings.workspaces_root / "channel-registry.yaml",
        ),
        actor_system=actor_system,
        team_manager=team_manager,
        channel_parser_registry=ChannelParserRegistry(channels_config={}),
        team_catalog=catalogs[0],
        agent_catalog=catalogs[1],
        tool_catalog=catalogs[2],
        template_catalog=catalogs[3],
    )


def _build_actor_layer(
    event_store: YamlEventStore,
    service_registry: LocalServiceRegistry,
) -> tuple[ActorSystem, TeamManager]:
    """Build ActorSystem and TeamManager."""
    shared_subscribers: list[EventSubscriber] = [TelemetrySubscriber()]
    actor_system = ActorSystem()
    team_manager = TeamManager(
        actor_system=actor_system,
        event_store=event_store,
        service_registry=service_registry,
        subscribers=shared_subscribers,
    )
    return actor_system, team_manager


def _build_catalogs(
    settings: ServerSettings,
) -> tuple[TeamCatalog, AgentCatalog, ToolCatalog, TemplateCatalog]:
    """Build YAML-backed catalog services."""
    catalog_root = settings.catalog_path or settings.workspaces_root / "catalog"
    template_catalog = TemplateCatalog(
        repository=YamlTemplateCatalogRepository(catalog_dir=catalog_root / "templates"),
    )
    tool_catalog = ToolCatalog(
        repository=YamlToolCatalogRepository(catalog_dir=catalog_root / "tools"),
    )
    agent_catalog = AgentCatalog(
        repository=YamlAgentCatalogRepository(catalog_dir=catalog_root / "agents"),
        template_catalog=template_catalog,
        tool_catalog=tool_catalog,
    )
    team_catalog = TeamCatalog(
        repository=YamlTeamCatalogRepository(catalog_dir=catalog_root / "teams"),
        agent_catalog=agent_catalog,
    )
    return team_catalog, agent_catalog, tool_catalog, template_catalog
