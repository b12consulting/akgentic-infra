"""Community-tier service wiring — assembles all adapters for single-process deployment."""

from __future__ import annotations

from akgentic.core import ActorSystem, EventSubscriber
from akgentic.infra.adapters.local_placement import LocalPlacement
from akgentic.infra.adapters.local_service_registry import LocalServiceRegistry
from akgentic.infra.adapters.no_auth import NoAuth
from akgentic.infra.adapters.telemetry_subscriber import TelemetrySubscriber
from akgentic.infra.server.deps import CommunityServices
from akgentic.infra.server.settings import ServerSettings
from akgentic.team.manager import TeamManager
from akgentic.team.repositories.yaml import YamlEventStore


def wire_community(settings: ServerSettings) -> CommunityServices:
    """Assemble community-tier services for single-process deployment.

    Assembly order:
    1. EventStore (YamlEventStore)
    2. ServiceRegistry (LocalServiceRegistry)
    3. Shared subscribers ([TelemetrySubscriber])
    4. ActorSystem
    5. TeamManager
    6. PlacementStrategy (LocalPlacement)
    7. AuthStrategy (NoAuth)

    Args:
        settings: Server configuration

    Returns:
        Fully wired CommunityServices container
    """
    event_store = YamlEventStore(data_dir=settings.workspaces_root)
    service_registry = LocalServiceRegistry()
    shared_subscribers: list[EventSubscriber] = [TelemetrySubscriber()]
    actor_system = ActorSystem()
    team_manager = TeamManager(
        actor_system=actor_system,
        event_store=event_store,
        service_registry=service_registry,
        subscribers=shared_subscribers,
    )
    placement = LocalPlacement()
    auth = NoAuth()

    return CommunityServices(
        placement=placement,
        service_registry=service_registry,
        auth=auth,
        event_store=event_store,
        team_manager=team_manager,
    )
