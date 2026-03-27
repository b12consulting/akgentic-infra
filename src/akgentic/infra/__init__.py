"""akgentic-infra: Infrastructure backend protocols and community-tier implementations."""

from __future__ import annotations

from pkgutil import extend_path

from akgentic.infra.adapters import (
    LocalPlacement,
    LocalServiceRegistry,
    NoAuth,
    TelemetrySubscriber,
)
from akgentic.infra.protocols import (
    AuthStrategy,
    ChannelMessage,
    ChannelParser,
    ChannelRegistry,
    HealthMonitor,
    InteractionChannelAdapter,
    InteractionChannelIngestion,
    PlacementStrategy,
    RecoveryPolicy,
)
from akgentic.infra.server.app import create_app
from akgentic.infra.server.deps import CommunityServices, TierServices
from akgentic.infra.server.models import (
    CreateTeamRequest,
    TeamListResponse,
    TeamResponse,
)
from akgentic.infra.server.services.team_service import TeamService
from akgentic.infra.server.settings import ServerSettings
from akgentic.infra.wiring import wire_community

__path__ = extend_path(__path__, __name__)

__all__ = [
    # Protocols
    "AuthStrategy",
    "ChannelMessage",
    "ChannelParser",
    "ChannelRegistry",
    "HealthMonitor",
    "InteractionChannelAdapter",
    "InteractionChannelIngestion",
    "PlacementStrategy",
    "RecoveryPolicy",
    # Adapters
    "LocalPlacement",
    "LocalServiceRegistry",
    "NoAuth",
    "TelemetrySubscriber",
    # Server
    "CommunityServices",
    "CreateTeamRequest",
    "ServerSettings",
    "TeamListResponse",
    "TeamResponse",
    "TeamService",
    "TierServices",
    "create_app",
    # Wiring
    "wire_community",
]
