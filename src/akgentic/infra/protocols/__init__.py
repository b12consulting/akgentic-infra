"""Infrastructure protocols — stable abstractions for community and tier implementations."""

from __future__ import annotations

from akgentic.infra.protocols.auth import AuthStrategy
from akgentic.infra.protocols.channels import (
    ChannelMessage,
    ChannelParser,
    ChannelRegistry,
    InteractionChannelAdapter,
    InteractionChannelIngestion,
)
from akgentic.infra.protocols.health import HealthMonitor
from akgentic.infra.protocols.placement import PlacementStrategy
from akgentic.infra.protocols.recovery import RecoveryPolicy

__all__ = [
    "AuthStrategy",
    "ChannelMessage",
    "ChannelParser",
    "ChannelRegistry",
    "HealthMonitor",
    "InteractionChannelAdapter",
    "InteractionChannelIngestion",
    "PlacementStrategy",
    "RecoveryPolicy",
]
