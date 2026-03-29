"""Adapters module — community-tier implementations of infrastructure protocols."""

from __future__ import annotations

from akgentic.infra.adapters.channel_dispatcher import InteractionChannelDispatcher
from akgentic.infra.adapters.channel_parser_registry import (
    ChannelConfig,
    ChannelParserRegistry,
)
from akgentic.infra.adapters.local_ingestion import LocalIngestion
from akgentic.infra.adapters.local_placement import LocalPlacement
from akgentic.infra.adapters.local_runtime_cache import LocalRuntimeCache
from akgentic.infra.adapters.local_service_registry import LocalServiceRegistry
from akgentic.infra.adapters.local_team_handle import LocalTeamHandle
from akgentic.infra.adapters.local_worker_handle import LocalWorkerHandle
from akgentic.infra.adapters.no_auth import NoAuth
from akgentic.infra.adapters.telemetry_subscriber import TelemetrySubscriber
from akgentic.infra.adapters.websocket_subscriber import WebSocketEventSubscriber
from akgentic.infra.adapters.yaml_channel_registry import YamlChannelRegistry

__all__ = [
    "ChannelConfig",
    "ChannelParserRegistry",
    "InteractionChannelDispatcher",
    "LocalIngestion",
    "LocalPlacement",
    "LocalRuntimeCache",
    "LocalServiceRegistry",
    "LocalTeamHandle",
    "LocalWorkerHandle",
    "NoAuth",
    "TelemetrySubscriber",
    "WebSocketEventSubscriber",
    "YamlChannelRegistry",
]
