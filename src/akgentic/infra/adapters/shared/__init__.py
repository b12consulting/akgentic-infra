"""Shared adapters — tier-agnostic implementations reusable by all deployment profiles.

Adapters in this package are independent of deployment topology. They integrate
with external services, provide dispatch infrastructure, or implement logic that
every tier needs identically. A future pro or enterprise tier imports from here
without modification.
"""

from __future__ import annotations

from akgentic.infra.adapters.shared.channel_dispatcher import ChannelDispatcher
from akgentic.infra.adapters.shared.channel_parser_registry import (
    ChannelConfig,
    ChannelParserRegistry,
)
from akgentic.infra.adapters.shared.channel_router import ChannelRouteContext, DefaultChannelRouter
from akgentic.infra.adapters.shared.event_stream_subscriber import EventStreamSubscriber
from akgentic.infra.adapters.shared.keep_every_tree_policy import KeepEveryTreePolicy
from akgentic.infra.adapters.shared.null_event_stream import NullEventStream, NullStreamReader
from akgentic.infra.adapters.shared.runtime_cache_eviction_subscriber import (
    RuntimeCacheEvictionSubscriber,
)
from akgentic.infra.adapters.shared.team_tree_only_policy import TeamTreeOnlyPolicy
from akgentic.infra.adapters.shared.telegram_adapter import TelegramChannelAdapter
from akgentic.infra.adapters.shared.telegram_parser import TelegramChannelParser
from akgentic.infra.adapters.shared.telemetry_subscriber import TelemetrySubscriber
from akgentic.infra.adapters.shared.workspace_deletion_mode import workspace_deletion_policy_for

__all__ = [
    "ChannelConfig",
    "ChannelParserRegistry",
    "ChannelRouteContext",
    "DefaultChannelRouter",
    "EventStreamSubscriber",
    "ChannelDispatcher",
    "KeepEveryTreePolicy",
    "NullEventStream",
    "NullStreamReader",
    "RuntimeCacheEvictionSubscriber",
    "TelegramChannelAdapter",
    "TeamTreeOnlyPolicy",
    "TelegramChannelParser",
    "TelemetrySubscriber",
    "workspace_deletion_policy_for",
]
