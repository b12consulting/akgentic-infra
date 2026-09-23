"""Adapters module — re-exports from the channels, shared and community sub-packages.

This top-level ``__init__.py`` provides backwards-compatible imports. All
adapters can be imported directly from ``akgentic.infra.adapters`` or from
their own sub-package:

- ``channels/`` — the interaction-channel layer: registry, router, dispatcher,
  and a parser/adapter pair per service.
- ``shared/`` — tier-agnostic pieces that are not channels: policies and event
  subscribers.
- ``community/`` — the single-process tier's own implementations.
"""

from __future__ import annotations

from akgentic.infra.adapters.channels import (
    ChannelConfig,
    ChannelParserRegistry,
    ChannelRouteContext,
    DefaultChannelRouter,
    InteractionChannelDispatcher,
    SignalChannelAdapter,
    SignalChannelParser,
    TeamsChannelAdapter,
    TeamsChannelParser,
    TelegramChannelAdapter,
    TelegramChannelParser,
)
from akgentic.infra.adapters.community import (
    LocalPlacement,
    LocalRuntimeCache,
    LocalTeamHandle,
    LocalWorkerHandle,
    NoAuth,
    YamlChannelRegistry,
)
from akgentic.infra.adapters.shared import TelemetrySubscriber

__all__ = [
    "ChannelConfig",
    "ChannelParserRegistry",
    "ChannelRouteContext",
    "DefaultChannelRouter",
    "InteractionChannelDispatcher",
    "LocalPlacement",
    "LocalRuntimeCache",
    "LocalTeamHandle",
    "LocalWorkerHandle",
    "NoAuth",
    "SignalChannelAdapter",
    "SignalChannelParser",
    "TeamsChannelAdapter",
    "TeamsChannelParser",
    "TelegramChannelAdapter",
    "TelegramChannelParser",
    "TelemetrySubscriber",
    "YamlChannelRegistry",
]
