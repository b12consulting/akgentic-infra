"""Shared adapters — tier-agnostic implementations reusable by all deployment profiles.

Adapters in this package are independent of deployment topology. They integrate
with external services, provide dispatch infrastructure, or implement logic that
every tier needs identically. A future pro or enterprise tier imports from here
without modification.

The interaction-channel layer used to live here and now has its own package,
``akgentic.infra.adapters.channels``. Three modules keep a re-exporting stub at
the old path because the sibling deployment packages import them that way, and
those are separate repositories with their own branch and PR — see each stub.
"""

from __future__ import annotations

from akgentic.infra.adapters.shared.event_stream_subscriber import EventStreamSubscriber
from akgentic.infra.adapters.shared.runtime_cache_eviction_subscriber import (
    RuntimeCacheEvictionSubscriber,
)
from akgentic.infra.adapters.shared.telemetry_subscriber import TelemetrySubscriber

__all__ = [
    "EventStreamSubscriber",
    "RuntimeCacheEvictionSubscriber",
    "TelemetrySubscriber",
]
