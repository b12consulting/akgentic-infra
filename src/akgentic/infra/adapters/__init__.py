"""Adapters module — community-tier implementations of infrastructure protocols."""

from __future__ import annotations

from akgentic.infra.adapters.local_placement import LocalPlacement
from akgentic.infra.adapters.local_service_registry import LocalServiceRegistry
from akgentic.infra.adapters.no_auth import NoAuth
from akgentic.infra.adapters.telemetry_subscriber import TelemetrySubscriber

__all__ = [
    "LocalPlacement",
    "LocalServiceRegistry",
    "NoAuth",
    "TelemetrySubscriber",
]
