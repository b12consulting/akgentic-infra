"""Tests for the worker FastAPI lifespan calling TelemetrySubscriber.close().

Story 27.1 AC #7: after ``WorkerLifecycle.shutdown()`` returns, the worker
lifespan must call ``services.telemetry_subscriber.close()`` exactly once.
When ``services.telemetry_subscriber is None`` the lifespan must not raise.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from akgentic.core import ActorSystem
from akgentic.team.manager import TeamManager
from akgentic.team.ports import EventStore, NullServiceRegistry

from akgentic.infra.adapters.shared.telemetry_subscriber import TelemetrySubscriber
from akgentic.infra.protocols.runtime_cache import RuntimeCache
from akgentic.infra.protocols.worker_handle import WorkerHandle
from akgentic.infra.worker.app import _lifespan, create_worker_app
from akgentic.infra.worker.deps import WorkerServices
from akgentic.infra.worker.settings import WorkerSettings


def _build_services(
    telemetry_subscriber: TelemetrySubscriber | None,
) -> WorkerServices:
    """Build a WorkerServices with all-mocked deps and the given subscriber."""
    return WorkerServices(
        team_manager=MagicMock(spec=TeamManager),
        actor_system=MagicMock(spec=ActorSystem),
        event_store=MagicMock(spec=EventStore),
        service_registry=NullServiceRegistry(),
        runtime_cache=MagicMock(spec=RuntimeCache),
        worker_handle=MagicMock(spec=WorkerHandle),
        telemetry_subscriber=telemetry_subscriber,
    )


class TestLifespanTelemetryClose:
    """Story 27.1 AC #7: worker lifespan invokes telemetry_subscriber.close() on shutdown."""

    @pytest.mark.asyncio
    async def test_lifespan_calls_close_exactly_once(self) -> None:
        """Entering and exiting the lifespan calls close() exactly once."""
        mock_subscriber = MagicMock(spec=TelemetrySubscriber)
        services = _build_services(telemetry_subscriber=mock_subscriber)
        settings = WorkerSettings(shutdown_drain_timeout=0, shutdown_pre_drain_delay=0)
        app = create_worker_app(services, settings)

        async with _lifespan(app):
            mock_subscriber.close.assert_not_called()

        mock_subscriber.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_lifespan_handles_none_subscriber(self) -> None:
        """If telemetry_subscriber is None, the lifespan exits cleanly without raising."""
        services = _build_services(telemetry_subscriber=None)
        settings = WorkerSettings(shutdown_drain_timeout=0, shutdown_pre_drain_delay=0)
        app = create_worker_app(services, settings)

        # Lifespan must not raise on shutdown when no subscriber is wired.
        async with _lifespan(app):
            pass

    @pytest.mark.asyncio
    async def test_close_called_after_lifecycle_shutdown(self) -> None:
        """close() runs AFTER WorkerLifecycle.shutdown() returns.

        Order matters: per-team ``on_stop`` notifications (driven by
        ``WorkerLifecycle.shutdown()``) might still emit telemetry spans;
        draining the daemon before they finish would lose them.
        """
        mock_subscriber = MagicMock(spec=TelemetrySubscriber)
        services = _build_services(telemetry_subscriber=mock_subscriber)
        settings = WorkerSettings(shutdown_drain_timeout=0, shutdown_pre_drain_delay=0)
        app = create_worker_app(services, settings)

        call_order: list[str] = []
        app.state.lifecycle.shutdown = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda drain_timeout: call_order.append("lifecycle.shutdown"),
        )
        mock_subscriber.close.side_effect = lambda: call_order.append("subscriber.close")

        # The MagicMock above replaces an async method with a sync one. Wrap to async.
        original_shutdown = app.state.lifecycle.shutdown

        async def _shutdown(drain_timeout: int) -> None:
            original_shutdown(drain_timeout)

        app.state.lifecycle.shutdown = _shutdown  # type: ignore[method-assign]

        async with _lifespan(app):
            pass

        assert call_order == ["lifecycle.shutdown", "subscriber.close"]
