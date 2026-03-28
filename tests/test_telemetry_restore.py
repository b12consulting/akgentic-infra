"""Tests for TelemetrySubscriber restore-awareness (Story 4.2 additions)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from akgentic.infra.adapters.telemetry_subscriber import TelemetrySubscriber


class TestTelemetryRestoreAwareness:
    """AC #5, #6: TelemetrySubscriber supports set_restoring to skip logfire spans."""

    def test_on_message_calls_logfire_for_live_events(self) -> None:
        """AC #5: logfire.info is called for live (non-restoring) events."""
        subscriber = TelemetrySubscriber()
        msg = MagicMock()
        msg.__class__.__name__ = "StartMessage"

        with patch("akgentic.infra.adapters.telemetry_subscriber.logfire") as mock_lf:
            subscriber.on_message(msg)
            mock_lf.info.assert_called_once()

    def test_restoring_skips_logfire_emission(self) -> None:
        """AC #6: set_restoring(True) suppresses logfire.info calls."""
        subscriber = TelemetrySubscriber()
        subscriber.set_restoring(True)
        msg = MagicMock()
        msg.__class__.__name__ = "StartMessage"

        with patch("akgentic.infra.adapters.telemetry_subscriber.logfire") as mock_lf:
            subscriber.on_message(msg)
            mock_lf.info.assert_not_called()

    def test_restoring_false_resumes_logfire(self) -> None:
        """set_restoring(False) resumes normal logfire emission."""
        subscriber = TelemetrySubscriber()
        subscriber.set_restoring(True)
        subscriber.set_restoring(False)
        msg = MagicMock()
        msg.__class__.__name__ = "StartMessage"

        with patch("akgentic.infra.adapters.telemetry_subscriber.logfire") as mock_lf:
            subscriber.on_message(msg)
            mock_lf.info.assert_called_once()

    def test_on_stop_does_not_raise(self) -> None:
        """on_stop completes without error."""
        subscriber = TelemetrySubscriber()
        subscriber.on_stop()
