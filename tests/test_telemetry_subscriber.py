"""Tests for TelemetrySubscriber adapter."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from akgentic.infra.adapters.telemetry_subscriber import TelemetrySubscriber


class TestTelemetrySubscriberProtocolCompliance:
    """AC5: TelemetrySubscriber implements EventSubscriber protocol."""

    def test_has_on_message_method(self) -> None:
        """TelemetrySubscriber exposes on_message."""
        subscriber = TelemetrySubscriber()
        assert callable(subscriber.on_message)

    def test_has_on_stop_method(self) -> None:
        """TelemetrySubscriber exposes on_stop."""
        subscriber = TelemetrySubscriber()
        assert callable(subscriber.on_stop)

    def test_on_message_signature_matches_protocol(self) -> None:
        """on_message has msg parameter matching EventSubscriber."""
        sig = inspect.signature(TelemetrySubscriber.on_message)
        assert "msg" in sig.parameters

    def test_on_stop_signature_matches_protocol(self) -> None:
        """on_stop has no parameters beyond self."""
        sig = inspect.signature(TelemetrySubscriber.on_stop)
        params = [p for p in sig.parameters if p != "self"]
        assert len(params) == 0


class TestTelemetrySubscriberBehavior:
    """AC6: TelemetrySubscriber logs/traces events."""

    def test_on_message_does_not_raise(self) -> None:
        """on_message processes a mock message without error."""
        subscriber = TelemetrySubscriber()
        msg = MagicMock()
        msg.__class__.__name__ = "StartMessage"
        subscriber.on_message(msg)

    def test_on_stop_does_not_raise(self) -> None:
        """on_stop completes without error."""
        subscriber = TelemetrySubscriber()
        subscriber.on_stop()
