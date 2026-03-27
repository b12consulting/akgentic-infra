"""Validate protocol definitions are structurally correct."""

from __future__ import annotations

import inspect
import uuid
from typing import Protocol, get_type_hints


def test_placement_strategy_is_protocol() -> None:
    """PlacementStrategy uses typing.Protocol base."""
    from akgentic.infra.protocols import PlacementStrategy

    assert Protocol in inspect.getmro(PlacementStrategy)


def test_placement_strategy_has_select_worker() -> None:
    """PlacementStrategy defines select_worker with team_id parameter."""
    from akgentic.infra.protocols import PlacementStrategy

    assert hasattr(PlacementStrategy, "select_worker")
    sig = inspect.signature(PlacementStrategy.select_worker)
    assert "team_id" in sig.parameters


def test_auth_strategy_is_protocol() -> None:
    """AuthStrategy uses typing.Protocol base."""
    from akgentic.infra.protocols import AuthStrategy

    assert Protocol in inspect.getmro(AuthStrategy)


def test_auth_strategy_has_authenticate() -> None:
    """AuthStrategy defines authenticate with request parameter."""
    from akgentic.infra.protocols import AuthStrategy

    assert hasattr(AuthStrategy, "authenticate")
    sig = inspect.signature(AuthStrategy.authenticate)
    assert "request" in sig.parameters


def test_recovery_policy_is_protocol() -> None:
    """RecoveryPolicy uses typing.Protocol base."""
    from akgentic.infra.protocols import RecoveryPolicy

    assert Protocol in inspect.getmro(RecoveryPolicy)


def test_recovery_policy_has_recover() -> None:
    """RecoveryPolicy defines recover with instance_id and team_ids parameters."""
    from akgentic.infra.protocols import RecoveryPolicy

    assert hasattr(RecoveryPolicy, "recover")
    sig = inspect.signature(RecoveryPolicy.recover)
    assert "instance_id" in sig.parameters
    assert "team_ids" in sig.parameters


def test_health_monitor_is_protocol() -> None:
    """HealthMonitor uses typing.Protocol base."""
    from akgentic.infra.protocols import HealthMonitor

    assert Protocol in inspect.getmro(HealthMonitor)


def test_health_monitor_has_check_health() -> None:
    """HealthMonitor defines check_health method."""
    from akgentic.infra.protocols import HealthMonitor

    assert hasattr(HealthMonitor, "check_health")
    sig = inspect.signature(HealthMonitor.check_health)
    # Only self parameter
    assert len(sig.parameters) == 1


def test_interaction_channel_adapter_is_protocol() -> None:
    """InteractionChannelAdapter uses typing.Protocol base."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    assert Protocol in inspect.getmro(InteractionChannelAdapter)


def test_interaction_channel_adapter_has_send() -> None:
    """InteractionChannelAdapter defines send with channel_id and message parameters."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    assert hasattr(InteractionChannelAdapter, "send")
    sig = inspect.signature(InteractionChannelAdapter.send)
    assert "channel_id" in sig.parameters
    assert "message" in sig.parameters


def test_interaction_channel_ingestion_is_protocol() -> None:
    """InteractionChannelIngestion uses typing.Protocol base."""
    from akgentic.infra.protocols import InteractionChannelIngestion

    assert Protocol in inspect.getmro(InteractionChannelIngestion)


def test_interaction_channel_ingestion_has_route_inbound() -> None:
    """InteractionChannelIngestion defines route_inbound with channel_id and content."""
    from akgentic.infra.protocols import InteractionChannelIngestion

    assert hasattr(InteractionChannelIngestion, "route_inbound")
    sig = inspect.signature(InteractionChannelIngestion.route_inbound)
    assert "channel_id" in sig.parameters
    assert "content" in sig.parameters


def test_channel_parser_is_protocol() -> None:
    """ChannelParser uses typing.Protocol base."""
    from akgentic.infra.protocols import ChannelParser

    assert Protocol in inspect.getmro(ChannelParser)


def test_channel_parser_has_parse() -> None:
    """ChannelParser defines async parse with payload parameter."""
    from akgentic.infra.protocols import ChannelParser

    assert hasattr(ChannelParser, "parse")
    sig = inspect.signature(ChannelParser.parse)
    assert "payload" in sig.parameters
    assert inspect.iscoroutinefunction(ChannelParser.parse)


def test_channel_registry_is_protocol() -> None:
    """ChannelRegistry uses typing.Protocol base."""
    from akgentic.infra.protocols import ChannelRegistry

    assert Protocol in inspect.getmro(ChannelRegistry)


def test_channel_registry_has_find_team() -> None:
    """ChannelRegistry defines async find_team with channel_id and sender_id."""
    from akgentic.infra.protocols import ChannelRegistry

    assert hasattr(ChannelRegistry, "find_team")
    sig = inspect.signature(ChannelRegistry.find_team)
    assert "channel_id" in sig.parameters
    assert "sender_id" in sig.parameters
    assert inspect.iscoroutinefunction(ChannelRegistry.find_team)


def test_placement_strategy_return_type() -> None:
    """PlacementStrategy.select_worker returns uuid.UUID."""
    from akgentic.infra.protocols import PlacementStrategy

    hints = get_type_hints(PlacementStrategy.select_worker)
    assert hints["return"] is uuid.UUID


def test_auth_strategy_return_type() -> None:
    """AuthStrategy.authenticate returns str | None."""
    from akgentic.infra.protocols import AuthStrategy

    hints = get_type_hints(AuthStrategy.authenticate)
    assert hints["return"] == str | None


def test_recovery_policy_return_type() -> None:
    """RecoveryPolicy.recover returns None."""
    from akgentic.infra.protocols import RecoveryPolicy

    hints = get_type_hints(RecoveryPolicy.recover)
    assert hints["return"] is type(None)


def test_health_monitor_return_type() -> None:
    """HealthMonitor.check_health returns list[uuid.UUID]."""
    from akgentic.infra.protocols import HealthMonitor

    hints = get_type_hints(HealthMonitor.check_health)
    assert hints["return"] == list[uuid.UUID]


def test_interaction_channel_adapter_return_type() -> None:
    """InteractionChannelAdapter.send returns None."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    hints = get_type_hints(InteractionChannelAdapter.send)
    assert hints["return"] is type(None)


def test_interaction_channel_ingestion_return_type() -> None:
    """InteractionChannelIngestion.route_inbound returns None."""
    from akgentic.infra.protocols import InteractionChannelIngestion

    hints = get_type_hints(InteractionChannelIngestion.route_inbound)
    assert hints["return"] is type(None)


def test_channel_parser_return_type() -> None:
    """ChannelParser.parse returns ChannelMessage."""
    from akgentic.infra.protocols import ChannelMessage, ChannelParser

    hints = get_type_hints(ChannelParser.parse)
    assert hints["return"] is ChannelMessage


def test_channel_registry_return_type() -> None:
    """ChannelRegistry.find_team returns uuid.UUID | None."""
    from akgentic.infra.protocols import ChannelRegistry

    hints = get_type_hints(ChannelRegistry.find_team)
    assert hints["return"] == uuid.UUID | None


def test_channel_message_is_pydantic_model() -> None:
    """ChannelMessage is a Pydantic BaseModel with correct fields."""
    from pydantic import BaseModel

    from akgentic.infra.protocols import ChannelMessage

    assert issubclass(ChannelMessage, BaseModel)
    fields = ChannelMessage.model_fields
    assert "channel_id" in fields
    assert "sender_id" in fields
    assert "content" in fields
    assert "metadata" in fields


def test_channel_message_field_descriptions() -> None:
    """ChannelMessage fields have descriptions."""
    from akgentic.infra.protocols import ChannelMessage

    for name, field_info in ChannelMessage.model_fields.items():
        assert field_info.description is not None, f"Field {name} missing description"


def test_channel_message_metadata_defaults_empty() -> None:
    """ChannelMessage metadata defaults to empty dict."""
    from akgentic.infra.protocols import ChannelMessage

    msg = ChannelMessage(channel_id="slack", sender_id="u1", content="hello")
    assert msg.metadata == {}
