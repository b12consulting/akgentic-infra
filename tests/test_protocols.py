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


# --- InteractionChannelAdapter ---


def test_interaction_channel_adapter_is_protocol() -> None:
    """InteractionChannelAdapter uses typing.Protocol base."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    assert Protocol in inspect.getmro(InteractionChannelAdapter)


def test_interaction_channel_adapter_has_matches() -> None:
    """InteractionChannelAdapter defines matches with msg parameter."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    assert hasattr(InteractionChannelAdapter, "matches")
    sig = inspect.signature(InteractionChannelAdapter.matches)
    assert "msg" in sig.parameters


def test_interaction_channel_adapter_has_deliver() -> None:
    """InteractionChannelAdapter defines deliver with msg parameter."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    assert hasattr(InteractionChannelAdapter, "deliver")
    sig = inspect.signature(InteractionChannelAdapter.deliver)
    assert "msg" in sig.parameters


def test_interaction_channel_adapter_has_on_stop() -> None:
    """InteractionChannelAdapter defines on_stop with team_id parameter."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    assert hasattr(InteractionChannelAdapter, "on_stop")
    sig = inspect.signature(InteractionChannelAdapter.on_stop)
    assert "team_id" in sig.parameters


def test_interaction_channel_adapter_matches_returns_bool() -> None:
    """InteractionChannelAdapter.matches has bool return annotation."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    sig = inspect.signature(InteractionChannelAdapter.matches)
    assert sig.return_annotation is bool or sig.return_annotation == "bool"


def test_interaction_channel_adapter_deliver_returns_none() -> None:
    """InteractionChannelAdapter.deliver has None return annotation."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    sig = inspect.signature(InteractionChannelAdapter.deliver)
    assert sig.return_annotation is None or sig.return_annotation == "None"


def test_interaction_channel_adapter_on_stop_returns_none() -> None:
    """InteractionChannelAdapter.on_stop returns None."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    hints = get_type_hints(InteractionChannelAdapter.on_stop)
    assert hints["return"] is type(None)


def test_interaction_channel_adapter_structural_subtyping() -> None:
    """A concrete class satisfying InteractionChannelAdapter is recognized."""
    from akgentic.infra.protocols import InteractionChannelAdapter

    class FakeAdapter:
        def matches(self, msg: object) -> bool:
            return True

        def deliver(self, msg: object) -> None:
            pass

        def on_stop(self, team_id: uuid.UUID) -> None:
            pass

    assert isinstance(FakeAdapter(), InteractionChannelAdapter)


# --- InteractionChannelIngestion ---


def test_interaction_channel_ingestion_is_protocol() -> None:
    """InteractionChannelIngestion uses typing.Protocol base."""
    from akgentic.infra.protocols import InteractionChannelIngestion

    assert Protocol in inspect.getmro(InteractionChannelIngestion)


def test_interaction_channel_ingestion_has_route_reply() -> None:
    """InteractionChannelIngestion defines async route_reply."""
    from akgentic.infra.protocols import InteractionChannelIngestion

    assert hasattr(InteractionChannelIngestion, "route_reply")
    sig = inspect.signature(InteractionChannelIngestion.route_reply)
    assert "team_id" in sig.parameters
    assert "content" in sig.parameters
    assert "original_message_id" in sig.parameters
    assert inspect.iscoroutinefunction(InteractionChannelIngestion.route_reply)


def test_interaction_channel_ingestion_has_initiate_team() -> None:
    """InteractionChannelIngestion defines async initiate_team."""
    from akgentic.infra.protocols import InteractionChannelIngestion

    assert hasattr(InteractionChannelIngestion, "initiate_team")
    sig = inspect.signature(InteractionChannelIngestion.initiate_team)
    assert "content" in sig.parameters
    assert "channel_user_id" in sig.parameters
    assert "catalog_entry_id" in sig.parameters
    assert inspect.iscoroutinefunction(InteractionChannelIngestion.initiate_team)


def test_interaction_channel_ingestion_route_reply_returns_none() -> None:
    """InteractionChannelIngestion.route_reply returns None."""
    from akgentic.infra.protocols import InteractionChannelIngestion

    hints = get_type_hints(InteractionChannelIngestion.route_reply)
    assert hints["return"] is type(None)


def test_interaction_channel_ingestion_initiate_team_returns_uuid() -> None:
    """InteractionChannelIngestion.initiate_team returns UUID."""
    from akgentic.infra.protocols import InteractionChannelIngestion

    hints = get_type_hints(InteractionChannelIngestion.initiate_team)
    assert hints["return"] is uuid.UUID


def test_interaction_channel_ingestion_structural_subtyping() -> None:
    """A concrete class satisfying InteractionChannelIngestion is recognized."""
    from akgentic.infra.protocols import InteractionChannelIngestion

    class FakeIngestion:
        async def route_reply(
            self,
            team_id: uuid.UUID,
            content: str,
            original_message_id: str | None = None,
        ) -> None:
            pass

        async def initiate_team(
            self,
            content: str,
            channel_user_id: str,
            catalog_entry_id: str,
        ) -> uuid.UUID:
            return uuid.uuid4()

    assert isinstance(FakeIngestion(), InteractionChannelIngestion)


# --- ChannelParser ---


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


def test_channel_parser_has_channel_name_property() -> None:
    """ChannelParser defines channel_name property."""
    from akgentic.infra.protocols import ChannelParser

    assert hasattr(ChannelParser, "channel_name")


def test_channel_parser_has_default_catalog_entry_property() -> None:
    """ChannelParser defines default_catalog_entry property."""
    from akgentic.infra.protocols import ChannelParser

    assert hasattr(ChannelParser, "default_catalog_entry")


def test_channel_parser_return_type() -> None:
    """ChannelParser.parse returns ChannelMessage."""
    from akgentic.infra.protocols import ChannelMessage, ChannelParser

    hints = get_type_hints(ChannelParser.parse)
    assert hints["return"] is ChannelMessage


# --- ChannelRegistry ---


def test_channel_registry_is_protocol() -> None:
    """ChannelRegistry uses typing.Protocol base."""
    from akgentic.infra.protocols import ChannelRegistry

    assert Protocol in inspect.getmro(ChannelRegistry)


def test_channel_registry_has_register() -> None:
    """ChannelRegistry defines async register with channel, channel_user_id, team_id."""
    from akgentic.infra.protocols import ChannelRegistry

    assert hasattr(ChannelRegistry, "register")
    sig = inspect.signature(ChannelRegistry.register)
    assert "channel" in sig.parameters
    assert "channel_user_id" in sig.parameters
    assert "team_id" in sig.parameters
    assert inspect.iscoroutinefunction(ChannelRegistry.register)


def test_channel_registry_has_find_team() -> None:
    """ChannelRegistry defines async find_team with channel and channel_user_id."""
    from akgentic.infra.protocols import ChannelRegistry

    assert hasattr(ChannelRegistry, "find_team")
    sig = inspect.signature(ChannelRegistry.find_team)
    assert "channel" in sig.parameters
    assert "channel_user_id" in sig.parameters
    assert inspect.iscoroutinefunction(ChannelRegistry.find_team)


def test_channel_registry_has_deregister() -> None:
    """ChannelRegistry defines async deregister with channel and channel_user_id."""
    from akgentic.infra.protocols import ChannelRegistry

    assert hasattr(ChannelRegistry, "deregister")
    sig = inspect.signature(ChannelRegistry.deregister)
    assert "channel" in sig.parameters
    assert "channel_user_id" in sig.parameters
    assert inspect.iscoroutinefunction(ChannelRegistry.deregister)


def test_channel_registry_find_team_return_type() -> None:
    """ChannelRegistry.find_team returns uuid.UUID | None."""
    from akgentic.infra.protocols import ChannelRegistry

    hints = get_type_hints(ChannelRegistry.find_team)
    assert hints["return"] == uuid.UUID | None


def test_channel_registry_register_returns_none() -> None:
    """ChannelRegistry.register returns None."""
    from akgentic.infra.protocols import ChannelRegistry

    hints = get_type_hints(ChannelRegistry.register)
    assert hints["return"] is type(None)


def test_channel_registry_deregister_returns_none() -> None:
    """ChannelRegistry.deregister returns None."""
    from akgentic.infra.protocols import ChannelRegistry

    hints = get_type_hints(ChannelRegistry.deregister)
    assert hints["return"] is type(None)


def test_channel_registry_structural_subtyping() -> None:
    """A concrete class satisfying ChannelRegistry is recognized."""
    from akgentic.infra.protocols import ChannelRegistry

    class FakeRegistry:
        async def register(
            self, channel: str, channel_user_id: str, team_id: uuid.UUID
        ) -> None:
            pass

        async def find_team(
            self, channel: str, channel_user_id: str
        ) -> uuid.UUID | None:
            return None

        async def deregister(self, channel: str, channel_user_id: str) -> None:
            pass

    assert isinstance(FakeRegistry(), ChannelRegistry)


# --- ChannelMessage ---


def test_channel_message_is_pydantic_model() -> None:
    """ChannelMessage is a Pydantic BaseModel with correct fields."""
    from pydantic import BaseModel

    from akgentic.infra.protocols import ChannelMessage

    assert issubclass(ChannelMessage, BaseModel)
    fields = ChannelMessage.model_fields
    assert "content" in fields
    assert "channel_user_id" in fields
    assert "team_id" in fields
    assert "message_id" in fields


def test_channel_message_field_descriptions() -> None:
    """ChannelMessage fields have descriptions."""
    from akgentic.infra.protocols import ChannelMessage

    for name, field_info in ChannelMessage.model_fields.items():
        assert field_info.description is not None, f"Field {name} missing description"


def test_channel_message_optional_defaults() -> None:
    """ChannelMessage team_id and message_id default to None."""
    from akgentic.infra.protocols import ChannelMessage

    msg = ChannelMessage(content="hello", channel_user_id="u1")
    assert msg.team_id is None
    assert msg.message_id is None


def test_channel_message_with_all_fields() -> None:
    """ChannelMessage can be created with all fields."""
    from akgentic.infra.protocols import ChannelMessage

    tid = uuid.uuid4()
    msg = ChannelMessage(
        content="hello",
        channel_user_id="u1",
        team_id=tid,
        message_id="msg-123",
    )
    assert msg.content == "hello"
    assert msg.channel_user_id == "u1"
    assert msg.team_id == tid
    assert msg.message_id == "msg-123"


# --- Non-channel protocols (unchanged) ---


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
