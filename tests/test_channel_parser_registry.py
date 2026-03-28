"""Tests for ChannelParserRegistry — FQCN-based channel parser/adapter resolution."""

from __future__ import annotations

import uuid

import pytest

from akgentic.infra.adapters.channel_parser_registry import (
    ChannelConfig,
    ChannelParserRegistry,
    import_class,
)
from akgentic.infra.protocols.channels import ChannelMessage, JsonValue

# --- Stub implementations for testing ---


class StubWhatsAppParser:
    """Test stub satisfying ChannelParser protocol."""

    @property
    def channel_name(self) -> str:
        return "whatsapp"

    @property
    def default_catalog_entry(self) -> str:
        return "default-whatsapp-team"

    async def parse(self, payload: dict[str, JsonValue]) -> ChannelMessage:
        return ChannelMessage(
            content=str(payload.get("text", "")),
            channel_user_id=str(payload.get("from", "")),
        )


class StubWhatsAppAdapter:
    """Test stub satisfying InteractionChannelAdapter protocol."""

    def matches(self, msg: object) -> bool:
        return False

    def deliver(self, msg: object) -> None:
        pass

    def on_stop(self, team_id: uuid.UUID) -> None:
        pass


class StubSlackParser:
    """Test stub for a second channel parser."""

    @property
    def channel_name(self) -> str:
        return "slack"

    @property
    def default_catalog_entry(self) -> str:
        return "default-slack-team"

    async def parse(self, payload: dict[str, JsonValue]) -> ChannelMessage:
        return ChannelMessage(
            content=str(payload.get("text", "")),
            channel_user_id=str(payload.get("user", "")),
        )


class StubSlackAdapter:
    """Test stub for a second channel adapter."""

    def matches(self, msg: object) -> bool:
        return False

    def deliver(self, msg: object) -> None:
        pass

    def on_stop(self, team_id: uuid.UUID) -> None:
        pass


# --- import_class tests ---


def test_import_class_resolves_known_class() -> None:
    """import_class() resolves a known class from its FQCN."""
    cls = import_class("akgentic.infra.protocols.channels.ChannelMessage")
    assert cls is ChannelMessage


def test_import_class_invalid_fqcn_no_dot() -> None:
    """import_class() raises ImportError for FQCN without a module path."""
    with pytest.raises(ImportError, match="Invalid FQCN"):
        import_class("NoDots")


def test_import_class_module_not_found() -> None:
    """import_class() raises ImportError for non-existent module."""
    with pytest.raises(ImportError, match="not found"):
        import_class("nonexistent.module.SomeClass")


def test_import_class_class_not_found() -> None:
    """import_class() raises ImportError for missing class in valid module."""
    with pytest.raises(ImportError, match="not found"):
        import_class("akgentic.infra.protocols.channels.NonExistentClass")


# --- ChannelParserRegistry tests ---


def _make_config() -> dict[str, ChannelConfig]:
    """Build a channels config pointing to test stubs in this module."""
    this_module = "tests.test_channel_parser_registry"
    return {
        "whatsapp": ChannelConfig(
            parser_fqcn=f"{this_module}.StubWhatsAppParser",
            adapter_fqcn=f"{this_module}.StubWhatsAppAdapter",
        ),
        "slack": ChannelConfig(
            parser_fqcn=f"{this_module}.StubSlackParser",
            adapter_fqcn=f"{this_module}.StubSlackAdapter",
        ),
    }


def test_registry_loads_parsers_and_adapters() -> None:
    """ChannelParserRegistry resolves parsers and adapters from config."""
    registry = ChannelParserRegistry(_make_config())
    assert len(registry.channel_names()) == 2
    assert len(registry.get_adapters()) == 2


def test_get_parser_returns_correct_parser() -> None:
    """get_parser() returns the parser indexed by channel_name."""
    registry = ChannelParserRegistry(_make_config())
    parser = registry.get_parser("whatsapp")
    assert parser is not None
    assert parser.channel_name == "whatsapp"


def test_get_parser_returns_none_for_unknown() -> None:
    """get_parser() returns None for an unregistered channel name."""
    registry = ChannelParserRegistry(_make_config())
    assert registry.get_parser("telegram") is None


def test_get_adapters_returns_all_adapters() -> None:
    """get_adapters() returns all resolved adapter instances."""
    registry = ChannelParserRegistry(_make_config())
    adapters = registry.get_adapters()
    assert len(adapters) == 2
    # Verify they are distinct instances
    assert adapters[0] is not adapters[1]


def test_get_adapters_returns_copy() -> None:
    """get_adapters() returns a copy, not the internal list."""
    registry = ChannelParserRegistry(_make_config())
    adapters = registry.get_adapters()
    adapters.clear()
    assert len(registry.get_adapters()) == 2


def test_channel_names_returns_all_names() -> None:
    """channel_names() returns all registered channel names."""
    registry = ChannelParserRegistry(_make_config())
    names = registry.channel_names()
    assert sorted(names) == ["slack", "whatsapp"]


def test_registry_with_empty_config() -> None:
    """ChannelParserRegistry with empty config has no parsers or adapters."""
    registry = ChannelParserRegistry({})
    assert registry.channel_names() == []
    assert registry.get_adapters() == []


def test_registry_invalid_parser_fqcn() -> None:
    """ChannelParserRegistry raises ImportError for invalid parser FQCN."""
    config = {
        "bad": ChannelConfig(
            parser_fqcn="nonexistent.module.BadParser",
            adapter_fqcn="tests.test_channel_parser_registry.StubWhatsAppAdapter",
        ),
    }
    with pytest.raises(ImportError):
        ChannelParserRegistry(config)


def test_registry_invalid_adapter_fqcn() -> None:
    """ChannelParserRegistry raises ImportError for invalid adapter FQCN."""
    config = {
        "bad": ChannelConfig(
            parser_fqcn="tests.test_channel_parser_registry.StubWhatsAppParser",
            adapter_fqcn="nonexistent.module.BadAdapter",
        ),
    }
    with pytest.raises(ImportError):
        ChannelParserRegistry(config)


def test_channel_config_is_pydantic_model() -> None:
    """ChannelConfig is a Pydantic model with correct fields."""
    from pydantic import BaseModel

    assert issubclass(ChannelConfig, BaseModel)
    fields = ChannelConfig.model_fields
    assert "parser_fqcn" in fields
    assert "adapter_fqcn" in fields
