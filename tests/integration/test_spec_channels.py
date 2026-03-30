"""Integration tests — channel spec compliance: multi-adapter, form-encoded, config passthrough.

Validates ADR-002 remediation for story 6.1 channel subsystem fixes.
"""

from __future__ import annotations

import uuid

import pytest
from akgentic.core.actor_address_impl import ActorAddressProxy
from akgentic.core.messages import SentMessage
from akgentic.core.messages.message import UserMessage
from fastapi.testclient import TestClient

from akgentic.infra.adapters.channel_dispatcher import InteractionChannelDispatcher
from akgentic.infra.adapters.channel_parser_registry import (
    ChannelConfig,
    ChannelParserRegistry,
)

from .test_channels import StubChannelAdapter

pytestmark = [pytest.mark.integration, pytest.mark.llm]


# ---------------------------------------------------------------------------
# AC #3 — Multi-adapter dispatch
# ---------------------------------------------------------------------------


class TestMultiAdapterDispatch:
    """Verify dispatcher delivers to ALL matching adapters (no break short-circuit)."""

    def test_two_adapters_both_receive_message(self) -> None:
        """AC #3: Register 2 StubChannelAdapters, dispatch, verify both received."""
        adapter_a = StubChannelAdapter()
        adapter_b = StubChannelAdapter()
        dispatcher = InteractionChannelDispatcher(
            team_id=uuid.uuid4(),
            adapters=[adapter_a, adapter_b],
        )

        # Build a SentMessage to dispatch
        addr_dict = {
            "__actor_address__": True,
            "__actor_type__": "test",
            "agent_id": str(uuid.uuid4()),
            "name": "@Test",
            "role": "Test",
            "team_id": str(uuid.uuid4()),
            "squad_id": str(uuid.uuid4()),
            "user_message": False,
        }
        proxy = ActorAddressProxy(addr_dict)
        inner = UserMessage(content="test message")
        sent = SentMessage(message=inner, sender=proxy, recipient=proxy)

        dispatcher.on_message(sent)

        assert len(adapter_a.delivered) == 1, "Adapter A should receive the message"
        assert len(adapter_b.delivered) == 1, "Adapter B should receive the message"
        assert adapter_a.delivered[0] is sent
        assert adapter_b.delivered[0] is sent


# ---------------------------------------------------------------------------
# AC #3 — Form-encoded webhook payloads
# ---------------------------------------------------------------------------


class TestFormEncodedWebhook:
    """Verify form-encoded webhook payloads are parsed correctly."""

    def test_form_encoded_webhook_returns_204(
        self,
        channel_client: TestClient,
    ) -> None:
        """AC #3: POST form-encoded to /webhook/{channel} returns 204.

        Uses an existing team via reply flow to avoid creating a new team
        with in-flight LLM calls that would block teardown.
        """
        import time

        # Create a team first so we can use reply flow (no LLM initiation)
        create_resp = channel_client.post(
            "/teams/",
            json={"catalog_entry_id": "test-team"},
        )
        assert create_resp.status_code == 201
        team_id = create_resp.json()["team_id"]

        resp = channel_client.post(
            "/webhook/test-channel",
            data={
                "content": "form test message",
                "channel_user_id": "form-user-1",
                "team_id": team_id,
            },
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 204

        # Stop team to allow clean teardown
        channel_client.post(f"/teams/{team_id}/stop")
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# AC #3 — ChannelConfig passthrough
# ---------------------------------------------------------------------------


class StubParserWithConfig:
    """Parser stub that captures constructor config args."""

    received_config: dict[str, str]

    def __init__(self, **kwargs: str) -> None:
        StubParserWithConfig.received_config = dict(kwargs)

    @property
    def channel_name(self) -> str:
        return "config-test-channel"

    @property
    def default_catalog_entry(self) -> str:
        return "test-team"

    async def parse(self, payload: dict[str, object]) -> None:
        pass  # pragma: no cover


class StubAdapterWithConfig:
    """Adapter stub that captures constructor config args."""

    received_config: dict[str, str]

    def __init__(self, **kwargs: str) -> None:
        StubAdapterWithConfig.received_config = dict(kwargs)

    def matches(self, msg: SentMessage) -> bool:
        return True  # pragma: no cover

    def deliver(self, msg: SentMessage) -> None:
        pass  # pragma: no cover

    def on_stop(self, team_id: uuid.UUID) -> None:
        pass  # pragma: no cover


class TestChannelConfigPassthrough:
    """Verify ChannelConfig.config values reach adapter/parser constructors."""

    def test_config_dict_reaches_constructors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AC #3: Configure channel with config dict, verify it reaches constructors."""
        # Monkeypatch import_class to return our stubs
        monkeypatch.setattr(
            "akgentic.infra.adapters.channel_parser_registry.import_class",
            lambda fqcn: StubParserWithConfig if "Parser" in fqcn else StubAdapterWithConfig,
        )

        config = {
            "test-chan": ChannelConfig(
                parser_fqcn="fake.module.Parser",
                adapter_fqcn="fake.module.Adapter",
                config={"key": "value", "api_token": "secret123"},
            )
        }

        ChannelParserRegistry(channels_config=config)

        assert StubParserWithConfig.received_config == {
            "key": "value",
            "api_token": "secret123",
        }
        assert StubAdapterWithConfig.received_config == {
            "key": "value",
            "api_token": "secret123",
        }
