"""Tests for TeamsChannelAdapter."""

from __future__ import annotations

import json
import uuid
from typing import Any, cast

import httpx
import pytest
from akgentic.core.actor_address_impl import ActorAddressProxy
from akgentic.core.messages.message import UserMessage
from akgentic.core.messages.orchestrator import SentMessage

from akgentic.infra.adapters.channels.teams_adapter import TeamsChannelAdapter
from akgentic.infra.protocols.channels import ChannelAddress, ChannelBinding

TEAM_ID = uuid.uuid4()
APP_ID = "00000000-0000-0000-0000-000000000000"
TENANT_ID = "72f988bf-0000-0000-0000-000000000000"
CONVERSATION_ID = "a:1abcDEF"
SERVICE_URL = "https://smba.trafficmanager.net/emea/"


def _make_addr(role: str = "UserProxy", name: str = "@Human", is_user_proxy: bool = True) -> Any:
    return ActorAddressProxy(
        {
            "__actor_address__": True,
            "__actor_type__": "akgentic.core.actor_address_impl.ActorAddressProxy",
            "agent_id": str(uuid.uuid4()),
            "name": name,
            "role": role,
            "team_id": str(uuid.uuid4()),
            "squad_id": str(uuid.uuid4()),
            "user_message": False,
            "is_user_proxy": is_user_proxy,
        }
    )


def _make_sent_message(
    content: str = "The answer is 42.",
    is_user_proxy: bool = True,
    sender_name: str = "@Expert",
) -> SentMessage:
    recipient = _make_addr(is_user_proxy=is_user_proxy)
    sender = _make_addr(role="assistant", name=sender_name, is_user_proxy=False)
    return SentMessage(
        message=UserMessage(content=content, sender=sender), recipient=recipient, sender=sender
    )


def _binding(channel: str = "teams", metadata: dict | None = None) -> ChannelBinding:
    return ChannelBinding(
        channel=channel,
        channel_user_id=CONVERSATION_ID,
        team_id=TEAM_ID,
        agent_name="@Human",
        metadata=metadata if metadata is not None else {"service_url": SERVICE_URL},
    )


class _Transport(httpx.BaseTransport):
    """Answers the token endpoint and the connector, recording both."""

    def __init__(
        self, token_status: int = 200, send_status: int = 200, expires_in: int = 3600
    ) -> None:
        self.requests: list[httpx.Request] = []
        self._token_status = token_status
        self._send_status = send_status
        self._expires_in = expires_in

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if "login.microsoftonline.com" in str(request.url):
            return httpx.Response(
                self._token_status,
                json={"access_token": "tok-abc", "expires_in": self._expires_in},
            )
        return httpx.Response(self._send_status, json={"id": "1616989574409"})

    def sends(self) -> list[httpx.Request]:
        return [r for r in self.requests if "login.microsoftonline.com" not in str(r.url)]

    def tokens(self) -> list[httpx.Request]:
        return [r for r in self.requests if "login.microsoftonline.com" in str(r.url)]


def _make_adapter(
    transport: httpx.BaseTransport | None = None, **kwargs: str
) -> TeamsChannelAdapter:
    config: dict[str, str] = {
        "teams_app_id": APP_ID,
        "teams_app_password": "secret",
        "teams_app_type": "SingleTenant",
        "teams_tenant_id": TENANT_ID,
    }
    config.update(kwargs)
    adapter = TeamsChannelAdapter(**config)
    if transport is not None:
        adapter._client = httpx.Client(transport=transport, timeout=5.0)
    return adapter


# ---------------------------------------------------------------------------
# Tenancy — the classic Teams misconfiguration
# ---------------------------------------------------------------------------


class TestTenancy:
    """Wrong authority = inbound works, every reply 401s, error says nothing."""

    def test_single_tenant_uses_its_own_tenant(self) -> None:
        adapter = _make_adapter()
        assert adapter._token_url() == (
            f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
        )

    def test_multi_tenant_uses_the_shared_authority(self) -> None:
        adapter = _make_adapter(teams_app_type="MultiTenant", teams_tenant_id="")
        assert "botframework.com" in adapter._token_url()

    def test_app_type_is_matched_case_insensitively(self) -> None:
        # Azure shows "SingleTenant"; humans type "singletenant".
        adapter = _make_adapter(teams_app_type="singletenant")
        assert TENANT_ID in adapter._token_url()

    def test_single_tenant_without_a_tenant_id_fails_loudly(self) -> None:
        # Otherwise it 401s on every reply forever, saying nothing useful.
        with pytest.raises(ValueError, match="teams_tenant_id"):
            _make_adapter(teams_app_type="SingleTenant", teams_tenant_id="")


# ---------------------------------------------------------------------------
# matches()
# ---------------------------------------------------------------------------


class TestMatches:
    def test_user_proxy_on_teams_matches(self) -> None:
        assert _make_adapter().matches(_make_sent_message(), _binding()) is True

    def test_non_user_proxy_does_not_match(self) -> None:
        msg = _make_sent_message(is_user_proxy=False)
        assert _make_adapter().matches(msg, _binding()) is False

    def test_foreign_channel_does_not_match(self) -> None:
        assert _make_adapter().matches(_make_sent_message(), _binding(channel="signal")) is False

    def test_raising_recipient_returns_false(self) -> None:
        class _Raising:
            @property
            def recipient(self) -> Any:
                raise RuntimeError("boom")

        assert _make_adapter().matches(cast(SentMessage, _Raising()), _binding()) is False


# ---------------------------------------------------------------------------
# deliver()
# ---------------------------------------------------------------------------


class TestDeliver:
    def test_posts_to_the_conversations_activities_endpoint(self) -> None:
        transport = _Transport()
        _make_adapter(transport).deliver(_make_sent_message(), _binding())
        sent = transport.sends()[0]
        assert str(sent.url) == f"{SERVICE_URL}v3/conversations/{CONVERSATION_ID}/activities"

    def test_uses_the_bearer_token(self) -> None:
        transport = _Transport()
        _make_adapter(transport).deliver(_make_sent_message(), _binding())
        assert transport.sends()[0].headers["Authorization"] == "Bearer tok-abc"

    def test_body_is_a_message_activity_attributed_to_its_sender(self) -> None:
        transport = _Transport()
        _make_adapter(transport).deliver(_make_sent_message(), _binding())
        body = json.loads(transport.sends()[0].content)
        assert body["type"] == "message"
        assert "@Expert" in body["text"]
        assert "The answer is 42." in body["text"]

    def test_service_url_comes_from_the_binding_not_a_global(self) -> None:
        # Bot Framework hosts conversations regionally; the value only ever
        # arrives on the inbound activity.
        transport = _Transport()
        apac = "https://smba.trafficmanager.net/apac/"
        _make_adapter(transport).deliver(
            _make_sent_message(), _binding(metadata={"service_url": apac})
        )
        assert str(transport.sends()[0].url).startswith(apac)

    def test_no_service_url_sends_nothing(self) -> None:
        # Refusing beats guessing: there is no default region.
        transport = _Transport()
        _make_adapter(transport).deliver(_make_sent_message(), _binding(metadata={}))
        assert transport.sends() == []

    def test_configured_fallback_is_used_when_the_binding_has_none(self) -> None:
        transport = _Transport()
        adapter = _make_adapter(transport, teams_service_url=SERVICE_URL)
        adapter.deliver(_make_sent_message(), _binding(metadata={}))
        assert len(transport.sends()) == 1

    def test_blank_content_sends_nothing(self) -> None:
        transport = _Transport()
        _make_adapter(transport).deliver(_make_sent_message(content="   "), _binding())
        assert transport.sends() == []

    def test_token_is_cached_across_messages(self) -> None:
        transport = _Transport()
        adapter = _make_adapter(transport)
        adapter.deliver(_make_sent_message(), _binding())
        adapter.deliver(_make_sent_message(), _binding())
        assert len(transport.tokens()) == 1
        assert len(transport.sends()) == 2

    def test_expired_token_is_refetched(self) -> None:
        # expires_in below the refresh margin means "already stale".
        transport = _Transport(expires_in=1)
        adapter = _make_adapter(transport)
        adapter.deliver(_make_sent_message(), _binding())
        adapter.deliver(_make_sent_message(), _binding())
        assert len(transport.tokens()) == 2

    def test_token_failure_drops_the_message_without_raising(self, caplog: Any) -> None:
        transport = _Transport(token_status=401)
        with caplog.at_level("ERROR"):
            _make_adapter(transport).deliver(_make_sent_message(), _binding())
        assert transport.sends() == []
        assert "teams_app_type" in caplog.text

    def test_api_error_is_logged_not_raised(self, caplog: Any) -> None:
        transport = _Transport(send_status=403)
        with caplog.at_level("ERROR"):
            _make_adapter(transport).deliver(_make_sent_message(), _binding())
        assert "Teams API error 403" in caplog.text

    def test_transport_error_is_logged_not_raised(self, caplog: Any) -> None:
        class _Exploding(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError("down")

        # deliver() runs in a Pykka actor thread — a raise here kills it.
        with caplog.at_level("ERROR"):
            _make_adapter(_Exploding()).deliver(_make_sent_message(), _binding())
        assert "Teams token request failed" in caplog.text


# ---------------------------------------------------------------------------
# deliver_notice()
# ---------------------------------------------------------------------------


class TestDeliverNotice:
    def test_posts_the_notice(self) -> None:
        transport = _Transport()
        address = ChannelAddress(
            channel="teams",
            channel_user_id=CONVERSATION_ID,
            metadata={"service_url": SERVICE_URL},
        )
        _make_adapter(transport).deliver_notice(address, "Started a new session")
        assert json.loads(transport.sends()[0].content)["text"] == "Started a new session"

    def test_notice_uses_the_addresss_service_url(self) -> None:
        # The notice path has no binding, which is why metadata lives on the
        # address rather than only on the binding.
        transport = _Transport()
        apac = "https://smba.trafficmanager.net/apac/"
        address = ChannelAddress(
            channel="teams", channel_user_id=CONVERSATION_ID, metadata={"service_url": apac}
        )
        _make_adapter(transport).deliver_notice(address, "ok")
        assert str(transport.sends()[0].url).startswith(apac)

    def test_foreign_channel_is_ignored(self) -> None:
        transport = _Transport()
        address = ChannelAddress(channel="telegram", channel_user_id="987654321")
        _make_adapter(transport).deliver_notice(address, "ok")
        assert transport.requests == []


# ---------------------------------------------------------------------------
# on_stop() and protocol conformance
# ---------------------------------------------------------------------------


class TestOnStop:
    def test_client_and_token_survive_a_team_stop(self) -> None:
        transport = _Transport()
        adapter = _make_adapter(transport)
        adapter.deliver(_make_sent_message(), _binding())
        adapter.on_stop(TEAM_ID)
        adapter.deliver(_make_sent_message(), _binding())
        assert len(transport.sends()) == 2
        assert len(transport.tokens()) == 1


class TestProtocolConformance:
    def test_satisfies_interaction_channel_adapter(self) -> None:
        from akgentic.infra.protocols.channels import InteractionChannelAdapter

        assert isinstance(_make_adapter(), InteractionChannelAdapter)
