"""Tests for SignalChannelAdapter."""

from __future__ import annotations

import json
import uuid
from typing import Any, cast

import httpx
import pytest
from akgentic.core.actor_address_impl import ActorAddressProxy
from akgentic.core.messages.message import UserMessage
from akgentic.core.messages.orchestrator import SentMessage

from akgentic.infra.adapters.channels.signal_adapter import SignalChannelAdapter
from akgentic.infra.protocols.channels import ChannelAddress, ChannelBinding

TEAM_ID = uuid.uuid4()
BOT_NUMBER = "+32471111111"
USER_NUMBER = "+32470000000"
GROUP_RECIPIENT = "group.Ci0KIF9hYmNkZWZnaGlqa2xtbm9w"
API_URL = "http://signal-api:8080"

# ---------------------------------------------------------------------------
# Helpers (following test_telegram_adapter.py patterns)
# ---------------------------------------------------------------------------


def _make_addr(
    role: str = "UserProxy",
    name: str = "@HumanProxy_0",
    is_user_proxy: bool = True,
) -> ActorAddressProxy:
    # `role` and `is_user_proxy` are set independently so tests can pair any
    # role string with either structural outcome.
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
    role: str = "UserProxy",
    content: str = "Hello from the agent!",
    is_user_proxy: bool = True,
    sender_name: str = "@Manager",
) -> SentMessage:
    recipient = _make_addr(role=role, is_user_proxy=is_user_proxy)
    sender = _make_addr(role="assistant", name=sender_name, is_user_proxy=False)
    inner = UserMessage(content=content, sender=sender)
    return SentMessage(message=inner, recipient=recipient, sender=sender)


def _binding(
    channel: str = "signal",
    channel_user_id: str = USER_NUMBER,
    team_id: uuid.UUID | None = None,
    agent_name: str = "@HumanProxy_0",
) -> ChannelBinding:
    """A binding naming the destination chat — the value the address cannot carry."""
    return ChannelBinding(
        channel=channel,
        channel_user_id=channel_user_id,
        team_id=team_id or TEAM_ID,
        agent_name=agent_name,
    )


class _CaptureTransport(httpx.BaseTransport):
    """Captures requests and returns configurable responses."""

    def __init__(self, status_code: int = 201, body: dict | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._status_code = status_code
        self._body = body if body is not None else {"timestamp": "1711800000000"}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(status_code=self._status_code, json=self._body)

    def sent(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


class _ExplodingTransport(httpx.BaseTransport):
    """Every send fails at the transport layer."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("signal-cli is down")


def _make_adapter(
    transport: httpx.BaseTransport | None = None,
) -> SignalChannelAdapter:
    """Create the adapter with an optional mock transport."""
    adapter = SignalChannelAdapter(signal_api_url=API_URL, signal_number=BOT_NUMBER)
    if transport is not None:
        adapter._client = httpx.Client(base_url=API_URL, transport=transport)
    return adapter


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """The adapter takes its transport coordinates from channel config."""

    def test_trailing_slash_is_trimmed(self) -> None:
        # httpx joins "/v2/send" against the base URL; a trailing slash here
        # plus the leading one on the path would yield "//v2/send".
        adapter = SignalChannelAdapter(signal_api_url="http://signal-api:8080/")
        assert str(adapter._client.base_url) == "http://signal-api:8080"

    def test_extra_config_kwargs_are_tolerated(self) -> None:
        # The registry passes one config dict to parser, router AND adapter.
        adapter = SignalChannelAdapter(
            signal_api_url=API_URL,
            signal_number=BOT_NUMBER,
            default_catalog_entry="my-team",
        )
        assert adapter._number == BOT_NUMBER


# ---------------------------------------------------------------------------
# matches()
# ---------------------------------------------------------------------------


class TestMatchesUserProxy:
    """A recipient that is structurally a user proxy matches."""

    def test_user_proxy_matches(self) -> None:
        assert _make_adapter().matches(_make_sent_message(is_user_proxy=True), _binding()) is True

    def test_user_proxy_matches_with_unrelated_role(self) -> None:
        msg = _make_sent_message(role="operator", is_user_proxy=True)
        assert _make_adapter().matches(msg, _binding()) is True


class TestMatchesNonUserProxy:
    """A recipient that is not a user proxy does not match — this adapter's rule.

    The dispatcher stopped filtering by recipient, so these are the specs that
    keep a Signal chat out of the team's internal traffic.
    """

    def test_agent_role_does_not_match(self) -> None:
        msg = _make_sent_message(role="assistant", is_user_proxy=False)
        assert _make_adapter().matches(msg, _binding()) is False

    def test_user_proxy_role_string_alone_does_not_match(self) -> None:
        msg = _make_sent_message(role="UserProxy", is_user_proxy=False)
        assert _make_adapter().matches(msg, _binding()) is False


class TestMatchesChannel:
    """The channel comparison is what stops a Telegram chat id reaching signal-cli."""

    def test_foreign_binding_does_not_match(self) -> None:
        msg = _make_sent_message(is_user_proxy=True)
        foreign = _binding(channel="telegram", channel_user_id="987654321")
        assert _make_adapter().matches(msg, foreign) is False

    def test_foreign_binding_loses_even_for_a_user_proxy(self) -> None:
        # Both halves are needed: this one satisfies the recipient check and
        # must still be refused.
        transport = _CaptureTransport()
        adapter = _make_adapter(transport)
        assert adapter.matches(_make_sent_message(), _binding(channel="slack")) is False
        assert transport.requests == []


class _RaisingMessage:
    """Stands in for a message whose ``recipient`` access blows up."""

    @property
    def recipient(self) -> Any:
        raise RuntimeError("message exploded")


class TestMatchesGuard:
    """A raising recipient yields False rather than propagating.

    ``matches`` runs in a Pykka actor thread, so an exception escaping here
    would take the dispatch with it.
    """

    def test_raising_recipient_access_returns_false(self) -> None:
        adapter = _make_adapter()
        assert adapter.matches(cast(SentMessage, _RaisingMessage()), _binding()) is False


# ---------------------------------------------------------------------------
# deliver()
# ---------------------------------------------------------------------------


class TestDeliver:
    """deliver() posts one v2/send naming the bound chat."""

    def test_posts_to_v2_send(self) -> None:
        transport = _CaptureTransport()
        _make_adapter(transport).deliver(_make_sent_message(), _binding())
        assert len(transport.requests) == 1
        assert transport.requests[0].url.path == "/v2/send"

    def test_body_carries_bot_number_and_bound_recipient(self) -> None:
        transport = _CaptureTransport()
        _make_adapter(transport).deliver(_make_sent_message(), _binding())
        body = transport.sent()[0]
        assert body["number"] == BOT_NUMBER
        assert body["recipients"] == [USER_NUMBER]

    def test_message_is_attributed_to_its_sender(self) -> None:
        transport = _CaptureTransport()
        msg = _make_sent_message(content="The answer is 42.", sender_name="@Expert")
        _make_adapter(transport).deliver(msg, _binding())
        body = transport.sent()[0]
        assert "@Expert" in body["message"]
        assert "The answer is 42." in body["message"]

    def test_group_recipient_passes_through_untouched(self) -> None:
        # The parser already encoded the group in signal-cli-rest-api's own
        # recipient syntax, so the adapter needs no branch — and cannot get it
        # wrong.
        transport = _CaptureTransport()
        binding = _binding(channel_user_id=GROUP_RECIPIENT)
        _make_adapter(transport).deliver(_make_sent_message(), binding)
        assert transport.sent()[0]["recipients"] == [GROUP_RECIPIENT]

    def test_blank_content_sends_nothing(self) -> None:
        # Attribution would make every message non-blank, so the check has to
        # happen before the prefix is prepended.
        transport = _CaptureTransport()
        _make_adapter(transport).deliver(_make_sent_message(content="   "), _binding())
        assert transport.requests == []

    def test_missing_content_attribute_sends_nothing(self) -> None:
        transport = _CaptureTransport()
        adapter = _make_adapter(transport)
        msg = _make_sent_message()
        object.__setattr__(msg, "message", object())
        adapter.deliver(msg, _binding())
        assert transport.requests == []

    def test_created_is_not_logged_as_an_error(self, caplog: Any) -> None:
        # signal-cli-rest-api answers a send with 201, so an ``== 200`` check
        # would log an error on every message that actually went out.
        transport = _CaptureTransport(status_code=201)
        with caplog.at_level("ERROR"):
            _make_adapter(transport).deliver(_make_sent_message(), _binding())
        assert "Signal API error" not in caplog.text

    def test_api_error_is_logged_not_raised(self, caplog: Any) -> None:
        transport = _CaptureTransport(status_code=400, body={"error": "Unregistered user"})
        with caplog.at_level("ERROR"):
            _make_adapter(transport).deliver(_make_sent_message(), _binding())
        assert "Signal API error 400" in caplog.text

    def test_transport_error_is_logged_not_raised(self, caplog: Any) -> None:
        # deliver() runs in a Pykka actor thread — a raise here kills it.
        with caplog.at_level("ERROR"):
            _make_adapter(_ExplodingTransport()).deliver(_make_sent_message(), _binding())
        assert "Failed to post to Signal chat" in caplog.text

    def test_closed_client_is_logged_not_raised(self, caplog: Any) -> None:
        # A closed client raises RuntimeError, which is NOT an httpx.HTTPError.
        adapter = _make_adapter(_CaptureTransport())
        adapter._client.close()
        with caplog.at_level("ERROR"):
            adapter.deliver(_make_sent_message(), _binding())
        assert "Failed to post to Signal chat" in caplog.text


# ---------------------------------------------------------------------------
# deliver_notice()
# ---------------------------------------------------------------------------


class TestDeliverNotice:
    """Notices are fanned out to every adapter, so the channel check is load-bearing."""

    def test_posts_the_notice_verbatim(self) -> None:
        transport = _CaptureTransport()
        address = ChannelAddress(channel="signal", channel_user_id=USER_NUMBER)
        _make_adapter(transport).deliver_notice(address, "Started a new session")
        body = transport.sent()[0]
        assert body["message"] == "Started a new session"
        assert body["recipients"] == [USER_NUMBER]

    def test_a_binding_satisfies_the_address(self) -> None:
        transport = _CaptureTransport()
        _make_adapter(transport).deliver_notice(_binding(), "ok")
        assert len(transport.requests) == 1

    def test_foreign_channel_is_ignored(self) -> None:
        transport = _CaptureTransport()
        address = ChannelAddress(channel="telegram", channel_user_id="987654321")
        _make_adapter(transport).deliver_notice(address, "ok")
        assert transport.requests == []

    def test_blank_notice_sends_nothing(self) -> None:
        transport = _CaptureTransport()
        address = ChannelAddress(channel="signal", channel_user_id=USER_NUMBER)
        _make_adapter(transport).deliver_notice(address, "  \n ")
        assert transport.requests == []

    def test_transport_error_is_logged_not_raised(self, caplog: Any) -> None:
        # The command has already taken effect; a raise turns it into a 500 the
        # pump retries.
        address = ChannelAddress(channel="signal", channel_user_id=USER_NUMBER)
        with caplog.at_level("ERROR"):
            _make_adapter(_ExplodingTransport()).deliver_notice(address, "ok")
        assert "Failed to post to Signal chat" in caplog.text


# ---------------------------------------------------------------------------
# on_stop()
# ---------------------------------------------------------------------------


class TestOnStop:
    """The httpx client is process-scoped and must survive one team's stop."""

    def test_client_stays_usable_after_a_team_stops(self) -> None:
        transport = _CaptureTransport()
        adapter = _make_adapter(transport)
        adapter.on_stop(TEAM_ID)
        adapter.deliver(_make_sent_message(), _binding())
        assert len(transport.requests) == 1


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """The registry isinstance-checks the adapter at load; assert it here too."""

    def test_satisfies_interaction_channel_adapter(self) -> None:
        from akgentic.infra.protocols.channels import InteractionChannelAdapter

        assert isinstance(_make_adapter(), InteractionChannelAdapter)


# ---------------------------------------------------------------------------
# Which bot account the reply leaves from
# ---------------------------------------------------------------------------

OTHER_BOT_NUMBER = "+32460000000"


class TestSendingAccount:
    """A reply must leave from the account the message arrived on."""

    def test_binding_metadata_wins_over_the_configured_default(self) -> None:
        transport = _CaptureTransport()
        binding = _binding()
        binding.metadata = {"account": OTHER_BOT_NUMBER}
        _make_adapter(transport).deliver(_make_sent_message(), binding)
        assert transport.sent()[0]["number"] == OTHER_BOT_NUMBER

    def test_configured_default_used_when_address_carries_none(self) -> None:
        transport = _CaptureTransport()
        _make_adapter(transport).deliver(_make_sent_message(), _binding())
        assert transport.sent()[0]["number"] == BOT_NUMBER

    def test_notices_honour_it_too(self) -> None:
        # The notice path has no binding. An acknowledgement from the wrong bot
        # number is as wrong as an agent message from one.
        transport = _CaptureTransport()
        address = ChannelAddress(
            channel="signal",
            channel_user_id=USER_NUMBER,
            metadata={"account": OTHER_BOT_NUMBER},
        )
        _make_adapter(transport).deliver_notice(address, "Started a new session")
        assert transport.sent()[0]["number"] == OTHER_BOT_NUMBER

    @pytest.mark.parametrize("value", [42, "", None])
    def test_non_string_account_falls_back(self, value: object) -> None:
        transport = _CaptureTransport()
        binding = _binding()
        binding.metadata = {"account": value}
        _make_adapter(transport).deliver(_make_sent_message(), binding)
        assert transport.sent()[0]["number"] == BOT_NUMBER
