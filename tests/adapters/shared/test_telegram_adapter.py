"""Tests for TelegramChannelAdapter."""

from __future__ import annotations

import json
import uuid
from typing import Any, cast

import httpx
from akgentic.core.actor_address_impl import ActorAddressProxy
from akgentic.core.messages.orchestrator import SentMessage

from akgentic.infra.adapters.shared.telegram_adapter import TelegramChannelAdapter
from akgentic.infra.protocols.channels import ChannelAddress, ChannelBinding

TEAM_ID = uuid.uuid4()

# ---------------------------------------------------------------------------
# Helpers (following test_channel_dispatcher.py patterns)
# ---------------------------------------------------------------------------


def _make_addr(
    role: str = "UserProxy",
    name: str = "987654321",
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
    name: str = "987654321",
    content: str = "Hello from the agent!",
    is_user_proxy: bool = True,
) -> SentMessage:
    recipient = _make_addr(role=role, name=name, is_user_proxy=is_user_proxy)
    sender = _make_addr(role="assistant", name="agent-1", is_user_proxy=False)
    from akgentic.core.messages.message import UserMessage

    inner = UserMessage(content=content, sender=sender)
    return SentMessage(message=inner, recipient=recipient, sender=sender)


def _binding(
    channel: str = "telegram",
    channel_user_id: str = "987654321",
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


# ---------------------------------------------------------------------------
# Mock transport for httpx
# ---------------------------------------------------------------------------


class _CaptureTransport(httpx.BaseTransport):
    """Captures requests and returns configurable responses."""

    def __init__(self, status_code: int = 200, body: dict | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._status_code = status_code
        self._body = body or {"ok": True, "result": {}}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            status_code=self._status_code,
            json=self._body,
        )


def _make_adapter(
    transport: httpx.BaseTransport | None = None,
) -> TelegramChannelAdapter:
    """Create adapter with optional mock transport."""
    adapter = TelegramChannelAdapter(bot_token="test-token-123")
    if transport is not None:
        adapter._client = httpx.Client(
            base_url="https://api.telegram.org/bottest-token-123/",
            transport=transport,
        )
    return adapter


# ---------------------------------------------------------------------------
# AC 1: matches() returns True for a user proxy, whatever its role string
# ---------------------------------------------------------------------------


class TestMatchesUserProxy:
    """AC 1: recipient is structurally a user proxy → matches() returns True."""

    def test_user_proxy_matches(self) -> None:
        adapter = _make_adapter()
        msg = _make_sent_message(role="UserProxy", is_user_proxy=True)
        assert adapter.matches(msg, _binding()) is True

    def test_user_proxy_matches_with_unrelated_role(self) -> None:
        adapter = _make_adapter()
        msg = _make_sent_message(role="operator", is_user_proxy=True)
        assert adapter.matches(msg, _binding()) is True

    def test_user_proxy_matches_with_empty_role(self) -> None:
        adapter = _make_adapter()
        msg = _make_sent_message(role="", is_user_proxy=True)
        assert adapter.matches(msg, _binding()) is True


# ---------------------------------------------------------------------------
# AC 2: matches() returns False for a non-user-proxy, even if it is *named*
#       "UserProxy" — the check is structural, not string-based
# ---------------------------------------------------------------------------


class TestMatchesNonUserProxy:
    """A recipient that is not a user proxy does not match — this adapter's rule.

    The dispatcher stopped filtering by recipient, so these are the specs that
    keep a Telegram chat out of the team's internal traffic. A binding naming
    an ordinary member is honoured by the lookup and then carries nothing
    here, which is what makes ``/register <team-id> @Expert`` inert rather
    than a way to watch a team work.
    """

    def test_agent_role_does_not_match(self) -> None:
        adapter = _make_adapter()
        msg = _make_sent_message(role="assistant", is_user_proxy=False)
        assert adapter.matches(msg, _binding()) is False

    def test_tester_role_does_not_match(self) -> None:
        adapter = _make_adapter()
        msg = _make_sent_message(role="tester", is_user_proxy=False)
        assert adapter.matches(msg, _binding()) is False

    def test_user_proxy_role_string_alone_does_not_match(self) -> None:
        adapter = _make_adapter()
        msg = _make_sent_message(role="UserProxy", is_user_proxy=False)
        assert adapter.matches(msg, _binding()) is False


# ---------------------------------------------------------------------------
# AC 3: matches() swallows errors raised while reading the recipient
# ---------------------------------------------------------------------------


class _RaisingRecipient:
    """Stands in for a recipient whose ``is_user_proxy`` access blows up."""

    @property
    def is_user_proxy(self) -> bool:
        raise RuntimeError("recipient exploded")


class _RaisingRecipientMessage:
    """Stands in for a message carrying a recipient that blows up."""

    recipient = _RaisingRecipient()


class _RaisingMessage:
    """Stands in for a message whose ``recipient`` access blows up."""

    @property
    def recipient(self) -> Any:
        raise RuntimeError("message exploded")


class TestMatchesGuard:
    """AC 3: a raising recipient yields False rather than propagating.

    ``matches`` runs in a Pykka actor thread, so an exception escaping here
    would take the dispatch with it.
    """

    def test_raising_recipient_access_returns_false(self) -> None:
        adapter = _make_adapter()
        assert adapter.matches(cast(SentMessage, _RaisingMessage()), _binding()) is False

    def test_raising_is_user_proxy_returns_false(self) -> None:
        adapter = _make_adapter()
        assert adapter.matches(cast(SentMessage, _RaisingRecipientMessage()), _binding()) is False


# ---------------------------------------------------------------------------
# deliver() POSTs to Telegram API
# ---------------------------------------------------------------------------


class TestDeliver:
    """deliver() sends the correct POST to Telegram sendMessage."""

    def test_posts_to_send_message(self) -> None:
        transport = _CaptureTransport()
        adapter = _make_adapter(transport=transport)
        msg = _make_sent_message(name="@HumanProxy_0", content="Test reply")

        adapter.deliver(msg, _binding(channel_user_id="987654321"))

        assert len(transport.requests) == 1
        req = transport.requests[0]
        assert str(req.url).endswith("/sendMessage")
        body = json.loads(req.content)
        assert body["chat_id"] == "987654321"
        assert body["text"] == "You received a message from agent-1: \n\nTest reply"

    def test_the_message_is_attributed_to_its_sender(self) -> None:
        """A chat can be bound to any member, so who is talking is not implicit.

        Before ``/register``, everything a chat received came to its entry
        point, and the sender was always the same agent. Now the chat may hold
        a conversation with a named member, and an unattributed line would read
        as the bot's own words.
        """
        transport = _CaptureTransport()
        adapter = _make_adapter(transport=transport)
        msg = _make_sent_message(name="@HumanProxy_0", content="Here is the joke")

        adapter.deliver(msg, _binding())

        body = json.loads(transport.requests[0].content)
        assert body["text"].startswith("You received a message from agent-1:")
        assert body["text"].endswith("Here is the joke")

    def test_an_agent_with_nothing_to_say_posts_nothing(self) -> None:
        """The attribution must not turn an empty message into a delivered one.

        ``_post`` drops blank text, but a prefix makes every message non-blank,
        so the check runs before the prefix is built. Otherwise a chat receives
        "You received a message from X:" with no message under it.
        """
        transport = _CaptureTransport()
        adapter = _make_adapter(transport=transport)

        adapter.deliver(_make_sent_message(name="@HumanProxy_0", content="   "), _binding())
        adapter.deliver(_make_sent_message(name="@HumanProxy_0", content=""), _binding())

        assert transport.requests == []

    def test_binding_wins_over_recipient_name(self) -> None:
        """The chat id comes from the binding, never from the recipient's name.

        ``ActorAddress.name`` is ``actor.config.name`` off the TeamCard —
        ``human_support``, ``@HumanProxy_0``. Posting it takes a Telegram 400.
        The two values differ here so the assertion means something.
        """
        transport = _CaptureTransport()
        adapter = _make_adapter(transport=transport)
        msg = _make_sent_message(name="human_support", content="Test reply")

        adapter.deliver(msg, _binding(channel_user_id="123456789"))

        body = json.loads(transport.requests[0].content)
        assert body["chat_id"] == "123456789"


# ---------------------------------------------------------------------------
# deliver() handles errors without raising
# ---------------------------------------------------------------------------


class TestDeliverError:
    """Telegram API error → logged, no exception raised."""

    def test_api_error_does_not_raise(self) -> None:
        transport = _CaptureTransport(
            status_code=400,
            body={"ok": False, "description": "Bad Request: chat not found"},
        )
        adapter = _make_adapter(transport=transport)
        msg = _make_sent_message()

        # Should not raise
        adapter.deliver(msg, _binding())


# ---------------------------------------------------------------------------
# matches() rejects a binding belonging to another channel
# ---------------------------------------------------------------------------


class TestMatchesForeignChannel:
    """A binding for another channel is not this adapter's to deliver.

    With two channels configured, a recipient-only check accepts a Slack
    binding and POSTs a Slack user id to Telegram — the same wrong-POST defect
    one channel over.
    """

    def test_slack_binding_does_not_match(self) -> None:
        adapter = _make_adapter()
        msg = _make_sent_message(is_user_proxy=True)

        assert adapter.matches(msg, _binding(channel="slack")) is False

    def test_telegram_binding_still_matches(self) -> None:
        adapter = _make_adapter()
        msg = _make_sent_message(is_user_proxy=True)

        assert adapter.matches(msg, _binding(channel="telegram")) is True


# ---------------------------------------------------------------------------
# on_stop() releases nothing process-scoped
# ---------------------------------------------------------------------------


class TestOnStop:
    """on_stop() must NOT close the httpx client.

    The inverse of the spec that used to live here. The client is
    process-scoped and shared by every team, so closing it on one team's stop
    mutes all the others — silently, since ``deliver()`` logs and swallows its
    own errors.
    """

    def test_on_stop_leaves_the_client_open(self) -> None:
        adapter = _make_adapter()

        adapter.on_stop(uuid.uuid4())

        assert adapter._client.is_closed is False

    def test_transport_survives_another_teams_stop(self) -> None:
        """Team A stops; a delivery for team B still reaches the transport."""
        transport = _CaptureTransport()
        adapter = _make_adapter(transport=transport)
        team_a = uuid.uuid4()
        team_b = uuid.uuid4()

        adapter.on_stop(team_a)
        adapter.deliver(
            _make_sent_message(content="still reachable"),
            _binding(channel_user_id="555", team_id=team_b),
        )

        assert len(transport.requests) == 1
        body = json.loads(transport.requests[0].content)
        assert body["chat_id"] == "555"


# ---------------------------------------------------------------------------
# deliver_notice() answers one chat, named by an address
# ---------------------------------------------------------------------------


class TestDeliverNotice:
    """A channel-layer acknowledgement reaches the chat the address names."""

    def test_notice_addressed_by_a_bare_address_posts_to_that_chat(self) -> None:
        """The unbound paths have no team, and must still be able to answer."""
        transport = _CaptureTransport()
        adapter = _make_adapter(transport=transport)

        adapter.deliver_notice(
            ChannelAddress(channel="telegram", channel_user_id="12345"),
            "no active session",
        )

        assert len(transport.requests) == 1
        body = json.loads(transport.requests[0].content)
        assert body["chat_id"] == "12345"
        assert body["text"] == "no active session"

    def test_notice_addressed_by_a_binding_posts_to_the_same_chat(self) -> None:
        """A binding *is* an address — every bound caller passes its own record."""
        transport = _CaptureTransport()
        adapter = _make_adapter(transport=transport)

        adapter.deliver_notice(_binding(channel_user_id="777"), "released")

        assert len(transport.requests) == 1
        body = json.loads(transport.requests[0].content)
        assert body["chat_id"] == "777"

    def test_notice_for_another_channel_posts_nothing(self) -> None:
        """G7: notices fan out to every adapter, so each one filters by channel.

        Without the comparison this adapter posts a Slack chat id to Telegram —
        the wrong-POST defect one channel over.
        """
        transport = _CaptureTransport()
        adapter = _make_adapter(transport=transport)

        adapter.deliver_notice(
            ChannelAddress(channel="slack", channel_user_id="U123"),
            "released",
        )

        assert transport.requests == []

    def test_a_closed_client_does_not_raise_out_of_a_notice(self) -> None:
        """A closed client raises RuntimeError, which is not an httpx.HTTPError.

        On this path the command has already taken effect, so an escaping error
        would turn a success into a 500 and a channel retry loop.
        """
        adapter = _make_adapter(transport=_CaptureTransport())
        adapter._client.close()

        adapter.deliver_notice(
            ChannelAddress(channel="telegram", channel_user_id="12345"),
            "released",
        )

    def test_a_closed_client_does_not_raise_out_of_deliver_either(self) -> None:
        """The same guard protects the actor thread ``deliver`` runs on."""
        adapter = _make_adapter(transport=_CaptureTransport())
        adapter._client.close()

        adapter.deliver(_make_sent_message(content="hi"), _binding())
