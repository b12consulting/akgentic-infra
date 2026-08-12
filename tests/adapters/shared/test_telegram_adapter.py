"""Tests for TelegramChannelAdapter."""

from __future__ import annotations

import json
import uuid
from typing import Any, cast

import httpx
from akgentic.core.actor_address_impl import ActorAddressProxy
from akgentic.core.messages.orchestrator import SentMessage

from akgentic.infra.adapters.shared.telegram_adapter import TelegramChannelAdapter

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
        assert adapter.matches(msg) is True

    def test_user_proxy_matches_with_unrelated_role(self) -> None:
        adapter = _make_adapter()
        msg = _make_sent_message(role="operator", is_user_proxy=True)
        assert adapter.matches(msg) is True

    def test_user_proxy_matches_with_empty_role(self) -> None:
        adapter = _make_adapter()
        msg = _make_sent_message(role="", is_user_proxy=True)
        assert adapter.matches(msg) is True


# ---------------------------------------------------------------------------
# AC 2: matches() returns False for a non-user-proxy, even if it is *named*
#       "UserProxy" — the check is structural, not string-based
# ---------------------------------------------------------------------------


class TestMatchesNonUserProxy:
    """AC 2: recipient is not a user proxy → matches() returns False."""

    def test_agent_role_does_not_match(self) -> None:
        adapter = _make_adapter()
        msg = _make_sent_message(role="assistant", is_user_proxy=False)
        assert adapter.matches(msg) is False

    def test_tester_role_does_not_match(self) -> None:
        adapter = _make_adapter()
        msg = _make_sent_message(role="tester", is_user_proxy=False)
        assert adapter.matches(msg) is False

    def test_user_proxy_role_string_alone_does_not_match(self) -> None:
        adapter = _make_adapter()
        msg = _make_sent_message(role="UserProxy", is_user_proxy=False)
        assert adapter.matches(msg) is False


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
    """AC 3: a raising recipient yields False rather than propagating."""

    def test_raising_recipient_access_returns_false(self) -> None:
        adapter = _make_adapter()
        assert adapter.matches(cast(SentMessage, _RaisingMessage())) is False

    def test_raising_is_user_proxy_returns_false(self) -> None:
        adapter = _make_adapter()
        assert adapter.matches(cast(SentMessage, _RaisingRecipientMessage())) is False


# ---------------------------------------------------------------------------
# deliver() POSTs to Telegram API
# ---------------------------------------------------------------------------


class TestDeliver:
    """deliver() sends the correct POST to Telegram sendMessage."""

    def test_posts_to_send_message(self) -> None:
        transport = _CaptureTransport()
        adapter = _make_adapter(transport=transport)
        msg = _make_sent_message(name="987654321", content="Test reply")

        adapter.deliver(msg)

        assert len(transport.requests) == 1
        req = transport.requests[0]
        assert str(req.url).endswith("/sendMessage")
        body = json.loads(req.content)
        assert body["chat_id"] == "987654321"
        assert body["text"] == "Test reply"


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
        adapter.deliver(msg)


# ---------------------------------------------------------------------------
# on_stop() cleanup
# ---------------------------------------------------------------------------


class TestOnStop:
    """on_stop() closes the httpx client without error."""

    def test_on_stop_closes_client(self) -> None:
        adapter = _make_adapter()
        adapter.on_stop(uuid.uuid4())  # Should not raise
