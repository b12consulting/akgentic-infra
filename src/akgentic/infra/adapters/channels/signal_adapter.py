"""SignalChannelAdapter — delivers outbound messages via signal-cli-rest-api."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from akgentic.core.messages import SentMessage
    from akgentic.infra.protocols.channels import ChannelAddress, ChannelBinding

logger = logging.getLogger(__name__)

# Mirrors ``SignalChannelParser.channel_name``. A literal here rather than a
# Protocol member: an adapter does not advertise its channel the way a parser
# does, so the comparison belongs to the one implementation that knows the
# answer (ADR-043 §D6).
SIGNAL_CHANNEL = "signal"

# signal-cli-rest-api's send endpoint. v2 is the one that takes a recipient
# list and returns the sent message's timestamp.
_SEND_PATH = "/v2/send"

# Shown in place of the sender's name when a message carries no sender. Every
# message a team emits has one, so this stands for a malformed event rather
# than an expected case — it names the gap instead of posting a bare colon.
_UNKNOWN_SENDER = "an unknown sender"


class SignalChannelAdapter:
    """Delivers outbound agent messages to Signal chats via signal-cli-rest-api.

    Satisfies the ``InteractionChannelAdapter`` protocol via structural
    subtyping.

    Transport:
        Signal has no Bot API. Messages go out through a ``signal-cli`` daemon
        fronted by ``signal-cli-rest-api``, reached over plain HTTP on the
        deployment's own network. ``signal_number`` is the *bot's* registered
        account — the ``number`` field of every send, and not something the
        binding can supply, since a binding names the recipient.

    Addressing:
        ``binding.channel_user_id`` is passed through as the sole recipient,
        untouched. The parser has already encoded a group as
        ``group.<groupId>``, which is signal-cli-rest-api's own recipient
        syntax, so a group and a 1:1 chat travel the same path and this adapter
        needs no branch — and cannot get the branch wrong.

    ``matches()`` returns ``True`` when the ``SentMessage`` recipient is
    structurally a ``UserProxy`` actor — or a subclass such as ``HumanProxy`` —
    **and** the binding names this adapter's own channel. Without the second
    half, a deployment configuring Signal alongside Telegram would have this
    adapter post a Telegram chat id to signal-cli. Without the first, a chat
    bound to an ordinary member would receive the team's internal traffic: the
    dispatcher stopped filtering by recipient, so this adapter is where that
    rule lives.

    This adapter is process-scoped: one instance serves every team, and its
    httpx client is never closed on a single team's stop.

    Args:
        signal_api_url: Base URL of the signal-cli-rest-api instance.
        signal_number: Default bot account to send as, in E.164. Used when the
            conversation's address carries no ``account`` — which is every
            conversation in a single-account deployment, and none in a
            multi-account one once the pump has seen them.
    """

    def __init__(
        self,
        signal_api_url: str = "http://localhost:8080",
        signal_number: str = "",
        **_kwargs: str,
    ) -> None:
        self._number = signal_number
        self._client = httpx.Client(base_url=signal_api_url.rstrip("/"), timeout=10.0)

    def matches(self, msg: SentMessage, binding: ChannelBinding) -> bool:
        """Check if this adapter should deliver the message.

        Returns True when the recipient actor is a ``UserProxy``, or a subclass
        such as ``HumanProxy``, indicating the message is headed to a human
        participant — and the binding belongs to Signal. The recipient check is
        structural rather than a comparison against the recipient's ``role``
        string, so a team is free to name its human-in-the-loop member anything.

        Args:
            msg: The outbound message to check.
            binding: The recipient agent's channel binding.

        Returns:
            True if the recipient is a UserProxy agent bound to Signal.
        """
        if binding.channel != SIGNAL_CHANNEL:
            return False
        try:
            return msg.recipient.is_user_proxy
        except Exception:  # noqa: BLE001
            return False

    def _sending_account(self, address: ChannelAddress) -> str:
        """The bot account a reply to this conversation must leave from.

        One signal-cli daemon may hold several registered accounts, and the pump
        polls all of them, so "which number are we" is a per-conversation fact
        rather than a global one. The parser puts the receiving account in the
        address metadata; ``signal_number`` is the fallback, and is the whole
        answer for a single-account deployment.

        Taking the address rather than the binding is what makes notices work:
        ``deliver_notice`` has no binding, and an acknowledgement sent from the
        wrong bot number is as wrong as an agent message sent from one.

        Args:
            address: The conversation. A ``ChannelBinding`` satisfies this.

        Returns:
            The account to send as, possibly empty when neither is configured —
            signal-cli rejects that, which is the correct loud failure.
        """
        account = address.metadata.get("account")
        if isinstance(account, str) and account:
            return account
        return self._number

    def _post(self, account: str, recipient: str, text: str) -> None:
        """POST one send call, logging every failure rather than raising.

        The single place this adapter talks to signal-cli, so a message and a
        notice cannot drift apart in how they are sent or how failures are
        handled.

        Success is ``response.is_success`` and not ``== 200``:
        signal-cli-rest-api answers a send with **201 Created** carrying the
        sent timestamp, so an equality check would log an error on every
        message that actually went out.

        ``RuntimeError`` is caught alongside ``httpx.HTTPError`` because a
        closed or otherwise unusable client raises ``RuntimeError("Cannot send
        a request, as the client has been closed.")``, which is **not** an
        ``httpx.HTTPError``. On the ``deliver`` path that escape crashes an
        actor thread; on the ``deliver_notice`` path it turns a command that
        already took effect into a 500 and a retry loop.

        A blank body is dropped here rather than sent, and this is the one place
        both callers pass through, so neither can reintroduce it.

        Args:
            account: The bot account to send as — the ``number`` field.
            recipient: The Signal chat to post to — a number, a UUID, or
                ``group.<groupId>``.
            text: The message body. Blank means nothing to say, so nothing is
                sent.
        """
        if not text.strip():
            logger.debug("Nothing to deliver to Signal chat %s — blank body", recipient)
            return
        try:
            response = self._client.post(
                _SEND_PATH,
                json={"message": text, "number": account, "recipients": [recipient]},
            )
            if not response.is_success:
                logger.error(
                    "Signal API error %d: %s",
                    response.status_code,
                    response.text,
                )
        except (httpx.HTTPError, RuntimeError):
            logger.exception("Failed to post to Signal chat %s", recipient)

    def deliver(self, msg: SentMessage, binding: ChannelBinding) -> None:
        """Deliver an outbound message to a Signal chat.

        The recipient comes from ``binding.channel_user_id``; the recipient
        actor's ``name`` is the TeamCard's agent name and means nothing to
        signal-cli.

        Logs transport errors and a closed client without raising — delivery
        failures must not crash the caller, which here is a Pykka actor thread.

        Args:
            msg: The message to deliver.
            binding: The recipient agent's channel binding, naming the chat.
        """
        recipient = binding.channel_user_id
        # No ``or str(msg.message)`` fallback: an empty ``content`` is falsy, so
        # that idiom quietly posts the message model's repr into a human's chat.
        text = getattr(msg.message, "content", "") or ""
        # An agent with nothing to say produces nothing. ``_post`` drops blank
        # text, but the attribution below would make every message non-blank,
        # so the check has to happen before it is prepended.
        if not text.strip():
            logger.debug("Agent produced no text; nothing delivered to chat %s", recipient)
            return
        sender_name = msg.sender.name if msg.sender else _UNKNOWN_SENDER
        logger.debug("Delivering message to Signal chat %s", recipient)
        self._post(
            self._sending_account(binding),
            recipient,
            f"You received a message from {sender_name}: \n\n{text}",
        )

    def deliver_notice(self, address: ChannelAddress, text: str) -> None:
        """Deliver a channel-layer acknowledgement to a Signal chat.

        Called from the FastAPI route, not an actor thread, and needing no
        ``matches()``: the address names the destination outright. Notices are
        fanned out to every configured adapter, so the channel comparison is
        what stops a Telegram chat id being posted here — the same check
        ``matches()`` performs on a binding.

        Args:
            address: The chat to answer. A ``ChannelBinding`` satisfies this.
            text: The acknowledgement text.
        """
        if address.channel != SIGNAL_CHANNEL:
            return
        logger.debug("Delivering notice to Signal chat %s", address.channel_user_id)
        self._post(self._sending_account(address), address.channel_user_id, text)

    def on_stop(self, team_id: uuid.UUID) -> None:
        """Note that a team stopped; release nothing.

        This adapter holds no per-team state. Its httpx client is
        **process-scoped** and shared by every team, so closing it here would
        mute every other conversation — silently, because ``deliver()`` logs
        and swallows its errors.

        Args:
            team_id: The team being stopped.
        """
        logger.debug("SignalAdapter stopped: team_id=%s", team_id)
