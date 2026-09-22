"""TelegramChannelAdapter — delivers outbound messages via the Telegram Bot API."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from akgentic.core.messages import SentMessage
    from akgentic.infra.protocols.channels import ChannelAddress, ChannelBinding

logger = logging.getLogger(__name__)

# Mirrors ``TelegramChannelParser.channel_name``. A literal here rather than a
# Protocol member: an adapter does not advertise its channel the way a parser
# does, so the comparison belongs to the one implementation that knows the
# answer (ADR-043 §D6).
TELEGRAM_CHANNEL = "telegram"

# Shown in place of the sender's name when a message carries no sender. Every
# message a team emits has one, so this stands for a malformed event rather
# than an expected case — it names the gap instead of posting a bare colon.
_UNKNOWN_SENDER = "an unknown sender"


class TelegramChannelAdapter:
    """Delivers outbound agent messages to Telegram chats via the Bot API.

    Satisfies the ``InteractionChannelAdapter`` protocol via structural
    subtyping.

    ``matches()`` returns ``True`` when the ``SentMessage`` recipient is
    structurally a ``UserProxy`` actor — or a subclass such as
    ``HumanProxy`` — **and** the binding names this adapter's own channel.
    Without the second half, a deployment configuring two channels would have
    this adapter accept a Slack binding and post a Slack user id to Telegram.
    Without the first, a chat bound to an ordinary member would receive the
    team's internal traffic: the dispatcher stopped filtering by recipient, so
    this adapter is where that rule lives.

    ``deliver()`` attributes the message to its sender — a chat can be bound
    to any member, so "who is talking" is no longer implicit — and sends a
    synchronous POST to the Telegram ``sendMessage`` endpoint, addressing
    ``binding.channel_user_id`` — the chat the inbound
    message arrived from, and the only place that value exists on the outbound
    path. ``deliver_notice()`` posts a channel-layer acknowledgement to an
    address, which a binding also satisfies. Both go through one ``_post``, so
    they cannot drift apart in how they send or how they handle failure.

    This adapter is process-scoped: one instance serves every team, and its
    httpx client is never closed on a single team's stop.

    Args:
        bot_token: Telegram Bot API token (from @BotFather).
        default_catalog_entry: Ignored by the adapter (used by parser only,
            but passed via shared ``ChannelConfig.config``).
    """

    def __init__(self, bot_token: str = "", **_kwargs: str) -> None:
        self._bot_token = bot_token
        self._client = httpx.Client(
            base_url=f"https://api.telegram.org/bot{bot_token}/",
            timeout=10.0,
        )

    def matches(self, msg: SentMessage, binding: ChannelBinding) -> bool:
        """Check if this adapter should deliver the message.

        Returns True when the recipient actor is a ``UserProxy``, or a
        subclass such as ``HumanProxy``, indicating the message is headed to a
        human participant — and the binding belongs to Telegram. The recipient
        check is structural rather than a comparison against the recipient's
        ``role`` string, so a team is free to name its human-in-the-loop
        member anything.

        **This adapter is now the only thing enforcing that rule.** The
        dispatcher no longer drops a message whose recipient is not a user
        proxy: it finds the binding and offers the message to every adapter,
        leaving each to say what it will carry. A chat is a place for humans,
        so Telegram carries a message to a human's seat and not the team's
        internal traffic — even where a binding names an ordinary member,
        which ``/register`` permits. Another channel may decide otherwise
        without changing the dispatcher.

        The channel comparison is not optional either: messages are offered to
        every configured adapter, so without it a deployment running two
        channels would post a Slack chat id to Telegram.

        Args:
            msg: The outbound message to check.
            binding: The recipient agent's channel binding.

        Returns:
            True if the recipient is a UserProxy agent bound to Telegram.
        """
        if binding.channel != TELEGRAM_CHANNEL:
            return False
        try:
            return msg.recipient.is_user_proxy
        except Exception:  # noqa: BLE001
            return False

    def _post(self, chat_id: str, text: str) -> None:
        """POST one ``sendMessage`` call, logging every failure rather than raising.

        The single place this adapter talks to Telegram, so a message and a
        notice cannot drift apart in how they are sent or how failures are
        handled.

        ``RuntimeError`` is caught alongside ``httpx.HTTPError`` because a closed
        or otherwise unusable client raises ``RuntimeError("Cannot send a
        request, as the client has been closed.")``, which is **not** an
        ``httpx.HTTPError``. On the ``deliver`` path that escape crashes an actor
        thread; on the ``deliver_notice`` path it turns a command that already
        took effect into a 500 and a channel retry loop.

        A blank body is dropped here rather than sent. Telegram rejects an
        empty ``text`` outright, so posting one buys a 400 and a log line in
        place of a message nobody could have read — and this is the one place
        both callers pass through, so neither can reintroduce it.

        Args:
            chat_id: The Telegram chat to post to.
            text: The message body. Blank means nothing to say, so nothing is sent.
        """
        if not text.strip():
            logger.debug("Nothing to deliver to Telegram chat %s — blank body", chat_id)
            return
        try:
            response = self._client.post(
                "sendMessage",
                json={"chat_id": chat_id, "text": text},
            )
            if response.status_code != 200:
                logger.error(
                    "Telegram API error %d: %s",
                    response.status_code,
                    response.text,
                )
        except (httpx.HTTPError, RuntimeError):
            logger.exception("Failed to post to Telegram chat %s", chat_id)

    def deliver(self, msg: SentMessage, binding: ChannelBinding) -> None:
        """Deliver an outbound message to a Telegram chat.

        Posts to the Telegram ``sendMessage`` API. The ``chat_id`` comes from
        ``binding.channel_user_id``; the recipient's ``name`` is the TeamCard's
        agent name and posting it yields a Telegram 400.

        Logs transport errors and a closed client without raising — delivery
        failures must not crash the caller, which here is a Pykka actor thread.

        Args:
            msg: The message to deliver.
            binding: The recipient agent's channel binding, naming the chat.
        """
        chat_id = binding.channel_user_id
        # No ``or str(msg.message)`` fallback: an empty ``content`` is falsy, so
        # that idiom quietly posted the message model's repr into a human's chat.
        text = getattr(msg.message, "content", "") or ""
        # An agent with nothing to say produces nothing. ``_post`` drops blank
        # text, but the attribution below would make every message non-blank,
        # so the check has to happen before it is prepended.
        if not text.strip():
            logger.debug("Agent produced no text; nothing delivered to chat %s", chat_id)
            return
        sender_name = msg.sender.name if msg.sender else _UNKNOWN_SENDER
        logger.debug("Delivering message to Telegram chat %s", chat_id)
        self._post(chat_id, f"You received a message from {sender_name}: \n\n{text}")

    def deliver_notice(self, address: ChannelAddress, text: str) -> None:
        """Deliver a channel-layer acknowledgement to a Telegram chat.

        Called from the FastAPI route, not an actor thread, and needing no
        ``matches()``: the address names the destination outright. Notices are
        fanned out to every configured adapter, so the channel comparison is
        what stops a Slack chat id being posted here — the same check
        ``matches()`` performs on a binding.

        Logs transport errors and a closed client without raising: the command
        has already taken effect, and a raise here would turn it into a 500 the
        channel retries.

        Args:
            address: The chat to answer. A ``ChannelBinding`` satisfies this.
            text: The acknowledgement text.
        """
        if address.channel != TELEGRAM_CHANNEL:
            return
        logger.debug("Delivering notice to Telegram chat %s", address.channel_user_id)
        self._post(address.channel_user_id, text)

    def on_stop(self, team_id: uuid.UUID) -> None:
        """Note that a team stopped; release nothing.

        This adapter holds no per-team state. Its httpx client is
        **process-scoped** and shared by every team, so closing it here would
        mute every other conversation — silently, because ``deliver()`` logs
        and swallows its errors.

        Args:
            team_id: The team being stopped.
        """
        logger.debug("TelegramAdapter stopped: team_id=%s", team_id)
