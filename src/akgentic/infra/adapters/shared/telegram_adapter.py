"""TelegramChannelAdapter — delivers outbound messages via the Telegram Bot API."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from akgentic.core.messages import SentMessage
    from akgentic.infra.protocols.channels import ChannelBinding

logger = logging.getLogger(__name__)

# Mirrors ``TelegramChannelParser.channel_name``. A literal here rather than a
# Protocol member: an adapter does not advertise its channel the way a parser
# does, so the comparison belongs to the one implementation that knows the
# answer (ADR-043 §D6).
TELEGRAM_CHANNEL = "telegram"


class TelegramChannelAdapter:
    """Delivers outbound agent messages to Telegram chats via the Bot API.

    Satisfies the ``InteractionChannelAdapter`` protocol via structural
    subtyping.

    ``matches()`` returns ``True`` when the ``SentMessage`` recipient is
    structurally a ``UserProxy`` actor — or a subclass such as
    ``HumanProxy`` — **and** the binding names this adapter's own channel.
    Without the second half, a deployment configuring two channels would have
    this adapter accept a Slack binding and post a Slack user id to Telegram.

    ``deliver()`` sends a synchronous POST to the Telegram ``sendMessage``
    endpoint, addressing ``binding.channel_user_id`` — the chat the inbound
    message arrived from, and the only place that value exists on the outbound
    path.

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
        subclass such as ``HumanProxy``, indicating the message is headed
        to a human participant — and the binding belongs to Telegram. The
        recipient check is structural rather than a comparison against the
        recipient's ``role`` string, so a team is free to name its
        human-in-the-loop member anything.

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

    def deliver(self, msg: SentMessage, binding: ChannelBinding) -> None:
        """Deliver an outbound message to a Telegram chat.

        Posts to the Telegram ``sendMessage`` API. The ``chat_id`` comes from
        ``binding.channel_user_id``; the recipient's ``name`` is the TeamCard's
        agent name and posting it yields a Telegram 400.

        Logs errors without raising — delivery failures must not crash
        the caller.

        Args:
            msg: The message to deliver.
            binding: The recipient agent's channel binding, naming the chat.
        """
        chat_id = binding.channel_user_id
        text = getattr(msg.message, "content", None) or str(msg.message)
        logger.debug("Delivering message to Telegram chat %s", chat_id)

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
        except httpx.HTTPError:
            logger.exception("Failed to deliver message to Telegram chat %s", chat_id)

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
