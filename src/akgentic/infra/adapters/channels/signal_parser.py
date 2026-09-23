"""SignalChannelParser — parses signal-cli envelopes into ChannelMessage."""

from __future__ import annotations

import logging

from akgentic.infra.adapters.channels.channel_commands import parse_leading_command
from akgentic.infra.protocols.channels import ChannelMessage, JsonValue

logger = logging.getLogger(__name__)

# Prefix signal-cli-rest-api uses to name a group where a recipient is expected.
# The parser writes it into ``channel_user_id`` so the adapter can pass that
# value straight through as a recipient with no branch of its own — the
# encoding is the transport's, not one invented here.
GROUP_RECIPIENT_PREFIX = "group."


def _quoted_text(quote: JsonValue) -> str | None:
    """Return the text of the quoted message, or None.

    signal-cli sends the quoted message's own text under ``dataMessage.quote``,
    so it is here for free. Everything but a mapping carrying a ``text`` string
    reads as "no quotation": a quote of an attachment-only message has nothing
    a router could act on, and that is ordinary input rather than a malformed
    payload.

    Args:
        quote: The raw ``dataMessage["quote"]`` value, of any shape.

    Returns:
        The quoted text, or None when the message quotes nothing or something
        textless.
    """
    if not isinstance(quote, dict):
        return None
    text = quote.get("text")
    return text if isinstance(text, str) else None


def _group_id(group_info: JsonValue) -> str | None:
    """Return the group id from ``groupInfo``, or None for a 1:1 chat."""
    if not isinstance(group_info, dict):
        return None
    group_id = group_info.get("groupId")
    return group_id if isinstance(group_id, str) and group_id else None


def _self_ids(envelope: dict[str, JsonValue], account: JsonValue) -> set[str]:
    """Every identifier that names the linked account itself.

    A sync message carries the destination as a number while the account may be
    known by UUID, or the reverse, so all of them are collected and compared as
    a set rather than picking one and hoping it matches.
    """
    ids = {
        account,
        envelope.get("sourceNumber"),
        envelope.get("source"),
        envelope.get("sourceUuid"),
    }
    return {value for value in ids if isinstance(value, str) and value}


def _note_to_self(
    envelope: dict[str, JsonValue], account: JsonValue
) -> dict[str, JsonValue] | None:
    """Return a Note-to-Self sync's inner message, or None.

    **Why only Note to Self.** When signal-cli is a *linked device* rather than
    a separately registered number, the operator's own messages never arrive as
    a ``dataMessage``: everything they type is synced to the device as
    ``syncMessage.sentMessage``. Without this, a linked-device deployment can
    only be driven by *other* people, and the obvious single-device test — send
    yourself a message — does nothing.

    Accepting **every** sync would be far worse than accepting none. The device
    is synced a copy of every message the human sends to anybody, so the bot
    would start a team for each of their private conversations and answer into
    them. So exactly one case is admitted: a sync whose destination is the
    account itself, which is Signal's own "Note to Self" and cannot be anything
    the operator did not address to this bot.

    A sync to a **group** is refused for the same reason: it would hijack every
    group the operator posts in.

    Args:
        envelope: The ``envelope`` mapping.
        account: The raw ``payload["account"]`` value, of any shape.

    Returns:
        The inner ``sentMessage`` mapping when the sync is a Note to Self,
        None for every other sync and for no sync at all.
    """
    sync = envelope.get("syncMessage")
    if not isinstance(sync, dict):
        return None
    sent = sync.get("sentMessage")
    if not isinstance(sent, dict):
        return None
    if _group_id(sent.get("groupInfo")) is not None:
        return None
    mine = _self_ids(envelope, account)
    for key in ("destinationNumber", "destination", "destinationUuid"):
        destination = sent.get(key)
        if isinstance(destination, str) and destination in mine:
            return sent
    return None


def _binding_metadata(account: JsonValue) -> dict[str, JsonValue] | None:
    """Carry the receiving bot account onto the binding, or None when absent.

    One signal-cli daemon may hold **several** registered accounts, and the pump
    polls each of them. A reply must leave from the account the message arrived
    on: answering from a different bot number reaches the human as a message
    from a stranger, in a conversation thread they have never seen.

    The account is the only place that fact exists. ``channel_user_id`` names
    the *human*, and the adapter's configured number is a single global default,
    so without this the second account's conversations would all be answered by
    the first. signal-cli puts it in every receive payload for free.

    Args:
        account: The raw ``payload["account"]`` value, of any shape.

    Returns:
        ``{"account": <number>}``, or None when the payload names no account —
        in which case the adapter falls back to its configured number, which is
        correct for the single-account case.
    """
    if isinstance(account, str) and account:
        return {"account": account}
    return None


def _sender_id(envelope: dict[str, JsonValue]) -> str | None:
    """Return the stable identifier of the human who sent the envelope.

    Preference order is ``sourceNumber`` → ``source`` → ``sourceUuid``. All
    three are accepted as recipients by the send API, so any of them round-trips
    — but only the FIRST one present is used, because the value becomes the
    binding key.

    The trap that ordering avoids: a deployment where some envelopes carry a
    number and others only a UUID would give one human two ``channel_user_id``
    values, hence two bindings and two teams, with nothing in the log saying
    why. signal-cli is consistent per account, so in practice the first
    preference either always resolves or never does.

    Args:
        envelope: The ``envelope`` mapping from the receive payload.

    Returns:
        The sender identifier, or None when the envelope names no source.
    """
    for key in ("sourceNumber", "source", "sourceUuid"):
        value = envelope.get(key)
        if isinstance(value, str) and value:
            return value
    return None


class SignalChannelParser:
    """Parses one inbound signal-cli envelope into a normalized ChannelMessage.

    Satisfies the ``ChannelParser`` protocol via structural subtyping.

    Unlike Telegram, Signal pushes nothing: there is no webhook to register.
    Envelopes are drawn from a ``signal-cli`` daemon — over the JSON-RPC socket
    or ``signal-cli-rest-api``'s ``/v1/receive`` — by a pump that POSTs each one
    to ``/webhook/signal``. This parser is indifferent to which: it reads the
    envelope that arrives and nothing about how it got here.

    Two shapes are routable, both needing text. A plain ``dataMessage`` — a
    message somebody sent this account — and a ``syncMessage.sentMessage``
    addressed to the account itself, which is Signal's **Note to Self** and the
    only way to drive a linked device from the operator's own phone (see
    ``_note_to_self``, which explains why no other sync is admitted).

    Everything else raises ``ValueError`` — receipts, typing notifications,
    edits, reactions, attachment-only messages, and syncs of messages sent to
    anybody else. The webhook route logs and drops those, since no retry can
    turn any of them into text.

    Addressing:
        ``channel_user_id`` is the **chat**, not the person. In a 1:1 chat that
        is the sender's own identifier; in a group it is
        ``group.<groupId>``. The distinction is not cosmetic — keying a group
        message on its sender would deliver the team's reply privately to one
        member instead of to the group everyone is reading.

    Args:
        default_catalog_entry: Catalog entry ID for initiating new teams.
    """

    def __init__(self, default_catalog_entry: str = "default", **_kwargs: str) -> None:
        self._default_catalog_entry = default_catalog_entry

    @property
    def channel_name(self) -> str:
        """The channel name this parser handles."""
        return "signal"

    @property
    def default_catalog_entry(self) -> str:
        """Default catalog entry ID for new team initiation."""
        return self._default_catalog_entry

    async def parse(self, payload: dict[str, JsonValue]) -> ChannelMessage:
        """Parse one signal-cli receive payload into a ChannelMessage.

        Args:
            payload: A single signal-cli envelope. Expected structure::

                {
                    "envelope": {
                        "source": "+32470000000",
                        "sourceNumber": "+32470000000",
                        "sourceUuid": "8f1c...",
                        "sourceName": "Geoff",
                        "sourceDevice": 1,
                        "timestamp": 1711800000000,
                        "dataMessage": {
                            "timestamp": 1711800000000,
                            "message": "Hello",
                            "quote": {"id": 1711799000000, "text": "earlier"},
                            "groupInfo": {"groupId": "Ci0K...", "type": "DELIVER"}
                        }
                    },
                    "account": "+32471111111"
                }

        Returns:
            Parsed ChannelMessage with content, chat id, message id, the leading
            slash command when the text opens with one, and the quoted text when
            the message quotes something.

        Raises:
            ValueError: If the payload carries no textual ``dataMessage``, or
                names no chat to answer.
        """
        logger.debug("Signal parser payload: %s", payload)

        envelope = payload.get("envelope")
        if not isinstance(envelope, dict):
            msg = "Signal payload does not contain an 'envelope' field"
            raise ValueError(msg)

        # A plain ``dataMessage``, or a Note-to-Self sync. An ``editMessage``
        # nests a dataMessage of its own and is deliberately NOT read through:
        # that would re-route a correction as a fresh message, the same reason
        # the Telegram parser ignores ``edited_message``.
        data_message = envelope.get("dataMessage")
        if not isinstance(data_message, dict):
            data_message = _note_to_self(envelope, payload.get("account"))
        if data_message is None:
            # Name what DID arrive. "no dataMessage" alone sends the reader
            # hunting, and the overwhelmingly common cause — a receipt, a typing
            # notification, or a sync of a message sent to somebody else — is
            # identified by the key that is present.
            msg = f"Signal envelope carries no routable message (keys: {sorted(envelope)})"
            raise ValueError(msg)

        text = data_message.get("message")
        if not isinstance(text, str) or not text:
            # Reactions and attachment-only messages both land here: signal-cli
            # sends them as a message whose ``message`` is null or empty.
            msg = f"Signal message carries no text (keys: {sorted(data_message)})"
            raise ValueError(msg)

        chat_id = _resolve_chat_id(envelope, data_message)

        message_timestamp = data_message.get("timestamp") or envelope.get("timestamp")

        logger.debug("Parsing Signal envelope: chat_id=%s", chat_id)
        # ``content`` stays the full original text, command word and all: a
        # command the channel layer does not consume must reach the team looking
        # exactly as the user typed it.
        return ChannelMessage(
            content=text,
            channel_user_id=chat_id,
            channel_message_id=str(message_timestamp) if message_timestamp is not None else None,
            command=parse_leading_command(text),
            quoted_text=_quoted_text(data_message.get("quote")),
            binding_metadata=_binding_metadata(payload.get("account")),
        )


def _resolve_chat_id(envelope: dict[str, JsonValue], data_message: dict[str, JsonValue]) -> str:
    """Return the chat to bind and answer — the group when there is one, else the sender.

    Args:
        envelope: The ``envelope`` mapping.
        data_message: The ``dataMessage`` mapping.

    Returns:
        ``group.<groupId>`` for a group message, the sender's identifier for a
        1:1 one.

    Raises:
        ValueError: If neither is present, which leaves nowhere to reply.
    """
    group_id = _group_id(data_message.get("groupInfo"))
    if group_id is not None:
        return f"{GROUP_RECIPIENT_PREFIX}{group_id}"
    sender = _sender_id(envelope)
    if sender is None:
        msg = "Signal envelope names neither a group nor a source to reply to"
        raise ValueError(msg)
    return sender
