"""TelegramChannelParser — parses Telegram Bot API webhook Updates into ChannelMessage."""

from __future__ import annotations

import logging

from akgentic.infra.protocols.channels import ChannelCommand, ChannelMessage, JsonValue

logger = logging.getLogger(__name__)

# Telegram's documented MessageEntity type for a leading ``/command`` token.
_BOT_COMMAND_ENTITY = "bot_command"


def _parse_command(text: str, entities: JsonValue) -> ChannelCommand | None:
    """Lift the leading ``/command`` token out of ``text``, or return None.

    The marker is channel-specific: Telegram reports the token as a
    ``bot_command`` MessageEntity rather than leaving the reader to guess from a
    leading slash. Only an entity that is **first** and at ``offset == 0``
    counts — a ``/slash`` mid-sentence is text (ADR-043 §D10).

    Entity offsets are UTF-16 code units, which diverge from Python string
    indices only for text *before* the offset. This offset is always 0 and the
    command token is always ASCII, so plain slicing is correct here.

    Args:
        text: The message's full text, returned to the caller untouched.
        entities: The raw ``message["entities"]`` value, of any shape.

    Returns:
        The parsed command, or None for any other entity shape — absent, not a
        list, first entity not a ``bot_command``, a non-zero offset, or a
        non-integer length.
    """
    if not isinstance(entities, list) or not entities:
        return None
    first = entities[0]
    if not isinstance(first, dict) or first.get("type") != _BOT_COMMAND_ENTITY:
        return None
    if first.get("offset") != 0:
        return None
    length = first.get("length")
    # ``bool`` is an ``int`` subclass, and a YAML/JSON ``true`` here would slice
    # one character rather than being rejected.
    if not isinstance(length, int) or isinstance(length, bool):
        return None

    token = text[:length]
    # Telegram includes the ``@botname`` suffix inside the entity's length, so a
    # group-chat ``/new@some_bot`` would otherwise yield a name matching nothing.
    name = token.removeprefix("/").split("@", 1)[0].lower()
    # ``lstrip`` drops the single separating space; the remainder is verbatim —
    # not lowercased, internal whitespace preserved.
    return ChannelCommand(name=name, rest=text[length:].lstrip())


def _quoted_text(reply_to_message: JsonValue) -> str | None:
    """Return the text of the replied-to message, or None.

    Telegram sends the whole quoted message under ``reply_to_message``, not
    merely its id, so the text is here for free. Everything but a mapping
    carrying a ``text`` string reads as "no quotation": a reply to a photo, a
    sticker or a service message has nothing a router could act on, and that is
    an ordinary input rather than a malformed payload.

    Args:
        reply_to_message: The raw ``message["reply_to_message"]`` value, of any
            shape.

    Returns:
        The quoted text, or None when the message replies to nothing or to
        something textless.
    """
    if not isinstance(reply_to_message, dict):
        return None
    text = reply_to_message.get("text")
    return text if isinstance(text, str) else None


class TelegramChannelParser:
    """Parses inbound Telegram webhook Update payloads into normalized ChannelMessage.

    Satisfies the ``ChannelParser`` protocol via structural subtyping.

    Telegram sends Updates as JSON POST to the configured webhook URL.
    This parser extracts the text message content, chat ID (used as the
    channel user identifier), and message ID from the Update payload.

    Only text messages (``message.text``) are supported. Non-text updates
    (edited messages, channel posts, photos, etc.) raise ``ValueError``.

    Args:
        bot_token: Telegram Bot API token (unused by parser, but passed via
            ``ChannelConfig.config`` shared with the adapter).
        default_catalog_entry: Catalog entry ID for initiating new teams.
    """

    def __init__(self, default_catalog_entry: str = "default", **_kwargs: str) -> None:
        self._default_catalog_entry = default_catalog_entry

    @property
    def channel_name(self) -> str:
        """The channel name this parser handles."""
        return "telegram"

    @property
    def default_catalog_entry(self) -> str:
        """Default catalog entry ID for new team initiation."""
        return self._default_catalog_entry

    async def parse(self, payload: dict[str, JsonValue]) -> ChannelMessage:
        """Parse a Telegram Update payload into a ChannelMessage.

        Args:
            payload: Raw Telegram Update JSON. Expected structure::

                {
                    "update_id": 123456,
                    "message": {
                        "message_id": 4,
                        "from": {
                            "id": 8892740599,
                            "is_bot": false,
                            "first_name": "John",
                            "last_name": "Doe",
                            "language_code": "en"
                        },
                        "chat": {
                            "id": 8892740599,
                            "first_name": "John",
                            "last_name": "Doe",
                            "type": "private"
                        },
                        "date": 1789587220,
                        "text": "Hello"
                    }
                }

        Returns:
            Parsed ChannelMessage with content, chat ID, message ID, and the
            leading slash command when the payload's entities name one.

        Raises:
            ValueError: If the payload does not contain a text message.
        """

        logger.debug("Telegram parser payload: %s", payload)

        message = payload.get("message")
        if not isinstance(message, dict):
            msg = "Telegram Update does not contain a 'message' field"
            raise ValueError(msg)

        text = message.get("text")
        if not isinstance(text, str):
            msg = "Telegram message does not contain a 'text' field"
            raise ValueError(msg)

        chat = message.get("chat")
        if not isinstance(chat, dict):
            msg = "Telegram message does not contain a 'chat' field"
            raise ValueError(msg)

        chat_id = chat.get("id")
        if chat_id is None:
            msg = "Telegram chat does not contain an 'id' field"
            raise ValueError(msg)

        message_id = message.get("message_id")

        logger.debug("Parsing Telegram update: chat_id=%s", chat_id)
        # ``content`` stays the full original text, command word and all: a
        # command the channel layer does not consume must reach the team looking
        # exactly as the user typed it.
        return ChannelMessage(
            content=text,
            channel_user_id=str(chat_id),
            channel_message_id=str(message_id) if message_id is not None else None,
            command=_parse_command(text, message.get("entities")),
            quoted_text=_quoted_text(message.get("reply_to_message")),
        )
