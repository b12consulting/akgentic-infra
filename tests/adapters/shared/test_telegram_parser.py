"""Tests for TelegramChannelParser."""

from __future__ import annotations

import pytest

from akgentic.infra.adapters.shared.telegram_parser import TelegramChannelParser

# ---------------------------------------------------------------------------
# Sample Telegram Update payloads
# ---------------------------------------------------------------------------

VALID_TEXT_UPDATE: dict = {
    "update_id": 123456789,
    "message": {
        "message_id": 42,
        "from": {"id": 111222333, "is_bot": False, "first_name": "Geoff"},
        "chat": {"id": 987654321, "type": "private"},
        "date": 1711800000,
        "text": "Hello, bot!",
    },
}

EDITED_MESSAGE_UPDATE: dict = {
    "update_id": 123456790,
    "edited_message": {
        "message_id": 42,
        "from": {"id": 111222333, "is_bot": False, "first_name": "Geoff"},
        "chat": {"id": 987654321, "type": "private"},
        "date": 1711800000,
        "edit_date": 1711800060,
        "text": "Hello, bot! (edited)",
    },
}

PHOTO_MESSAGE_UPDATE: dict = {
    "update_id": 123456791,
    "message": {
        "message_id": 43,
        "from": {"id": 111222333, "is_bot": False, "first_name": "Geoff"},
        "chat": {"id": 987654321, "type": "private"},
        "date": 1711800000,
        "photo": [{"file_id": "abc123", "width": 100, "height": 100}],
    },
}


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestChannelName:
    """channel_name property returns 'telegram'."""

    def test_channel_name(self) -> None:
        parser = TelegramChannelParser()
        assert parser.channel_name == "telegram"


class TestDefaultCatalogEntry:
    """default_catalog_entry returns the configured value."""

    def test_default_value(self) -> None:
        parser = TelegramChannelParser()
        assert parser.default_catalog_entry == "default"

    def test_custom_value(self) -> None:
        parser = TelegramChannelParser(default_catalog_entry="my-team")
        assert parser.default_catalog_entry == "my-team"


# ---------------------------------------------------------------------------
# parse() — happy path
# ---------------------------------------------------------------------------


class TestParseValidTextMessage:
    """AC 1: Valid Telegram text Update → correct ChannelMessage."""

    @pytest.mark.asyncio
    async def test_extracts_content(self) -> None:
        parser = TelegramChannelParser()
        result = await parser.parse(VALID_TEXT_UPDATE)
        assert result.content == "Hello, bot!"

    @pytest.mark.asyncio
    async def test_extracts_channel_user_id(self) -> None:
        parser = TelegramChannelParser()
        result = await parser.parse(VALID_TEXT_UPDATE)
        assert result.channel_user_id == "987654321"

    @pytest.mark.asyncio
    async def test_extracts_message_id(self) -> None:
        parser = TelegramChannelParser()
        result = await parser.parse(VALID_TEXT_UPDATE)
        assert result.message_id == "42"

    @pytest.mark.asyncio
    async def test_team_id_is_none(self) -> None:
        parser = TelegramChannelParser()
        result = await parser.parse(VALID_TEXT_UPDATE)
        assert result.team_id is None


# ---------------------------------------------------------------------------
# parse() — error cases
# ---------------------------------------------------------------------------


class TestParseNoMessage:
    """AC 2: Update with no 'message' key raises ValueError."""

    @pytest.mark.asyncio
    async def test_edited_message_raises(self) -> None:
        parser = TelegramChannelParser()
        with pytest.raises(ValueError, match="does not contain a 'message' field"):
            await parser.parse(EDITED_MESSAGE_UPDATE)

    @pytest.mark.asyncio
    async def test_empty_payload_raises(self) -> None:
        parser = TelegramChannelParser()
        with pytest.raises(ValueError, match="does not contain a 'message' field"):
            await parser.parse({})


class TestParseNoText:
    """AC 2: Message without 'text' key raises ValueError."""

    @pytest.mark.asyncio
    async def test_photo_only_raises(self) -> None:
        parser = TelegramChannelParser()
        with pytest.raises(ValueError, match="does not contain a 'text' field"):
            await parser.parse(PHOTO_MESSAGE_UPDATE)


class TestParseMissingChat:
    """Message without 'chat' raises ValueError."""

    @pytest.mark.asyncio
    async def test_no_chat_raises(self) -> None:
        parser = TelegramChannelParser()
        payload: dict = {
            "update_id": 1,
            "message": {"message_id": 1, "text": "hello"},
        }
        with pytest.raises(ValueError, match="does not contain a 'chat' field"):
            await parser.parse(payload)


# ---------------------------------------------------------------------------
# The bot_command entity fills ChannelMessage.command
# ---------------------------------------------------------------------------


def _command_update(
    text: str,
    length: int,
    offset: int = 0,
    entity_type: str = "bot_command",
) -> dict:
    """A Telegram Update whose message carries one MessageEntity."""
    return {
        "update_id": 1,
        "message": {
            "message_id": 7,
            "chat": {"id": 987654321, "type": "private"},
            "date": 1711800000,
            "text": text,
            "entities": [{"type": entity_type, "offset": offset, "length": length}],
        },
    }


class TestParseCommand:
    """A ``bot_command`` entity at offset 0 is lifted into ``command``."""

    async def test_command_name_and_rest(self) -> None:
        parser = TelegramChannelParser()

        msg = await parser.parse(_command_update("/new hi", length=4))

        assert msg.command is not None
        assert msg.command.name == "new"
        assert msg.command.rest == "hi"
        assert msg.content == "/new hi"

    async def test_command_name_is_lowercased(self) -> None:
        parser = TelegramChannelParser()

        msg = await parser.parse(_command_update("/NEW hi", length=4))

        assert msg.command is not None
        assert msg.command.name == "new"
        assert msg.content == "/NEW hi"

    async def test_bot_mention_suffix_is_stripped_from_the_name(self) -> None:
        """Telegram puts ``@botname`` inside the entity's length.

        Without the ``@`` split a group-chat ``/new@some_bot`` yields the name
        ``new@some_bot``, matches nothing, and degrades silently to content.
        """
        parser = TelegramChannelParser()

        msg = await parser.parse(_command_update("/new@some_bot hi", length=13))

        assert msg.command is not None
        assert msg.command.name == "new"
        assert msg.command.rest == "hi"
        assert msg.content == "/new@some_bot hi"

    async def test_command_alone_has_empty_rest(self) -> None:
        parser = TelegramChannelParser()

        msg = await parser.parse(_command_update("/new", length=4))

        assert msg.command is not None
        assert msg.command.name == "new"
        assert msg.command.rest == ""
        assert msg.content == "/new"

    async def test_rest_is_verbatim_after_one_separator(self) -> None:
        """Internal spacing and case survive; only the separating space goes."""
        parser = TelegramChannelParser()

        msg = await parser.parse(_command_update("/new Fix the   invoice", length=4))

        assert msg.command is not None
        assert msg.command.rest == "Fix the   invoice"
        assert msg.content == "/new Fix the   invoice"

    async def test_plain_text_has_no_command(self) -> None:
        parser = TelegramChannelParser()

        msg = await parser.parse(VALID_TEXT_UPDATE)

        assert msg.command is None
        assert msg.content == "Hello, bot!"

    async def test_entity_at_a_non_zero_offset_is_not_a_command(self) -> None:
        """A ``/slash`` mid-sentence is text, not a command."""
        parser = TelegramChannelParser()

        msg = await parser.parse(_command_update("see /new later", length=4, offset=4))

        assert msg.command is None
        assert msg.content == "see /new later"

    async def test_a_first_entity_that_is_not_a_bot_command_is_ignored(self) -> None:
        parser = TelegramChannelParser()

        msg = await parser.parse(_command_update("/new hi", length=4, entity_type="bold"))

        assert msg.command is None
        assert msg.content == "/new hi"

    async def test_a_non_integer_length_is_ignored(self) -> None:
        parser = TelegramChannelParser()
        payload = _command_update("/new hi", length=4)
        payload["message"]["entities"][0]["length"] = "4"

        msg = await parser.parse(payload)

        assert msg.command is None
        assert msg.content == "/new hi"

    async def test_a_non_list_entities_field_is_ignored(self) -> None:
        parser = TelegramChannelParser()
        payload = _command_update("/new hi", length=4)
        payload["message"]["entities"] = {"type": "bot_command"}

        msg = await parser.parse(payload)

        assert msg.command is None
        assert msg.content == "/new hi"
