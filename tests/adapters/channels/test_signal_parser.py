"""Tests for SignalChannelParser."""

from __future__ import annotations

import pytest

from akgentic.infra.adapters.channels.signal_parser import SignalChannelParser

# ---------------------------------------------------------------------------
# Sample signal-cli receive payloads
# ---------------------------------------------------------------------------

BOT_NUMBER = "+32471111111"
USER_NUMBER = "+32470000000"
USER_UUID = "8f1c9e2a-0000-4000-8000-000000000001"
GROUP_ID = "Ci0KIF9hYmNkZWZnaGlqa2xtbm9w"


def _envelope(
    data_message: dict | None = None,
    *,
    source_number: str | None = USER_NUMBER,
    source: str | None = USER_NUMBER,
    source_uuid: str | None = USER_UUID,
    **extra: object,
) -> dict:
    """One signal-cli receive payload, with absent keys genuinely absent."""
    envelope: dict = {"sourceName": "Geoff", "sourceDevice": 1, "timestamp": 1711800000000}
    if source_number is not None:
        envelope["sourceNumber"] = source_number
    if source is not None:
        envelope["source"] = source
    if source_uuid is not None:
        envelope["sourceUuid"] = source_uuid
    if data_message is not None:
        envelope["dataMessage"] = data_message
    envelope.update(extra)
    return {"envelope": envelope, "account": BOT_NUMBER}


def _data_message(text: str | None = "Hello, bot!", **extra: object) -> dict:
    message: dict = {"timestamp": 1711800000000, "expiresInSeconds": 0}
    message["message"] = text
    message.update(extra)
    return message


VALID_TEXT_ENVELOPE = _envelope(_data_message())


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestChannelName:
    """channel_name property returns 'signal'."""

    def test_channel_name(self) -> None:
        assert SignalChannelParser().channel_name == "signal"


class TestDefaultCatalogEntry:
    """default_catalog_entry returns the configured value."""

    def test_default_value(self) -> None:
        assert SignalChannelParser().default_catalog_entry == "default"

    def test_custom_value(self) -> None:
        assert SignalChannelParser(default_catalog_entry="my-team").default_catalog_entry == (
            "my-team"
        )

    def test_extra_config_kwargs_are_tolerated(self) -> None:
        # The registry passes one config dict to parser, router AND adapter, so
        # the parser is handed the adapter's keys and must not choke on them.
        parser = SignalChannelParser(
            default_catalog_entry="my-team",
            signal_api_url="http://localhost:8080",
            signal_number=BOT_NUMBER,
        )
        assert parser.default_catalog_entry == "my-team"


# ---------------------------------------------------------------------------
# parse() — happy path
# ---------------------------------------------------------------------------


class TestParseValidTextMessage:
    """A textual dataMessage becomes a ChannelMessage."""

    async def test_content_and_ids(self) -> None:
        message = await SignalChannelParser().parse(VALID_TEXT_ENVELOPE)
        assert message.content == "Hello, bot!"
        assert message.channel_user_id == USER_NUMBER
        assert message.channel_message_id == "1711800000000"

    async def test_no_command_and_no_quote_by_default(self) -> None:
        message = await SignalChannelParser().parse(VALID_TEXT_ENVELOPE)
        assert message.command is None
        assert message.quoted_text is None

    async def test_message_id_falls_back_to_envelope_timestamp(self) -> None:
        payload = _envelope(_data_message())
        del payload["envelope"]["dataMessage"]["timestamp"]
        message = await SignalChannelParser().parse(payload)
        assert message.channel_message_id == "1711800000000"

    async def test_message_id_is_none_when_nothing_is_timestamped(self) -> None:
        payload = _envelope(_data_message())
        del payload["envelope"]["dataMessage"]["timestamp"]
        del payload["envelope"]["timestamp"]
        message = await SignalChannelParser().parse(payload)
        assert message.channel_message_id is None


# ---------------------------------------------------------------------------
# parse() — chat identity
# ---------------------------------------------------------------------------


class TestChatIdentity:
    """channel_user_id names the CHAT, not the person."""

    async def test_group_wins_over_sender(self) -> None:
        # The whole point: answering a group message must reach the group, not
        # the one member who typed it.
        payload = _envelope(_data_message(groupInfo={"groupId": GROUP_ID, "type": "DELIVER"}))
        message = await SignalChannelParser().parse(payload)
        assert message.channel_user_id == f"group.{GROUP_ID}"

    async def test_source_number_preferred(self) -> None:
        message = await SignalChannelParser().parse(VALID_TEXT_ENVELOPE)
        assert message.channel_user_id == USER_NUMBER

    async def test_falls_back_to_source_then_uuid(self) -> None:
        no_number = await SignalChannelParser().parse(
            _envelope(_data_message(), source_number=None, source="LEGACY-SOURCE")
        )
        assert no_number.channel_user_id == "LEGACY-SOURCE"

        uuid_only = await SignalChannelParser().parse(
            _envelope(_data_message(), source_number=None, source=None)
        )
        assert uuid_only.channel_user_id == USER_UUID

    async def test_blank_source_is_skipped(self) -> None:
        message = await SignalChannelParser().parse(
            _envelope(_data_message(), source_number="", source="")
        )
        assert message.channel_user_id == USER_UUID

    async def test_empty_group_id_falls_through_to_sender(self) -> None:
        payload = _envelope(_data_message(groupInfo={"groupId": "", "type": "DELIVER"}))
        message = await SignalChannelParser().parse(payload)
        assert message.channel_user_id == USER_NUMBER

    async def test_non_mapping_group_info_falls_through_to_sender(self) -> None:
        payload = _envelope(_data_message(groupInfo="not-a-mapping"))
        message = await SignalChannelParser().parse(payload)
        assert message.channel_user_id == USER_NUMBER

    async def test_no_group_and_no_source_is_refused(self) -> None:
        payload = _envelope(_data_message(), source_number=None, source=None, source_uuid=None)
        with pytest.raises(ValueError, match="neither a group nor a source"):
            await SignalChannelParser().parse(payload)


# ---------------------------------------------------------------------------
# parse() — commands
# ---------------------------------------------------------------------------


class TestCommandParsing:
    """Signal carries no markup, so the command is read from the text itself."""

    async def test_bare_command(self) -> None:
        message = await SignalChannelParser().parse(_envelope(_data_message("/status")))
        assert message.command is not None
        assert message.command.name == "status"
        assert message.command.rest == ""

    async def test_command_with_rest(self) -> None:
        message = await SignalChannelParser().parse(
            _envelope(_data_message("/new  Plan my Q3   budget"))
        )
        assert message.command is not None
        assert message.command.name == "new"
        # Verbatim after the separating whitespace: not lowercased, internal
        # spacing preserved.
        assert message.command.rest == "Plan my Q3   budget"

    async def test_command_name_is_lowercased(self) -> None:
        message = await SignalChannelParser().parse(_envelope(_data_message("/STATUS")))
        assert message.command is not None
        assert message.command.name == "status"

    async def test_content_keeps_the_command_word(self) -> None:
        message = await SignalChannelParser().parse(_envelope(_data_message("/new hello")))
        assert message.content == "/new hello"

    @pytest.mark.parametrize(
        "text",
        [
            "not a /command",
            "/path/to/file is broken",
            "/2fa please",
            "/",
            "//double",
            " /leading-space",
        ],
    )
    async def test_non_commands(self, text: str) -> None:
        message = await SignalChannelParser().parse(_envelope(_data_message(text)))
        assert message.command is None


# ---------------------------------------------------------------------------
# parse() — quotes
# ---------------------------------------------------------------------------


class TestQuotedText:
    """dataMessage.quote.text, when it is there and is a string."""

    async def test_quote_text_is_lifted(self) -> None:
        payload = _envelope(
            _data_message("who?", quote={"id": 1711799000000, "text": "Started team abc"})
        )
        message = await SignalChannelParser().parse(payload)
        assert message.quoted_text == "Started team abc"

    @pytest.mark.parametrize("quote", ["not-a-mapping", {"id": 1}, {"text": 42}, None])
    async def test_quote_without_usable_text(self, quote: object) -> None:
        payload = _envelope(_data_message("who?", quote=quote))
        message = await SignalChannelParser().parse(payload)
        assert message.quoted_text is None


# ---------------------------------------------------------------------------
# parse() — rejected payloads
# ---------------------------------------------------------------------------


class TestRejectedPayloads:
    """Everything a retry could never turn into text raises ValueError."""

    async def test_missing_envelope(self) -> None:
        with pytest.raises(ValueError, match="'envelope' field"):
            await SignalChannelParser().parse({"account": BOT_NUMBER})

    async def test_non_mapping_envelope(self) -> None:
        with pytest.raises(ValueError, match="'envelope' field"):
            await SignalChannelParser().parse({"envelope": "nope"})

    async def test_receipt_message(self) -> None:
        payload = _envelope(None, receiptMessage={"when": 1711800000000, "isDelivery": True})
        with pytest.raises(ValueError, match="no routable message"):
            await SignalChannelParser().parse(payload)

    async def test_typing_message(self) -> None:
        payload = _envelope(None, typingMessage={"action": "STARTED"})
        with pytest.raises(ValueError, match="no routable message"):
            await SignalChannelParser().parse(payload)

    async def test_edit_message_is_not_read_through(self) -> None:
        # An editMessage nests a dataMessage of its own. Reading through to it
        # would re-route a correction as a fresh message.
        payload = _envelope(
            None,
            editMessage={"targetSentTimestamp": 1711799000000, "dataMessage": _data_message("v2")},
        )
        with pytest.raises(ValueError, match="no routable message"):
            await SignalChannelParser().parse(payload)

    @pytest.mark.parametrize("text", [None, "", 42])
    async def test_textless_data_message(self, text: object) -> None:
        # Reactions and attachment-only messages both arrive this way.
        payload = _envelope(_data_message(text))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="carries no text"):
            await SignalChannelParser().parse(payload)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """The registry isinstance-checks the parser at load; assert it here too.

    A structurally-typed implementation that loses a member fails at wiring
    time with a TypeError naming the FQCN, which is a poor place to find out.
    """

    def test_satisfies_channel_parser(self) -> None:
        from akgentic.infra.protocols.channels import ChannelParser

        assert isinstance(SignalChannelParser(), ChannelParser)


# ---------------------------------------------------------------------------
# parse() — Note-to-Self syncs (linked-device operation)
# ---------------------------------------------------------------------------


def _sync_envelope(sent: dict, *, account: str = USER_NUMBER) -> dict:
    """A sync of a message THIS account sent — what a linked device receives."""
    return {
        "envelope": {
            "source": USER_NUMBER,
            "sourceNumber": USER_NUMBER,
            "sourceUuid": USER_UUID,
            "timestamp": 1711800000000,
            "syncMessage": {"sentMessage": sent},
        },
        "account": account,
    }


def _sent(text: str = "Hello, bot!", **extra: object) -> dict:
    sent: dict = {"timestamp": 1711800000000, "message": text}
    sent.update(extra)
    return sent


class TestNoteToSelfAccepted:
    """A sync addressed to the account itself is the linked-device inbound path."""

    async def test_destination_number_matches_account(self) -> None:
        payload = _sync_envelope(_sent(destinationNumber=USER_NUMBER))
        message = await SignalChannelParser().parse(payload)
        assert message.content == "Hello, bot!"
        assert message.channel_user_id == USER_NUMBER

    async def test_destination_uuid_matches_the_source_uuid(self) -> None:
        # The destination may be a UUID while the account is a number; the
        # comparison is against the whole set of self identifiers.
        payload = _sync_envelope(_sent(destinationUuid=USER_UUID))
        message = await SignalChannelParser().parse(payload)
        assert message.channel_user_id == USER_NUMBER

    async def test_commands_and_quotes_work_the_same(self) -> None:
        payload = _sync_envelope(
            _sent("/new plan the offsite", destinationNumber=USER_NUMBER, quote={"text": "earlier"})
        )
        message = await SignalChannelParser().parse(payload)
        assert message.command is not None
        assert message.command.name == "new"
        assert message.quoted_text == "earlier"


class TestOtherSyncsRefused:
    """Accepting every sync would answer into the operator's private chats."""

    async def test_sync_to_someone_else_is_refused(self) -> None:
        # The device is synced a copy of EVERY message the human sends. This is
        # the case that must never start a team.
        payload = _sync_envelope(_sent(destinationNumber="+32479999999"))
        with pytest.raises(ValueError, match="no routable message"):
            await SignalChannelParser().parse(payload)

    async def test_sync_to_a_group_is_refused(self) -> None:
        payload = _sync_envelope(
            _sent(destinationNumber=USER_NUMBER, groupInfo={"groupId": GROUP_ID})
        )
        with pytest.raises(ValueError, match="no routable message"):
            await SignalChannelParser().parse(payload)

    async def test_sync_with_no_destination_is_refused(self) -> None:
        payload = _sync_envelope(_sent())
        with pytest.raises(ValueError, match="no routable message"):
            await SignalChannelParser().parse(payload)

    async def test_read_receipt_sync_is_refused(self) -> None:
        payload = {
            "envelope": {
                "sourceNumber": USER_NUMBER,
                "syncMessage": {"readMessages": [{"sender": USER_NUMBER}]},
            },
            "account": USER_NUMBER,
        }
        with pytest.raises(ValueError, match="no routable message"):
            await SignalChannelParser().parse(payload)


class TestDiagnostics:
    """The rejection message names what DID arrive, so the log explains itself."""

    async def test_envelope_keys_are_reported(self) -> None:
        payload = _envelope(None, receiptMessage={"isDelivery": True})
        with pytest.raises(ValueError, match="receiptMessage"):
            await SignalChannelParser().parse(payload)

    async def test_message_keys_are_reported(self) -> None:
        payload = _envelope(_data_message(None, reaction={"emoji": "👍"}))
        with pytest.raises(ValueError, match="reaction"):
            await SignalChannelParser().parse(payload)


# ---------------------------------------------------------------------------
# parse() — which bot account received this
# ---------------------------------------------------------------------------


class TestReceivingAccount:
    """One daemon may hold several bot accounts; a reply must use the right one."""

    async def test_account_is_carried_as_binding_metadata(self) -> None:
        message = await SignalChannelParser().parse(VALID_TEXT_ENVELOPE)
        assert message.binding_metadata == {"account": BOT_NUMBER}

    async def test_second_account_is_carried_distinctly(self) -> None:
        payload = _envelope(_data_message())
        payload["account"] = "+32460000000"
        message = await SignalChannelParser().parse(payload)
        assert message.binding_metadata == {"account": "+32460000000"}

    async def test_note_to_self_carries_it_too(self) -> None:
        payload = _sync_envelope(_sent(destinationNumber=USER_NUMBER), account=USER_NUMBER)
        message = await SignalChannelParser().parse(payload)
        assert message.binding_metadata == {"account": USER_NUMBER}

    @pytest.mark.parametrize("account", [None, "", 42])
    async def test_absent_account_leaves_it_unset(self, account: object) -> None:
        # None means "no per-conversation account", and the adapter falls back
        # to its configured number — correct for a single-account deployment.
        payload = _envelope(_data_message())
        if account is None:
            del payload["account"]
        else:
            payload["account"] = account
        message = await SignalChannelParser().parse(payload)
        assert message.binding_metadata is None
