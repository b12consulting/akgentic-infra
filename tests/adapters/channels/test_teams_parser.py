"""Tests for TeamsChannelParser."""

from __future__ import annotations

import pytest

from akgentic.infra.adapters.channels.teams_parser import TeamsChannelParser

BOT_ID = "28:00000000-0000-0000-0000-000000000000"
CONVERSATION_ID = "a:1abcDEF"
SERVICE_URL = "https://smba.trafficmanager.net/emea/"


def _mention(text: str = "<at>Akgentic</at>", mentioned_id: str = BOT_ID) -> dict:
    return {"type": "mention", "mentioned": {"id": mentioned_id}, "text": text}


def _activity(
    text: str | None = "hello",
    *,
    activity_type: str = "message",
    conversation_id: str | None = CONVERSATION_ID,
    entities: object = None,
    service_url: object = SERVICE_URL,
    **extra: object,
) -> dict:
    activity: dict = {
        "type": activity_type,
        "id": "1616989574408",
        "channelId": "msteams",
        "from": {"id": "29:1xyz", "name": "Geoff", "aadObjectId": "8f1c"},
        "recipient": {"id": BOT_ID, "name": "Akgentic"},
    }
    if conversation_id is not None:
        activity["conversation"] = {"conversationType": "personal", "id": conversation_id}
    if text is not None:
        activity["text"] = text
    if entities is not None:
        activity["entities"] = entities
    if service_url is not None:
        activity["serviceUrl"] = service_url
    activity.update(extra)
    return activity


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_channel_name(self) -> None:
        assert TeamsChannelParser().channel_name == "teams"

    def test_default_catalog_entry(self) -> None:
        assert TeamsChannelParser(default_catalog_entry="my-team").default_catalog_entry == (
            "my-team"
        )

    def test_extra_config_kwargs_are_tolerated(self) -> None:
        # The registry passes one config dict to parser, router AND adapter.
        parser = TeamsChannelParser(default_catalog_entry="t", teams_app_id="x", teams_app_type="y")
        assert parser.default_catalog_entry == "t"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestParseMessage:
    async def test_personal_message(self) -> None:
        message = await TeamsChannelParser().parse(_activity("hello there"))
        assert message.content == "hello there"
        assert message.channel_user_id == CONVERSATION_ID
        assert message.channel_message_id == "1616989574408"

    async def test_conversation_is_the_chat_not_the_person(self) -> None:
        # A Teams channel is ONE conversation shared by everyone in it, so it
        # gets one binding — the same split as a Signal group.
        activity = _activity("hi", conversation_id="19:channelthread@thread.tacv2")
        message = await TeamsChannelParser().parse(activity)
        assert message.channel_user_id == "19:channelthread@thread.tacv2"

    async def test_command_is_parsed_from_plain_text(self) -> None:
        message = await TeamsChannelParser().parse(_activity("/status"))
        assert message.command is not None
        assert message.command.name == "status"

    async def test_quoting_is_never_available(self) -> None:
        # Teams carries replyToId but not the replied-to text, so /register
        # cannot read a team id out of a reply as it can on Telegram.
        message = await TeamsChannelParser().parse(_activity("hi", replyToId="1616989574000"))
        assert message.quoted_text is None


# ---------------------------------------------------------------------------
# Mention stripping
# ---------------------------------------------------------------------------


class TestMentionStripping:
    """In a channel the bot is only invoked by @-mention, so it is addressing."""

    async def test_bot_mention_is_removed(self) -> None:
        activity = _activity("<at>Akgentic</at> hello", entities=[_mention()])
        message = await TeamsChannelParser().parse(activity)
        assert message.content == "hello"

    async def test_agent_routing_survives_the_strip(self) -> None:
        # The router only routes on a recipient at the START of the text, so a
        # leftover bot mention would send every message to the default agent.
        activity = _activity("<at>Akgentic</at> @Expert look at this", entities=[_mention()])
        message = await TeamsChannelParser().parse(activity)
        assert message.content == "@Expert look at this"

    async def test_command_survives_the_strip(self) -> None:
        activity = _activity("<at>Akgentic</at> /new plan the offsite", entities=[_mention()])
        message = await TeamsChannelParser().parse(activity)
        assert message.command is not None
        assert message.command.name == "new"
        assert message.command.rest == "plan the offsite"

    async def test_other_peoples_mentions_are_kept(self) -> None:
        # "ask @Alice about the invoice" means something different without her.
        activity = _activity(
            "<at>Akgentic</at> ask <at>Alice</at> about it",
            entities=[_mention(), _mention("<at>Alice</at>", mentioned_id="29:alice")],
        )
        message = await TeamsChannelParser().parse(activity)
        assert message.content == "ask <at>Alice</at> about it"

    async def test_falls_back_to_tag_stripping_without_entities(self) -> None:
        # Not a shape Teams should send, but raw markup must never reach a prompt.
        activity = _activity("<at>Akgentic</at> hello", entities=None)
        message = await TeamsChannelParser().parse(activity)
        assert message.content == "hello"

    async def test_mention_only_message_is_refused(self) -> None:
        activity = _activity("<at>Akgentic</at>", entities=[_mention()])
        with pytest.raises(ValueError, match="no text"):
            await TeamsChannelParser().parse(activity)


# ---------------------------------------------------------------------------
# serviceUrl
# ---------------------------------------------------------------------------


class TestServiceUrl:
    """Outbound delivery is impossible without it, and it arrives only inbound."""

    async def test_service_url_is_carried_as_binding_metadata(self) -> None:
        message = await TeamsChannelParser().parse(_activity())
        assert message.binding_metadata == {"service_url": SERVICE_URL}

    async def test_a_different_region_is_carried_distinctly(self) -> None:
        other = "https://smba.trafficmanager.net/apac/"
        message = await TeamsChannelParser().parse(_activity(service_url=other))
        assert message.binding_metadata == {"service_url": other}

    @pytest.mark.parametrize("value", [None, "", 42])
    async def test_absent_service_url_leaves_it_unset(self, value: object) -> None:
        message = await TeamsChannelParser().parse(_activity(service_url=value))
        assert message.binding_metadata is None


# ---------------------------------------------------------------------------
# Rejected activities
# ---------------------------------------------------------------------------


class TestRejectedActivities:
    @pytest.mark.parametrize(
        "activity_type",
        ["conversationUpdate", "messageReaction", "typing", "installationUpdate", "invoke"],
    )
    async def test_non_message_activities(self, activity_type: str) -> None:
        with pytest.raises(ValueError, match="not a message"):
            await TeamsChannelParser().parse(_activity(activity_type=activity_type))

    async def test_missing_conversation(self) -> None:
        with pytest.raises(ValueError, match="'conversation' field"):
            await TeamsChannelParser().parse(_activity(conversation_id=None))

    async def test_blank_conversation_id(self) -> None:
        with pytest.raises(ValueError, match="'id' field"):
            await TeamsChannelParser().parse(_activity(conversation_id=""))

    async def test_textless_activity_reports_its_keys(self) -> None:
        # A card action or file-only post — ordinary input, not a malformed
        # payload, and the keys say which.
        with pytest.raises(ValueError, match="value"):
            await TeamsChannelParser().parse(_activity(text=None, value={"action": "submit"}))


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_channel_parser(self) -> None:
        from akgentic.infra.protocols.channels import ChannelParser

        assert isinstance(TeamsChannelParser(), ChannelParser)
