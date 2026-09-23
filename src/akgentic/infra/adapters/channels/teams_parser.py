"""TeamsChannelParser — parses Bot Framework Activities into ChannelMessage."""

from __future__ import annotations

import logging
import re

from akgentic.infra.adapters.channels.channel_commands import parse_leading_command
from akgentic.infra.protocols.channels import ChannelMessage, JsonValue

logger = logging.getLogger(__name__)

# The only Activity type carrying something a team can act on. Teams also sends
# ``conversationUpdate`` (someone joined), ``messageReaction``, ``typing``,
# ``installationUpdate`` and ``invoke`` — none of which has text, and none of
# which a retry could turn into text.
MESSAGE_ACTIVITY = "message"

# Teams wraps every @-mention in ``<at>Display Name</at>``, including the one
# addressing the bot. Used only as a fallback: the entity list is authoritative
# and says *which* mention is the bot's.
_AT_TAG_RE = re.compile(r"<at\b[^>]*>.*?</at>", re.IGNORECASE | re.DOTALL)

_MENTION_ENTITY = "mention"


def _bot_mention_texts(entities: JsonValue, bot_id: JsonValue) -> list[str]:
    """Return the literal ``<at>…</at>`` spans that address the bot.

    Only the **bot's own** mention is a candidate for removal. Mentions of
    other people are part of what the user wrote and belong to the team: a
    message reading "ask @Alice about the invoice" means something different
    with the name taken out.

    Args:
        entities: The raw ``activity["entities"]`` value, of any shape.
        bot_id: The raw ``activity["recipient"]["id"]`` value — the bot, since
            an inbound activity is addressed *to* it.

    Returns:
        The exact substrings to remove, possibly empty.
    """
    if not isinstance(entities, list) or not isinstance(bot_id, str):
        return []
    texts: list[str] = []
    for entity in entities:
        if not isinstance(entity, dict) or entity.get("type") != _MENTION_ENTITY:
            continue
        mentioned = entity.get("mentioned")
        if not isinstance(mentioned, dict) or mentioned.get("id") != bot_id:
            continue
        text = entity.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
    return texts


def _strip_bot_mention(text: str, entities: JsonValue, bot_id: JsonValue) -> str:
    """Remove the bot's own @-mention from the text, leaving everything else.

    In a Teams channel or group chat the bot is only invoked when @-mentioned,
    so the mention is addressing, not content — it is present in *every*
    message and would otherwise be prepended to everything the team ever reads.

    Removing it also restores the router's leading-recipient rule: the user
    types ``@Akgentic @Expert look at this`` and the team must see
    ``@Expert look at this``, with ``@Expert`` **first**, or the message is not
    routed to the Expert at all.

    The entity list is authoritative. The regex fallback only runs when the
    activity carries no usable entities — a shape Teams is not supposed to
    send, but one that would otherwise leak raw ``<at>`` markup into an agent's
    prompt.

    Args:
        text: The activity's text, with markup.
        entities: The raw ``activity["entities"]`` value.
        bot_id: The raw ``activity["recipient"]["id"]`` value.

    Returns:
        The text without the bot's mention, stripped of the whitespace the
        removal leaves behind.
    """
    mentions = _bot_mention_texts(entities, bot_id)
    if mentions:
        for mention in mentions:
            text = text.replace(mention, " ")
    elif _AT_TAG_RE.search(text):
        logger.debug("No mention entity matched the bot; falling back to tag stripping")
        text = _AT_TAG_RE.sub(" ", text)
    return text.strip()


def _binding_metadata(service_url: JsonValue) -> dict[str, JsonValue] | None:
    """Carry the conversation's ``serviceUrl`` onto the binding, or None.

    **Outbound delivery is impossible without it.** Bot Framework does not
    publish one global endpoint: each conversation names the regional service
    that hosts it (``https://smba.trafficmanager.net/emea/`` and siblings), and
    the value arrives *only* on the inbound activity. Microsoft's own guidance
    is to store it per conversation, which is what the binding is.

    Args:
        service_url: The raw ``activity["serviceUrl"]`` value, of any shape.

    Returns:
        ``{"service_url": <url>}``, or None when the activity names none — in
        which case the adapter falls back to its configured default and says so.
    """
    if isinstance(service_url, str) and service_url:
        return {"service_url": service_url}
    return None


class TeamsChannelParser:
    """Parses one inbound Bot Framework Activity into a normalized ChannelMessage.

    Satisfies the ``ChannelParser`` protocol via structural subtyping.

    Teams is a **push** channel like Telegram: Bot Framework POSTs activities to
    the messaging endpoint configured on the Azure Bot resource. Unlike
    Telegram, that endpoint is set *in Azure* rather than by a call this process
    makes, so nothing here registers anything.

    Addressing:
        ``channel_user_id`` is ``conversation.id``, which identifies the
        **conversation** and not the person. A 1:1 chat gives each human their
        own conversation and therefore their own team; a Teams channel or group
        chat is one conversation shared by everyone in it, so it gets one
        binding and one agent seat — the same split as a Signal 1:1 versus a
        Signal group.

    Scopes:
        In ``personal`` scope every message reaches the bot. In ``team`` and
        ``groupChat`` scope Teams delivers only messages that @-mention it, and
        that mention is stripped here before the text reaches the team.

    No quoting:
        ``quoted_text`` is always None. A Teams reply carries ``replyToId`` but
        not the replied-to *text*, and fetching it would need Microsoft Graph.
        The practical consequence is that ``/register`` cannot read a team id
        out of a replied-to message here as it can on Telegram and Signal — the
        id has to be typed.

    Args:
        default_catalog_entry: Catalog entry ID for initiating new teams.
    """

    def __init__(self, default_catalog_entry: str = "default", **_kwargs: str) -> None:
        self._default_catalog_entry = default_catalog_entry

    @property
    def channel_name(self) -> str:
        """The channel name this parser handles."""
        return "teams"

    @property
    def default_catalog_entry(self) -> str:
        """Default catalog entry ID for new team initiation."""
        return self._default_catalog_entry

    async def parse(self, payload: dict[str, JsonValue]) -> ChannelMessage:
        """Parse a Bot Framework Activity into a ChannelMessage.

        Args:
            payload: Raw Activity JSON. Expected structure::

                {
                    "type": "message",
                    "id": "1616989574408",
                    "timestamp": "2026-09-23T05:06:14.408Z",
                    "serviceUrl": "https://smba.trafficmanager.net/emea/",
                    "channelId": "msteams",
                    "from": {"id": "29:1abc...", "name": "Geoff",
                             "aadObjectId": "8f1c..."},
                    "conversation": {"conversationType": "personal",
                                     "tenantId": "72f9...", "id": "a:1xyz..."},
                    "recipient": {"id": "28:<bot-app-id>", "name": "Akgentic"},
                    "text": "<at>Akgentic</at> hello",
                    "entities": [{"type": "mention",
                                  "mentioned": {"id": "28:<bot-app-id>"},
                                  "text": "<at>Akgentic</at>"}]
                }

        Returns:
            Parsed ChannelMessage with the bot's @-mention removed, the
            conversation as the chat id, the leading slash command when the
            text opens with one, and the conversation's ``serviceUrl`` as
            binding metadata.

        Raises:
            ValueError: If the activity is not a text message, or names no
                conversation to answer.
        """
        logger.debug("Teams parser payload: %s", payload)

        activity_type = payload.get("type")
        if activity_type != MESSAGE_ACTIVITY:
            msg = f"Teams activity is a {activity_type!r}, not a message"
            raise ValueError(msg)

        conversation = payload.get("conversation")
        if not isinstance(conversation, dict):
            msg = "Teams activity does not contain a 'conversation' field"
            raise ValueError(msg)

        conversation_id = conversation.get("id")
        if not isinstance(conversation_id, str) or not conversation_id:
            msg = "Teams conversation does not contain an 'id' field"
            raise ValueError(msg)

        raw_text = payload.get("text")
        if not isinstance(raw_text, str):
            # A card action or a file-only post: an ordinary Teams message with
            # nothing this parser can route, not a malformed payload.
            msg = f"Teams message carries no text (keys: {sorted(payload)})"
            raise ValueError(msg)

        recipient = payload.get("recipient")
        bot_id = recipient.get("id") if isinstance(recipient, dict) else None
        text = _strip_bot_mention(raw_text, payload.get("entities"), bot_id)
        if not text:
            # An @-mention and nothing else. The user got the bot's attention
            # and said nothing; forwarding an empty prompt makes an agent guess.
            msg = "Teams message is an @-mention with no text"
            raise ValueError(msg)

        message_id = payload.get("id")
        logger.debug("Parsing Teams activity: conversation=%s", conversation_id)
        return ChannelMessage(
            content=text,
            channel_user_id=conversation_id,
            channel_message_id=str(message_id) if message_id is not None else None,
            command=parse_leading_command(text),
            binding_metadata=_binding_metadata(payload.get("serviceUrl")),
        )
