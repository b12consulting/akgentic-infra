"""Interaction channels — everything that carries a message to and from a human.

One place for the whole channel layer, rather than scattered through
``adapters.shared``:

- the **seam**: ``ChannelParserRegistry`` resolves a channel's parser, router
  and adapter from config, ``DefaultChannelRouter`` decides what an inbound
  message does, and ``InteractionChannelDispatcher`` carries outbound messages
  back.
- the **channels**: a parser and an adapter per service — Telegram, Signal,
  Microsoft Teams.
- the **shared rules** a markup-less channel needs, such as reading a leading
  ``/command`` out of plain text.

These are tier-agnostic. The same concrete classes serve the community,
department and enterprise profiles; what differs between tiers is the registry
that stores bindings, not the channel.

``adapters.shared`` keeps what is not a channel — policies and event
subscribers — and re-exports three modules from here for the sibling
deployment packages that still import them by the old path.
"""

from __future__ import annotations

from akgentic.infra.adapters.channels.channel_commands import parse_leading_command
from akgentic.infra.adapters.channels.channel_dispatcher import InteractionChannelDispatcher
from akgentic.infra.adapters.channels.channel_parser_registry import (
    ChannelConfig,
    ChannelParserRegistry,
    import_class,
)
from akgentic.infra.adapters.channels.channel_router import (
    ChannelRouteContext,
    DefaultChannelRouter,
)
from akgentic.infra.adapters.channels.signal_adapter import SignalChannelAdapter
from akgentic.infra.adapters.channels.signal_parser import SignalChannelParser
from akgentic.infra.adapters.channels.teams_adapter import TeamsChannelAdapter
from akgentic.infra.adapters.channels.teams_parser import TeamsChannelParser
from akgentic.infra.adapters.channels.telegram_adapter import TelegramChannelAdapter
from akgentic.infra.adapters.channels.telegram_parser import TelegramChannelParser

__all__ = [
    "ChannelConfig",
    "ChannelParserRegistry",
    "ChannelRouteContext",
    "DefaultChannelRouter",
    "InteractionChannelDispatcher",
    "SignalChannelAdapter",
    "SignalChannelParser",
    "TeamsChannelAdapter",
    "TeamsChannelParser",
    "TelegramChannelAdapter",
    "TelegramChannelParser",
    "import_class",
    "parse_leading_command",
]
