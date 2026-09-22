"""Channel parser registry — resolves and holds channel parsers/adapters from config."""

from __future__ import annotations

import importlib
import logging

from pydantic import BaseModel, Field

from akgentic.infra.adapters.shared.channel_router import DefaultChannelRouter
from akgentic.infra.protocols.channels import (
    ChannelParser,
    InteractionChannelAdapter,
    InteractionChannelRouter,
)

logger = logging.getLogger(__name__)


class ChannelConfig(BaseModel):
    """Configuration for a single interaction channel."""

    parser_fqcn: str = Field(description="Fully-qualified class name of the ChannelParser")
    router_fqcn: str | None = Field(
        default=None,
        description=(
            "Fully-qualified class name of the InteractionChannelRouter; "
            "DefaultChannelRouter when unset"
        ),
    )
    adapter_fqcn: str = Field(
        description="Fully-qualified class name of the InteractionChannelAdapter"
    )
    config: dict[str, str] = Field(
        default_factory=dict,
        description="Extra kwargs passed to parser, router and adapter constructors",
    )


def import_class(fqcn: str) -> type:
    """Dynamically import a class by its fully-qualified dotted name.

    Args:
        fqcn: Fully-qualified class name, e.g. "acme.parsers.WhatsAppParser".

    Returns:
        The resolved class object.

    Raises:
        ImportError: If the module or class cannot be found.
    """
    module_path, _, class_name = fqcn.rpartition(".")
    if not module_path:
        msg = f"Invalid FQCN '{fqcn}': no module path"
        raise ImportError(msg)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        msg = f"Module '{module_path}' not found for FQCN '{fqcn}'"
        raise ImportError(msg) from exc
    try:
        cls: type = getattr(module, class_name)
        return cls
    except AttributeError as exc:
        msg = f"Class '{class_name}' not found in module '{module_path}'"
        raise ImportError(msg) from exc


def _load_router(fqcn: str, config: dict[str, str]) -> InteractionChannelRouter:
    """Resolve and instantiate the router a channel names."""
    router = import_class(fqcn)(**config)
    if not isinstance(router, InteractionChannelRouter):
        msg = f"Class '{fqcn}' does not satisfy InteractionChannelRouter protocol"
        raise TypeError(msg)
    return router


class ChannelParserRegistry:
    """Resolves FQCNs from channel configuration and holds parsers/routers/adapters.

    Parsers and routers are indexed by ``channel_name``; adapters are collected
    into a list for use by ``InteractionChannelDispatcher`` (story 4.2).
    """

    def __init__(self, channels_config: dict[str, ChannelConfig]) -> None:
        self._parsers: dict[str, ChannelParser] = {}
        self._routers: dict[str, InteractionChannelRouter] = {}
        self._adapters: list[InteractionChannelAdapter] = []
        self._default_router: InteractionChannelRouter = DefaultChannelRouter()
        self._load(channels_config)

    def _load(self, channels_config: dict[str, ChannelConfig]) -> None:
        """Resolve FQCNs and instantiate parsers, routers and adapters."""
        for _channel_key, cfg in channels_config.items():
            logger.info(
                "Loading channel: %s (parser=%s, router=%s, adapter=%s)",
                _channel_key,
                cfg.parser_fqcn,
                cfg.router_fqcn,
                cfg.adapter_fqcn,
            )
            parser_cls = import_class(cfg.parser_fqcn)
            adapter_cls = import_class(cfg.adapter_fqcn)

            parser = parser_cls(**cfg.config)
            if not isinstance(parser, ChannelParser):
                msg = f"Class '{cfg.parser_fqcn}' does not satisfy ChannelParser protocol"
                raise TypeError(msg)

            adapter = adapter_cls(**cfg.config)
            if not isinstance(adapter, InteractionChannelAdapter):
                msg = (
                    f"Class '{cfg.adapter_fqcn}' does not satisfy "
                    f"InteractionChannelAdapter protocol"
                )
                raise TypeError(msg)

            self._parsers[parser.channel_name] = parser
            if cfg.router_fqcn is not None:
                self._routers[parser.channel_name] = _load_router(cfg.router_fqcn, cfg.config)
            self._adapters.append(adapter)
        logger.debug("Channel parser registry loaded: %d channel(s)", len(self._parsers))

    def get_parser(self, channel_name: str) -> ChannelParser | None:
        """Return the parser for the given channel name, or None."""
        parser = self._parsers.get(channel_name)
        logger.debug("Parser lookup: channel=%s, found=%s", channel_name, parser is not None)
        return parser

    def get_router(self, channel_name: str) -> InteractionChannelRouter:
        """Return the channel's router, or the default router when it names none.

        Never None: a channel with a parser always routes. A subclass that
        registers parsers without routers therefore keeps today's behaviour
        rather than losing every channel to a missing router.
        """
        return self._routers.get(channel_name, self._default_router)

    def get_adapters(self) -> list[InteractionChannelAdapter]:
        """Return all resolved adapters."""
        return list(self._adapters)

    def channel_names(self) -> list[str]:
        """Return all registered channel names."""
        return list(self._parsers.keys())
