"""Deprecated path — moved to ``akgentic.infra.adapters.channels.channel_parser_registry``.

Kept because ``akgentic-infra-enterprise`` imports ``ChannelParserRegistry`` and
``import_class`` from here, in its **source**, and that is a separate repository
with its own branch, PR and CI. Deleting this file is a cross-submodule change,
not a tidy-up.

To remove: update these three, then delete this file.

- ``enterprise/server/modules/webhook_delivery.py``
- ``enterprise/server/modules/channel_registry.py``
- ``enterprise/channels/parser_registry.py``
"""

from __future__ import annotations

from akgentic.infra.adapters.channels.channel_parser_registry import (
    ChannelConfig,
    ChannelParserRegistry,
    import_class,
)

__all__ = ["ChannelConfig", "ChannelParserRegistry", "import_class"]
