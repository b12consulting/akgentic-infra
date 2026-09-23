"""Deprecated path — moved to ``akgentic.infra.adapters.channels.channel_router``.

Kept because ``akgentic-infra-department`` imports from here in two tests, and
that is a separate repository with its own branch, PR and CI.

To remove: update these two, then delete this file.

- ``department/tests/unit/test_channel_reply_reaches_worker.py``
- ``department/tests/integration/test_webhook_to_agent_reply.py``
"""

from __future__ import annotations

from akgentic.infra.adapters.channels.channel_router import (
    ChannelRouteContext,
    DefaultChannelRouter,
    NoDefaultRecipientError,
)

__all__ = ["ChannelRouteContext", "DefaultChannelRouter", "NoDefaultRecipientError"]
