"""Deprecated path — moved to ``akgentic.infra.adapters.channels.telegram_parser``.

Kept because this module path appears as a **configuration string** in
``akgentic-infra-enterprise``, where it is resolved at runtime by ``import_class``
rather than by an import statement — so nothing would fail at build time and the
break would surface as a channel that silently stops parsing.

To remove: update these two, then delete this file.

- ``enterprise/deploy/k8s/helm/values.yuma-vanilla.yaml``
- ``enterprise/tests/channels/test_bootstrap_startup_fallback.py``
"""

from __future__ import annotations

from akgentic.infra.adapters.channels.telegram_parser import TelegramChannelParser

__all__ = ["TelegramChannelParser"]
