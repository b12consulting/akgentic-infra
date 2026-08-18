"""Composition modules for the modular app assembly (epic 57).

Each module here implements the ``AppModule`` contract from
``akgentic.infra.server.assembly`` and is composed into a tier's app by
``build_app``. ``CoreModule`` is the community tier's base module; department
and enterprise adoption compositions build their module lists around it — an
override module (e.g. an enterprise route override) is simply listed before
it, everything else after.
"""

from __future__ import annotations

from akgentic.infra.server.modules.core import CoreModule

__all__ = ["CoreModule"]
