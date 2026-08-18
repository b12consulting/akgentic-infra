"""Composition modules for the modular app assembly (epic 57).

Each module here implements the ``AppModule`` contract from
``akgentic.infra.server.assembly`` and is composed into a tier's app by
``build_app``. ``CoreModule`` is the community tier's base module; department
and enterprise adoption compositions list it first and layer their own modules
around it.
"""

from __future__ import annotations

from akgentic.infra.server.modules.core import CoreModule

__all__ = ["CoreModule"]
