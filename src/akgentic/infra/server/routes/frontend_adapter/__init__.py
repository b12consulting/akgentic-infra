"""Frontend adapter plugin system for the akgentic-infra server.

Defines the ``FrontendAdapter`` protocol and a dynamic loader utility that
resolves adapter classes by fully-qualified dotted name (FQDN) at server
start-up.  This allows external packages to supply their own route
translations and WebSocket event wrappers without modifying server code.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

from fastapi import FastAPI
from pydantic import BaseModel, Field

from akgentic.team.models import PersistedEvent

__all__ = ["FrontendAdapter", "WrappedWsEvent", "load_frontend_adapter"]


class WrappedWsEvent(BaseModel):
    """WebSocket event wrapped by a frontend adapter for client delivery.

    Encapsulates the adapter-transformed payload so that the return type
    crossing the adapter → WebSocket boundary is a proper Pydantic model
    rather than a raw dict.
    """

    payload: dict[str, Any] = Field(
        description="Adapter-transformed event data in the frontend's expected format",
    )


@runtime_checkable
class FrontendAdapter(Protocol):
    """Protocol for frontend compatibility adapters.

    Implementations translate V2 API routes and WebSocket events into
    formats expected by a specific frontend client (e.g. Angular V1).
    """

    def register_routes(self, app: FastAPI) -> None:
        """Mount adapter-specific HTTP routes onto the FastAPI application.

        Args:
            app: The FastAPI application instance to add routes to.
        """
        ...

    def wrap_ws_event(self, event: PersistedEvent) -> WrappedWsEvent:
        """Translate a persisted event into a frontend-specific payload.

        Args:
            event: The V2 persisted event to translate.

        Returns:
            A WrappedWsEvent containing the event in the frontend's expected format.
        """
        ...


def load_frontend_adapter(fqdn: str) -> FrontendAdapter:
    """Dynamically load a frontend adapter class by fully-qualified dotted name.

    Args:
        fqdn: Fully-qualified class name, e.g. ``"acme_corp.compat.AcmeAdapter"``.

    Returns:
        An instance of the adapter class that satisfies the ``FrontendAdapter``
        protocol.

    Raises:
        ImportError: If the module or class cannot be found.
        TypeError: If the resolved class does not implement ``FrontendAdapter``.
    """
    module_path, _, class_name = fqdn.rpartition(".")
    if not module_path:
        raise ImportError(f"Cannot load frontend adapter '{fqdn}': invalid FQDN (no module path)")

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(f"Cannot load frontend adapter '{fqdn}': {exc}") from exc

    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"Cannot load frontend adapter '{fqdn}': {exc}") from exc

    instance = cls()
    if not isinstance(instance, FrontendAdapter):
        raise TypeError(f"Class '{fqdn}' does not implement FrontendAdapter protocol")

    return instance
