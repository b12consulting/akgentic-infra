"""Angular V1 frontend adapter — REST endpoint translations.

Implements the FrontendAdapter protocol to translate V1 REST endpoints
to V2 service calls, allowing the existing Angular V1 frontend to
communicate with the V2 backend.
"""

from __future__ import annotations

from fastapi import FastAPI

from akgentic.infra.server.routes.frontend_adapter import WrappedWsEvent
from akgentic.infra.server.routes.frontend_adapter.angular_v1.router import (
    human_input_router,
    llm_context_router,
    messages_router,
    process_router,
    states_router,
)
from akgentic.infra.server.routes.frontend_adapter.angular_v1.ws import wrap_event
from akgentic.team.models import PersistedEvent

__all__ = ["AngularV1Adapter"]


class AngularV1Adapter:
    """Frontend adapter for the Angular V1 client.

    Satisfies the FrontendAdapter protocol by mounting V1-compatible
    REST routes and providing a minimal WebSocket event wrapper.
    """

    def register_routes(self, app: FastAPI) -> None:
        """Mount V1 REST routes onto the FastAPI application."""
        app.include_router(process_router)
        app.include_router(human_input_router)
        app.include_router(messages_router)
        app.include_router(llm_context_router)
        app.include_router(states_router)

    def wrap_ws_event(self, event: PersistedEvent) -> WrappedWsEvent:
        """Translate a V2 persisted event into a V1 WebSocket envelope."""
        return wrap_event(event)
