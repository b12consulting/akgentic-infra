"""Tests for Angular V1 adapter — models and REST route translations (Story 3.2a)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from akgentic.core.messages.message import Message, UserMessage
from akgentic.core.messages.orchestrator import (
    ReceivedMessage,
)
from akgentic.team.models import PersistedEvent, Process, TeamCard, TeamStatus
from fastapi import FastAPI
from fastapi.testclient import TestClient

from akgentic.infra.server.app import create_app
from akgentic.infra.server.routes.frontend_adapter import FrontendAdapter, WrappedWsEvent
from akgentic.infra.server.routes.frontend_adapter.angular_v1 import AngularV1Adapter
from akgentic.infra.server.routes.frontend_adapter.angular_v1.models import (
    V1ActorAddress,
    V1LlmContextEntry,
    V1MessageEntry,
    V1ProcessContext,
    V1ProcessList,
    V1ProcessParams,
    V1StateEntry,
)
from akgentic.infra.server.settings import ServerSettings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 28, 12, 0, 0, tzinfo=UTC)
_TEAM_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _make_agent_card(name: str = "@Human", role: str = "Human") -> Any:
    """Create a minimal AgentCard-like mock."""
    card = MagicMock()
    card.config.name = name
    card.role = role
    return card


def _make_team_card(name: str = "Test Team") -> MagicMock:
    """Create a minimal TeamCard mock."""
    card = MagicMock(spec=TeamCard)
    card.name = name
    return card


def _make_process(
    team_id: uuid.UUID = _TEAM_ID,
    status: TeamStatus = TeamStatus.RUNNING,
    name: str = "Test Team",
) -> Process:
    """Create a Process fixture."""
    team_card = _make_team_card(name)
    return Process(
        team_id=team_id,
        team_card=team_card,
        status=status,
        user_id="anonymous",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_persisted_event(
    event: Message,
    team_id: uuid.UUID = _TEAM_ID,
    sequence: int = 1,
) -> PersistedEvent:
    """Create a PersistedEvent fixture."""
    return PersistedEvent(
        team_id=team_id,
        sequence=sequence,
        event=event,
        timestamp=_NOW,
    )


@pytest.fixture()
def v1_client() -> TestClient:
    """TestClient with Angular V1 adapter loaded via mock services."""
    mock_services = MagicMock()
    mock_team_service = MagicMock()
    settings = ServerSettings(
        frontend_adapter="akgentic.infra.server.routes.frontend_adapter.angular_v1.AngularV1Adapter",
    )
    app = create_app(mock_services, mock_team_service, settings=settings)
    return TestClient(app)


@pytest.fixture()
def mock_service(v1_client: TestClient) -> MagicMock:
    """Extract the mock TeamService from the test client's app."""
    return v1_client.app.state.team_service  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Task 4: V1 response model tests (AC #8, #9)
# ---------------------------------------------------------------------------


class TestV1Models:
    """Verify V1 response models."""

    def test_v1_process_context_from_process(self) -> None:
        """V1ProcessContext can be constructed from V2 Process data."""
        ctx = V1ProcessContext(
            id=str(_TEAM_ID),
            type="Test Team",
            status="running",
            created_at=_NOW.isoformat(),
            updated_at=_NOW.isoformat(),
            params={},
        )
        assert ctx.id == str(_TEAM_ID)
        assert ctx.type == "Test Team"
        assert ctx.status == "running"
        assert ctx.params == {}

    def test_v1_process_context_serialization_roundtrip(self) -> None:
        """V1ProcessContext survives JSON serialization roundtrip."""
        ctx = V1ProcessContext(
            id=str(_TEAM_ID),
            type="Test Team",
            status="running",
            created_at=_NOW.isoformat(),
            updated_at=_NOW.isoformat(),
            params={},
        )
        data = ctx.model_dump()
        restored = V1ProcessContext.model_validate(data)
        assert restored == ctx

    def test_v1_message_entry_construction(self) -> None:
        """V1MessageEntry can be constructed from event data."""
        entry = V1MessageEntry(
            id=str(uuid.uuid4()),
            sender="@Human",
            content="hello",
            timestamp=_NOW.isoformat(),
            type="user",
        )
        assert entry.content == "hello"
        assert entry.type == "user"

    def test_v1_message_entry_serialization_roundtrip(self) -> None:
        """V1MessageEntry survives JSON roundtrip."""
        entry = V1MessageEntry(
            id=str(uuid.uuid4()),
            sender="@Manager",
            content="response",
            timestamp=_NOW.isoformat(),
            type="agent",
        )
        data = entry.model_dump()
        restored = V1MessageEntry.model_validate(data)
        assert restored == entry

    def test_v1_actor_address_serialization(self) -> None:
        """V1ActorAddress serializes correctly."""
        addr = V1ActorAddress(name="@Human", role="Human")
        data = addr.model_dump()
        assert data == {"name": "@Human", "role": "Human"}
        restored = V1ActorAddress.model_validate(data)
        assert restored == addr

    def test_v1_process_params_serialization(self) -> None:
        """V1ProcessParams serializes correctly."""
        params = V1ProcessParams(
            type="test-team",
            agents=[V1ActorAddress(name="@Human", role="Human")],
        )
        data = params.model_dump()
        restored = V1ProcessParams.model_validate(data)
        assert restored == params

    def test_v1_process_list_serialization(self) -> None:
        """V1ProcessList serializes correctly."""
        ctx = V1ProcessContext(
            id=str(_TEAM_ID),
            type="Test Team",
            status="running",
            created_at=_NOW.isoformat(),
            updated_at=_NOW.isoformat(),
        )
        pl = V1ProcessList(processes=[ctx])
        data = pl.model_dump()
        restored = V1ProcessList.model_validate(data)
        assert len(restored.processes) == 1
        assert restored.processes[0].id == str(_TEAM_ID)

    def test_v1_llm_context_entry_serialization(self) -> None:
        """V1LlmContextEntry serializes correctly."""
        entry = V1LlmContextEntry(
            role="user",
            content="hello",
            timestamp=_NOW.isoformat(),
        )
        data = entry.model_dump()
        restored = V1LlmContextEntry.model_validate(data)
        assert restored == entry
        assert restored.role == "user"

    def test_v1_state_entry_serialization(self) -> None:
        """V1StateEntry serializes correctly."""
        entry = V1StateEntry(
            agent="@Manager",
            state={"status": "active"},
            timestamp=_NOW.isoformat(),
        )
        data = entry.model_dump()
        restored = V1StateEntry.model_validate(data)
        assert restored == entry
        assert restored.agent == "@Manager"


# ---------------------------------------------------------------------------
# Task 3: AngularV1Adapter tests (AC #8)
# ---------------------------------------------------------------------------


class TestAngularV1Adapter:
    """Verify AngularV1Adapter satisfies FrontendAdapter protocol."""

    def test_satisfies_frontend_adapter_protocol(self) -> None:
        """AngularV1Adapter is recognized as a FrontendAdapter."""
        adapter = AngularV1Adapter()
        assert isinstance(adapter, FrontendAdapter)

    def test_register_routes_adds_v1_paths(self) -> None:
        """register_routes adds V1 route paths to the app."""
        adapter = AngularV1Adapter()
        app = FastAPI()
        adapter.register_routes(app)
        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/process/{type}" in paths
        assert "/process/" in paths or "/process" in paths
        assert "/messages/{id}" in paths
        assert "/llm_context/{id}" in paths
        assert "/states/{id}" in paths

    def test_wrap_ws_event_returns_wrapped_event(self) -> None:
        """wrap_ws_event returns a WrappedWsEvent with payload."""
        adapter = AngularV1Adapter()
        msg = UserMessage(content="hello")
        event = _make_persisted_event(msg)
        result = adapter.wrap_ws_event(event)
        assert isinstance(result, WrappedWsEvent)
        assert isinstance(result.payload, dict)

    def test_adapter_fqdn_loading(self) -> None:
        """Adapter can be loaded via FQDN through load_frontend_adapter."""
        from akgentic.infra.server.routes.frontend_adapter import load_frontend_adapter

        adapter = load_frontend_adapter(
            "akgentic.infra.server.routes.frontend_adapter.angular_v1.AngularV1Adapter"
        )
        assert isinstance(adapter, FrontendAdapter)


# ---------------------------------------------------------------------------
# Task 5: V1 REST route translation tests (AC #1-7, #9)
# ---------------------------------------------------------------------------


class TestV1ProcessRoutes:
    """Test V1 process endpoint translations."""

    def test_create_process(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC #1: POST /process/{type} creates team via V2 service."""
        mock_service.create_team.return_value = _make_process()
        resp = v1_client.post("/process/test-team")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == str(_TEAM_ID)
        assert data["type"] == "Test Team"
        assert data["status"] == "running"
        assert data["params"] == {}
        mock_service.create_team.assert_called_once_with(
            catalog_entry_id="test-team", user_id="anonymous",
        )

    def test_create_process_not_found(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """POST /process/{type} with unknown type returns 404."""
        from akgentic.catalog.models.errors import EntryNotFoundError

        mock_service.create_team.side_effect = EntryNotFoundError("bad-type")
        resp = v1_client.post("/process/bad-type")
        assert resp.status_code == 404

    def test_list_processes(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC #7: GET /process lists teams with V1 response shape."""
        mock_service.list_teams.return_value = [_make_process()]
        resp = v1_client.get("/process/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["processes"]) == 1
        assert data["processes"][0]["type"] == "Test Team"

    def test_list_processes_empty(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """GET /process returns empty list when no teams."""
        mock_service.list_teams.return_value = []
        resp = v1_client.get("/process/")
        assert resp.status_code == 200
        assert resp.json()["processes"] == []

    def test_get_process(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC #7: GET /process/{id} gets team with V1 response shape."""
        mock_service.get_team.return_value = _make_process()
        resp = v1_client.get(f"/process/{_TEAM_ID}")
        assert resp.status_code == 200
        assert resp.json()["id"] == str(_TEAM_ID)

    def test_get_process_not_found(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """GET /process/{id} returns 404 for unknown team."""
        mock_service.get_team.return_value = None
        resp = v1_client.get(f"/process/{_TEAM_ID}")
        assert resp.status_code == 404

    def test_send_message(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC #2: PATCH /process/{id} sends message via V2 service."""
        mock_service.send_message.return_value = None
        resp = v1_client.patch(
            f"/process/{_TEAM_ID}",
            json={"content": "hello"},
        )
        assert resp.status_code == 200
        mock_service.send_message.assert_called_once_with(_TEAM_ID, "hello")

    def test_send_message_not_found(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """PATCH /process/{id} returns 404 for unknown team."""
        mock_service.send_message.side_effect = ValueError("Team not found")
        resp = v1_client.patch(
            f"/process/{_TEAM_ID}",
            json={"content": "hello"},
        )
        assert resp.status_code == 404

    def test_send_message_not_running(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """PATCH /process/{id} returns 409 when team is not running."""
        mock_service.send_message.side_effect = ValueError("Team is not running")
        resp = v1_client.patch(
            f"/process/{_TEAM_ID}",
            json={"content": "hello"},
        )
        assert resp.status_code == 409

    def test_delete_process(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC #7: DELETE /process/{id} deletes team via V2 service."""
        mock_service.delete_team.return_value = None
        resp = v1_client.delete(f"/process/{_TEAM_ID}")
        assert resp.status_code == 200
        mock_service.delete_team.assert_called_once_with(_TEAM_ID)

    def test_delete_process_not_found(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """DELETE /process/{id} returns 404 for unknown team."""
        mock_service.delete_team.side_effect = ValueError("not found")
        resp = v1_client.delete(f"/process/{_TEAM_ID}")
        assert resp.status_code == 404

    def test_archive_process(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC #6: DELETE /process/{id}/archive stops team via V2 service."""
        mock_service.stop_team.return_value = None
        resp = v1_client.delete(f"/process/{_TEAM_ID}/archive")
        assert resp.status_code == 200
        mock_service.stop_team.assert_called_once_with(_TEAM_ID)

    def test_archive_process_not_found(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """DELETE /process/{id}/archive returns 404 for unknown team."""
        mock_service.stop_team.side_effect = ValueError("not found")
        resp = v1_client.delete(f"/process/{_TEAM_ID}/archive")
        assert resp.status_code == 404

    def test_archive_process_conflict(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """DELETE /process/{id}/archive returns 409 for already stopped team."""
        mock_service.stop_team.side_effect = ValueError("already stopped")
        resp = v1_client.delete(f"/process/{_TEAM_ID}/archive")
        assert resp.status_code == 409

    def test_restore_process(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC #7: POST /process/{id}/restore restores team via V2 service."""
        mock_service.restore_team.return_value = _make_process()
        resp = v1_client.post(f"/process/{_TEAM_ID}/restore")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == str(_TEAM_ID)
        mock_service.restore_team.assert_called_once_with(_TEAM_ID)

    def test_restore_process_not_found(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """POST /process/{id}/restore returns 404 for unknown team."""
        mock_service.restore_team.side_effect = ValueError("not found")
        resp = v1_client.post(f"/process/{_TEAM_ID}/restore")
        assert resp.status_code == 404

    def test_restore_process_conflict(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """POST /process/{id}/restore returns 409 when already running."""
        mock_service.restore_team.side_effect = ValueError("already running")
        resp = v1_client.post(f"/process/{_TEAM_ID}/restore")
        assert resp.status_code == 409


class TestV1HumanInputRoute:
    """Test V1 human input endpoint translation."""

    def test_process_human_input(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC #7: POST /process_human_input/{id}/human/{proxy} routes human input."""
        mock_service.process_human_input.return_value = None
        resp = v1_client.post(
            f"/process_human_input/{_TEAM_ID}/human/some-proxy",
            json={"content": "yes", "message_id": "msg-123"},
        )
        assert resp.status_code == 200
        mock_service.process_human_input.assert_called_once_with(
            _TEAM_ID, "yes", "msg-123",
        )

    def test_process_human_input_not_found(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """POST /process_human_input returns 404 for unknown team."""
        mock_service.process_human_input.side_effect = ValueError("not found")
        resp = v1_client.post(
            f"/process_human_input/{_TEAM_ID}/human/proxy",
            json={"content": "yes", "message_id": "msg-123"},
        )
        assert resp.status_code == 404


class TestV1MessagesRoute:
    """Test V1 messages endpoint translation."""

    def test_get_messages(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC #3: GET /messages/{id} returns event-sourced messages in V1 format."""
        user_msg = UserMessage(content="hello")
        mock_service.get_events.return_value = [
            _make_persisted_event(user_msg, sequence=1),
        ]
        resp = v1_client.get(f"/messages/{_TEAM_ID}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["content"] == "hello"
        assert data[0]["type"] == "user"

    def test_get_messages_not_found(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """GET /messages/{id} returns 404 for unknown team."""
        mock_service.get_events.side_effect = ValueError("not found")
        resp = v1_client.get(f"/messages/{_TEAM_ID}")
        assert resp.status_code == 404

    def test_get_messages_filters_non_content_events(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """GET /messages/{id} filters out events without content."""
        user_msg = UserMessage(content="hello")
        received = ReceivedMessage(message_id=uuid.uuid4())
        mock_service.get_events.return_value = [
            _make_persisted_event(user_msg, sequence=1),
            _make_persisted_event(received, sequence=2),
        ]
        resp = v1_client.get(f"/messages/{_TEAM_ID}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["content"] == "hello"


class TestV1LlmContextRoute:
    """Test V1 LLM context endpoint translation."""

    def test_get_llm_context(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC #4: GET /llm_context/{id} returns LLM context in V1 format."""
        user_msg = UserMessage(content="hello")
        mock_service.get_events.return_value = [
            _make_persisted_event(user_msg, sequence=1),
        ]
        resp = v1_client.get(f"/llm_context/{_TEAM_ID}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["role"] == "user"
        assert data[0]["content"] == "hello"

    def test_get_llm_context_not_found(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """GET /llm_context/{id} returns 404 for unknown team."""
        mock_service.get_events.side_effect = ValueError("not found")
        resp = v1_client.get(f"/llm_context/{_TEAM_ID}")
        assert resp.status_code == 404


class TestV1StatesRoute:
    """Test V1 states endpoint translation."""

    def test_get_states_not_found(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC #5: GET /states/{id} returns 404 for unknown team."""
        mock_service.get_events.side_effect = ValueError("not found")
        resp = v1_client.get(f"/states/{_TEAM_ID}")
        assert resp.status_code == 404

    def test_get_states_empty(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """GET /states/{id} returns empty list when no state events."""
        user_msg = UserMessage(content="hello")
        mock_service.get_events.return_value = [
            _make_persisted_event(user_msg, sequence=1),
        ]
        resp = v1_client.get(f"/states/{_TEAM_ID}")
        assert resp.status_code == 200
        assert resp.json() == []


class TestV1ErrorCases:
    """Test error handling across V1 routes."""

    def test_invalid_uuid_process_get(self, v1_client: TestClient) -> None:
        """Invalid UUID in path returns 422."""
        resp = v1_client.get("/process/not-a-uuid")
        assert resp.status_code == 422

    def test_invalid_uuid_messages(self, v1_client: TestClient) -> None:
        """Invalid UUID in messages path returns 422."""
        resp = v1_client.get("/messages/not-a-uuid")
        assert resp.status_code == 422

    def test_invalid_uuid_llm_context(self, v1_client: TestClient) -> None:
        """Invalid UUID in llm_context path returns 422."""
        resp = v1_client.get("/llm_context/not-a-uuid")
        assert resp.status_code == 422

    def test_invalid_uuid_states(self, v1_client: TestClient) -> None:
        """Invalid UUID in states path returns 422."""
        resp = v1_client.get("/states/not-a-uuid")
        assert resp.status_code == 422
