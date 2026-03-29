"""Tests for Angular V1 adapter — models and REST route translations (Story 3.2a)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import Message, ResultMessage, UserMessage
from akgentic.core.messages.orchestrator import (
    ReceivedMessage,
    SentMessage,
    StateChangedMessage,
)
from akgentic.team.models import PersistedEvent, Process, TeamCard, TeamStatus
from fastapi import FastAPI
from fastapi.testclient import TestClient

from akgentic.infra.server.app import _build_app
from akgentic.infra.server.routes.frontend_adapter import FrontendAdapter, WrappedWsEvent
from akgentic.infra.server.routes.frontend_adapter.angular_v1 import AngularV1Adapter
from akgentic.infra.server.routes.frontend_adapter.angular_v1.models import (
    V1ActorAddress,
    V1ConfigEntry,
    V1DescriptionBody,
    V1FeedbackEntry,
    V1LlmContextEntry,
    V1MessageEntry,
    V1ProcessContext,
    V1ProcessList,
    V1ProcessParams,
    V1StateEntry,
    V1StateUpdateBody,
    V1StatusResponse,
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
    """Create a minimal TeamCard mock with entry_point for V1 translation."""
    entry_point = MagicMock()
    entry_point.card.config.name = "@Orchestrator"
    card = MagicMock(spec=TeamCard)
    card.name = name
    card.entry_point = entry_point
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
    app = _build_app(mock_services, mock_team_service, settings)
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
        """V1ActorAddress serializes correctly with all fields."""
        addr = V1ActorAddress(name="@Human", role="Human")
        data = addr.model_dump()
        assert data["name"] == "@Human"
        assert data["role"] == "Human"
        assert data["address"] == ""
        assert data["agent_id"] == ""
        assert data["squad_id"] == ""
        assert data["user_message"] == ""
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
        assert hasattr(result.payload, "type")

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

    def test_get_states_with_state_events(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC #5: GET /states/{id} returns state changes in V1 format."""
        sender = MagicMock()
        sender.name = "@Manager"
        state_msg = StateChangedMessage(state=BaseState())
        state_msg.sender = sender
        mock_service.get_events.return_value = [
            _make_persisted_event(state_msg, sequence=1),
        ]
        resp = v1_client.get(f"/states/{_TEAM_ID}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["agent"] == "@Manager"
        assert isinstance(data[0]["state"], dict)
        assert data[0]["timestamp"] == _NOW.isoformat()


class TestV1MessageExtraction:
    """Test message content extraction and classification helper coverage."""

    def test_sent_message_content_extraction(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """SentMessage.message.content is extracted when outer has no content."""
        inner = UserMessage(content="inner content")
        recipient = MagicMock(spec=ActorAddress)
        recipient.name = "@Worker"
        sent = SentMessage(message=inner, recipient=recipient)
        mock_service.get_events.return_value = [
            _make_persisted_event(sent, sequence=1),
        ]
        resp = v1_client.get(f"/messages/{_TEAM_ID}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

    def test_result_message_classified_as_agent(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """ResultMessage events are classified as 'agent' type."""
        result_msg = ResultMessage(content="AI response")
        mock_service.get_events.return_value = [
            _make_persisted_event(result_msg, sequence=1),
        ]
        resp = v1_client.get(f"/messages/{_TEAM_ID}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["type"] == "agent"

    def test_llm_context_filters_non_content_events(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """GET /llm_context/{id} filters out events without displayable content."""
        user_msg = UserMessage(content="hello")
        received = ReceivedMessage(message_id=uuid.uuid4())
        mock_service.get_events.return_value = [
            _make_persisted_event(user_msg, sequence=1),
            _make_persisted_event(received, sequence=2),
        ]
        resp = v1_client.get(f"/llm_context/{_TEAM_ID}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["content"] == "hello"


class TestV1StatusResponseModel:
    """Test V1StatusResponse Pydantic model."""

    def test_v1_status_response_serialization(self) -> None:
        """V1StatusResponse serializes correctly."""
        resp = V1StatusResponse(status="ok")
        data = resp.model_dump()
        assert data == {"status": "ok"}
        restored = V1StatusResponse.model_validate(data)
        assert restored == resp


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


# ---------------------------------------------------------------------------
# Story 6.8: V1ProcessContext new fields (AC #1)
# ---------------------------------------------------------------------------


class TestV1ProcessContextNewFields:
    """Test V1ProcessContext new fields: orchestrator, running, config_name, user_id, user_email."""

    def test_new_fields_default_values(self) -> None:
        """New fields have sensible defaults."""
        ctx = V1ProcessContext(
            id=str(_TEAM_ID),
            type="Test",
            status="running",
            created_at=_NOW.isoformat(),
            updated_at=_NOW.isoformat(),
        )
        assert ctx.orchestrator == ""
        assert ctx.running is False
        assert ctx.config_name == ""
        assert ctx.user_id == ""
        assert ctx.user_email == ""

    def test_new_fields_populated(self) -> None:
        """New fields can be explicitly populated."""
        ctx = V1ProcessContext(
            id=str(_TEAM_ID),
            type="Test Team",
            status="running",
            created_at=_NOW.isoformat(),
            updated_at=_NOW.isoformat(),
            orchestrator="@Manager",
            running=True,
            config_name="my-team",
            user_id="user-1",
            user_email="user@example.com",
        )
        assert ctx.orchestrator == "@Manager"
        assert ctx.running is True
        assert ctx.config_name == "my-team"
        assert ctx.user_id == "user-1"
        assert ctx.user_email == "user@example.com"

    def test_new_fields_serialization_roundtrip(self) -> None:
        """V1ProcessContext with new fields survives roundtrip."""
        ctx = V1ProcessContext(
            id=str(_TEAM_ID),
            type="Test",
            status="running",
            created_at=_NOW.isoformat(),
            updated_at=_NOW.isoformat(),
            orchestrator="@Bot",
            running=True,
            config_name="team-cfg",
            user_id="u1",
            user_email="e@e.com",
        )
        data = ctx.model_dump()
        restored = V1ProcessContext.model_validate(data)
        assert restored == ctx

    def test_to_v1_process_context_populates_new_fields(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """GET /process/{id} response includes new fields from Process."""
        process = _make_process()
        mock_service.get_team.return_value = process
        resp = v1_client.get(f"/process/{_TEAM_ID}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["orchestrator"] == "@Orchestrator"
        assert data["running"] is True
        assert data["config_name"] == "Test Team"
        assert data["user_id"] == "anonymous"
        assert data["user_email"] == ""

    def test_running_false_when_stopped(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """Running field is False when team is stopped."""
        process = _make_process(status=TeamStatus.STOPPED)
        mock_service.get_team.return_value = process
        resp = v1_client.get(f"/process/{_TEAM_ID}")
        assert resp.status_code == 200
        assert resp.json()["running"] is False


# ---------------------------------------------------------------------------
# Story 6.8: V1ActorAddress new fields (AC #2)
# ---------------------------------------------------------------------------


class TestV1ActorAddressNewFields:
    """Test V1ActorAddress new fields."""

    def test_new_fields_default_values(self) -> None:
        """New fields default to empty strings."""
        addr = V1ActorAddress(name="@Agent", role="Worker")
        assert addr.actor_address == ""
        assert addr.address == ""
        assert addr.agent_id == ""
        assert addr.squad_id == ""
        assert addr.user_message == ""

    def test_new_fields_populated(self) -> None:
        """New fields can be explicitly populated."""
        addr = V1ActorAddress(
            name="@Agent",
            role="Worker",
            actor_address="agent-addr-1",
            address="agent-addr-1",
            agent_id="agent-1",
            squad_id=str(_TEAM_ID),
            user_message="hello",
        )
        assert addr.actor_address == "agent-addr-1"
        assert addr.address == "agent-addr-1"
        assert addr.agent_id == "agent-1"
        assert addr.squad_id == str(_TEAM_ID)
        assert addr.user_message == "hello"

    def test_alias_serialization(self) -> None:
        """V1ActorAddress serializes actor_address with alias __actor_address__."""
        addr = V1ActorAddress(name="@A", role="R", actor_address="addr-1")
        data = addr.model_dump(by_alias=True)
        assert "__actor_address__" in data
        assert data["__actor_address__"] == "addr-1"

    def test_alias_deserialization(self) -> None:
        """V1ActorAddress can be deserialized from __actor_address__ alias."""
        data = {
            "name": "@A",
            "role": "R",
            "__actor_address__": "addr-from-alias",
            "address": "",
            "agent_id": "",
            "squad_id": "",
            "user_message": "",
        }
        addr = V1ActorAddress.model_validate(data)
        assert addr.actor_address == "addr-from-alias"


# ---------------------------------------------------------------------------
# Story 6.8: New request/response models (AC #1, #2)
# ---------------------------------------------------------------------------


class TestNewRequestModels:
    """Test new request/response models added for story 6.8."""

    def test_v1_description_body(self) -> None:
        """V1DescriptionBody serializes correctly."""
        body = V1DescriptionBody(description="new desc")
        assert body.description == "new desc"
        data = body.model_dump()
        assert V1DescriptionBody.model_validate(data) == body

    def test_v1_state_update_body(self) -> None:
        """V1StateUpdateBody serializes correctly."""
        body = V1StateUpdateBody(content="new state")
        assert body.content == "new state"

    def test_v1_config_entry(self) -> None:
        """V1ConfigEntry serializes correctly."""
        entry = V1ConfigEntry(id="cfg-1", type="team", data={"name": "x"})
        data = entry.model_dump()
        restored = V1ConfigEntry.model_validate(data)
        assert restored == entry

    def test_v1_feedback_entry(self) -> None:
        """V1FeedbackEntry serializes correctly."""
        entry = V1FeedbackEntry(id="fb-1", content="great", rating=5)
        data = entry.model_dump()
        restored = V1FeedbackEntry.model_validate(data)
        assert restored == entry

    def test_v1_feedback_entry_defaults(self) -> None:
        """V1FeedbackEntry has sensible defaults."""
        entry = V1FeedbackEntry()
        assert entry.id == ""
        assert entry.content == ""
        assert entry.rating == 0


# ---------------------------------------------------------------------------
# Story 6.8: New REST endpoints (AC #3)
# ---------------------------------------------------------------------------


class TestV1DescriptionEndpoint:
    """Test PATCH /process/{id}/description endpoint."""

    def test_update_description_ok(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC3: PATCH /process/{id}/description returns ok."""
        mock_service.get_team.return_value = _make_process()
        resp = v1_client.patch(
            f"/process/{_TEAM_ID}/description",
            json={"description": "new desc"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_update_description_not_found(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """PATCH /process/{id}/description returns 404 for missing team."""
        mock_service.get_team.return_value = None
        resp = v1_client.patch(
            f"/process/{_TEAM_ID}/description",
            json={"description": "new desc"},
        )
        assert resp.status_code == 404


class TestV1RelaunchEndpoint:
    """Test POST /relaunch/{id}/message/{msgId} endpoint."""

    def test_relaunch_message_ok(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC3: POST /relaunch/{id}/message/{msgId} re-sends message."""
        msg = UserMessage(content="original message")
        mock_service.get_events.return_value = [
            _make_persisted_event(msg, sequence=1),
        ]
        mock_service.send_message.return_value = None
        resp = v1_client.post(f"/relaunch/{_TEAM_ID}/message/{msg.id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        mock_service.send_message.assert_called_once_with(_TEAM_ID, "original message")

    def test_relaunch_message_not_found_team(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """POST /relaunch/{id}/message/{msgId} returns 404 for unknown team."""
        mock_service.get_events.side_effect = ValueError("not found")
        resp = v1_client.post(f"/relaunch/{_TEAM_ID}/message/some-msg-id")
        assert resp.status_code == 404

    def test_relaunch_message_not_found_msg(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """POST /relaunch/{id}/message/{msgId} returns 404 for unknown msg."""
        mock_service.get_events.return_value = []
        resp = v1_client.post(f"/relaunch/{_TEAM_ID}/message/nonexistent")
        assert resp.status_code == 404


class TestV1StateUpdateEndpoint:
    """Test PATCH /state/{id}/of/{agent} endpoint."""

    def test_update_agent_state_ok(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC3: PATCH /state/{id}/of/{agent} sends to agent."""
        mock_handle = MagicMock()
        mock_service.get_handle.return_value = mock_handle
        resp = v1_client.patch(
            f"/state/{_TEAM_ID}/of/@Worker",
            json={"content": "update state"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        mock_handle.send_to.assert_called_once_with("@Worker", "update state")

    def test_update_agent_state_not_found(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """PATCH /state/{id}/of/{agent} returns 404 when no handle."""
        mock_service.get_handle.return_value = None
        resp = v1_client.patch(
            f"/state/{_TEAM_ID}/of/@Worker",
            json={"content": "update"},
        )
        assert resp.status_code == 404


class TestV1FeedbackEndpoints:
    """Test feedback stub endpoints."""

    def test_get_feedback_returns_empty(self, v1_client: TestClient) -> None:
        """AC3: GET /get-feedback returns empty list."""
        resp = v1_client.get("/get-feedback")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_set_feedback_returns_ok(self, v1_client: TestClient) -> None:
        """AC3: POST /set-feedback returns ok."""
        resp = v1_client.post(
            "/set-feedback",
            json={"id": "fb-1", "content": "great", "rating": 5},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestV1TeamConfigsEndpoint:
    """Test GET /team-configs endpoint."""

    def test_get_team_configs(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC3: GET /team-configs returns team catalog entries."""
        mock_entry = MagicMock()
        mock_entry.id = "team-1"
        mock_entry.model_dump.return_value = {"id": "team-1", "name": "My Team"}
        v1_client.app.state.services.team_catalog.list.return_value = [mock_entry]  # type: ignore[union-attr]
        resp = v1_client.get("/team-configs/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == "team-1"
        assert data[0]["type"] == "team"

    def test_get_team_configs_empty(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """GET /team-configs returns empty list when no entries."""
        v1_client.app.state.services.team_catalog.list.return_value = []  # type: ignore[union-attr]
        resp = v1_client.get("/team-configs/")
        assert resp.status_code == 200
        assert resp.json() == []


class TestV1ConfigEndpoints:
    """Test config CRUD endpoints."""

    def test_get_config_by_type(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC3: GET /config/{type} returns catalog entries."""
        mock_entry = MagicMock()
        mock_entry.id = "agent-1"
        mock_entry.model_dump.return_value = {"id": "agent-1", "name": "Bot"}
        v1_client.app.state.services.agent_catalog.list.return_value = [mock_entry]  # type: ignore[union-attr]
        resp = v1_client.get("/config/agent")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == "agent-1"
        assert data[0]["type"] == "agent"

    def test_get_config_unknown_type(self, v1_client: TestClient) -> None:
        """GET /config/{type} returns 400 for unknown type."""
        resp = v1_client.get("/config/unknown")
        assert resp.status_code == 400

    def test_delete_config_ok(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC3: DELETE /config deletes catalog entry."""
        mock_entry = MagicMock()
        v1_client.app.state.services.tool_catalog.get.return_value = mock_entry  # type: ignore[union-attr]
        resp = v1_client.request(
            "DELETE", "/config/",
            json={"id": "tool-1", "type": "tool", "data": {}},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_delete_config_not_found(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """DELETE /config returns 404 for missing entry."""
        v1_client.app.state.services.tool_catalog.get.return_value = None  # type: ignore[union-attr]
        resp = v1_client.request(
            "DELETE", "/config/",
            json={"id": "missing", "type": "tool", "data": {}},
        )
        assert resp.status_code == 404


class TestV1PutConfigEndpoint:
    """Test PUT /config endpoint."""

    def test_put_config_update_existing(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC3: PUT /config updates existing catalog entry."""
        mock_existing = MagicMock()
        mock_existing.__class__ = type(mock_existing)
        mock_existing.__class__.model_validate = MagicMock(return_value=mock_existing)
        v1_client.app.state.services.agent_catalog.get.return_value = mock_existing  # type: ignore[union-attr]
        resp = v1_client.put(
            "/config/",
            json={"id": "agent-1", "type": "agent", "data": {"name": "Bot"}},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        v1_client.app.state.services.agent_catalog.update.assert_called_once()  # type: ignore[union-attr]

    def test_put_config_create_new(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """AC3: PUT /config creates new catalog entry when not found."""
        mock_entry = MagicMock()
        mock_entry_cls = type(mock_entry)
        mock_entry_cls.model_validate = MagicMock(return_value=mock_entry)
        v1_client.app.state.services.agent_catalog.get.return_value = None  # type: ignore[union-attr]
        v1_client.app.state.services.agent_catalog.list.return_value = [mock_entry]  # type: ignore[union-attr]
        resp = v1_client.put(
            "/config/",
            json={"id": "new-1", "type": "agent", "data": {"name": "New"}},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        v1_client.app.state.services.agent_catalog.create.assert_called_once()  # type: ignore[union-attr]

    def test_put_config_empty_catalog_returns_400(
        self, v1_client: TestClient, mock_service: MagicMock,
    ) -> None:
        """PUT /config returns 400 when catalog is empty and entry type cannot be inferred."""
        v1_client.app.state.services.agent_catalog.get.return_value = None  # type: ignore[union-attr]
        v1_client.app.state.services.agent_catalog.list.return_value = []  # type: ignore[union-attr]
        resp = v1_client.put(
            "/config/",
            json={"id": "new-1", "type": "agent", "data": {}},
        )
        assert resp.status_code == 400


class TestV1AdapterNewRoutes:
    """Test that new routes are registered by AngularV1Adapter."""

    def test_new_routes_registered(self) -> None:
        """AngularV1Adapter registers all new route paths."""
        adapter = AngularV1Adapter()
        app = FastAPI()
        adapter.register_routes(app)
        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/config/{config_type}" in paths
        assert "/team-configs/" in paths
        assert "/get-feedback" in paths
        assert "/set-feedback" in paths
        assert "/relaunch/{id}/message/{msg_id}" in paths
        assert "/state/{id}/of/{agent}" in paths
        assert "/process/{id}/description" in paths
