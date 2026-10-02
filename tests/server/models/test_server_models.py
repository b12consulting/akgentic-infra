"""Tests for REST API request/response models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from akgentic.core.messages.message import UserMessage
from pydantic import ValidationError

from akgentic.infra.server.models import (
    MAX_TEAM_DESCRIPTION_LENGTH,
    CreateTeamRequest,
    EmitMessageRequest,
    EventListResponse,
    EventResponse,
    HumanInputRequest,
    SendMessageRequest,
    TeamDescriptionResponse,
    TeamListResponse,
    TeamResponse,
    UpdateTeamDescriptionRequest,
)


def _team_response(**overrides: object) -> TeamResponse:
    """A minimal valid ``TeamResponse``; ``overrides`` set the optional tail fields."""
    now = datetime.now(tz=UTC)
    return TeamResponse(
        team_id=uuid.uuid4(),
        name="Test",
        status="running",
        user_id="anonymous",
        created_at=now,
        updated_at=now,
        **overrides,  # type: ignore[arg-type]
    )


def test_create_team_request_minimal() -> None:
    """CreateTeamRequest requires only catalog_namespace."""
    req = CreateTeamRequest(catalog_namespace="test-team")
    assert req.catalog_namespace == "test-team"
    assert req.params == {}


def test_create_team_request_with_params() -> None:
    """CreateTeamRequest accepts optional params."""
    req = CreateTeamRequest(
        catalog_namespace="test-team",
        params={"key": "value"},
    )
    assert req.params == {"key": "value"}


def test_create_team_request_metadata_defaults_to_none() -> None:
    """metadata is optional — omitting it keeps today's behaviour."""
    assert CreateTeamRequest(catalog_namespace="test-team").metadata is None


def test_create_team_request_accepts_plain_json_metadata() -> None:
    """metadata is carried verbatim; the model itself applies no schema.

    The schema comes from the team's catalog entry, resolved server-side — which
    is why the request model cannot type this field beyond raw JSON.
    """
    req = CreateTeamRequest(
        catalog_namespace="acme-cases",
        metadata={"tenant": "acme", "case": "C-1234"},
    )
    assert req.metadata == {"tenant": "acme", "case": "C-1234"}


def test_team_response_metadata_defaults_to_none() -> None:
    """metadata is optional and additive — a client that ignores it is unaffected."""
    now = datetime.now(tz=UTC)
    resp = TeamResponse(
        team_id=uuid.uuid4(),
        name="Test",
        status="running",
        user_id="anonymous",
        created_at=now,
        updated_at=now,
    )
    assert resp.metadata is None
    assert resp.model_dump(mode="json")["metadata"] is None


def test_team_response_metadata_round_trips() -> None:
    """A populated metadata value survives serialization unchanged."""
    now = datetime.now(tz=UTC)
    resp = TeamResponse(
        team_id=uuid.uuid4(),
        name="Test",
        status="running",
        user_id="anonymous",
        created_at=now,
        updated_at=now,
        metadata={"tenant": "acme", "owner": {"email": "ops@contoso.example"}},
    )
    dumped = resp.model_dump(mode="json")
    assert dumped["metadata"]["tenant"] == "acme"
    assert dumped["metadata"]["owner"]["email"] == "ops@contoso.example"


def test_team_response_catalog_namespace_defaults_to_none() -> None:
    """catalog_namespace is optional and additive, and null is a real answer.

    A team not created from a catalog genuinely has none, so ``null`` is the
    value — never the empty string, never an absent key.
    """
    now = datetime.now(tz=UTC)
    resp = TeamResponse(
        team_id=uuid.uuid4(),
        name="Test",
        status="running",
        user_id="anonymous",
        created_at=now,
        updated_at=now,
    )
    assert resp.catalog_namespace is None
    dumped = resp.model_dump(mode="json")
    assert dumped["catalog_namespace"] is None
    assert dumped["catalog_namespace"] != ""


def test_team_response_catalog_namespace_round_trips() -> None:
    """A populated namespace survives serialization unchanged."""
    now = datetime.now(tz=UTC)
    resp = TeamResponse(
        team_id=uuid.uuid4(),
        name="Test",
        status="running",
        user_id="anonymous",
        created_at=now,
        updated_at=now,
        catalog_namespace="acme-cases",
    )
    assert resp.model_dump(mode="json")["catalog_namespace"] == "acme-cases"


def test_team_response_field_order_appends_catalog_namespace_last() -> None:
    """The wire shape is pinned: the new field is appended, and nothing is reordered.

    The frontend reads this body by key and the team-list mapper must keep
    working against a server predating the field, so a field added at the end is
    additive while one inserted in the middle is a breaking change to anything
    positional. Asserted as an ordered list rather than a set, because a set
    cannot see a reorder at all.
    """
    assert list(TeamResponse.model_fields) == [
        "team_id",
        "name",
        "status",
        "user_id",
        "created_at",
        "updated_at",
        "metadata",
        "catalog_namespace",
        "description",
    ]


# --- Story 80.1: TeamResponse.description, the wire field ---


def test_team_response_description_defaults_to_none() -> None:
    """A team that never set a description reads ``null``, never ``""``."""
    resp = _team_response()
    assert resp.description is None
    dumped = resp.model_dump(mode="json")
    assert dumped["description"] is None
    assert dumped["description"] != ""


def test_team_response_description_round_trips() -> None:
    """A populated description survives serialization unchanged."""
    resp = _team_response(description="Triage inbound acme cases")
    assert resp.model_dump(mode="json")["description"] == "Triage inbound acme cases"


def test_team_response_description_is_appended_last_on_every_surface() -> None:
    """AC11: appended after ``catalog_namespace`` on the model, the dump and the schema.

    Three surfaces because a client may read any of them: the field registry is
    what Pydantic iterates, the dump is the JSON body, the schema is the OpenAPI
    document. An insert in the middle fails all three; a reorder of one fails
    that one.
    """
    tail = ["catalog_namespace", "description"]
    assert list(TeamResponse.model_fields)[-2:] == tail
    assert list(_team_response().model_dump())[-2:] == tail
    assert list(TeamResponse.model_json_schema()["properties"])[-2:] == tail


def test_team_response_does_not_carry_the_origin() -> None:
    """Only the PATCH response says who wrote the description."""
    assert "origin" not in TeamResponse.model_fields


# --- Story 80.1: UpdateTeamDescriptionRequest normalisation ---


def test_update_description_request_strips_surrounding_whitespace() -> None:
    req = UpdateTeamDescriptionRequest(description="  Triage inbound acme cases  ")
    assert req.description == "Triage inbound acme cases"


def test_update_description_request_maps_blank_to_none() -> None:
    """All-whitespace is a clear, not an empty string that would read as a description."""
    assert UpdateTeamDescriptionRequest(description="   \n\t ").description is None
    assert UpdateTeamDescriptionRequest(description="").description is None


def test_update_description_request_passes_none_through() -> None:
    assert UpdateTeamDescriptionRequest(description=None).description is None


def test_update_description_request_accepts_exactly_the_cap_after_trimming() -> None:
    body = " " + "x" * MAX_TEAM_DESCRIPTION_LENGTH + " "
    assert UpdateTeamDescriptionRequest(description=body).description == "x" * 500


def test_update_description_request_rejects_one_over_the_cap_after_trimming() -> None:
    """Measured after trimming: surrounding whitespace neither helps nor hurts."""
    with pytest.raises(ValidationError, match=str(MAX_TEAM_DESCRIPTION_LENGTH)):
        UpdateTeamDescriptionRequest(description=" " + "x" * 501 + " ")


def test_update_description_request_requires_the_key() -> None:
    """``{}`` is not a clear: an accidental empty body cannot wipe a description."""
    with pytest.raises(ValidationError):
        UpdateTeamDescriptionRequest()  # type: ignore[call-arg]


def test_update_description_request_cap_is_five_hundred() -> None:
    """The decision names the number; the constant is where it lives."""
    assert MAX_TEAM_DESCRIPTION_LENGTH == 500


# --- Story 80.1: TeamDescriptionResponse ---


def test_team_description_response_requires_all_three_fields() -> None:
    now = datetime.now(tz=UTC)
    with pytest.raises(ValidationError):
        TeamDescriptionResponse(description="x", origin="user")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        TeamDescriptionResponse(description="x", updated_at=now)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        TeamDescriptionResponse(origin="user", updated_at=now)  # type: ignore[call-arg]


def test_team_description_response_carries_a_null_description_and_the_stamp() -> None:
    """A clear is a real answer: ``null`` with an owner and the stamp the write set."""
    now = datetime.now(tz=UTC)
    resp = TeamDescriptionResponse(description=None, origin="user", updated_at=now)
    assert resp.model_dump(mode="json") == {
        "description": None,
        "origin": "user",
        "updated_at": now.isoformat().replace("+00:00", "Z"),
    }


def test_team_response_serialization() -> None:
    """TeamResponse serializes all fields correctly."""
    tid = uuid.uuid4()
    now = datetime.now(tz=UTC)
    resp = TeamResponse(
        team_id=tid,
        name="Test",
        status="running",
        user_id="anonymous",
        created_at=now,
        updated_at=now,
    )
    data = resp.model_dump(mode="json")
    assert data["team_id"] == str(tid)
    assert data["status"] == "running"


def test_team_list_response_empty() -> None:
    """TeamListResponse can hold an empty list with a zero total_count."""
    resp = TeamListResponse(teams=[], total_count=0)
    assert resp.teams == []
    assert resp.total_count == 0


def test_team_list_response_requires_total_count() -> None:
    """total_count is required — omitting it raises a ValidationError."""
    with pytest.raises(ValidationError):
        TeamListResponse(teams=[])  # type: ignore[call-arg]


def test_team_list_response_total_count_round_trips() -> None:
    """total_count round-trips through serialization (full owned count before paging)."""
    resp = TeamListResponse(teams=[], total_count=1234)
    assert resp.total_count == 1234
    assert resp.model_dump(mode="json")["total_count"] == 1234


def test_team_list_response_with_items() -> None:
    """TeamListResponse serializes a list of TeamResponses plus the total."""
    tid = uuid.uuid4()
    now = datetime.now(tz=UTC)
    item = TeamResponse(
        team_id=tid,
        name="A",
        status="running",
        user_id="u",
        created_at=now,
        updated_at=now,
    )
    resp = TeamListResponse(teams=[item], total_count=1)
    assert len(resp.teams) == 1
    assert resp.teams[0].team_id == tid
    assert resp.total_count == 1


def test_send_message_request_content_path() -> None:
    """SendMessageRequest accepts a plain content string (message left unset)."""
    req = SendMessageRequest(content="hello")
    assert req.content == "hello"
    assert req.message is None


def test_send_message_request_message_path() -> None:
    """SendMessageRequest accepts a serialized Message envelope (content left unset)."""
    serialized = UserMessage(content="typed").model_dump(mode="json")
    req = SendMessageRequest(message=serialized)
    assert req.message == serialized
    assert req.content is None


def test_send_message_request_rejects_neither() -> None:
    """Neither content nor message set violates the exactly-one validator."""
    with pytest.raises(ValidationError):
        SendMessageRequest()


def test_send_message_request_rejects_both() -> None:
    """Both content and message set violates the exactly-one validator."""
    serialized = UserMessage(content="typed").model_dump(mode="json")
    with pytest.raises(ValidationError):
        SendMessageRequest(content="hello", message=serialized)


def test_emit_message_request_round_trips_serialized_message() -> None:
    """EmitMessageRequest holds a serialized Message dict (with __model__) as message."""
    serialized = UserMessage(content="banner").model_dump(mode="json")
    req = EmitMessageRequest(message=serialized)
    assert req.message == serialized
    assert req.message["__model__"] == "akgentic.core.messages.message.UserMessage"


def test_emit_message_request_requires_message() -> None:
    """message is required — omitting it raises a ValidationError."""
    with pytest.raises(ValidationError):
        EmitMessageRequest()  # type: ignore[call-arg]


def test_human_input_request() -> None:
    """HumanInputRequest requires content and message_id."""
    req = HumanInputRequest(content="yes", message_id="msg-123")
    assert req.content == "yes"
    assert req.message_id == "msg-123"


def test_event_response_serialization() -> None:
    """EventResponse serializes all fields correctly."""
    tid = uuid.uuid4()
    now = datetime.now(tz=UTC)
    resp = EventResponse(
        team_id=tid,
        sequence=1,
        event={"type": "UserMessage", "content": "hello"},
        timestamp=now,
    )
    data = resp.model_dump(mode="json")
    assert data["team_id"] == str(tid)
    assert data["sequence"] == 1
    assert data["event"]["type"] == "UserMessage"


def test_event_list_response_empty() -> None:
    """EventListResponse can hold an empty list."""
    resp = EventListResponse(events=[])
    assert resp.events == []


def test_event_list_response_with_items() -> None:
    """EventListResponse serializes a list of EventResponses."""
    tid = uuid.uuid4()
    now = datetime.now(tz=UTC)
    item = EventResponse(
        team_id=tid,
        sequence=0,
        event={"type": "test"},
        timestamp=now,
    )
    resp = EventListResponse(events=[item])
    assert len(resp.events) == 1
    assert resp.events[0].team_id == tid
