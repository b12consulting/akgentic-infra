"""Tests for LocalIngestion — community-tier InteractionChannelIngestion."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from akgentic.catalog.models.errors import CatalogValidationError, EntryNotFoundError
from akgentic.core.messages.message import UserMessage

from akgentic.infra.adapters.community.local_ingestion import LocalIngestion
from akgentic.infra.errors import MetadataValidationError
from akgentic.infra.protocols.channels import (
    InitiatedTeam,
    InteractionChannelIngestion,
    JsonValue,
)
from akgentic.infra.server.services.team_service import TeamService


def _make_mock_service() -> MagicMock:
    """Create a MagicMock TeamService with predictable return values."""
    mock = MagicMock(spec=TeamService)
    return mock


def _make_process_stub(
    team_id: uuid.UUID,
    entry_point_name: str = "@HumanProxy_0",
    entry_point_role: str = "human_proxy",
) -> MagicMock:
    """Create a stub Process with the given team_id and entry point.

    ``name`` is assigned after construction rather than passed to ``MagicMock``:
    the constructor treats ``name`` as the mock's own label, so
    ``MagicMock(name=...)`` would leave ``process.entry_point.name`` a child
    mock and the assertions below would compare two mocks.
    """
    process = MagicMock()
    process.team_id = team_id
    process.entry_point.name = entry_point_name
    process.entry_point.role = entry_point_role
    return process


async def test_local_ingestion_satisfies_protocol() -> None:
    """LocalIngestion structurally satisfies InteractionChannelIngestion."""
    mock_service = _make_mock_service()
    ingestion = LocalIngestion(mock_service)
    assert isinstance(ingestion, InteractionChannelIngestion)


async def test_send_message_calls_send_message() -> None:
    """send_message() delegates to team_service.send_message()."""
    mock_service = _make_mock_service()
    ingestion = LocalIngestion(mock_service)
    team_id = uuid.uuid4()

    await ingestion.send_message(team_id, "hello team")

    mock_service.send_message.assert_called_once_with(team_id, "hello team")


async def test_send_message_with_original_message_id() -> None:
    """send_message() with original_message_id still calls send_message."""
    mock_service = _make_mock_service()
    ingestion = LocalIngestion(mock_service)
    team_id = uuid.uuid4()

    await ingestion.send_message(team_id, "threaded reply", original_message_id="msg-123")

    mock_service.send_message.assert_called_once_with(team_id, "threaded reply")


async def test_create_team_creates_and_sends_nothing() -> None:
    """create_team() creates the team and sends it nothing — the caller binds first."""
    mock_service = _make_mock_service()
    new_team_id = uuid.uuid4()
    mock_service.create_team.return_value = _make_process_stub(new_team_id)
    ingestion = LocalIngestion(mock_service)

    result = await ingestion.create_team("user-42", "catalog-entry-1")

    mock_service.create_team.assert_called_once_with(
        "catalog-entry-1", user_id="user-42", metadata=None
    )
    mock_service.send_message.assert_not_called()
    assert result.team_id == new_team_id


async def test_create_team_returns_correct_uuid() -> None:
    """create_team() returns the UUID from the created process."""
    mock_service = _make_mock_service()
    expected_id = uuid.uuid4()
    mock_service.create_team.return_value = _make_process_stub(expected_id)
    ingestion = LocalIngestion(mock_service)

    result = await ingestion.create_team("user-1", "entry-1")

    assert result.team_id == expected_id
    assert isinstance(result.team_id, uuid.UUID)


async def test_create_team_returns_an_initiated_team() -> None:
    """create_team() answers with the model, not a bare id."""
    mock_service = _make_mock_service()
    mock_service.create_team.return_value = _make_process_stub(uuid.uuid4())
    ingestion = LocalIngestion(mock_service)

    result = await ingestion.create_team("user-1", "entry-1")

    assert isinstance(result, InitiatedTeam)


async def test_create_team_reports_the_entry_point_name_not_its_role() -> None:
    """entry_point_name is the spawned name — the key into the team's addresses.

    A role is shared by every member hired from the same card, so binding a
    channel to one would route an outbound message to whichever member the
    lookup happened to find. Reading ``entry_point.role`` here passes every
    other spec in this file; only this one says which attribute is meant.
    """
    mock_service = _make_mock_service()
    mock_service.create_team.return_value = _make_process_stub(
        uuid.uuid4(),
        entry_point_name="@HumanProxy_0",
        entry_point_role="human_proxy",
    )
    ingestion = LocalIngestion(mock_service)

    result = await ingestion.create_team("user-1", "entry-1")

    assert result.entry_point_name == "@HumanProxy_0"


async def test_create_team_reads_the_entry_point_off_the_process_it_already_has() -> None:
    """No second service call: create_team and send_message are the whole of it."""
    mock_service = _make_mock_service()
    mock_service.create_team.return_value = _make_process_stub(uuid.uuid4())
    ingestion = LocalIngestion(mock_service)

    await ingestion.create_team("user-1", "entry-1")

    called = [name for name, _args, _kwargs in mock_service.method_calls]
    assert called == ["create_team"]


async def test_create_team_forwards_metadata() -> None:
    """create_team() hands the metadata mapping to create_team untouched."""
    mock_service = _make_mock_service()
    mock_service.create_team.return_value = _make_process_stub(uuid.uuid4())
    ingestion = LocalIngestion(mock_service)
    metadata: dict[str, JsonValue] = {"tenant": "acme", "case": {"id": 7, "tags": ["a"]}}

    await ingestion.create_team("user-42", "catalog-entry-1", metadata)

    mock_service.create_team.assert_called_once_with(
        "catalog-entry-1", user_id="user-42", metadata=metadata
    )


async def test_create_team_propagates_metadata_validation_error() -> None:
    """A refused metadata body leaves create_team uncaught.

    ``MetadataValidationError`` is a ``ServerError`` carrying its own 422; a
    local ``except`` here would replace that answer with whatever this layer
    invented, on the channel surface only.
    """
    mock_service = _make_mock_service()
    mock_service.create_team.side_effect = MetadataValidationError("case.id must be an integer")
    ingestion = LocalIngestion(mock_service)

    with pytest.raises(MetadataValidationError, match="case.id must be an integer"):
        await ingestion.create_team("user-1", "entry-1", {"case": {"id": "seven"}})


async def test_send_message_passes_a_preformed_message_through_unchanged() -> None:
    """A pre-formed Message reaches send_message as the same object.

    Identity, not equality: a ``str(content)`` coercion would still produce an
    argument that compares equal to nothing useful, and any re-construction
    would silently drop what the typed message carries.
    """
    mock_service = _make_mock_service()
    ingestion = LocalIngestion(mock_service)
    team_id = uuid.uuid4()
    message = UserMessage(content="typed reply")

    await ingestion.send_message(team_id, message)

    sent = mock_service.send_message.call_args.args[1]
    assert sent is message


async def test_send_message_propagates_value_error() -> None:
    """send_message() propagates ValueError from team_service.send_message()."""
    mock_service = _make_mock_service()
    mock_service.send_message.side_effect = ValueError("Team not found")
    ingestion = LocalIngestion(mock_service)
    team_id = uuid.uuid4()

    with pytest.raises(ValueError, match="Team not found"):
        await ingestion.send_message(team_id, "hello")


async def test_create_team_lets_the_catalog_diagnosis_through(
    team_service: TeamService,
) -> None:
    """An invalid stored namespace reaches the app-level handler, message intact.

    ``create_team`` catches nothing and the webhook route catches nothing
    either, so this path answers 409 with the catalog's own text where it used
    to answer 404 — a free improvement from the create_team split, and one no
    other test holds. Add a local ``except`` here and the diagnosis is destroyed
    again on the channel surface only, silently.

    Driven against the real wired service and the on-disk broken namespace: a
    mocked ``create_team`` with a side effect would prove nothing about which
    exception the catalog actually raises.
    """
    ingestion = LocalIngestion(team_service)

    with pytest.raises(CatalogValidationError, match="ref marker"):
        await ingestion.create_team("user-42", "broken-team")


async def test_create_team_on_a_teamless_namespace_stays_a_not_found(
    team_service: TeamService,
) -> None:
    """A namespace with no team entry stays in the 404 family on this path too.

    ``CatalogTeamEntryMissingError`` subclasses ``EntryNotFoundError`` precisely
    so the catalog's app-level 404 handler keeps serving it here unchanged; only
    the message gets sharper.
    """
    ingestion = LocalIngestion(team_service)

    with pytest.raises(EntryNotFoundError, match="has no team entry"):
        await ingestion.create_team("user-42", "teamless")
