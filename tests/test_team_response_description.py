"""``TeamResponse.description``, held against BOTH producers.

``TeamResponse`` is produced twice — once in ``server/routes/teams.py`` and once
in ``worker/routes/teams.py`` — by two ``_process_to_response`` functions that
share a name and a job and import nothing from each other. A field filled on one
side and left at its default on the other reports ``null`` for every team that
side serves, and nothing on the server goes red when only the server producer
is updated, because no server test ever calls the worker's function.

These specs are the executable half of that rule for the description, written
as ``tests/test_team_response_name.py`` is for the name.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from akgentic.team.models import AgentCardRef, AgentRef, Process, TeamStatus

from akgentic.infra.server.models import TeamResponse
from akgentic.infra.server.routes.teams import _process_to_response as server_to_response
from akgentic.infra.worker.routes.teams import _process_to_response as worker_to_response

_PRODUCERS = pytest.mark.parametrize(
    "to_response",
    [server_to_response, worker_to_response],
    ids=["server", "worker"],
)


def _process(*, team_description: str | None) -> Process:
    """A minimal persisted ``Process`` carrying only what the producers read."""
    now = datetime.now(UTC)
    return Process(
        team_id=uuid.uuid4(),
        status=TeamStatus.RUNNING,
        user_id="user-1",
        created_at=now,
        updated_at=now,
        entry_point=AgentRef(name="@Manager", role="Manager"),
        agent_cards=[AgentCardRef(role="Manager", card_hash="0" * 64)],
        team_name="Case Triage",
        team_description=team_description,
    )


@_PRODUCERS
def test_a_described_team_reports_its_description(
    to_response: Callable[[Process], TeamResponse],
) -> None:
    """AC12: ``process.team_description`` reaches the wire on both producers."""
    process = _process(team_description="Case Triage")

    assert to_response(process).description == "Case Triage"


@_PRODUCERS
def test_an_undescribed_team_reports_null_not_empty(
    to_response: Callable[[Process], TeamResponse],
) -> None:
    """AC3 / AC12: ``None`` is a real answer and travels as ``null``."""
    process = _process(team_description=None)

    response = to_response(process)
    assert response.description is None
    assert response.model_dump(mode="json")["description"] is None
