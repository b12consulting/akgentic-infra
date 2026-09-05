"""The ``TeamResponse`` name fallback, held against BOTH producers.

``TeamResponse.name`` is produced twice — once in ``server/routes/teams.py`` and
once in ``worker/routes/teams.py`` — by two ``_process_to_response`` functions
that share a name and a job and import nothing from each other. Both resolve the
name as ``process.team_name or process.catalog_namespace or str(process.team_id)``.

All three rungs are reachable, because ``Process.team_name`` is ``str | None``:
a team projected off a nameless card carries ``None`` there, and a team created
outside a catalog carries ``None`` in ``catalog_namespace`` too. Yet the whole
suite only ever asserted the first rung, on a team whose card is named
``"Test Team"``. Collapsing the expression to ``process.team_name`` — the
obvious "simplification" once the nested card is gone — therefore left every
test green while a nameless team began reporting an empty name to the API.

These specs are the executable half of that rule, parametrized over both
producers so a fallback repaired on one side and dropped on the other cannot
pass.
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


def _process(*, team_name: str | None, catalog_namespace: str | None) -> Process:
    """A minimal persisted ``Process`` carrying only what the producers read.

    Built through the real constructor rather than ``model_construct``: the
    model validates that the entry point's role is one the projection actually
    carries a card ref for, so a hand-built ``Process`` has to be internally
    coherent to exist at all. Everything past that pair is irrelevant to the
    name and stays at its default.
    """
    now = datetime.now(UTC)
    return Process(
        team_id=uuid.uuid4(),
        status=TeamStatus.RUNNING,
        user_id="user-1",
        created_at=now,
        updated_at=now,
        entry_point=AgentRef(name="@Manager", role="Manager"),
        agent_cards=[AgentCardRef(role="Manager", card_hash="0" * 64)],
        team_name=team_name,
        catalog_namespace=catalog_namespace,
    )


@_PRODUCERS
def test_a_named_team_reports_its_projected_name(
    to_response: Callable[[Process], TeamResponse],
) -> None:
    """Rung 1: ``team_name`` wins over both fallbacks when it is set."""
    process = _process(team_name="Case Triage", catalog_namespace="acme-cases")

    assert to_response(process).name == "Case Triage"


@_PRODUCERS
def test_a_nameless_team_falls_back_to_its_catalog_namespace(
    to_response: Callable[[Process], TeamResponse],
) -> None:
    """Rung 2: a team projected off a nameless card displays as its namespace."""
    process = _process(team_name=None, catalog_namespace="acme-cases")

    assert to_response(process).name == "acme-cases"


@_PRODUCERS
def test_an_empty_name_falls_back_the_same_way_a_missing_one_does(
    to_response: Callable[[Process], TeamResponse],
) -> None:
    """``or`` is falsy-driven, not ``None``-driven, and that is deliberate.

    A card named ``""`` and a card with no name at all are the same team as far
    as a reader is concerned; reporting an empty string for one of them would be
    a blank row in the UI.
    """
    process = _process(team_name="", catalog_namespace="acme-cases")

    assert to_response(process).name == "acme-cases"


@_PRODUCERS
def test_a_nameless_team_outside_any_catalog_falls_back_to_its_id(
    to_response: Callable[[Process], TeamResponse],
) -> None:
    """Rung 3: the last rung is the id, so the name is never empty.

    This is the rung with no other source of truth — a team created outside a
    catalog from a nameless card has nothing else to display, and the response
    model requires a name.
    """
    process = _process(team_name=None, catalog_namespace=None)

    assert to_response(process).name == str(process.team_id)
