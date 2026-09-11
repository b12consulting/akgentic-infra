"""Unit tests for the pluggable per-team authorization port (ADR-035 Decision 8).

Covers ``OwnerOrAdminPolicy`` (the default rule), ``TeamAccessContext`` (the
neutral DTO), and the ``TierServices`` ``default_factory`` wiring. Behaviour
only — no ADR-string or docstring-presence assertions (Golden Rule #8).
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from akgentic.infra.adapters.shared.owner_or_admin_policy import OwnerOrAdminPolicy
from akgentic.infra.protocols.authz import TeamAccessContext, TeamAccessPolicy, TeamListFilter
from akgentic.infra.server.auth import RequestUser
from akgentic.infra.server.deps import CommunityServices


def _ctx(owner: str) -> TeamAccessContext:
    return TeamAccessContext(team_id=uuid.uuid4(), owner_user_id=owner)


class TestOwnerOrAdminPolicy:
    """The default owner-or-admin rule (byte-identical to the historical gate)."""

    async def test_owner_is_allowed(self) -> None:
        policy = OwnerOrAdminPolicy()
        assert await policy.is_allowed(ctx=_ctx("alice"), user=RequestUser(user_id="alice"))

    async def test_admin_non_owner_is_allowed(self) -> None:
        policy = OwnerOrAdminPolicy()
        user = RequestUser(user_id="bob", roles=["admin"])
        assert await policy.is_allowed(ctx=_ctx("alice"), user=user)

    async def test_non_owner_non_admin_is_denied(self) -> None:
        policy = OwnerOrAdminPolicy()
        assert not await policy.is_allowed(ctx=_ctx("alice"), user=RequestUser(user_id="mallory"))

    async def test_owner_filter_is_the_default_for_users_and_admins(self) -> None:
        policy = OwnerOrAdminPolicy()

        assert await policy.list_filters(user=RequestUser(user_id="alice")) == [
            TeamListFilter(user_id="alice")
        ]
        assert await policy.list_filters(
            user=RequestUser(user_id="root", roles=["admin"])
        ) == [TeamListFilter(user_id="root")]

    async def test_configured_admin_lists_without_a_store_constraint(self) -> None:
        policy = OwnerOrAdminPolicy(admin_list_all_teams=True)

        assert await policy.list_filters(
            user=RequestUser(user_id="root", roles=["admin"])
        ) == [TeamListFilter()]
        assert await policy.list_filters(user=RequestUser(user_id="alice")) == [
            TeamListFilter(user_id="alice")
        ]

    def test_satisfies_protocol(self) -> None:
        assert isinstance(OwnerOrAdminPolicy(), TeamAccessPolicy)

    def test_constructible_with_no_arguments(self) -> None:
        # No-arg construction is what makes it a valid ``default_factory``.
        assert isinstance(OwnerOrAdminPolicy(), OwnerOrAdminPolicy)


class TestTeamAccessContext:
    """The neutral DTO handed to a policy (never the team ``Process``)."""

    def test_round_trips_through_model_dump(self) -> None:
        team_id = uuid.uuid4()
        ctx = TeamAccessContext(
            team_id=team_id,
            owner_user_id="alice",
            metadata_indexes=["customer_id|1234"],
        )
        restored = TeamAccessContext.model_validate(ctx.model_dump())
        assert restored == ctx
        assert restored.team_id == team_id
        assert restored.owner_user_id == "alice"
        assert restored.metadata_indexes == ["customer_id|1234"]


class TestTeamListFilter:
    def test_fields_inside_one_clause_represent_owner_and_metadata(self) -> None:
        clause = TeamListFilter(
            user_id="alice",
            metadata={"customer_id": ["1234", "5678"]},
        )

        assert TeamListFilter.model_validate(clause.model_dump()) == clause

    def test_separate_clauses_can_represent_owner_or_metadata(self) -> None:
        clauses = [
            TeamListFilter(user_id="alice"),
            TeamListFilter(metadata={"customer_id": ["1234"]}),
        ]

        assert clauses[0].metadata is None
        assert clauses[1].user_id is None

    @pytest.mark.parametrize(
        "metadata",
        [{}, {"customer_id": []}, {"customer_id": [""]}],
    )
    def test_ambiguous_empty_metadata_is_rejected(
        self, metadata: dict[str, list[str]]
    ) -> None:
        with pytest.raises(ValidationError):
            TeamListFilter(metadata=metadata)


class TestTierServicesDefault:
    """The bundle default_factory supplies an ``OwnerOrAdminPolicy``."""

    def test_bundle_without_policy_defaults_to_owner_or_admin(
        self, community_services: CommunityServices
    ) -> None:
        assert isinstance(community_services.team_access_policy, OwnerOrAdminPolicy)
        assert isinstance(community_services.team_access_policy, TeamAccessPolicy)
