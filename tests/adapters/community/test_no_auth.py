"""Tests for the NoAuth community marker strategy."""

from __future__ import annotations

from akgentic.infra.adapters.community.no_auth import NoAuth
from akgentic.infra.server.auth import RequestUser, get_request_user


class TestNoAuthMarker:
    """NoAuth is the community marker strategy — it carries no auth behaviour."""

    def test_noauth_instantiates(self) -> None:
        """NoAuth can be constructed as the community tier's auth marker."""
        assert isinstance(NoAuth(), NoAuth)

    def test_noauth_has_no_authenticate_method(self) -> None:
        """The synchronous authenticate method is gone (collapsed onto the seam)."""
        assert not hasattr(NoAuth(), "authenticate")


class TestCommunityIdentitySeam:
    """Community identity resolves to anonymous through get_request_user."""

    def test_get_request_user_default_is_anonymous(self) -> None:
        """The community default returns an anonymous principal, never None."""
        user = get_request_user()
        assert isinstance(user, RequestUser)
        assert user.user_id == "anonymous"
