"""AuthStrategy protocol — authenticates and authorizes incoming server requests."""

from __future__ import annotations

from typing import Any, Protocol


class AuthStrategy(Protocol):
    """Authenticates and authorizes incoming server requests.

    Implementations: NoAuth (community), OAuth2Auth (department),
    SsoRbacAuth (enterprise).
    """

    def authenticate(self, request: Any) -> str | None:
        """Authenticate a request and return user identifier.

        Args:
            request: The incoming HTTP request

        Returns:
            User identifier string, or None if authentication fails
        """
        ...
