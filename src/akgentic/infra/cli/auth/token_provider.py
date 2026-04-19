"""Token provider protocol used by the CLI HTTP client factory.

Story 21.2 defines ONLY the Protocol. The concrete OIDC implementation
(``OidcTokenProvider``) lands in Story 21.3. By keeping this boundary as a
``typing.Protocol`` (not an ABC), tests can supply lightweight fakes without
inheriting from a framework class, and the production OIDC code never has to
know about the factory.

``@runtime_checkable`` is applied so tests may ``isinstance(x, TokenProvider)``
a fake if they want; the Protocol's only method is :meth:`get_access_token`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenProvider(Protocol):
    """Source of bearer tokens for authenticated CLI requests.

    Implementations are expected to handle their own caching / refresh policy.
    :func:`akgentic.infra.cli.http.build_http_client` calls
    :meth:`get_access_token` on every outgoing request, so a provider that
    caches internally picks up rotation transparently without the client being
    rebuilt.
    """

    def get_access_token(self) -> str:
        """Return a valid bearer token as a string.

        Implementations MUST NOT prefix the return value with ``"Bearer "`` —
        the factory formats the ``Authorization`` header.
        """
        ...
