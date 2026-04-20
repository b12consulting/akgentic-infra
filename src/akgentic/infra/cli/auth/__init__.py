"""CLI auth package.

Public API for pluggable access-token providers used by
:func:`akgentic.infra.cli.http.build_http_client`. Story 21.2 delivers only the
:class:`TokenProvider` Protocol (the boundary); Story 21.3 delivers the real
``OidcTokenProvider`` implementation.
"""

from akgentic.infra.cli.auth.token_provider import TokenProvider

__all__ = ["TokenProvider"]
