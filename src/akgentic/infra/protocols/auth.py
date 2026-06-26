"""AuthStrategy marker protocol for the wired tier authentication strategy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AuthStrategy(Protocol):
    """Marker protocol for a tier's wired authentication strategy.

    Authentication now flows through the ADR-023 ``get_request_user`` seam, not
    a synchronous ``authenticate`` method — this protocol is retained only as
    the ``TierServices.auth`` field type so each tier can wire its marker
    strategy (community ``NoAuth``, department ``OAuth2Auth``, enterprise
    ``SsoRbacAuth``). It carries no members.
    """
