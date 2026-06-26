"""NoAuth — community-tier authentication marker strategy."""

from __future__ import annotations


class NoAuth:
    """Community-tier marker strategy: no request-time authentication.

    Community identity flows solely through the ADR-023 ``get_request_user``
    seam, whose default resolves the anonymous principal. ``NoAuth`` carries no
    behaviour; it only marks the community tier's ``TierServices.auth`` slot.
    """
