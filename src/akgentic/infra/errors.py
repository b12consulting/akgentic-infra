"""Framework-agnostic server-error base carrying its own HTTP mapping.

``ServerError`` subclasses pin ``status_code`` (and optionally ``code``) as
class attributes; the single FastAPI handler in ``server/errors.py`` reads them
to build the response. No FastAPI import here — the base is a plain ``Exception``
carrying data, so ``protocols/`` and ``services/`` can raise it without dragging
in the web framework. See ADR-031 §Decision 1.
"""

from __future__ import annotations


class ServerError(Exception):
    """Base for infra-server domain errors that carry their own HTTP mapping.

    Subclasses pin ``status_code`` (and optionally ``code``) as class attributes;
    the single registered handler reads them. A call site may still override
    ``status_code``/``headers``/``code`` per instance for a one-off mapping.
    """

    status_code: int = 500
    code: str | None = None

    def __init__(
        self,
        detail: str,
        *,
        status_code: int | None = None,
        headers: dict[str, str] | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code
        self.headers = headers
        if code is not None:
            self.code = code


class PlacementConsistencyError(ServerError):
    """A team was created but is absent from the event store afterwards.

    Fires *after* a successful placement, so it is a ``ServerError`` but not a
    ``PlacementError`` (no worker-selection failure). See ADR-031 §Decision 4.
    """

    status_code = 502
    code = "placement_consistency"


class MetadataValidationError(ServerError):
    """A team-metadata request body was refused before anything was written.

    Its own type, not a bare ``ValueError``: the teams router maps ``ValueError``
    by string-matching the message to 404/409, which would turn a validation
    failure into a conflict. 422 matches FastAPI's own request-validation status.

    Raised by ``server/services/_metadata_payload.validate_metadata`` for a
    ``__model__`` key in the body, for metadata sent to a team that declares no
    contract, and for a body that fails the declared schema. See ADR-24 §D3.
    """

    status_code = 422
    code = "invalid_metadata"


_SHARED_WORKSPACE_REFUSED_DETAIL = (
    "This workspace is shared across principals, and no policy yet decides who may reach "
    "a shared workspace: that is akgentic-infra-auth's metadata-entitlement policy, which "
    "does not exist yet. Shared workspaces are refused to every caller until it does."
)


class SharedWorkspaceRefusedError(ServerError):
    """A workspace route was asked to open a tree under the shared scope.

    A shared tree has no owner, so whether the caller owns the team is the wrong
    question for it, and an admin's authority over principals does not answer
    it either. The right question is entitlement: may this principal assert the
    values the tree is keyed on. ``akgentic-infra-auth`` owns that policy, and
    until it exists every caller is refused, admins included.

    **403, not 404.** The 404-over-403 rule hides the existence of something the
    caller may not see. The caller has already passed the team gate, and the
    tree is one their own authorized team declares, so they know it exists. A
    404 would misstate the reason.

    The ``detail`` names the missing policy. Clients should match on ``code``,
    which is stable, and not on the wording.
    """

    status_code = 403
    code = "shared_workspace_entitlement_undecided"

    def __init__(self, detail: str = _SHARED_WORKSPACE_REFUSED_DETAIL) -> None:
        super().__init__(detail)


class TeamNotFoundError(ValueError):
    """The team does not exist in the system of record.

    A ``ValueError`` subclass on purpose: every existing ``except ValueError``
    around the team service keeps catching it unchanged, so routes can
    discriminate by type without any call site having to be migrated first.
    """


class TeamStateConflictError(ValueError):
    """The team exists, but its current state forbids the requested operation.

    Carries the service's own message ("is already stopped", "is not running",
    …) so a route can answer 409 with the real condition rather than reporting a
    live team as missing. Same ``ValueError`` base, same additive reasoning as
    ``TeamNotFoundError``.
    """
