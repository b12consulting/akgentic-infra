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
