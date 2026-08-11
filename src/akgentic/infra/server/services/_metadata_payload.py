"""The ``__model__`` rule for team metadata, in both directions.

Team metadata is plain JSON on the wire, inbound and outbound. This module owns
both halves of that rule, because they are the same traversal in opposite
directions and splitting them is how they drift apart:

- **Inbound** (:func:`validate_metadata`) — a request body carries no type name.
  The server resolves the type from the team's own ``TeamCard.metadata_type``
  and validates against that. A ``__model__`` key at any depth is refused.
- **Outbound** (:func:`dump_metadata`) — the persisted model is serialized and
  its ``__model__`` tags are stripped recursively before the value reaches a
  client.

Why the tag is refused inbound: ``deserialize_object`` resolves it by importing
the embedded dotted path with no allowlist, so honouring it from a request body
would be an arbitrary-import gadget, and catalog ADR-016 §D5 keeps dynamic type
resolution off the HTTP surface entirely. It cannot be left to Pydantic either —
unknown keys are ignored and ``SerializableBaseModel`` strips the tag for its own
declared class, so a tagged body would validate *cleanly* and the key would
vanish with no signal. The explicit scan below is the only thing that turns it
into a 422. It is refused rather than dropped so that a caller who believed they
were choosing the type is told they are not.

Why the tag is stripped outbound: it is a persistence concern. The store needs it
to reconstruct the concrete type; a client never reconstructs, and if it did that
would be client-side type resolution — the very thing the inbound rule forbids.
Emitting it would also break GET → modify → PATCH, which would 422 on the
server's own output.

Pure functions with no FastAPI import and no module-level state, so the same
callable serves the create route and the metadata-update route. Implements
ADR-24 §D3.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.infra.errors import MetadataValidationError

MODEL_TAG = "__model__"
"""The serializer's polymorphic type tag — never accepted from, nor emitted to, a client."""


def _contains_model_tag(value: object) -> bool:
    """Report whether ``__model__`` appears anywhere in *value*.

    Walks dicts and list elements to any depth: a metadata model may nest
    sub-models, so a top-level-only check would miss a tag one level down.
    """
    if isinstance(value, dict):
        return MODEL_TAG in value or any(_contains_model_tag(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_model_tag(item) for item in value)
    return False


def _describe_validation_failure(exc: ValidationError) -> str:
    """Name the offending field from a ``ValidationError``.

    Uses the first error's ``loc`` so the caller is told *which* field failed
    rather than a bare "invalid metadata".
    """
    errors = exc.errors()
    if not errors:  # pragma: no cover — Pydantic always reports at least one
        return "metadata does not match the team's declared schema"
    first = errors[0]
    field = ".".join(str(part) for part in first["loc"]) or "<root>"
    return f"metadata field '{field}' is invalid: {first['msg']}"


def validate_metadata(
    metadata_type: type[SerializableBaseModel] | None,
    raw: dict[str, Any] | None,
) -> SerializableBaseModel | None:
    """Validate a raw metadata body against a server-chosen type.

    The four steps run in this order, and the order is load-bearing: the
    ``__model__`` scan is unconditional and comes first, so an attempt to name a
    type is always answered with the ``__model__`` reason rather than a
    misleading "this team takes no metadata".

    Args:
        metadata_type: The type declared by the team's ``TeamCard``, resolved by
            the caller from deployment-controlled catalog data. ``None`` when the
            team declares no metadata contract.
        raw: The plain-JSON request body, or ``None``.

    Returns:
        A validated instance of *metadata_type*, or ``None`` when *raw* is empty
        — an absent or empty body is not an error, even for a team that declares
        a type. Metadata stays optional; a declared type constrains its shape,
        not its presence.

    Raises:
        MetadataValidationError: If the body carries a ``__model__`` key at any
            depth, if metadata is supplied for a team declaring no contract, or
            if the body fails the declared schema. Nothing is written in any
            case — the caller runs this before creating anything.
    """
    if _contains_model_tag(raw):
        msg = (
            f"metadata must not contain a '{MODEL_TAG}' key at any depth: the metadata "
            "type is chosen by the team's catalog entry, never by the request body"
        )
        raise MetadataValidationError(msg)
    if not raw:
        return None
    if metadata_type is None:
        msg = "this team declares no metadata contract, so metadata cannot be supplied"
        raise MetadataValidationError(msg)
    try:
        return metadata_type.model_validate(raw)
    except ValidationError as exc:
        # FastAPI's automatic 422 covers request *parsing* only; a ValidationError
        # raised inside a handler would surface as a 500 uncaught.
        raise MetadataValidationError(_describe_validation_failure(exc)) from exc


def _strip_model_tags(value: object) -> object:
    """Return *value* with every ``__model__`` key removed, at any depth."""
    if isinstance(value, dict):
        return {key: _strip_model_tags(item) for key, item in value.items() if key != MODEL_TAG}
    if isinstance(value, list):
        return [_strip_model_tags(item) for item in value]
    return value


def dump_metadata(metadata: SerializableBaseModel | None) -> dict[str, object] | None:
    """Serialize persisted metadata for the wire — plain fields, no type tag.

    Args:
        metadata: The persisted value, or ``None``.

    Returns:
        The JSON-mode dump with every ``__model__`` key stripped recursively, or
        ``None`` when the team carries no metadata. The result round-trips: it is
        accepted verbatim as the body of a subsequent create or update.
    """
    if metadata is None:
        return None
    return {
        key: _strip_model_tags(value)
        for key, value in metadata.model_dump(mode="json").items()
        if key != MODEL_TAG
    }
