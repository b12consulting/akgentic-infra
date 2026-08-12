"""Unit tests for the metadata payload helper — Story 53.1 Tasks 1 and 6.

Exercises each branch of ``validate_metadata``'s four-step order and the
outbound ``dump_metadata`` strip, independently of any route. Story 53-3 calls
the same two functions, so these tests guard both call sites.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from akgentic.infra.errors import MetadataValidationError
from akgentic.infra.server.services._metadata_payload import (
    MODEL_TAG,
    dump_metadata,
    validate_metadata,
)

from tests.fixtures.team_metadata import AcmeCaseMetadata, AcmeOwner, make_metadata_body

# --- validate_metadata: step 1, the unconditional __model__ scan ---


def test_top_level_model_tag_is_rejected() -> None:
    """A ``__model__`` key at the top level is refused, never honoured."""
    with pytest.raises(MetadataValidationError, match=MODEL_TAG):
        validate_metadata(AcmeCaseMetadata, {"__model__": "acme.Thing", "tenant": "acme"})


def test_nested_model_tag_is_rejected() -> None:
    """A ``__model__`` one level down is refused too — the scan recurses."""
    body = {"tenant": "acme", "case": "C-1", "owner": {"__model__": "acme.Owner"}}
    with pytest.raises(MetadataValidationError, match=MODEL_TAG):
        validate_metadata(AcmeCaseMetadata, body)


def test_model_tag_inside_list_element_is_rejected() -> None:
    """A ``__model__`` inside a list element is refused — lists are walked too."""
    body = {"tenant": "acme", "items": [{"ok": 1}, {"__model__": "acme.Thing"}]}
    with pytest.raises(MetadataValidationError, match=MODEL_TAG):
        validate_metadata(AcmeCaseMetadata, body)


def test_model_tag_beats_the_no_contract_error() -> None:
    """Step 1 runs before step 3: a tag wins even when the team declares no type.

    The type-naming attempt is the security-relevant condition, so it must always
    be the message the caller sees — not a misleading "this team takes no
    metadata", which would hide that the tag was noticed at all.
    """
    with pytest.raises(MetadataValidationError, match=MODEL_TAG):
        validate_metadata(None, {"__model__": "akgentic.infra.server.models.TeamResponse"})


def test_model_tag_with_a_real_importable_class_is_still_rejected() -> None:
    """The rejection is policy, not a failed import.

    The value is a real, importable, harmless class. A nonexistent dotted path
    would let this pass on an ``ImportError`` even if the server had honoured the
    tag — a false green that would hide the vulnerability this rule exists for.
    """
    body = {
        "__model__": "akgentic.infra.server.models.TeamResponse",
        "tenant": "acme",
        "case": "C-1",
    }
    with pytest.raises(MetadataValidationError, match=MODEL_TAG):
        validate_metadata(AcmeCaseMetadata, body)


# --- validate_metadata: step 2, an empty body is not an error ---


@pytest.mark.parametrize("raw", [None, {}])
def test_empty_body_yields_none_for_a_typed_card(raw: dict[str, object] | None) -> None:
    """A declared type constrains metadata's shape, not its presence."""
    assert validate_metadata(AcmeCaseMetadata, raw) is None


@pytest.mark.parametrize("raw", [None, {}])
def test_empty_body_yields_none_for_an_untyped_card(raw: dict[str, object] | None) -> None:
    """The no-contract guard is on truthiness: an empty body is never an error."""
    assert validate_metadata(None, raw) is None


# --- validate_metadata: step 3, no declared contract ---


def test_metadata_for_a_card_declaring_none_is_rejected() -> None:
    """A non-empty body against ``metadata_type=None`` names the missing contract."""
    with pytest.raises(MetadataValidationError, match="no metadata contract"):
        validate_metadata(None, {"tenant": "acme"})


# --- validate_metadata: step 4, schema validation ---


def test_valid_body_returns_an_instance_of_the_declared_type() -> None:
    """The happy path returns the card's type, carrying the sent values."""
    result = validate_metadata(AcmeCaseMetadata, {"tenant": "acme", "case": "C-1234"})
    assert isinstance(result, AcmeCaseMetadata)
    assert result.tenant == "acme"
    assert result.case == "C-1234"


def test_missing_required_field_names_the_offending_field() -> None:
    """A missing required field is named in the detail, not a bare 'invalid'."""
    with pytest.raises(MetadataValidationError, match="case") as exc_info:
        validate_metadata(AcmeCaseMetadata, {"tenant": "acme"})
    assert "case" in exc_info.value.detail


def test_wrong_scalar_type_names_the_offending_field() -> None:
    """A wrong scalar type is named in the detail too."""
    with pytest.raises(MetadataValidationError) as exc_info:
        validate_metadata(AcmeCaseMetadata, {"tenant": "acme", "case": {"not": "a string"}})
    assert "case" in exc_info.value.detail


def test_nested_field_path_is_rendered_dotted() -> None:
    """A nested failure reports the full ``loc`` path, not just the leaf."""
    body = {"tenant": "acme", "case": "C-1", "owner": {"squad": "contoso"}}
    with pytest.raises(MetadataValidationError) as exc_info:
        validate_metadata(AcmeCaseMetadata, body)
    assert "owner" in exc_info.value.detail
    assert "email" in exc_info.value.detail


def test_validation_error_carries_the_422_mapping() -> None:
    """The exception is its own type carrying 422 — not a ValueError the router
    would string-match into a 404 or a 409."""
    with pytest.raises(MetadataValidationError) as exc_info:
        validate_metadata(None, {"tenant": "acme"})
    assert exc_info.value.status_code == 422
    assert not isinstance(exc_info.value, ValueError)


# --- Statelessness ---


def test_validate_metadata_is_a_pure_function_of_its_arguments() -> None:
    """Repeated calls neither cache a resolved type nor mutate the input body."""
    body = {"tenant": "acme", "case": "C-1234"}
    snapshot = dict(body)
    first = validate_metadata(AcmeCaseMetadata, body)
    second = validate_metadata(AcmeCaseMetadata, body)
    assert first == second
    assert first is not second  # no memoised instance handed out twice
    assert body == snapshot


# --- dump_metadata: the outbound strip ---


def test_dump_of_none_is_none() -> None:
    """A team with no metadata dumps to ``None``, not an empty dict."""
    assert dump_metadata(None) is None


def test_dump_strips_the_top_level_tag() -> None:
    """The declared model's own ``__model__`` tag never reaches the wire."""
    dumped = dump_metadata(AcmeCaseMetadata(tenant="acme", case="C-1234"))
    assert dumped is not None
    assert MODEL_TAG not in dumped
    assert dumped["tenant"] == "acme"


def test_dump_strips_a_nested_tag() -> None:
    """A nested sub-model's tag is stripped too — the strip recurses.

    A top-level-only strip would leave this one behind and break the round-trip
    for any metadata model that nests.
    """
    metadata = AcmeCaseMetadata(
        tenant="acme",
        case="C-1234",
        owner=AcmeOwner(email="ops@contoso.example", squad="contoso-support"),
    )
    dumped = dump_metadata(metadata)
    assert dumped is not None
    owner = dumped["owner"]
    assert isinstance(owner, dict)
    assert MODEL_TAG not in owner
    assert owner["email"] == "ops@contoso.example"


def test_dump_strips_tags_inside_a_list_of_sub_models() -> None:
    """A tag inside a list element is stripped too — the strip walks lists.

    Only *indexed* fields are restricted to scalars, so a metadata model may
    legitimately hold a list of sub-models. Each element carries its own tag, and
    a strip that recursed into dicts but not lists would emit every one of them.
    """
    metadata = AcmeCaseMetadata(
        tenant="acme",
        case="C-1234",
        watchers=[
            AcmeOwner(email="first@contoso.example"),
            AcmeOwner(email="second@contoso.example", squad="contoso-support"),
        ],
    )
    dumped = dump_metadata(metadata)
    assert dumped is not None
    watchers = dumped["watchers"]
    assert isinstance(watchers, list)
    assert len(watchers) == 2
    assert all(MODEL_TAG not in watcher for watcher in watchers)
    assert [watcher["email"] for watcher in watchers] == [
        "first@contoso.example",
        "second@contoso.example",
    ]


def test_dump_output_is_accepted_verbatim_by_validate() -> None:
    """Round-trip: what the server emits, the server accepts back unchanged.

    This is the property that keeps GET → modify → PATCH working. Pinned
    behaviourally, so it survives a refactor that changes how the strip is done.
    """
    original = AcmeCaseMetadata(
        tenant="acme",
        case="C-1234",
        owner=AcmeOwner(email="ops@contoso.example"),
        watchers=[AcmeOwner(email="watcher@contoso.example")],
        note="escalated",
    )
    dumped = dump_metadata(original)
    assert dumped is not None
    revalidated = validate_metadata(AcmeCaseMetadata, dict(dumped))
    assert revalidated == original


def test_helper_module_has_no_web_framework_import() -> None:
    """AC #9: one callable, two call sites — so it must not depend on a route.

    Story 53-3 calls both functions from the PATCH path. A FastAPI import here
    would be the first step towards a second, route-shaped copy, and the copy
    that drifts is the one that stops rejecting ``__model__``.
    """
    import ast

    from akgentic.infra.server.services import _metadata_payload

    source = _metadata_payload.__file__
    assert source is not None
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)

    assert not any(name.split(".")[0] in {"fastapi", "starlette"} for name in imported)


def test_helper_module_holds_no_module_level_state() -> None:
    """AC #10: no cache to go stale, and nothing a replica could disagree about.

    The only module-level binding is the tag constant; a dict or list at module
    scope would be a per-process cache by another name.
    """
    from akgentic.infra.server.services import _metadata_payload

    mutable = {
        name: value
        for name, value in vars(_metadata_payload).items()
        if not name.startswith("__") and isinstance(value, dict | list | set)
    }
    assert mutable == {}


def test_make_metadata_body_factory_produces_a_legal_request_body() -> None:
    """The shared fixture factory emits a tag-free body the helper accepts."""
    body = make_metadata_body(case="C-9999")
    assert MODEL_TAG not in body
    validated = validate_metadata(AcmeCaseMetadata, body)
    assert isinstance(validated, AcmeCaseMetadata)
    assert validated.case == "C-9999"
