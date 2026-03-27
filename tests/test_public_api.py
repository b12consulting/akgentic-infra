"""Validate public API exports from akgentic.infra."""

from __future__ import annotations


def test_infra_exports() -> None:
    """All declared __all__ symbols are importable from akgentic.infra."""
    from akgentic import infra

    for name in infra.__all__:
        assert hasattr(infra, name), f"Missing export: {name}"


def test_protocols_exports() -> None:
    """All declared __all__ symbols are importable from akgentic.infra.protocols."""
    from akgentic.infra import protocols

    for name in protocols.__all__:
        assert hasattr(protocols, name), f"Missing export: {name}"


def test_infra_all_matches_protocols_all() -> None:
    """akgentic.infra.__all__ re-exports all protocols."""
    from akgentic import infra
    from akgentic.infra import protocols

    assert set(infra.__all__) == set(protocols.__all__)
