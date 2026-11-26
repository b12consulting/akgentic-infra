"""Validate public API exports from akgentic.infra."""

from __future__ import annotations

import subprocess
import sys

import pytest

_FRESH_IMPORT_TIMEOUT_SECONDS = 180

# Story 63.6: each surface imported first in a cold interpreter, then every name a
# client extension needs is read through it. ``OwnerOrAdminPolicy`` imports the
# server's auth module, and the server imports it back for its default policy, so
# a re-export added in the wrong place is an import cycle that only shows cold —
# by the time the suite runs, ``sys.modules`` already holds every module warm.
_FRESH_IMPORT_CHILD = """
import importlib
import sys

surface = importlib.import_module(sys.argv[1])
missing = [name for name in sys.argv[2:] if not hasattr(surface, name)]
assert not missing, "unresolved after a cold import: " + repr(missing)
print("OK")
"""


@pytest.mark.parametrize(
    ("module", "names"),
    [
        ("akgentic.infra.protocols", ["TeamAccessContext", "UserAccessContext", "TeamListFilter"]),
        ("akgentic.infra.adapters", ["OwnerOrAdminPolicy"]),
        ("akgentic.infra.adapters.shared", ["OwnerOrAdminPolicy"]),
        ("akgentic.infra.server", ["RequireAuthMiddleware", "MAX_PAGE_SIZE"]),
        (
            "akgentic.infra",
            ["TeamAccessContext", "UserAccessContext", "TeamListFilter", "OwnerOrAdminPolicy"],
        ),
    ],
)
def test_client_extension_names_import_cold(module: str, names: list[str]) -> None:
    """Each surface imports in a fresh interpreter with no cycle and binds its new names."""
    result = subprocess.run(
        [sys.executable, "-c", _FRESH_IMPORT_CHILD, module, *names],
        capture_output=True,
        text=True,
        timeout=_FRESH_IMPORT_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


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


def test_adapters_exports() -> None:
    """All declared __all__ symbols are importable from akgentic.infra.adapters."""
    from akgentic.infra import adapters

    for name in adapters.__all__:
        assert hasattr(adapters, name), f"Missing export: {name}"


def test_infra_all_includes_protocols() -> None:
    """akgentic.infra.__all__ includes all protocol exports."""
    from akgentic import infra
    from akgentic.infra import protocols

    for name in protocols.__all__:
        assert name in infra.__all__, f"Protocol {name} not in infra.__all__"


def test_infra_all_includes_adapters() -> None:
    """akgentic.infra.__all__ includes all adapter exports."""
    from akgentic import infra
    from akgentic.infra import adapters

    for name in adapters.__all__:
        assert name in infra.__all__, f"Adapter {name} not in infra.__all__"


def test_infra_all_includes_server_models() -> None:
    """akgentic.infra.__all__ exposes the server settings and tier-services symbols."""
    from akgentic import infra

    assert "ServerSettings" in infra.__all__
    assert "CommunitySettings" in infra.__all__
    assert "TierServices" in infra.__all__
    assert "CommunityServices" in infra.__all__


def test_infra_all_includes_wiring() -> None:
    """akgentic.infra.__all__ includes wire_community."""
    from akgentic import infra

    assert "wire_community" in infra.__all__


def test_infra_all_includes_server_app_and_models() -> None:
    """akgentic.infra.__all__ includes create_app, request/response models, and TeamService."""
    from akgentic import infra

    expected = (
        "create_app",
        "CreateTeamRequest",
        "TeamResponse",
        "TeamListResponse",
        "TeamService",
        "SendMessageRequest",
        "HumanInputRequest",
        "EventResponse",
        "EventListResponse",
    )
    for name in expected:
        assert name in infra.__all__, f"Missing export: {name}"


def test_server_module_exports() -> None:
    """akgentic.infra.server.__all__ includes all server-layer symbols."""
    from akgentic.infra import server

    for name in server.__all__:
        assert hasattr(server, name), f"Missing export: {name}"
