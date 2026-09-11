"""The ``?workspace_id=`` values the guard refuses and the ones it lets through (Story 70.3).

The guard in front of the selector is the tool's own ``leaf_segment``. It is a
traversal guard: it refuses what cannot be a single leaf directory, and it
leaves authorization to the membership check behind it. Both tables are defined
once, here, so the unit, gate and route specs cannot drift apart.

Every kind name and sidecar suffix is spelled through the tool's exported
constants and never as a literal. A spec written that way stays correct when the
tool renames one, and a literal would go red, or silently stop testing the
reserved name, on whichever tool spelling it did not match.
"""

from __future__ import annotations

import pytest
from akgentic.tool.workspace import (
    GIT_DIR_SUFFIX,
    ID_KIND,
    META_DIR_SUFFIX,
    METADATA_KIND,
    TEAM_KIND,
)

__all__ = [
    "PATH_SAFE_UNDECLARED_IDS",
    "REJECTED_WORKSPACE_IDS",
]

REJECTED_WORKSPACE_IDS = [
    pytest.param("../x", id="dot-dot-slash"),
    pytest.param("..", id="dot-dot"),
    pytest.param(".", id="dot"),
    pytest.param("a/b", id="slash"),
    pytest.param("/abs", id="absolute"),
    pytest.param("a\\b", id="backslash"),
    pytest.param("", id="empty"),
    pytest.param("a\x00b", id="nul"),
    pytest.param(".hidden", id="leading-dot"),
    pytest.param(TEAM_KIND, id="team-kind"),
    pytest.param(ID_KIND, id="id-kind"),
    pytest.param(METADATA_KIND, id="metadata-kind"),
    pytest.param(METADATA_KIND.upper(), id="metadata-kind-upper"),
    pytest.param(f"notes{GIT_DIR_SUFFIX}", id="git-suffix"),
    pytest.param(f"notes{META_DIR_SUFFIX}", id="meta-suffix"),
    pytest.param(f"notes{GIT_DIR_SUFFIX.upper()}", id="git-suffix-upper"),
]
"""Values ``leaf_segment`` refuses, so the guard answers 400 before any card is read.

The first eight were 400 under the old regex too. The rest reached the
membership check and got its 404: the regex admitted a leading dot, the kind
names and the sidecar suffixes, which the tool refuses as a leaf.
"""

PATH_SAFE_UNDECLARED_IDS = [
    pytest.param("a" * 129, id="129-chars"),
    pytest.param("%2E%2E%2Fx", id="encoded-traversal"),
    pytest.param("customer_id-Acme%20Corp__case_id-42", id="percent-leaf"),
]
"""Path-safe values the guard lets through, so a team that declares none of them answers 404.

The old regex refused all three with 400, on its length bound and its charset.
None of them is a traversal. ``%2E%2E%2Fx`` is a literal nine-character name:
nothing decodes the selector again after the query layer.
"""
