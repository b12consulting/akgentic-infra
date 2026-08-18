"""Story 57.7: ``modules`` and ``configure_process`` arguments on ``create_app``.

The tier-shaped sibling of the 57.5 composition proof: the same
CoreModule-plus-fake-tier composition, but THROUGH the public entry — proving
the invariant process globals (logging, catalog prefix policy) are applied for
a tier composition without calling any tier code, and that a custom
``configure_process`` hook observes both invariants already applied when it
runs (extension semantics, order pinned).
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from akgentic.catalog import allowed_prefixes
from akgentic.infra.server.app import create_app
from akgentic.infra.server.assembly import build_manifest
from akgentic.infra.server.modules import CoreModule
from tests.server.test_core_module_composition import FakeTierModule

if TYPE_CHECKING:
    from akgentic.infra.server.deps import CommunityServices
    from akgentic.infra.server.settings import CommunitySettings, ServerSettings


@pytest.fixture(autouse=True)
def _reset_root_logger() -> Generator[None, None, None]:
    """Snapshot/restore process-global logging state (handlers + level)."""
    root = logging.getLogger()
    original_level = root.level
    original_handlers = list(root.handlers)
    yield
    root.setLevel(original_level)
    root.handlers = original_handlers


class TestTierShapedCompositionThroughCreateApp:
    """AC 1: tier composition through the public entry, invariants applied."""

    def test_tier_surface_and_invariant_globals(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """Tier route + middleware serve, community surface intact, globals set."""
        settings = seeded_settings.model_copy(update={"catalog_model_type_prefixes": ["acme."]})
        # Force a known pre-state: earlier suite tests routinely leave the root
        # logger at INFO, so INFO-after-call proves nothing without this. With
        # WARNING forced, the closing level assertion can only be satisfied by
        # create_app's own configure_logging (mirrors the AC 3 hook test).
        logging.getLogger().setLevel(logging.WARNING)
        app = create_app(
            community_services,
            settings,
            modules=[
                CoreModule(services=community_services, settings=settings),
                FakeTierModule(),
            ],
        )
        manifest = build_manifest(app)
        reference = build_manifest(create_app(community_services, settings))
        assert set(reference.routes) <= set(manifest.routes)
        assert "GET /tier/ping" in manifest.routes
        assert "_TierMiddleware" in manifest.middleware
        with TestClient(app) as client:
            assert client.get("/readiness").status_code == 200
            assert client.get("/tier/ping").json() == {"tier": "fake"}
        # Both invariant process globals were applied by create_app itself —
        # no tier code was called to produce either observation.
        assert logging.getLogger().level == logging.getLevelNamesMapping()[settings.log_level]
        assert allowed_prefixes() == ("akgentic.", "acme.")


class TestConfigureProcessHook:
    """AC 3: extension semantics — the hook runs once, after both invariants."""

    def test_hook_runs_once_after_both_invariants(
        self,
        seeded_settings: CommunitySettings,
        community_services: CommunityServices,
    ) -> None:
        """When invoked, the hook observes logging AND prefixes already applied."""
        settings = seeded_settings.model_copy(
            update={"catalog_model_type_prefixes": ["acme."], "log_level": "DEBUG"}
        )
        # Force a known pre-state so DEBUG-at-hook-time can only have come
        # from create_app's own configure_logging call.
        logging.getLogger().setLevel(logging.WARNING)
        observed: list[tuple[ServerSettings, int, tuple[str, ...]]] = []

        def recording_hook(hook_settings: ServerSettings) -> None:
            observed.append((hook_settings, logging.getLogger().level, allowed_prefixes()))

        create_app(community_services, settings, configure_process=recording_hook)

        assert len(observed) == 1
        hook_settings, root_level_at_hook, prefixes_at_hook = observed[0]
        assert hook_settings is settings
        assert root_level_at_hook == logging.DEBUG
        assert prefixes_at_hook == ("akgentic.", "acme.")
