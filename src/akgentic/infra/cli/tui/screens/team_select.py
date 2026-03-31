"""Full-screen team selection menu with pagination."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Input, Static

if TYPE_CHECKING:
    from akgentic.infra.cli.client import ApiClient, CatalogTeamInfo, TeamInfo

_log = logging.getLogger(__name__)

PAGE_SIZE = 5

_CSS_PATH = Path(__file__).parent.parent / "styles" / "team_select.tcss"


def _short_id(team_id: str) -> str:
    """Truncate a UUID to the first 13 characters."""
    return team_id[:13] if len(team_id) > 13 else team_id


class TeamSelectScreen(Screen[str | None]):
    """Full-screen team selection with pagination for running/stopped teams."""

    CSS_PATH = _CSS_PATH

    def __init__(self, client: ApiClient | None = None) -> None:
        super().__init__()
        self._client = client
        self._page: int = 0
        self._running_teams: list[TeamInfo] = []
        self._stopped_teams: list[TeamInfo] = []
        self._catalog: list[CatalogTeamInfo] = []

    def compose(self) -> ComposeResult:
        """Compose the team selection layout."""
        yield Static("Akgentic -- Team Selection", id="team-header")
        yield VerticalScroll(id="team-list")
        yield Static("", id="team-hints")
        yield Input(placeholder="> ", id="team-input")

    def on_mount(self) -> None:
        """Fetch data and render the first page."""
        self._fetch_data_worker()

    @work(thread=True)
    def _fetch_data_worker(self) -> None:
        """Fetch teams and catalog data in a background thread."""
        running, stopped, catalog = self._fetch_data()
        self.app.call_from_thread(self._on_data_loaded, running, stopped, catalog)

    def _fetch_data(
        self,
    ) -> tuple[list[TeamInfo], list[TeamInfo], list[CatalogTeamInfo]]:
        """Call API to get teams and catalog. Returns (running, stopped, catalog)."""
        from akgentic.infra.cli.client import ApiError

        running: list[TeamInfo] = []
        stopped: list[TeamInfo] = []
        catalog: list[CatalogTeamInfo] = []
        if self._client is None:
            return running, stopped, catalog
        try:
            teams = self._client.list_teams()
            running = [t for t in teams if t.status == "running"]
            stopped = [t for t in teams if t.status == "stopped"]
        except ApiError:
            _log.debug("Failed to fetch teams")
        try:
            catalog = self._client.list_catalog_teams()
        except ApiError:
            _log.debug("Failed to fetch catalog")
        return running, stopped, catalog

    def _on_data_loaded(
        self,
        running: list[TeamInfo],
        stopped: list[TeamInfo],
        catalog: list[CatalogTeamInfo],
    ) -> None:
        """Store fetched data and render."""
        self._running_teams = running
        self._stopped_teams = stopped
        self._catalog = catalog
        self._render_page()

    def _max_pages(self) -> int:
        """Return total number of pages based on running and stopped lists."""
        max_items = max(len(self._running_teams), len(self._stopped_teams))
        if max_items == 0:
            return 1
        return math.ceil(max_items / PAGE_SIZE)

    def _render_page(self) -> None:
        """Clear and re-render the team list for the current page."""
        container = self.query_one("#team-list", VerticalScroll)
        container.remove_children()

        start = self._page * PAGE_SIZE
        end = start + PAGE_SIZE

        # -- Running teams --
        if self._running_teams:
            r_page = self._running_teams[start:end]
            r_total = len(self._running_teams)
            r_end = min(start + len(r_page), r_total)
            header_text = f"Running teams ({start + 1}-{r_end} of {r_total}):"
            container.mount(Static(header_text, classes="section-header"))
            for i, team in enumerate(r_page):
                global_idx = start + i + 1
                line = Text()
                line.append(f"  \\[{global_idx}]", style="bold cyan")
                line.append(f"  {team.name}", style="bold")
                line.append(f"  {_short_id(team.team_id)}", style="dim")
                line.append("  > running", style="green")
                container.mount(Static(line, classes="team-entry"))
        else:
            container.mount(Static("Running teams: (none)", classes="section-header"))

        container.mount(Static(""))

        # -- Stopped teams --
        if self._stopped_teams:
            s_page = self._stopped_teams[start:end]
            s_total = len(self._stopped_teams)
            s_end = min(start + len(s_page), s_total)
            header_text = f"Stopped teams ({start + 1}-{s_end} of {s_total}):"
            container.mount(Static(header_text, classes="section-header"))
            for i, team in enumerate(s_page):
                global_idx = start + i + 1
                line = Text()
                line.append(f"  \\[s{global_idx}]", style="bold yellow")
                line.append(f"  {team.name}", style="bold")
                line.append(f"  {_short_id(team.team_id)}", style="dim")
                line.append("  || stopped", style="yellow")
                container.mount(Static(line, classes="team-entry"))
        else:
            container.mount(Static("Stopped teams: (none)", classes="section-header"))

        container.mount(Static(""))

        # -- Catalog entries --
        if self._catalog:
            container.mount(Static("Create new:", classes="section-header"))
            for entry in self._catalog:
                line = Text()
                line.append(f"  \\[c {entry.id}]", style="bold magenta")
                line.append(f"  {entry.description}", style="dim")
                container.mount(Static(line, classes="team-entry"))
        else:
            container.mount(Static("Create new: (none)", classes="section-header"))

        # -- Update hints --
        self._update_hints()

    def _update_hints(self) -> None:
        """Update the hint bar based on current page state."""
        hints_parts: list[str] = []
        if self._running_teams:
            start = self._page * PAGE_SIZE + 1
            end = min(start + PAGE_SIZE - 1, len(self._running_teams))
            hints_parts.append(f"[{start}-{end}] connect")
        if self._stopped_teams:
            start = self._page * PAGE_SIZE + 1
            end = min(start + PAGE_SIZE - 1, len(self._stopped_teams))
            hints_parts.append(f"[s{start}-s{end}] restore")
        if self._catalog:
            hints_parts.append("[c <name>] create")
        if self._max_pages() > 1:
            hints_parts.append("<-> page")
        hints_parts.append("[q] quit")
        hint_text = "  ".join(hints_parts)
        try:
            self.query_one("#team-hints", Static).update(hint_text)
        except Exception:  # noqa: BLE001
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input from the Input widget."""
        text = event.value.strip()
        event.input.value = ""

        if not text:
            return

        if text == "q":
            self.dismiss(None)
            return

        if text == "n":
            self._next_page()
            return

        if text == "p":
            self._prev_page()
            return

        if text.isdigit():
            self._select_running(int(text))
            return

        if text.startswith("s") and text[1:].isdigit():
            self._select_stopped(int(text[1:]))
            return

        if text.startswith("c "):
            name = text[2:].strip()
            if name:
                self._create_team(name)
                return

        self._show_error(f"Unknown command: {text}")

    def on_key(self, event: object) -> None:
        """Handle arrow key presses for pagination."""
        from textual.events import Key

        if not isinstance(event, Key):
            return
        if event.key == "right":
            self._next_page()
        elif event.key == "left":
            self._prev_page()

    def _next_page(self) -> None:
        """Advance to the next page if available."""
        if self._page < self._max_pages() - 1:
            self._page += 1
            self._render_page()

    def _prev_page(self) -> None:
        """Go back one page if not on the first page."""
        if self._page > 0:
            self._page -= 1
            self._render_page()

    def _select_running(self, number: int) -> None:
        """Select a running team by global index."""
        idx = number - 1
        if 0 <= idx < len(self._running_teams):
            self.dismiss(self._running_teams[idx].team_id)
        else:
            self._show_error(f"Invalid selection: {number}")

    def _select_stopped(self, number: int) -> None:
        """Select a stopped team by global index, restore it."""
        idx = number - 1
        if 0 <= idx < len(self._stopped_teams):
            team = self._stopped_teams[idx]
            self._restore_team_worker(team.team_id)
        else:
            self._show_error(f"Invalid selection: s{number}")

    @work(thread=True)
    def _restore_team_worker(self, team_id: str) -> None:
        """Restore a stopped team in a background thread."""
        from akgentic.infra.cli.client import ApiError

        if self._client is None:
            return
        try:
            self._client.restore_team(team_id)
            self.app.call_from_thread(self.dismiss, team_id)
        except ApiError as exc:
            self.app.call_from_thread(self._show_error, f"Failed to restore team: {exc}")

    def _create_team(self, catalog_entry_id: str) -> None:
        """Create a team from a catalog entry."""
        self._create_team_worker(catalog_entry_id)

    @work(thread=True)
    def _create_team_worker(self, catalog_entry_id: str) -> None:
        """Create a team in a background thread."""
        from akgentic.infra.cli.client import ApiError

        if self._client is None:
            return
        try:
            team = self._client.create_team(catalog_entry_id)
            self.app.call_from_thread(self.dismiss, team.team_id)
        except ApiError as exc:
            self.app.call_from_thread(self._show_error, f"Failed to create team: {exc}")

    def _show_error(self, message: str) -> None:
        """Display an error message in the team list area."""
        container = self.query_one("#team-list", VerticalScroll)
        error_widget = Static(
            Text(message, style="bold red"),
            classes="error-message",
        )
        container.mount(error_widget)
