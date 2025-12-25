"""Textual TUI for Tuxedo."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, Static, Tree
from textual.widgets.tree import TreeNode

from tuxedo.clustering import PaperClusterer
from tuxedo.models import Cluster, ClusterView, Paper

if TYPE_CHECKING:
    from tuxedo.project import Project


# ============================================================================
# Confirmation Dialog
# ============================================================================


class ConfirmDialog(ModalScreen[bool]):
    """A modal dialog for confirming actions."""

    CSS = """
    ConfirmDialog {
        align: center middle;
    }

    #dialog {
        width: 60;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    #dialog .title {
        text-style: bold;
        margin-bottom: 1;
    }

    #dialog .message {
        margin-bottom: 1;
    }

    #dialog .buttons {
        margin-top: 1;
        align: center middle;
    }

    #dialog Button {
        margin: 0 1;
    }
    """

    def __init__(self, title: str, message: str):
        super().__init__()
        self.title_text = title
        self.message_text = message

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self.title_text, classes="title")
            yield Label(self.message_text, classes="message")
            with Horizontal(classes="buttons"):
                yield Button("Yes", variant="error", id="yes")
                yield Button("No", variant="primary", id="no")

    @on(Button.Pressed, "#yes")
    def on_yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def on_no(self) -> None:
        self.dismiss(False)

    def key_escape(self) -> None:
        self.dismiss(False)

    def key_y(self) -> None:
        self.dismiss(True)

    def key_n(self) -> None:
        self.dismiss(False)


# ============================================================================
# View Selection Screen
# ============================================================================


class ViewListItem(ListItem):
    """A list item representing a cluster view."""

    def __init__(self, view: ClusterView, cluster_count: int):
        super().__init__()
        self.view = view
        self.cluster_count = cluster_count

    def compose(self) -> ComposeResult:
        yield Label(f"[bold]{self.view.name}[/bold]")
        prompt_short = self.view.prompt[:50] + "..." if len(self.view.prompt) > 50 else self.view.prompt
        yield Label(f"[dim]{self.cluster_count} clusters | {prompt_short}[/dim]")


class NewViewItem(ListItem):
    """A list item for creating a new view."""

    def compose(self) -> ComposeResult:
        yield Label("[bold green]+ New Clustering...[/bold green]")
        yield Label("[dim]Organize papers with a different prompt or focus[/dim]")


class ViewSelectionScreen(Screen):
    """Screen for selecting or creating a cluster view."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("n", "new_view", "New"),
        Binding("enter", "select", "Select"),
        Binding("d", "delete", "Delete"),
        Binding("escape", "cancel_form", "Cancel", show=False),
    ]

    CSS = """
    ViewSelectionScreen {
        align: center middle;
    }

    #view-list-container {
        width: 80%;
        max-width: 100;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    #view-list-container .title {
        text-align: center;
        text-style: bold;
        padding: 1;
        margin-bottom: 1;
    }

    #view-list-container .subtitle {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }

    ListView {
        height: auto;
        max-height: 20;
    }

    ListItem {
        padding: 1 2;
    }

    ListItem:hover {
        background: $primary-darken-2;
    }

    ListView > ListItem.--highlight {
        background: $primary;
    }

    #new-view-form {
        display: none;
        padding: 1;
        margin-top: 1;
        border-top: solid $primary-darken-2;
    }

    #new-view-form.visible {
        display: block;
    }

    #new-view-form .form-label {
        margin-bottom: 1;
        text-style: bold;
    }

    #new-view-form Input {
        margin-bottom: 1;
    }

    #new-view-form .buttons {
        margin-top: 1;
    }

    #new-view-form .hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    """

    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self._pending_delete: ClusterView | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="view-list-container"):
            yield Static("Select a Clustering View", classes="title")
            paper_count = self.project.paper_count()
            yield Static(f"[dim]{paper_count} papers in project[/dim]", classes="subtitle")
            yield ListView(id="view-list")
            with Vertical(id="new-view-form"):
                yield Label("Create New Clustering View", classes="form-label")
                yield Label("Different prompts organize papers in different ways", classes="hint")
                yield Input(placeholder="View name (e.g., 'By Methodology')", id="new-view-name")
                yield Input(placeholder="Clustering prompt (leave empty for research question)", id="new-view-prompt")
                with Horizontal(classes="buttons"):
                    yield Button("Create", variant="primary", id="create-btn")
                    yield Button("Cancel", id="cancel-btn")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_list()

    def _refresh_list(self) -> None:
        """Refresh the view list."""
        list_view = self.query_one("#view-list", ListView)
        list_view.clear()

        views = self.project.get_views()
        for view in views:
            count = self.project.cluster_count(view.id)
            list_view.append(ViewListItem(view, count))

        list_view.append(NewViewItem())

        # Focus list if no views
        if not views:
            list_view.focus()

    @on(ListView.Selected)
    def on_list_selected(self, event: ListView.Selected) -> None:
        """Handle list item selection."""
        if isinstance(event.item, NewViewItem):
            self._show_new_view_form()
        elif isinstance(event.item, ViewListItem):
            self.app.push_screen(ClusterScreen(self.project, event.item.view))

    def _show_new_view_form(self) -> None:
        """Show the new view form."""
        form = self.query_one("#new-view-form")
        form.add_class("visible")
        self.query_one("#new-view-name", Input).focus()

    def _hide_new_view_form(self) -> None:
        """Hide the new view form."""
        form = self.query_one("#new-view-form")
        form.remove_class("visible")
        self.query_one("#new-view-name", Input).value = ""
        self.query_one("#new-view-prompt", Input).value = ""
        self.query_one("#view-list", ListView).focus()

    def action_cancel_form(self) -> None:
        """Cancel the new view form if visible."""
        form = self.query_one("#new-view-form")
        if "visible" in form.classes:
            self._hide_new_view_form()

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self._hide_new_view_form()

    @on(Button.Pressed, "#create-btn")
    def on_create(self) -> None:
        self._create_new_view()

    def action_new_view(self) -> None:
        self._show_new_view_form()

    def action_delete(self) -> None:
        """Delete the selected view with confirmation."""
        list_view = self.query_one("#view-list", ListView)
        if list_view.highlighted_child and isinstance(list_view.highlighted_child, ViewListItem):
            view = list_view.highlighted_child.view
            self._pending_delete = view

            def handle_confirm(confirmed: bool) -> None:
                if confirmed and self._pending_delete:
                    self.project.delete_view(self._pending_delete.id)
                    self.notify(f"Deleted '{self._pending_delete.name}'")
                    self._refresh_list()
                self._pending_delete = None

            self.app.push_screen(
                ConfirmDialog("Delete View", f"Delete '{view.name}' and its clusters?"),
                handle_confirm,
            )

    @work(thread=True)
    def _create_new_view(self) -> None:
        """Create a new clustering view."""
        name = self.query_one("#new-view-name", Input).value.strip()
        prompt = self.query_one("#new-view-prompt", Input).value.strip()

        if not name:
            views = self.project.get_views()
            name = f"View {len(views) + 1}"

        if not prompt:
            prompt = self.project.config.research_question

        self.call_from_thread(self.notify, f"Clustering papers for '{name}'...")

        try:
            # Create view and cluster
            view = self.project.create_view(name=name, prompt=prompt)
            papers = self.project.get_papers()

            if not papers:
                self.call_from_thread(self.notify, "No papers to cluster", severity="warning")
                return

            clusterer = PaperClusterer()
            clusters = clusterer.cluster_papers(papers, prompt)
            self.project.save_clusters(view.id, clusters)

            self.call_from_thread(self._hide_new_view_form)
            self.call_from_thread(self._refresh_list)
            self.call_from_thread(self.notify, f"Created '{name}' with {len(clusters)} clusters")
        except Exception as e:
            self.call_from_thread(self.notify, f"Failed to create view: {e}", severity="error")


# ============================================================================
# Cluster View Screen
# ============================================================================


class PaperDetail(Static):
    """Widget to display paper details."""

    DEFAULT_CSS = """
    PaperDetail {
        background: $surface;
        padding: 1 2;
        border: solid $primary;
        height: 100%;
        overflow-y: auto;
    }

    PaperDetail .title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    PaperDetail .meta {
        color: $text-muted;
        margin-bottom: 1;
    }

    PaperDetail .section-header {
        text-style: bold;
        color: $secondary;
        margin-top: 1;
    }

    PaperDetail .abstract {
        color: $text;
    }

    PaperDetail .keywords {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(self, paper: Paper | None = None):
        super().__init__()
        self.paper = paper

    def compose(self) -> ComposeResult:
        if self.paper:
            yield Static(self.paper.title, classes="title")

            # Authors and year
            authors = ", ".join(a.name for a in self.paper.authors[:3])
            if len(self.paper.authors) > 3:
                authors += f" et al. ({len(self.paper.authors)} authors)"
            meta = f"{authors}" if authors else ""
            if self.paper.year:
                meta += f" ({self.paper.year})" if meta else str(self.paper.year)
            if self.paper.doi:
                meta += f" | DOI: {self.paper.doi}"
            if meta:
                yield Static(meta, classes="meta")

            # Abstract
            if self.paper.abstract:
                yield Static("Abstract", classes="section-header")
                yield Static(self.paper.abstract, classes="abstract")

            # Keywords
            if self.paper.keywords:
                keywords = ", ".join(self.paper.keywords[:10])
                yield Static(f"Keywords: {keywords}", classes="keywords")
        else:
            yield Static("Select a paper to view details", classes="meta")


class ClusterTree(Tree):
    """Tree widget for displaying paper clusters."""

    DEFAULT_CSS = """
    ClusterTree {
        background: $surface;
        padding: 1;
        scrollbar-gutter: stable;
    }

    ClusterTree > .tree--guides {
        color: $primary-darken-2;
    }

    ClusterTree > .tree--cursor {
        background: $primary;
        color: $text;
    }
    """

    def __init__(self, view_name: str, papers: list[Paper], clusters: list[Cluster], filter_text: str = ""):
        super().__init__(view_name)
        self.all_papers = papers
        self.clusters = clusters
        self.filter_text = filter_text.lower()
        self._paper_map: dict[str, Paper] = {p.id: p for p in papers}

    def on_mount(self) -> None:
        self._build_tree()
        self.root.expand_all()

    def _build_tree(self) -> None:
        self.root.remove_children()

        if not self.clusters:
            self.root.add_leaf("No clusters yet")
            return

        for cluster in self.clusters:
            self._add_cluster_node(self.root, cluster)

    def _matches_filter(self, paper: Paper) -> bool:
        """Check if a paper matches the current filter."""
        if not self.filter_text:
            return True
        text = self.filter_text
        return (
            text in paper.title.lower()
            or any(text in a.name.lower() for a in paper.authors)
            or (paper.abstract and text in paper.abstract.lower())
            or any(text in k.lower() for k in paper.keywords)
        )

    def _add_cluster_node(self, parent: TreeNode, cluster: Cluster) -> None:
        # Filter papers
        matching_papers = [
            pid for pid in cluster.paper_ids
            if pid in self._paper_map and self._matches_filter(self._paper_map[pid])
        ]

        # Count total including subclusters (filtered)
        total = len(matching_papers)
        for sub in cluster.subclusters:
            sub_matching = [
                pid for pid in sub.paper_ids
                if pid in self._paper_map and self._matches_filter(self._paper_map[pid])
            ]
            total += len(sub_matching)

        # Skip empty clusters when filtering
        if self.filter_text and total == 0:
            return

        node = parent.add(f"[bold]{cluster.name}[/bold] ({total})", data={"type": "cluster", "cluster": cluster})

        for paper_id in matching_papers:
            paper = self._paper_map[paper_id]
            # Show year in tree if available
            year_suffix = f" [{paper.year}]" if paper.year else ""
            node.add_leaf(f"{paper.display_title}{year_suffix}", data={"type": "paper", "paper": paper})

        for subcluster in cluster.subclusters:
            self._add_cluster_node(node, subcluster)

    def get_selected_paper(self) -> Paper | None:
        node = self.cursor_node
        if node and node.data and node.data.get("type") == "paper":
            return node.data.get("paper")
        return None

    def refresh_filter(self, filter_text: str) -> None:
        """Rebuild tree with new filter."""
        self.filter_text = filter_text.lower()
        self._build_tree()
        self.root.expand_all()


class ClusterScreen(Screen):
    """Screen for viewing clusters in a specific view."""

    BINDINGS = [
        Binding("q", "back", "Back"),
        Binding("o", "open_pdf", "Open PDF"),
        Binding("/", "search", "Search"),
        Binding("escape", "clear_search", "Clear", show=False),
        Binding("e", "expand_all", "Expand"),
        Binding("c", "collapse_all", "Collapse"),
        Binding("?", "help", "Help"),
    ]

    CSS = """
    ClusterScreen #main-container {
        layout: horizontal;
        height: 1fr;
    }

    ClusterScreen #tree-container {
        width: 50%;
        height: 100%;
        border-right: solid $primary-darken-2;
    }

    ClusterScreen #detail-container {
        width: 50%;
        height: 100%;
        padding: 0 1;
    }

    ClusterScreen .section-title {
        text-style: bold;
        background: $primary-darken-2;
        padding: 0 1;
        margin-bottom: 1;
    }

    ClusterScreen .view-header {
        background: $primary-darken-3;
        padding: 1 2;
        dock: top;
    }

    ClusterScreen .status-bar {
        background: $surface-darken-1;
        padding: 0 2;
        height: 1;
        dock: bottom;
    }

    ClusterScreen #search-container {
        display: none;
        dock: top;
        height: 3;
        background: $surface;
        padding: 0 1;
    }

    ClusterScreen #search-container.visible {
        display: block;
    }

    ClusterScreen #search-input {
        width: 100%;
    }
    """

    def __init__(self, project: Project, view: ClusterView):
        super().__init__()
        self.project = project
        self.view = view
        self.papers = project.get_papers()
        self.clusters = project.get_clusters(view.id)
        self._tree: ClusterTree | None = None
        self._detail: Container | None = None
        self._filter_text = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"[bold]{self.view.name}[/bold] | {self.view.prompt[:60]}...", classes="view-header")
        with Container(id="search-container"):
            yield Input(placeholder="Search papers by title, author, abstract...", id="search-input")
        with Horizontal(id="main-container"):
            with Vertical(id="tree-container"):
                yield Static("Literature Structure", classes="section-title")
                self._tree = ClusterTree(self.view.name, self.papers, self.clusters)
                yield self._tree
            with Vertical(id="detail-container"):
                yield Static("Paper Details", classes="section-title")
                self._detail = Container(id="paper-detail-wrapper")
                self._detail.mount(PaperDetail())
                yield self._detail
        yield Static(f"{len(self.papers)} papers | {len(self.clusters)} clusters | Press / to search", classes="status-bar")
        yield Footer()

    @on(Tree.NodeHighlighted)
    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        if self._tree and self._detail:
            paper = self._tree.get_selected_paper()
            self._detail.remove_children()
            self._detail.mount(PaperDetail(paper))

    @on(Input.Changed, "#search-input")
    def on_search_changed(self, event: Input.Changed) -> None:
        """Filter tree as user types."""
        if self._tree:
            self._filter_text = event.value
            self._tree.refresh_filter(event.value)

    @on(Input.Submitted, "#search-input")
    def on_search_submitted(self, event: Input.Submitted) -> None:
        """Focus tree after search."""
        if self._tree:
            self._tree.focus()

    def action_search(self) -> None:
        """Show search input."""
        container = self.query_one("#search-container")
        container.add_class("visible")
        self.query_one("#search-input", Input).focus()

    def action_clear_search(self) -> None:
        """Clear search and hide input."""
        container = self.query_one("#search-container")
        if "visible" in container.classes:
            container.remove_class("visible")
            search_input = self.query_one("#search-input", Input)
            search_input.value = ""
            if self._tree:
                self._tree.refresh_filter("")
                self._tree.focus()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_expand_all(self) -> None:
        if self._tree:
            self._tree.root.expand_all()

    def action_collapse_all(self) -> None:
        if self._tree:
            self._tree.root.collapse_all()

    def action_open_pdf(self) -> None:
        if not self._tree:
            return

        paper = self._tree.get_selected_paper()
        if not paper:
            self.notify("No paper selected", severity="warning")
            return

        pdf_path = self.project.get_pdf_path(paper)
        if not pdf_path.exists():
            self.notify(f"PDF not found: {pdf_path.name}", severity="error")
            return

        if sys.platform == "darwin":
            subprocess.Popen(["open", str(pdf_path)])
        elif sys.platform == "win32":
            subprocess.Popen(["start", "", str(pdf_path)], shell=True)
        else:
            subprocess.Popen(["xdg-open", str(pdf_path)])

        self.notify(f"Opening {pdf_path.name}")

    def action_help(self) -> None:
        self.notify(
            "/ Search | o Open PDF | e/c Expand/Collapse | q Back",
            timeout=5,
        )


# ============================================================================
# Main App
# ============================================================================


class TuxedoApp(App):
    """Main TUI application."""

    TITLE = "Tuxedo"

    def __init__(self, project: Project):
        super().__init__()
        self.project = project

    def on_mount(self) -> None:
        self.push_screen(ViewSelectionScreen(self.project))


def run_tui(project: Project) -> None:
    """Run the TUI application."""
    app = TuxedoApp(project)
    app.run()
