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
from textual.widgets import Button, Input, Label, ListItem, ListView, Select, Static, Tree
from textual.widgets.tree import TreeNode

from tuxedo.clustering import PaperClusterer
from tuxedo.models import Cluster, ClusterView, Paper

# Available models for clustering
MODEL_OPTIONS = [
    ("gpt-5.2", "GPT-5.2 (recommended)"),
    ("gpt-5.2-pro", "GPT-5.2 Pro"),
    ("gpt-5.1", "GPT-5.1"),
    ("gpt-5", "GPT-5"),
    ("gpt-4o", "GPT-4o"),
    ("gpt-4o-mini", "GPT-4o Mini (fast)"),
    ("o3", "o3 (reasoning)"),
    ("o3-mini", "o3 Mini"),
]

if TYPE_CHECKING:
    from tuxedo.project import Project


# ============================================================================
# Confirmation Dialog
# ============================================================================


class MoveToClusterDialog(ModalScreen[str | None]):
    """A modal dialog for selecting a target cluster."""

    CSS = """
    MoveToClusterDialog {
        align: center middle;
    }

    #move-dialog {
        width: 70;
        height: auto;
        max-height: 80%;
        background: $surface;
        padding: 1 2;
    }

    #move-dialog .title {
        text-style: bold;
        margin-bottom: 1;
    }

    #move-dialog ListView {
        height: auto;
        max-height: 15;
        margin-bottom: 1;
    }

    #move-dialog ListItem {
        padding: 0 1;
    }

    #move-dialog ListItem.--highlight {
        background: $surface-lighten-2;
    }
    """

    def __init__(self, clusters: list[Cluster], paper_title: str):
        super().__init__()
        self.clusters = clusters
        self.paper_title = paper_title
        self._cluster_map: dict[str, Cluster] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="move-dialog"):
            yield Label(f"Move: {self.paper_title[:50]}...", classes="title")
            yield Label("Select target cluster:", classes="message")
            yield ListView(id="cluster-list")

    def on_mount(self) -> None:
        list_view = self.query_one("#cluster-list", ListView)
        self._add_clusters(list_view, self.clusters, "")

    def _add_clusters(self, list_view: ListView, clusters: list[Cluster], prefix: str) -> None:
        for cluster in clusters:
            item_id = cluster.id
            self._cluster_map[item_id] = cluster
            item = ListItem(Label(f"{prefix}{cluster.name}"), id=f"cluster-{item_id}")
            list_view.append(item)
            self._add_clusters(list_view, cluster.subclusters, prefix + "  ")

    @on(ListView.Selected)
    def on_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id
        if item_id and item_id.startswith("cluster-"):
            cluster_id = item_id[8:]  # Remove "cluster-" prefix
            self.dismiss(cluster_id)

    def key_escape(self) -> None:
        self.dismiss(None)


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


class CreateClusterDialog(ModalScreen[dict | None]):
    """A modal dialog for creating a new cluster view."""

    CSS = """
    CreateClusterDialog {
        align: center middle;
    }

    #create-dialog {
        width: 80;
        height: auto;
        background: $surface;
        padding: 1 2;
    }

    #create-dialog .title {
        text-style: bold;
        margin-bottom: 1;
    }

    #create-dialog .hint {
        color: $text-muted;
        margin-bottom: 1;
    }

    #create-dialog Input {
        margin-bottom: 1;
    }

    #create-dialog Select {
        margin-bottom: 1;
        width: 100%;
    }

    #create-dialog .buttons {
        margin-top: 1;
    }

    #create-dialog Button {
        margin-right: 1;
    }
    """

    def __init__(self, default_prompt: str = ""):
        super().__init__()
        self.default_prompt = default_prompt

    def compose(self) -> ComposeResult:
        with Vertical(id="create-dialog"):
            yield Label("Create New Clustering View", classes="title")
            yield Label("Different prompts organize papers in different ways", classes="hint")
            yield Input(placeholder="View name (e.g., 'By Methodology')", id="new-view-name")
            yield Input(placeholder="Clustering prompt (leave empty for research question)", id="new-view-prompt")
            yield Input(placeholder="Include sections (e.g., 'method,results')", id="new-view-sections")
            yield Select(
                [(label, value) for value, label in MODEL_OPTIONS],
                value="gpt-5.2",
                id="new-view-model",
                prompt="Select model",
            )
            with Horizontal(classes="buttons"):
                yield Button("Create", variant="primary", id="create-btn")
                yield Button("Cancel", id="cancel-btn")

    def on_mount(self) -> None:
        self.query_one("#new-view-name", Input).focus()

    @on(Button.Pressed, "#create-btn")
    def on_create(self) -> None:
        name = self.query_one("#new-view-name", Input).value.strip()
        prompt = self.query_one("#new-view-prompt", Input).value.strip()
        sections = self.query_one("#new-view-sections", Input).value.strip()
        model_select = self.query_one("#new-view-model", Select)
        model = model_select.value if model_select.value != Select.BLANK else "gpt-5.2"

        self.dismiss({
            "name": name,
            "prompt": prompt or self.default_prompt,
            "sections": sections,
            "model": model,
        })

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)


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
        Binding("q", "quit", "Quit", priority=True),
        Binding("n", "new_view", "New", priority=True),
        Binding("enter", "select", "Select"),
        Binding("d", "delete", "Delete", priority=True),
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
        background: $surface-lighten-1;
    }

    ListView > ListItem.--highlight {
        background: $surface-lighten-2;
    }
    """

    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self._pending_delete: ClusterView | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="view-list-container"):
            yield Static("Select a Clustering View", classes="title")
            paper_count = self.project.paper_count()
            yield Static(f"[dim]{paper_count} papers in project[/dim]", classes="subtitle")
            yield ListView(id="view-list")

    def on_mount(self) -> None:
        self._refresh_list()
        self.query_one("#view-list", ListView).focus()

    def _refresh_list(self) -> None:
        """Refresh the view list."""
        list_view = self.query_one("#view-list", ListView)
        list_view.clear()

        views = self.project.get_views()
        for view in views:
            count = self.project.cluster_count(view.id)
            list_view.append(ViewListItem(view, count))

        list_view.append(NewViewItem())

    @on(ListView.Selected)
    def on_list_selected(self, event: ListView.Selected) -> None:
        """Handle list item selection."""
        if isinstance(event.item, NewViewItem):
            self.action_new_view()
        elif isinstance(event.item, ViewListItem):
            self.app.push_screen(ClusterScreen(self.project, event.item.view))

    def action_new_view(self) -> None:
        """Show the create cluster dialog."""
        def handle_create(result: dict | None) -> None:
            if result:
                self._create_new_view(result)

        self.app.push_screen(
            CreateClusterDialog(self.project.config.research_question),
            handle_create,
        )

    def action_quit(self) -> None:
        self.app.exit()

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
    def _create_new_view(self, config: dict) -> None:
        """Create a new clustering view."""
        name = config["name"]
        prompt = config["prompt"]
        sections_input = config["sections"]
        model = config["model"]

        # Parse section patterns
        section_patterns = None
        if sections_input:
            section_patterns = [s.strip() for s in sections_input.split(",") if s.strip()]

        if not name:
            views = self.project.get_views()
            name = f"View {len(views) + 1}"

        self.app.call_from_thread(self.notify, f"Clustering with {model}...")

        try:
            # Create view and cluster
            view = self.project.create_view(name=name, prompt=prompt)
            papers = self.project.get_papers()

            if not papers:
                self.app.call_from_thread(self.notify, "No papers to cluster", severity="warning")
                return

            clusterer = PaperClusterer(model=model)
            clusters = clusterer.cluster_papers(papers, prompt, include_sections=section_patterns)
            self.project.save_clusters(view.id, clusters)

            self.app.call_from_thread(self._refresh_list)
            self.app.call_from_thread(self.notify, f"Created '{name}' with {len(clusters)} clusters")
        except Exception as e:
            self.app.call_from_thread(self.notify, f"Failed to create view: {e}", severity="error")


# ============================================================================
# Cluster View Screen
# ============================================================================


class PaperDetail(Static):
    """Widget to display paper details."""

    DEFAULT_CSS = """
    PaperDetail {
        background: $surface;
        padding: 1 2;
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

    BINDINGS = [
        Binding("right", "expand_node", "Expand", show=False),
        Binding("left", "collapse_node", "Collapse", show=False),
    ]

    DEFAULT_CSS = """
    ClusterTree {
        background: $surface;
        padding: 1;
        scrollbar-gutter: stable;
        overflow-x: auto;
    }

    ClusterTree > .tree--guides {
        color: $text-muted;
    }

    ClusterTree > .tree--cursor {
        background: $surface-lighten-2;
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
        self.root.expand()  # Only show first level of themes

    def _build_tree(self) -> None:
        self.root.remove_children()

        if not self.clusters:
            self.root.add_leaf("No clusters yet")
            return

        # Collect all paper IDs used in clusters
        used_paper_ids: set[str] = set()
        self._collect_used_papers(self.clusters, used_paper_ids)

        for cluster in self.clusters:
            self._add_cluster_node(self.root, cluster)

        # Add uncategorized papers (sorted alphabetically)
        uncategorized = sorted(
            [p for p in self.all_papers if p.id not in used_paper_ids and self._matches_filter(p)],
            key=lambda p: p.title.lower()
        )
        if uncategorized:
            node = self.root.add(f"[bold italic]Uncategorized[/bold italic] ({len(uncategorized)})", data={"type": "uncategorized"})
            for paper in uncategorized:
                year_suffix = f" [{paper.year}]" if paper.year else ""
                node.add_leaf(f"[dim]{paper.title}{year_suffix}[/dim]", data={"type": "paper", "paper": paper})

    def _collect_used_papers(self, clusters: list[Cluster], used: set[str]) -> None:
        """Recursively collect all paper IDs used in clusters."""
        for cluster in clusters:
            used.update(cluster.paper_ids)
            self._collect_used_papers(cluster.subclusters, used)

    def get_expanded_state(self) -> set[str]:
        """Get the set of expanded cluster IDs."""
        expanded = set()
        self._collect_expanded(self.root, expanded)
        return expanded

    def _collect_expanded(self, node: TreeNode, expanded: set[str]) -> None:
        """Recursively collect expanded cluster IDs."""
        if node.is_expanded and node.data:
            if node.data.get("type") == "cluster":
                cluster = node.data.get("cluster")
                if cluster:
                    expanded.add(cluster.id)
            elif node.data.get("type") == "uncategorized":
                expanded.add("__uncategorized__")
        for child in node.children:
            self._collect_expanded(child, expanded)

    def restore_expanded_state(self, expanded: set[str]) -> None:
        """Restore expanded state from a set of cluster IDs."""
        self._restore_expanded(self.root, expanded)

    def _restore_expanded(self, node: TreeNode, expanded: set[str]) -> None:
        """Recursively restore expanded state."""
        if node.data:
            cluster_id = None
            if node.data.get("type") == "cluster":
                cluster = node.data.get("cluster")
                if cluster:
                    cluster_id = cluster.id
            elif node.data.get("type") == "uncategorized":
                cluster_id = "__uncategorized__"

            if cluster_id and cluster_id in expanded:
                node.expand()
        for child in node.children:
            self._restore_expanded(child, expanded)

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

        # Sort papers alphabetically by title
        sorted_papers = sorted(
            [self._paper_map[pid] for pid in matching_papers],
            key=lambda p: p.title.lower()
        )
        for paper in sorted_papers:
            year_suffix = f" [{paper.year}]" if paper.year else ""
            node.add_leaf(f"[dim]{paper.title}{year_suffix}[/dim]", data={"type": "paper", "paper": paper})

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

    def action_expand_node(self) -> None:
        """Expand the current node."""
        node = self.cursor_node
        if node and not node.is_expanded and node.children:
            node.expand()

    def action_collapse_node(self) -> None:
        """Collapse current node, or move to parent if already collapsed."""
        node = self.cursor_node
        if not node:
            return
        if node.is_expanded and node.children:
            node.collapse()
        elif node.parent and node.parent != self.root:
            # Move to parent and collapse it
            self.select_node(node.parent)
            node.parent.collapse()


class ClusterScreen(Screen):
    """Screen for viewing clusters in a specific view."""

    BINDINGS = [
        Binding("q", "back", "Back", priority=True),
        Binding("o", "open_pdf", "Open PDF"),
        Binding("m", "move_paper", "Move"),
        Binding("/", "search", "Search", priority=True),
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
        width: 60%;
        height: 100%;
    }

    ClusterScreen #detail-container {
        width: 40%;
        height: 100%;
        padding: 0 1;
    }

    ClusterScreen .section-title {
        text-style: bold;
        color: $text-muted;
        padding: 0 1;
        margin-bottom: 1;
    }

    ClusterScreen .view-header {
        color: $text-muted;
        padding: 1 2;
        dock: top;
    }

    ClusterScreen .status-bar {
        color: $text-muted;
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
        yield Static(f"[bold]{self.view.name}[/bold] | {self.view.prompt[:60]}...", classes="view-header")
        with Container(id="search-container"):
            yield Input(placeholder="Search papers by title, author, abstract...", id="search-input")
        with Horizontal(id="main-container"):
            with Vertical(id="tree-container"):
                self._tree = ClusterTree(self.view.name, self.papers, self.clusters)
                yield self._tree
            with Vertical(id="detail-container"):
                with Container(id="paper-detail-wrapper") as self._detail:
                    yield PaperDetail()
        yield Static(f"{len(self.papers)} papers | {len(self.clusters)} clusters | Press / to search", classes="status-bar")

    def on_mount(self) -> None:
        """Focus the tree when screen opens."""
        if self._tree:
            self._tree.focus()

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

    def key_q(self) -> None:
        """Handle q key to go back."""
        self.app.pop_screen()

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

    def action_move_paper(self) -> None:
        """Move selected paper to a different cluster."""
        if not self._tree:
            return

        paper = self._tree.get_selected_paper()
        if not paper:
            self.notify("Select a paper to move", severity="warning")
            return

        def handle_move(target_cluster_id: str | None) -> None:
            if target_cluster_id and paper:
                self.project.move_paper_to_cluster(self.view.id, paper.id, target_cluster_id)
                # Refresh clusters and tree, preserving expanded state
                self.clusters = self.project.get_clusters(self.view.id)
                if self._tree:
                    expanded = self._tree.get_expanded_state()
                    self._tree.clusters = self.clusters
                    self._tree._build_tree()
                    self._tree.restore_expanded_state(expanded)
                self.notify(f"Moved paper to cluster")

        self.app.push_screen(
            MoveToClusterDialog(self.clusters, paper.title),
            handle_move,
        )

    def action_help(self) -> None:
        self.notify(
            "/ Search | o Open | m Move | e/c Expand/Collapse | q Back",
            timeout=5,
        )


# ============================================================================
# Main App
# ============================================================================


class TuxedoApp(App):
    """Main TUI application."""

    TITLE = "Tuxedo"
    dark = True  # Always use dark mode

    def __init__(self, project: Project):
        super().__init__()
        self.project = project

    def on_mount(self) -> None:
        self.push_screen(ViewSelectionScreen(self.project))


def run_tui(project: Project) -> None:
    """Run the TUI application."""
    app = TuxedoApp(project)
    app.run()
