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
from textual.widgets import (
    Button,
    Footer,
    Input,
    Label,
    ListItem,
    ListView,
    LoadingIndicator,
    Select,
    Static,
    Tree,
)
from textual.widgets.tree import TreeNode

from tuxedo.clustering import PaperClusterer
from tuxedo.models import Cluster, ClusterView, Paper, PaperAnswer, Question

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

# Auto-discovery mode options
AUTO_MODE_OPTIONS = [
    ("", "Use research question/prompt"),
    ("themes", "Auto: Discover themes"),
    ("methodology", "Auto: Group by methodology"),
    ("domain", "Auto: Group by domain"),
    ("temporal", "Auto: Temporal evolution"),
    ("findings", "Auto: Group by findings"),
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
        border: none;
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


class ClusteringProgressScreen(ModalScreen[None]):
    """A modal screen showing clustering progress.

    This screen blocks interaction while clustering is in progress,
    displaying a loading indicator and status message.
    """

    CSS = """
    ClusteringProgressScreen {
        align: center middle;
    }

    #progress-dialog {
        width: 60;
        height: auto;
        background: $surface;
        padding: 2 3;
    }

    #progress-dialog .title {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }

    #progress-dialog .status {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }

    #progress-dialog LoadingIndicator {
        width: 100%;
        height: 3;
    }
    """

    def __init__(self, title: str = "Clustering Papers"):
        super().__init__()
        self.title_text = title
        self._status = "Initializing..."

    def compose(self) -> ComposeResult:
        with Vertical(id="progress-dialog"):
            yield Label(self.title_text, classes="title")
            yield LoadingIndicator()
            yield Label(self._status, classes="status", id="status-label")

    def update_status(self, status: str) -> None:
        """Update the status message."""
        self._status = status
        try:
            label = self.query_one("#status-label", Label)
            label.update(status)
        except Exception:
            pass  # Screen may not be fully mounted yet


class EditPaperDialog(ModalScreen[dict | None]):
    """A modal dialog for editing paper metadata."""

    CSS = """
    EditPaperDialog {
        align: center middle;
    }

    #edit-dialog {
        width: 95%;
        max-width: 120;
        height: auto;
        max-height: 85%;
        background: $surface;
        padding: 1 2;
        overflow-y: auto;
        border: solid $primary;
    }

    #edit-dialog .title {
        text-style: bold;
        text-align: center;
        padding: 1;
        background: $primary;
        color: $text;
        margin-bottom: 1;
    }

    #edit-dialog .hint {
        color: $text-muted;
        text-align: center;
        margin-bottom: 1;
    }

    #edit-dialog .field-label {
        color: $text-muted;
        height: 1;
        margin: 0;
        padding: 0;
    }

    #edit-dialog Input {
        margin: 0 0 1 0;
    }

    #edit-dialog .buttons {
        margin-top: 1;
        height: auto;
        align: center middle;
    }

    #edit-dialog .buttons Button {
        margin: 0 1;
        border: none;
    }

    #edit-dialog .row {
        layout: horizontal;
        height: auto;
        margin: 0;
    }

    #edit-dialog .row > Vertical {
        width: 1fr;
        padding-right: 1;
        height: auto;
    }

    #edit-dialog .row3 {
        layout: horizontal;
        height: auto;
        margin: 0;
    }

    #edit-dialog .row3 > Vertical {
        width: 1fr;
        padding-right: 1;
        height: auto;
    }
    """

    def __init__(self, paper: Paper):
        super().__init__()
        self.paper = paper

    def compose(self) -> ComposeResult:
        with Vertical(id="edit-dialog"):
            yield Label("Edit Paper Metadata", classes="title")
            yield Label(
                f"{self.paper.title[:80]}{'...' if len(self.paper.title) > 80 else ''}",
                classes="hint",
            )

            yield Label("Title", classes="field-label")
            yield Input(value=self.paper.title, id="edit-title")

            yield Label("Authors (comma-separated)", classes="field-label")
            authors_str = ", ".join(a.name for a in self.paper.authors)
            yield Input(value=authors_str, id="edit-authors")

            yield Label("Abstract", classes="field-label")
            yield Input(value=self.paper.abstract or "", id="edit-abstract")

            with Horizontal(classes="row"):
                with Vertical():
                    yield Label("Journal", classes="field-label")
                    yield Input(value=self.paper.journal or "", id="edit-journal")
                with Vertical():
                    yield Label("Conference/Booktitle", classes="field-label")
                    yield Input(value=self.paper.booktitle or "", id="edit-booktitle")

            with Horizontal(classes="row3"):
                with Vertical():
                    yield Label("Year", classes="field-label")
                    yield Input(
                        value=str(self.paper.year) if self.paper.year else "", id="edit-year"
                    )
                with Vertical():
                    yield Label("Volume", classes="field-label")
                    yield Input(value=self.paper.volume or "", id="edit-volume")
                with Vertical():
                    yield Label("Issue", classes="field-label")
                    yield Input(value=self.paper.number or "", id="edit-number")
                with Vertical():
                    yield Label("Pages", classes="field-label")
                    yield Input(value=self.paper.pages or "", id="edit-pages")

            with Horizontal(classes="row"):
                with Vertical():
                    yield Label("DOI", classes="field-label")
                    yield Input(value=self.paper.doi or "", id="edit-doi")
                with Vertical():
                    yield Label("arXiv ID", classes="field-label")
                    yield Input(value=self.paper.arxiv_id or "", id="edit-arxiv")

            with Horizontal(classes="row"):
                with Vertical():
                    yield Label("Publisher", classes="field-label")
                    yield Input(value=self.paper.publisher or "", id="edit-publisher")
                with Vertical():
                    yield Label("URL", classes="field-label")
                    yield Input(value=self.paper.url or "", id="edit-url")

            yield Label("Keywords (comma-separated)", classes="field-label")
            keywords_str = ", ".join(self.paper.keywords)
            yield Input(value=keywords_str, id="edit-keywords")

            with Horizontal(classes="buttons"):
                yield Button("Save", variant="primary", id="save-btn")
                yield Button("Cancel", id="cancel-btn")

    def on_mount(self) -> None:
        self.query_one("#edit-title", Input).focus()

    @on(Button.Pressed, "#save-btn")
    def on_save(self) -> None:
        from tuxedo.models import Author

        updates = {}

        # Title
        title = self.query_one("#edit-title", Input).value.strip()
        if title and title != self.paper.title:
            updates["title"] = title

        # Authors
        authors_str = self.query_one("#edit-authors", Input).value.strip()
        if authors_str:
            new_authors = [
                Author(name=name.strip()) for name in authors_str.split(",") if name.strip()
            ]
            current_authors = [a.name for a in self.paper.authors]
            new_author_names = [a.name for a in new_authors]
            if new_author_names != current_authors:
                updates["authors"] = new_authors

        # Abstract
        abstract = self.query_one("#edit-abstract", Input).value.strip()
        if abstract != (self.paper.abstract or ""):
            updates["abstract"] = abstract or None

        # Year
        year_str = self.query_one("#edit-year", Input).value.strip()
        if year_str:
            try:
                year = int(year_str)
                if year != self.paper.year:
                    updates["year"] = year
            except ValueError:
                pass
        elif self.paper.year:
            updates["year"] = None

        # DOI
        doi = self.query_one("#edit-doi", Input).value.strip()
        if doi != (self.paper.doi or ""):
            updates["doi"] = doi or None

        # Journal
        journal = self.query_one("#edit-journal", Input).value.strip()
        if journal != (self.paper.journal or ""):
            updates["journal"] = journal or None

        # Booktitle
        booktitle = self.query_one("#edit-booktitle", Input).value.strip()
        if booktitle != (self.paper.booktitle or ""):
            updates["booktitle"] = booktitle or None

        # Publisher
        publisher = self.query_one("#edit-publisher", Input).value.strip()
        if publisher != (self.paper.publisher or ""):
            updates["publisher"] = publisher or None

        # Volume
        volume = self.query_one("#edit-volume", Input).value.strip()
        if volume != (self.paper.volume or ""):
            updates["volume"] = volume or None

        # Number
        number = self.query_one("#edit-number", Input).value.strip()
        if number != (self.paper.number or ""):
            updates["number"] = number or None

        # Pages
        pages = self.query_one("#edit-pages", Input).value.strip()
        if pages != (self.paper.pages or ""):
            updates["pages"] = pages or None

        # arXiv ID
        arxiv_id = self.query_one("#edit-arxiv", Input).value.strip()
        if arxiv_id != (self.paper.arxiv_id or ""):
            updates["arxiv_id"] = arxiv_id or None

        # URL
        url = self.query_one("#edit-url", Input).value.strip()
        if url != (self.paper.url or ""):
            updates["url"] = url or None

        # Keywords
        keywords_str = self.query_one("#edit-keywords", Input).value.strip()
        new_keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
        if new_keywords != self.paper.keywords:
            updates["keywords"] = new_keywords

        self.dismiss(updates if updates else None)

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)


class ReclusterDialog(ModalScreen[str | None]):
    """A modal dialog for providing recluster feedback."""

    CSS = """
    ReclusterDialog {
        align: center middle;
    }

    #recluster-dialog {
        width: 80;
        height: auto;
        background: $surface;
        padding: 1 2;
    }

    #recluster-dialog .title {
        text-style: bold;
        margin-bottom: 1;
    }

    #recluster-dialog .hint {
        color: $text-muted;
        margin-bottom: 1;
    }

    #recluster-dialog Input {
        margin-bottom: 1;
    }

    #recluster-dialog .buttons {
        margin-top: 1;
    }

    #recluster-dialog Button {
        margin-right: 1;
        border: none;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="recluster-dialog"):
            yield Label("Recluster with Feedback", classes="title")
            yield Label("Describe how you'd like the clusters reorganized:", classes="hint")
            yield Input(
                placeholder="e.g., 'Merge the ML and DL clusters' or 'Split methodology by quantitative/qualitative'",
                id="feedback-input",
            )
            with Horizontal(classes="buttons"):
                yield Button("Recluster", variant="primary", id="recluster-btn")
                yield Button("Cancel", id="cancel-btn")

    def on_mount(self) -> None:
        self.query_one("#feedback-input", Input).focus()

    @on(Button.Pressed, "#recluster-btn")
    def on_recluster(self) -> None:
        feedback = self.query_one("#feedback-input", Input).value.strip()
        if feedback:
            self.dismiss(feedback)
        else:
            self.notify("Please provide feedback", severity="warning")

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)


class RenameClusterDialog(ModalScreen[dict | None]):
    """A modal dialog for renaming a cluster."""

    CSS = """
    RenameClusterDialog {
        align: center middle;
    }

    #rename-dialog {
        width: 80;
        height: auto;
        background: $surface;
        padding: 1 2;
    }

    #rename-dialog .title {
        text-style: bold;
        margin-bottom: 1;
    }

    #rename-dialog .hint {
        color: $text-muted;
        margin-bottom: 1;
    }

    #rename-dialog .field-label {
        color: $text-muted;
        margin-top: 1;
    }

    #rename-dialog Input {
        margin-bottom: 1;
    }

    #rename-dialog .buttons {
        margin-top: 1;
    }

    #rename-dialog Button {
        margin-right: 1;
        border: none;
    }
    """

    def __init__(self, cluster: Cluster):
        super().__init__()
        self.cluster = cluster

    def compose(self) -> ComposeResult:
        with Vertical(id="rename-dialog"):
            yield Label("Rename Cluster", classes="title")
            yield Label(f"Editing: {self.cluster.name}", classes="hint")
            yield Label("Name", classes="field-label")
            yield Input(value=self.cluster.name, id="rename-name")
            yield Label("Description", classes="field-label")
            yield Input(value=self.cluster.description or "", id="rename-description")
            with Horizontal(classes="buttons"):
                yield Button("Save", variant="primary", id="save-btn")
                yield Button("Cancel", id="cancel-btn")

    def on_mount(self) -> None:
        self.query_one("#rename-name", Input).focus()

    @on(Button.Pressed, "#save-btn")
    def on_save(self) -> None:
        name = self.query_one("#rename-name", Input).value.strip()
        description = self.query_one("#rename-description", Input).value.strip()

        if not name:
            self.notify("Name cannot be empty", severity="warning")
            return

        result = {"name": name}
        if description != (self.cluster.description or ""):
            result["description"] = description or None

        self.dismiss(result)

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)


class RenameViewDialog(ModalScreen[str | None]):
    """A modal dialog for renaming a cluster view."""

    CSS = """
    RenameViewDialog {
        align: center middle;
    }

    #rename-view-dialog {
        width: 70;
        height: auto;
        background: $surface;
        padding: 1 2;
    }

    #rename-view-dialog .title {
        text-style: bold;
        margin-bottom: 1;
    }

    #rename-view-dialog .hint {
        color: $text-muted;
        margin-bottom: 1;
    }

    #rename-view-dialog Input {
        margin-bottom: 1;
    }

    #rename-view-dialog .buttons {
        margin-top: 1;
    }

    #rename-view-dialog Button {
        margin-right: 1;
        border: none;
    }
    """

    def __init__(self, view: ClusterView):
        super().__init__()
        self.view = view

    def compose(self) -> ComposeResult:
        with Vertical(id="rename-view-dialog"):
            yield Label("Rename View", classes="title")
            yield Label(f"Current: {self.view.name}", classes="hint")
            yield Input(value=self.view.name, id="rename-view-name")
            with Horizontal(classes="buttons"):
                yield Button("Save", variant="primary", id="save-btn")
                yield Button("Cancel", id="cancel-btn")

    def on_mount(self) -> None:
        self.query_one("#rename-view-name", Input).focus()

    @on(Button.Pressed, "#save-btn")
    def on_save(self) -> None:
        name = self.query_one("#rename-view-name", Input).value.strip()
        if not name:
            self.notify("Name cannot be empty", severity="warning")
            return
        self.dismiss(name)

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)


EXPORT_FORMATS = [
    ("markdown", "Markdown - Hierarchical outline"),
    ("bibtex", "BibTeX - LaTeX bibliography"),
    ("csv", "CSV - Spreadsheet format"),
    ("ris", "RIS - Reference managers"),
    ("json", "JSON - Structured data"),
    ("latex", "LaTeX - Document skeleton"),
]


class ExportDialog(ModalScreen[dict | None]):
    """A modal dialog for exporting a view."""

    CSS = """
    ExportDialog {
        align: center middle;
    }

    #export-dialog {
        width: 70;
        height: auto;
        background: $surface;
        padding: 1 2;
        border: solid $primary;
    }

    #export-dialog .title {
        text-style: bold;
        margin-bottom: 1;
    }

    #export-dialog .hint {
        color: $text-muted;
        margin-bottom: 1;
    }

    #export-dialog .field-label {
        color: $text-muted;
        margin-top: 1;
    }

    #export-dialog Select {
        width: 100%;
    }

    #export-dialog Input {
        margin-top: 1;
    }

    #export-dialog .buttons {
        margin-top: 1;
    }

    #export-dialog Button {
        margin-right: 1;
        border: none;
    }
    """

    def __init__(self, view_name: str):
        super().__init__()
        self.view_name = view_name

    def compose(self) -> ComposeResult:
        with Vertical(id="export-dialog"):
            yield Label("Export View", classes="title")
            yield Label(f"Exporting: {self.view_name}", classes="hint")
            yield Label("Format", classes="field-label")
            yield Select(
                [(label, value) for value, label in EXPORT_FORMATS],
                value="markdown",
                id="export-format",
            )
            yield Label("Output file (leave empty for clipboard)", classes="field-label")
            yield Input(placeholder="e.g., export.md", id="export-path")
            with Horizontal(classes="buttons"):
                yield Button("Export", variant="primary", id="export-btn")
                yield Button("Cancel", id="cancel-btn")

    def on_mount(self) -> None:
        self.query_one("#export-format", Select).focus()

    @on(Button.Pressed, "#export-btn")
    def on_export(self) -> None:
        format_select = self.query_one("#export-format", Select)
        format_value = format_select.value if format_select.value != Select.BLANK else "markdown"
        path = self.query_one("#export-path", Input).value.strip()
        self.dismiss({"format": format_value, "path": path or None})

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)


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
        border: none;
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
            yield Select(
                [(label, value) for value, label in AUTO_MODE_OPTIONS],
                value="",
                id="new-view-auto",
                prompt="Clustering mode",
            )
            yield Input(
                placeholder="Custom prompt (ignored if auto mode selected)",
                id="new-view-prompt",
            )
            yield Input(
                placeholder="Categories (comma-separated, e.g., 'Quantitative, Qualitative, Mixed')",
                id="new-view-categories",
            )
            yield Input(
                placeholder="Include sections (e.g., 'method,results')", id="new-view-sections"
            )
            yield Input(
                placeholder="Batch size (e.g., 10) for large paper sets", id="new-view-batch"
            )
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
        auto_select = self.query_one("#new-view-auto", Select)
        auto_mode = auto_select.value if auto_select.value != Select.BLANK else ""
        prompt = self.query_one("#new-view-prompt", Input).value.strip()
        categories_input = self.query_one("#new-view-categories", Input).value.strip()
        sections = self.query_one("#new-view-sections", Input).value.strip()
        batch_input = self.query_one("#new-view-batch", Input).value.strip()
        model_select = self.query_one("#new-view-model", Select)
        model = model_select.value if model_select.value != Select.BLANK else "gpt-5.2"

        # Parse batch size
        batch_size = None
        if batch_input:
            try:
                batch_size = int(batch_input)
            except ValueError:
                pass

        # Parse categories
        categories = None
        if categories_input:
            categories = [c.strip() for c in categories_input.split(",") if c.strip()]

        self.dismiss(
            {
                "name": name,
                "prompt": prompt or self.default_prompt,
                "sections": sections,
                "model": model,
                "batch_size": batch_size,
                "auto_mode": auto_mode if auto_mode else None,
                "categories": categories,
            }
        )

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)


class AskQuestionDialog(ModalScreen[dict | None]):
    """A modal dialog for asking a question about papers.

    Args:
        total_papers: Total number of papers in the view
        cluster_name: Name of the selected cluster (if any)
        cluster_paper_count: Number of papers in the selected cluster (if any)
    """

    CSS = """
    AskQuestionDialog {
        align: center middle;
    }

    #ask-dialog {
        width: 80;
        height: auto;
        background: $surface;
        padding: 1 2;
    }

    #ask-dialog .title {
        text-style: bold;
        margin-bottom: 1;
    }

    #ask-dialog .hint {
        color: $text-muted;
        margin-bottom: 1;
    }

    #ask-dialog Input {
        margin-bottom: 1;
    }

    #ask-dialog Select {
        margin-bottom: 1;
        width: 100%;
    }

    #ask-dialog .buttons {
        margin-top: 1;
    }

    #ask-dialog Button {
        margin-right: 1;
        border: none;
    }
    """

    def __init__(
        self,
        total_papers: int = 0,
        cluster_name: str | None = None,
        cluster_paper_count: int = 0,
    ):
        super().__init__()
        self.total_papers = total_papers
        self.cluster_name = cluster_name
        self.cluster_paper_count = cluster_paper_count

    def compose(self) -> ComposeResult:
        with Vertical(id="ask-dialog"):
            yield Label("Ask a Question", classes="title")
            yield Label(
                "Ask a question to analyze across papers",
                classes="hint",
            )
            yield Input(
                placeholder="e.g., 'What methodology does this paper use?' or 'What are the main findings?'",
                id="question-input",
            )
            yield Select(
                [(label, value) for value, label in MODEL_OPTIONS],
                value="gpt-4o-mini",
                id="ask-model",
                prompt="Select model",
            )
            # Show scope selection if a cluster is selected
            if self.cluster_name and self.cluster_paper_count > 0:
                cluster_short = (
                    self.cluster_name[:20] + "..."
                    if len(self.cluster_name) > 20
                    else self.cluster_name
                )
                yield Select(
                    [
                        (f"All papers ({self.total_papers})", "all"),
                        (f"This cluster: {cluster_short} ({self.cluster_paper_count})", "cluster"),
                    ],
                    value="all",
                    id="ask-scope",
                    prompt="Analyze scope",
                )
            with Horizontal(classes="buttons"):
                yield Button("Analyze", variant="primary", id="analyze-btn")
                yield Button("Cancel", id="cancel-btn")

    def on_mount(self) -> None:
        self.query_one("#question-input", Input).focus()

    @on(Button.Pressed, "#analyze-btn")
    def on_analyze(self) -> None:
        question = self.query_one("#question-input", Input).value.strip()
        if not question:
            self.notify("Please enter a question", severity="warning")
            return

        model_select = self.query_one("#ask-model", Select)
        model = model_select.value if model_select.value != Select.BLANK else "gpt-4o-mini"

        # Get scope if available
        scope = "all"
        try:
            scope_select = self.query_one("#ask-scope", Select)
            if scope_select.value != Select.BLANK:
                scope = scope_select.value
        except Exception:
            pass  # Scope select not present

        self.dismiss({"question": question, "model": model, "scope": scope})

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)


class AnalysisProgressScreen(ModalScreen[None]):
    """A modal screen showing analysis progress.

    This screen blocks interaction while analysis is in progress,
    displaying a loading indicator and status message.
    """

    CSS = """
    AnalysisProgressScreen {
        align: center middle;
    }

    #analysis-dialog {
        width: 70;
        height: auto;
        background: $surface;
        padding: 2 3;
    }

    #analysis-dialog .title {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }

    #analysis-dialog .question {
        color: $text-muted;
        text-align: center;
        margin-bottom: 1;
    }

    #analysis-dialog .status {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }

    #analysis-dialog LoadingIndicator {
        width: 100%;
        height: 3;
    }
    """

    def __init__(self, question: str):
        super().__init__()
        self.question = question
        self._status = "Initializing..."

    def compose(self) -> ComposeResult:
        with Vertical(id="analysis-dialog"):
            yield Label("Analyzing Papers", classes="title")
            question_short = (
                self.question[:60] + "..." if len(self.question) > 60 else self.question
            )
            yield Label(f"Q: {question_short}", classes="question")
            yield LoadingIndicator()
            yield Label(self._status, classes="status", id="analysis-status")

    def update_status(self, status: str) -> None:
        """Update the status message."""
        self._status = status
        try:
            label = self.query_one("#analysis-status", Label)
            label.update(status)
        except Exception:
            pass  # Screen may not be fully mounted yet


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
        prompt_short = (
            self.view.prompt[:50] + "..." if len(self.view.prompt) > 50 else self.view.prompt
        )
        yield Label(f"[dim]{self.cluster_count} clusters | {prompt_short}[/dim]")


class NewViewItem(ListItem):
    """A list item for creating a new view."""

    def compose(self) -> ComposeResult:
        yield Label("[bold green]+ New Clustering...[/bold green]")
        yield Label("[dim]Organize papers with a different prompt or focus[/dim]")


class QuestionListItem(ListItem):
    """A list item representing a question."""

    def __init__(self, question: Question, answer_count: int):
        super().__init__()
        self.question = question
        self.answer_count = answer_count

    def compose(self) -> ComposeResult:
        text = self.question.text
        if len(text) > 60:
            text = text[:57] + "..."
        yield Label(f"[bold]{text}[/bold]")
        date_str = self.question.created_at.strftime("%Y-%m-%d")
        yield Label(f"[dim]{self.answer_count} answers | {date_str}[/dim]")


class ViewSelectionScreen(Screen):
    """Screen for selecting or creating a cluster view.

    This is the initial screen shown when launching the TUI. It displays
    a list of existing clustering views and allows users to:
    - Select a view to explore its clusters
    - Create a new clustering view with custom prompts
    - Rename existing views
    - Delete existing views

    Bindings:
        n: Create a new clustering view
        r: Rename the selected view
        d: Delete the selected view
        L: View application logs
        q: Quit the application
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("n", "new_view", "New", priority=True),
        Binding("enter", "select", "Select"),
        Binding("r", "rename", "Rename", priority=True),
        Binding("d", "delete", "Delete", priority=True),
        Binding("Q", "questions", "Questions"),
        Binding("L", "view_logs", "Logs"),
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
        self._progress_screen: ClusteringProgressScreen | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="view-list-container"):
            yield Static("Select a Clustering View", classes="title")
            paper_count = self.project.paper_count()
            yield Static(f"[dim]{paper_count} papers in project[/dim]", classes="subtitle")
            yield ListView(id="view-list")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_list()
        self.query_one("#view-list", ListView).focus()

    def _refresh_list(self) -> None:
        """Refresh the view list."""
        list_view = self.query_one("#view-list", ListView)
        list_view.clear()

        # New view option always at the top
        list_view.append(NewViewItem())

        views = self.project.get_views()
        for view in views:
            count = self.project.cluster_count(view.id)
            list_view.append(ViewListItem(view, count))

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
                # Show progress screen before starting clustering
                self._progress_screen = ClusteringProgressScreen()
                self.app.push_screen(self._progress_screen)
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

    def action_rename(self) -> None:
        """Rename the selected view."""
        list_view = self.query_one("#view-list", ListView)
        if list_view.highlighted_child and isinstance(list_view.highlighted_child, ViewListItem):
            view = list_view.highlighted_child.view

            def handle_rename(new_name: str | None) -> None:
                if new_name and new_name != view.name:
                    self.project.rename_view(view.id, new_name)
                    self.notify(f"Renamed to '{new_name}'")
                    self._refresh_list()

            self.app.push_screen(RenameViewDialog(view), handle_rename)

    def action_view_logs(self) -> None:
        """Open the log viewer screen."""
        self.app.push_screen(LogViewerScreen())

    def action_questions(self) -> None:
        """Open the questions screen."""
        self.app.push_screen(QuestionsScreen(self.project))

    def _update_progress(self, status: str) -> None:
        """Update the progress screen status from background thread."""
        if self._progress_screen:
            self.app.call_from_thread(self._progress_screen.update_status, status)

    def _dismiss_progress(self) -> None:
        """Dismiss the progress screen from background thread."""
        if self._progress_screen:
            self.app.call_from_thread(self.app.pop_screen)
            self._progress_screen = None

    @work(thread=True)
    def _create_new_view(self, config: dict) -> None:
        """Create a new clustering view."""
        name = config["name"]
        prompt = config["prompt"]
        sections_input = config["sections"]
        model = config["model"]
        batch_size = config.get("batch_size")
        auto_mode = config.get("auto_mode")
        categories = config.get("categories")

        # Parse section patterns
        section_patterns = None
        if sections_input:
            section_patterns = [s.strip() for s in sections_input.split(",") if s.strip()]

        # Handle different clustering modes
        if categories:
            # Guided clustering mode
            if not name:
                name = "Guided: " + ", ".join(categories[:3])
                if len(categories) > 3:
                    name += f" +{len(categories) - 3} more"
            prompt = f"Guided clustering with categories: {', '.join(categories)}"
        elif auto_mode:
            from tuxedo.clustering import AUTO_DISCOVERY_PROMPTS

            auto_focus = AUTO_DISCOVERY_PROMPTS.get(auto_mode, auto_mode)
            if not name:
                mode_name = (
                    auto_mode.capitalize() if auto_mode in AUTO_DISCOVERY_PROMPTS else "Custom"
                )
                name = f"Auto: {mode_name}"
            prompt = f"Auto-discovery: {auto_focus}"
        elif not name:
            views = self.project.get_views()
            name = f"View {len(views) + 1}"

        # Update progress screen with initial status
        if categories:
            self._update_progress(f"Guided clustering into {len(categories)} categories...")
        elif auto_mode:
            self._update_progress(f"Auto-discovering themes with {model}...")
        elif batch_size:
            self._update_progress(f"Clustering with {model} (batch size: {batch_size})...")
        else:
            self._update_progress(f"Clustering {self.project.paper_count()} papers with {model}...")

        try:
            # Create view and cluster
            view = self.project.create_view(name=name, prompt=prompt)
            papers = self.project.get_papers()

            if not papers:
                self._dismiss_progress()
                self.app.call_from_thread(self.notify, "No papers to cluster", severity="warning")
                return

            clusterer = PaperClusterer(model=model)

            # Progress callback for batch mode
            def progress_callback(batch_num: int, total: int, message: str) -> None:
                self._update_progress(message)

            clusters, relevance_scores = clusterer.cluster_papers(
                papers,
                prompt,
                include_sections=section_patterns,
                batch_size=batch_size,
                progress_callback=progress_callback if batch_size else None,
                auto_mode=auto_mode,
                categories=categories,
                allow_new_categories=True,  # TUI defaults to flexible mode
            )
            self.project.save_clusters(view.id, clusters)
            # Update paper relevance scores
            for paper_id, score in relevance_scores.items():
                self.project.update_paper(paper_id, {"relevance_score": score})

            self._dismiss_progress()
            self.app.call_from_thread(self._refresh_list)
            self.app.call_from_thread(
                self.notify, f"Created '{name}' with {len(clusters)} clusters"
            )
        except Exception as e:
            self._dismiss_progress()
            self.app.call_from_thread(self.notify, f"Failed to create view: {e}", severity="error")


# ============================================================================
# Questions Screen
# ============================================================================


class QuestionsScreen(Screen):
    """Screen for viewing and managing questions.

    Displays all questions asked across the project with their answer counts.
    Allows users to delete questions they no longer need.

    Bindings:
        d: Delete the selected question
        q: Go back
    """

    BINDINGS = [
        Binding("q", "back", "Back", priority=True),
        Binding("d", "delete", "Delete", priority=True),
    ]

    CSS = """
    QuestionsScreen {
        align: center middle;
    }

    #questions-container {
        width: 80%;
        max-width: 100;
        height: auto;
        max-height: 80%;
        background: $surface;
        padding: 1 2;
    }

    #questions-container .title {
        text-style: bold;
        margin-bottom: 1;
    }

    #questions-container .subtitle {
        color: $text-muted;
        margin-bottom: 1;
    }

    #questions-list {
        height: auto;
        max-height: 20;
    }

    #questions-list > ListItem {
        padding: 1 2;
    }

    #questions-list > ListItem:hover {
        background: $surface-lighten-1;
    }

    #questions-list > ListItem.--highlight {
        background: $surface-lighten-2;
    }
    """

    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self._pending_delete: Question | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="questions-container"):
            yield Static("Questions", classes="title")
            yield Static("[dim]Questions asked across all papers[/dim]", classes="subtitle")
            yield ListView(id="questions-list")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_list()
        self.query_one("#questions-list", ListView).focus()

    def _refresh_list(self) -> None:
        """Refresh the questions list."""
        list_view = self.query_one("#questions-list", ListView)
        list_view.clear()

        questions = self.project.get_questions()
        if not questions:
            list_view.append(
                ListItem(Label("[dim]No questions yet. Press 'a' in cluster view to ask.[/dim]"))
            )
            return

        for question in questions:
            count = self.project.get_question_answer_count(question.id)
            list_view.append(QuestionListItem(question, count))

    def action_back(self) -> None:
        """Go back to the previous screen."""
        self.app.pop_screen()

    def action_delete(self) -> None:
        """Delete the selected question."""
        list_view = self.query_one("#questions-list", ListView)
        if list_view.highlighted_child and isinstance(
            list_view.highlighted_child, QuestionListItem
        ):
            question = list_view.highlighted_child.question
            self._pending_delete = question

            def handle_confirm(confirmed: bool) -> None:
                if confirmed and self._pending_delete:
                    self.project.delete_question(self._pending_delete.id)
                    self.notify("Deleted question and its answers")
                    self._refresh_list()
                self._pending_delete = None

            # Truncate question for dialog
            q_short = question.text[:50] + "..." if len(question.text) > 50 else question.text
            self.app.push_screen(
                ConfirmDialog(
                    "Delete Question",
                    f"Delete '{q_short}' and all its answers?",
                ),
                handle_confirm,
            )


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

    PaperDetail .qa-section {
        margin-top: 2;
        padding-top: 1;
        border-top: solid $primary;
    }

    PaperDetail .qa-question {
        color: $primary;
        text-style: bold;
        margin-top: 1;
    }

    PaperDetail .qa-answer {
        color: $text;
        margin-left: 1;
    }

    PaperDetail .qa-meta {
        color: $text-muted;
        margin-left: 1;
    }
    """

    def __init__(
        self,
        paper: Paper | None = None,
        qa_pairs: list[tuple[Question, PaperAnswer]] | None = None,
    ):
        super().__init__()
        self.paper = paper
        self.qa_pairs = qa_pairs or []

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

            # Q&A Section
            if self.qa_pairs:
                yield Static("", classes="qa-section")
                yield Static("Questions & Answers", classes="section-header")
                for question, answer in self.qa_pairs:
                    yield Static(f"Q: {question.text}", classes="qa-question")
                    yield Static(f"A: {answer.answer}", classes="qa-answer")
                    if answer.sections_used:
                        sections = ", ".join(answer.sections_used)
                        confidence = f" | {answer.confidence}" if answer.confidence else ""
                        yield Static(f"(from: {sections}{confidence})", classes="qa-meta")
        else:
            yield Static("Select a paper to view details", classes="meta")


class ClusterDetail(Static):
    """Widget to display cluster details."""

    DEFAULT_CSS = """
    ClusterDetail {
        background: $surface;
        padding: 1 2;
        height: 100%;
        overflow-y: auto;
    }

    ClusterDetail .title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    ClusterDetail .meta {
        color: $text-muted;
        margin-bottom: 1;
    }

    ClusterDetail .description {
        color: $text;
        margin-top: 1;
    }

    ClusterDetail .papers-header {
        text-style: bold;
        color: $secondary;
        margin-top: 1;
    }

    ClusterDetail .paper-item {
        color: $text-muted;
        padding-left: 1;
    }
    """

    def __init__(self, cluster: Cluster, papers_by_id: dict[str, Paper]):
        super().__init__()
        self.cluster = cluster
        self.papers_by_id = papers_by_id

    def compose(self) -> ComposeResult:
        yield Static(f"📁 {self.cluster.name}", classes="title")

        # Paper count
        total_papers = len(self.cluster.paper_ids)
        for sub in self.cluster.subclusters:
            total_papers += len(sub.paper_ids)
        yield Static(f"{total_papers} papers in this cluster", classes="meta")

        # Description
        if self.cluster.description:
            yield Static("Description", classes="papers-header")
            yield Static(self.cluster.description, classes="description")

        # List papers in cluster
        if self.cluster.paper_ids:
            yield Static("Papers", classes="papers-header")
            for pid in self.cluster.paper_ids[:10]:
                if pid in self.papers_by_id:
                    paper = self.papers_by_id[pid]
                    year = f" ({paper.year})" if paper.year else ""
                    yield Static(f"• {paper.title[:60]}{year}", classes="paper-item")
            if len(self.cluster.paper_ids) > 10:
                yield Static(
                    f"  ... and {len(self.cluster.paper_ids) - 10} more",
                    classes="paper-item",
                )

        # Subclusters
        if self.cluster.subclusters:
            yield Static("Subclusters", classes="papers-header")
            for sub in self.cluster.subclusters:
                yield Static(f"• {sub.name} ({len(sub.paper_ids)} papers)", classes="paper-item")


class ClusterTree(Tree):
    """Tree widget for displaying paper clusters hierarchically.

    Displays clusters as expandable tree nodes with papers as leaves.
    Supports filtering by title, author, abstract, and keywords.
    Papers are sorted alphabetically within each cluster.

    Attributes:
        all_papers: List of all papers in the view
        clusters: List of top-level clusters
        filter_text: Current search filter text
    """

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

    ClusterTree > .tree--label {
        padding: 2 0 0 0;
    }
    """

    def __init__(
        self, view_name: str, papers: list[Paper], clusters: list[Cluster], filter_text: str = ""
    ):
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
            key=lambda p: p.title.lower(),
        )
        if uncategorized:
            node = self.root.add(
                f"[bold italic]Uncategorized[/bold italic] ({len(uncategorized)})",
                data={"type": "uncategorized"},
            )
            for paper in uncategorized:
                year_suffix = f" [{paper.year}]" if paper.year else ""
                relevance = f" [{paper.relevance_score}%]" if paper.relevance_score else ""
                node.add_leaf(
                    f"[dim]{paper.title}{year_suffix}{relevance}[/dim]",
                    data={"type": "paper", "paper": paper},
                )

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
            pid
            for pid in cluster.paper_ids
            if pid in self._paper_map and self._matches_filter(self._paper_map[pid])
        ]

        # Count total including subclusters (filtered)
        total = len(matching_papers)
        for sub in cluster.subclusters:
            sub_matching = [
                pid
                for pid in sub.paper_ids
                if pid in self._paper_map and self._matches_filter(self._paper_map[pid])
            ]
            total += len(sub_matching)

        # Skip empty clusters when filtering
        if self.filter_text and total == 0:
            return

        node = parent.add(
            f"[bold]{cluster.name}[/bold] ({total})", data={"type": "cluster", "cluster": cluster}
        )

        # Sort papers alphabetically by title
        sorted_papers = sorted(
            [self._paper_map[pid] for pid in matching_papers], key=lambda p: p.title.lower()
        )
        for paper in sorted_papers:
            year_suffix = f" [{paper.year}]" if paper.year else ""
            relevance = f" [{paper.relevance_score}%]" if paper.relevance_score else ""
            node.add_leaf(
                f"[dim]{paper.title}{year_suffix}{relevance}[/dim]",
                data={"type": "paper", "paper": paper},
            )

        for subcluster in cluster.subclusters:
            self._add_cluster_node(node, subcluster)

    def get_selected_paper(self) -> Paper | None:
        node = self.cursor_node
        if node and node.data and node.data.get("type") == "paper":
            return node.data.get("paper")
        return None

    def get_selected_cluster(self) -> Cluster | None:
        node = self.cursor_node
        if node and node.data and node.data.get("type") == "cluster":
            return node.data.get("cluster")
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
    """Screen for viewing and managing clusters in a specific view.

    This screen displays a tree view of clusters on the left and paper
    details on the right. Users can navigate clusters, search papers,
    and perform various actions on papers and clusters.

    Bindings:
        /: Open search to filter papers
        o: Open paper's DOI or search Google Scholar
        p: Open the local PDF file
        m: Move paper to a different cluster
        E: Edit paper metadata
        d: Delete paper from project
        r: Rename the selected cluster
        R: Recluster papers with feedback
        a: Ask a question about all papers
        x: Export view to file or clipboard
        e: Expand all tree nodes
        c: Collapse all tree nodes
        L: View application logs
        q: Go back to view selection
    """

    BINDINGS = [
        Binding("q", "back", "Back", priority=True),
        Binding("o", "open_web", "Open Web"),
        Binding("p", "open_pdf", "PDF"),
        Binding("m", "move_paper", "Move"),
        Binding("E", "edit_paper", "Edit"),
        Binding("d", "delete_paper", "Delete"),
        Binding("r", "rename_cluster", "Rename"),
        Binding("R", "recluster", "Recluster"),
        Binding("a", "ask_question", "Ask Q"),
        Binding("x", "export", "Export"),
        Binding("/", "search", "Search", priority=True),
        Binding("escape", "clear_search", "Clear", show=False),
        Binding("e", "expand_all", "Expand"),
        Binding("c", "collapse_all", "Collapse"),
        Binding("Q", "questions", "Questions"),
        Binding("L", "view_logs", "Logs"),
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
        self._papers_by_id = {p.id: p for p in self.papers}
        self._tree: ClusterTree | None = None
        self._detail: Container | None = None
        self._filter_text = ""
        self._analysis_progress: AnalysisProgressScreen | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            f"[bold]{self.view.name}[/bold] | {self.view.prompt[:60]}... | {len(self.papers)} papers, {len(self.clusters)} clusters",
            classes="view-header",
        )
        with Container(id="search-container"):
            yield Input(
                placeholder="Search papers by title, author, abstract...", id="search-input"
            )
        with Horizontal(id="main-container"):
            with Vertical(id="tree-container"):
                self._tree = ClusterTree(self.view.name, self.papers, self.clusters)
                yield self._tree
            with Vertical(id="detail-container"):
                with Container(id="paper-detail-wrapper") as self._detail:
                    yield PaperDetail()
        yield Footer()

    def on_mount(self) -> None:
        """Focus the tree when screen opens."""
        if self._tree:
            self._tree.focus()

    @on(Tree.NodeHighlighted)
    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        if self._tree and self._detail:
            self._detail.remove_children()

            # Check if a paper is selected
            paper = self._tree.get_selected_paper()
            if paper:
                # Load Q&A pairs for this paper
                qa_pairs = self.project.get_answers_with_questions(paper.id)
                self._detail.mount(PaperDetail(paper, qa_pairs))
                return

            # Check if a cluster is selected
            cluster = self._tree.get_selected_cluster()
            if cluster:
                self._detail.mount(ClusterDetail(cluster, self._papers_by_id))
                return

            # Nothing specific selected
            self._detail.mount(PaperDetail(None))

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

    def action_open_web(self) -> None:
        """Open paper webpage (DOI) or search Google Scholar."""
        import urllib.parse

        if not self._tree:
            return

        paper = self._tree.get_selected_paper()
        if not paper:
            self.notify("No paper selected", severity="warning")
            return

        # Prefer DOI URL, then paper URL, then Google Scholar search
        if paper.doi:
            url = f"https://doi.org/{paper.doi}"
            self.notify(f"Opening DOI: {paper.doi}")
        elif paper.url:
            url = paper.url
            self.notify("Opening URL")
        else:
            # Fallback to Google Scholar search
            query = urllib.parse.quote(paper.title)
            url = f"https://scholar.google.com/scholar?q={query}"
            self.notify("Searching Google Scholar...")

        if sys.platform == "darwin":
            subprocess.Popen(["open", url])
        elif sys.platform == "win32":
            subprocess.Popen(["start", "", url], shell=True)
        else:
            subprocess.Popen(["xdg-open", url])

    def action_open_pdf(self) -> None:
        """Open the local PDF file."""
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
                self.notify("Moved paper to cluster")

        self.app.push_screen(
            MoveToClusterDialog(self.clusters, paper.title),
            handle_move,
        )

    def action_edit_paper(self) -> None:
        """Edit selected paper's metadata."""
        if not self._tree:
            return

        paper = self._tree.get_selected_paper()
        if not paper:
            self.notify("Select a paper to edit", severity="warning")
            return

        def handle_edit(result: dict | None) -> None:
            if result and paper:
                self.project.update_paper(paper.id, result)
                # Refresh paper detail view with updated paper
                updated_paper = self.project.db.get_paper(paper.id)
                if updated_paper:
                    # Update the paper in our local list and lookup dict
                    for i, p in enumerate(self.papers):
                        if p.id == paper.id:
                            self.papers[i] = updated_paper
                            break
                    self._papers_by_id[paper.id] = updated_paper

                    # Refresh tree to show updated title
                    if self._tree:
                        expanded = self._tree.get_expanded_state()
                        self._tree.all_papers = self.papers
                        self._tree._paper_map = self._papers_by_id
                        self._tree._build_tree()
                        self._tree.restore_expanded_state(expanded)

                    # Refresh detail panel
                    if self._detail:
                        self._detail.remove_children()
                        self._detail.mount(PaperDetail(updated_paper))

                self.notify("Paper updated")

        self.app.push_screen(EditPaperDialog(paper), handle_edit)

    def action_delete_paper(self) -> None:
        """Delete selected paper from the project."""
        if not self._tree:
            return

        paper = self._tree.get_selected_paper()
        if not paper:
            self.notify("Select a paper to delete", severity="warning")
            return

        def handle_confirm(confirmed: bool) -> None:
            if confirmed and paper:
                self.project.delete_paper(paper.id)

                # Remove paper from local lists
                self.papers = [p for p in self.papers if p.id != paper.id]
                self._papers_by_id.pop(paper.id, None)

                # Refresh clusters and tree
                self.clusters = self.project.get_clusters(self.view.id)
                if self._tree:
                    expanded = self._tree.get_expanded_state()
                    self._tree.all_papers = self.papers
                    self._tree._paper_map = self._papers_by_id
                    self._tree.clusters = self.clusters
                    self._tree._build_tree()
                    self._tree.restore_expanded_state(expanded)

                # Clear detail panel
                if self._detail:
                    self._detail.remove_children()
                    self._detail.mount(PaperDetail(None))

                self.notify(f"Deleted '{paper.display_title}'")

        self.app.push_screen(
            ConfirmDialog(
                "Delete Paper", f"Delete '{paper.display_title}'? This cannot be undone."
            ),
            handle_confirm,
        )

    def action_recluster(self) -> None:
        """Recluster papers with feedback."""
        if not self.clusters:
            self.notify("No clusters to reorganize", severity="warning")
            return

        def handle_feedback(feedback: str | None) -> None:
            if feedback:
                self._do_recluster(feedback)

        self.app.push_screen(ReclusterDialog(), handle_feedback)

    @work(thread=True)
    def _do_recluster(self, feedback: str) -> None:
        """Perform reclustering in background thread."""
        self.app.call_from_thread(self.notify, "Reclustering papers...")

        try:
            clusterer = PaperClusterer()
            new_clusters, relevance_scores = clusterer.recluster(
                papers=self.papers,
                research_question=self.project.config.research_question,
                feedback=feedback,
                current_clusters=self.clusters,
            )

            # Save new clusters and update relevance scores
            self.project.save_clusters(self.view.id, new_clusters)
            for paper_id, score in relevance_scores.items():
                self.project.update_paper(paper_id, {"relevance_score": score})
            self.clusters = new_clusters
            # Refresh papers to get updated relevance scores
            self.papers = self.project.get_papers()
            self._papers_by_id = {p.id: p for p in self.papers}

            # Update tree on main thread
            def refresh_tree() -> None:
                if self._tree:
                    self._tree.clusters = self.clusters
                    self._tree._build_tree()
                    self._tree.root.expand()

            self.app.call_from_thread(refresh_tree)
            self.app.call_from_thread(self.notify, f"Reclustered into {len(new_clusters)} clusters")
        except Exception as e:
            self.app.call_from_thread(self.notify, f"Recluster failed: {e}", severity="error")

    def action_ask_question(self) -> None:
        """Ask a question to analyze across papers."""
        if not self.papers:
            self.notify("No papers to analyze", severity="warning")
            return

        # Check if a cluster is selected
        cluster = self._tree.get_selected_cluster() if self._tree else None
        cluster_paper_ids: list[str] | None = None
        if cluster:
            cluster_paper_ids = cluster.paper_ids

        def handle_question(result: dict | None) -> None:
            if result:
                question_text = result["question"]
                model = result["model"]
                scope = result.get("scope", "all")

                # Determine which papers to analyze
                if scope == "cluster" and cluster_paper_ids:
                    papers_to_analyze = [p for p in self.papers if p.id in cluster_paper_ids]
                else:
                    papers_to_analyze = self.papers

                # Show progress screen
                self._analysis_progress = AnalysisProgressScreen(question_text)
                self.app.push_screen(self._analysis_progress)

                # Start analysis in background
                self._analyze_papers(question_text, model, papers_to_analyze)

        # Pass cluster context if available
        self.app.push_screen(
            AskQuestionDialog(
                total_papers=len(self.papers),
                cluster_name=cluster.name if cluster else None,
                cluster_paper_count=len(cluster_paper_ids) if cluster_paper_ids else 0,
            ),
            handle_question,
        )

    def _update_analysis_progress(self, status: str) -> None:
        """Update the analysis progress screen status from background thread."""
        if self._analysis_progress:
            self.app.call_from_thread(self._analysis_progress.update_status, status)

    def _dismiss_analysis_progress(self) -> None:
        """Dismiss the analysis progress screen from background thread."""
        if self._analysis_progress:
            self.app.call_from_thread(self.app.pop_screen)
            self._analysis_progress = None

    @work(thread=True)
    def _analyze_papers(
        self, question_text: str, model: str, papers: list[Paper] | None = None
    ) -> None:
        """Analyze papers to answer a question.

        Args:
            question_text: The question to answer
            model: The LLM model to use
            papers: Papers to analyze (defaults to all papers in view)
        """
        from tuxedo.analysis import PaperAnalyzer

        papers_to_analyze = papers if papers is not None else self.papers

        try:
            self._update_analysis_progress("Creating question...")

            # Create the question
            question = self.project.create_question(question_text)

            self._update_analysis_progress(f"Initializing {model}...")

            # Create analyzer
            analyzer = PaperAnalyzer(model=model)

            # Progress callback
            def progress_callback(current: int, total: int, message: str) -> None:
                self._update_analysis_progress(message)

            # Analyze papers
            answers = analyzer.analyze_papers(
                papers=papers_to_analyze,
                question=question_text,
                question_id=question.id,
                progress_callback=progress_callback,
            )

            # Save answers
            self._update_analysis_progress("Saving answers...")
            for answer in answers:
                self.project.save_answer(answer)

            self._dismiss_analysis_progress()

            # Refresh detail view if a paper is selected
            def refresh_detail() -> None:
                if self._tree and self._detail:
                    paper = self._tree.get_selected_paper()
                    if paper:
                        qa_pairs = self.project.get_answers_with_questions(paper.id)
                        self._detail.remove_children()
                        self._detail.mount(PaperDetail(paper, qa_pairs))

            self.app.call_from_thread(refresh_detail)
            self.app.call_from_thread(
                self.notify, f"Analyzed {len(answers)} papers for your question"
            )

        except Exception as e:
            self._dismiss_analysis_progress()
            self.app.call_from_thread(self.notify, f"Analysis failed: {e}", severity="error")

    def action_rename_cluster(self) -> None:
        """Rename the selected cluster."""
        if not self._tree:
            return

        cluster = self._tree.get_selected_cluster()
        if not cluster:
            self.notify("Select a cluster to rename", severity="warning")
            return

        def handle_rename(result: dict | None) -> None:
            if result and cluster:
                new_name = result["name"]
                new_description = result.get("description")
                self.project.rename_cluster(cluster.id, new_name, new_description)

                # Update local cluster object
                cluster.name = new_name
                if new_description is not None:
                    cluster.description = new_description

                # Refresh tree, preserving expanded state
                if self._tree:
                    expanded = self._tree.get_expanded_state()
                    self._tree._build_tree()
                    self._tree.restore_expanded_state(expanded)
                self.notify(f"Renamed cluster to '{new_name}'")

        self.app.push_screen(RenameClusterDialog(cluster), handle_rename)

    def action_help(self) -> None:
        self.notify(
            "Tip: Select a paper for o/p/m/E/d actions, or a cluster for r/R actions",
            timeout=5,
        )

    def action_view_logs(self) -> None:
        """Open the log viewer screen."""
        self.app.push_screen(LogViewerScreen())

    def action_questions(self) -> None:
        """Open the questions screen."""
        self.app.push_screen(QuestionsScreen(self.project))

    def action_export(self) -> None:
        """Export the current view to a file or clipboard."""

        def handle_export(result: dict | None) -> None:
            if not result:
                return

            from tuxedo.cli import (
                _export_bibtex,
                _export_csv,
                _export_json,
                _export_latex,
                _export_markdown,
                _export_ris,
            )

            format_type = result["format"]
            output_path = result["path"]

            # Generate export content
            export_funcs = {
                "markdown": _export_markdown,
                "bibtex": lambda v, c, p: _export_bibtex(v, c, p, include_abstract=False),
                "csv": _export_csv,
                "ris": _export_ris,
                "json": _export_json,
                "latex": _export_latex,
            }

            func = export_funcs.get(format_type)
            if not func:
                self.notify(f"Unknown format: {format_type}", severity="error")
                return

            content = func(self.view, self.clusters, self._papers_by_id)

            if output_path:
                # Write to file
                from pathlib import Path

                path = Path(output_path)
                path.write_text(content)
                self.notify(f"Exported to {path.name}")
            else:
                # Copy to clipboard
                try:
                    if sys.platform == "darwin":
                        subprocess.run(["pbcopy"], input=content.encode(), check=True)
                    elif sys.platform == "win32":
                        subprocess.run(["clip"], input=content.encode(), check=True)
                    else:
                        subprocess.run(
                            ["xclip", "-selection", "clipboard"],
                            input=content.encode(),
                            check=True,
                        )
                    self.notify(f"Copied {format_type} to clipboard")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    self.notify("Clipboard copy failed - save to file instead", severity="warning")

        self.app.push_screen(ExportDialog(self.view.name), handle_export)


# ============================================================================
# Log Viewer Screen
# ============================================================================


class LogViewerScreen(Screen):
    """Screen for viewing the current log file.

    Displays the contents of the current Tuxedo log file with auto-scroll
    to the bottom. Supports refreshing to see new log entries.

    Bindings:
        r: Refresh log content
        q: Go back to previous screen
    """

    BINDINGS = [
        Binding("q", "back", "Back", priority=True),
        Binding("r", "refresh", "Refresh"),
        Binding("g", "go_top", "Top"),
        Binding("G", "go_bottom", "Bottom"),
    ]

    CSS = """
    LogViewerScreen {
        background: $surface;
    }

    LogViewerScreen #log-header {
        dock: top;
        height: 3;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $primary;
    }

    LogViewerScreen #log-content {
        padding: 1 2;
        overflow-y: auto;
    }

    LogViewerScreen .log-line {
        color: $text;
    }

    LogViewerScreen .log-line-error {
        color: $error;
    }

    LogViewerScreen .log-line-warning {
        color: $warning;
    }

    LogViewerScreen .log-line-info {
        color: $text;
    }

    LogViewerScreen .log-line-debug {
        color: $text-muted;
    }
    """

    def __init__(self):
        super().__init__()
        self._log_path: str | None = None

    def compose(self) -> ComposeResult:
        from tuxedo.logging import get_log_file

        log_file = get_log_file()
        self._log_path = str(log_file) if log_file else "No log file"

        yield Static(
            f"[bold]Log Viewer[/bold] | {self._log_path}",
            id="log-header",
        )
        yield Vertical(id="log-content")
        yield Footer()

    def on_mount(self) -> None:
        """Load log content when screen opens."""
        self._load_log()
        # Scroll to bottom after a brief delay to ensure content is rendered
        self.set_timer(0.1, self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the log."""
        container = self.query_one("#log-content", Vertical)
        container.scroll_end(animate=False)

    def _load_log(self) -> None:
        """Load and display log file content."""
        from tuxedo.logging import get_log_file

        container = self.query_one("#log-content", Vertical)
        container.remove_children()

        log_file = get_log_file()
        if not log_file or not log_file.exists():
            container.mount(Static("[dim]No log file found[/dim]"))
            return

        try:
            content = log_file.read_text(encoding="utf-8")
            lines = content.splitlines()

            # Show last 500 lines to avoid performance issues
            if len(lines) > 500:
                lines = lines[-500:]
                container.mount(
                    Static(
                        f"[dim]... showing last 500 of {len(content.splitlines())} lines ...[/dim]"
                    )
                )

            for line in lines:
                # Color-code based on log level
                if "| ERROR" in line or "| CRITICAL" in line:
                    style_class = "log-line-error"
                elif "| WARNING" in line:
                    style_class = "log-line-warning"
                elif "| DEBUG" in line:
                    style_class = "log-line-debug"
                else:
                    style_class = "log-line-info"

                container.mount(Static(line, classes=f"log-line {style_class}"))

            if not lines:
                container.mount(Static("[dim]Log file is empty[/dim]"))

        except Exception as e:
            container.mount(Static(f"[red]Error reading log: {e}[/red]"))

    def action_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()

    def action_refresh(self) -> None:
        """Refresh log content."""
        self._load_log()
        self.set_timer(0.1, self._scroll_to_bottom)
        self.notify("Log refreshed")

    def action_go_top(self) -> None:
        """Scroll to top of log."""
        container = self.query_one("#log-content", Vertical)
        container.scroll_home(animate=False)

    def action_go_bottom(self) -> None:
        """Scroll to bottom of log."""
        self._scroll_to_bottom()


# ============================================================================
# Main App
# ============================================================================


class TuxedoApp(App):
    """Main TUI application for exploring literature review clusters.

    The application uses a screen-based navigation:
    1. ViewSelectionScreen: Choose or create clustering views
    2. ClusterScreen: Explore papers within a specific view

    Always uses dark mode for consistent appearance.
    """

    TITLE = "Tuxedo"
    dark = True  # Always use dark mode

    DEFAULT_CSS = """
    Button {
        border: none;
    }
    """

    def __init__(self, project: Project):
        super().__init__()
        self.project = project

    def on_mount(self) -> None:
        self.push_screen(ViewSelectionScreen(self.project))


def run_tui(project: Project) -> None:
    """Run the TUI application."""
    app = TuxedoApp(project)
    app.run()
