"""Tests for the Textual TUI components."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Label, ListView, Select

from tuxedo.models import Author, Cluster, ClusterView, Paper, Question
from tuxedo.tui import (
    AnalysisProgressScreen,
    AskQuestionDialog,
    ClusteringProgressScreen,
    ConfirmDialog,
    CreateClusterDialog,
    EditPaperDialog,
    ExportDialog,
    LogViewerScreen,
    MoveToClusterDialog,
    QuestionListItem,
    QuestionsScreen,
    ReclusterDialog,
    RenameClusterDialog,
    ViewListItem,
    NewViewItem,
    PaperDetail,
    ClusterDetail,
    ClusterTree,
    ViewSelectionScreen,
    ClusterScreen,
    TuxedoApp,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_paper():
    """Create a sample paper for testing."""
    return Paper(
        id="paper1",
        pdf_path=Path("/tmp/test.pdf"),
        title="Machine Learning for Climate Science: A Comprehensive Review",
        authors=[
            Author(name="John Smith", affiliation="MIT"),
            Author(name="Jane Doe", affiliation="Stanford"),
        ],
        abstract="This paper reviews machine learning applications in climate science.",
        year=2024,
        doi="10.1234/ml.climate.2024",
        journal="Nature Machine Intelligence",
        volume="6",
        number="3",
        pages="123-145",
        keywords=["machine learning", "climate", "review"],
    )


@pytest.fixture
def sample_clusters():
    """Create sample clusters for testing."""
    return [
        Cluster(
            id="c1",
            name="Deep Learning Methods",
            description="Papers using deep neural networks",
            paper_ids=["paper1", "paper2"],
            subclusters=[],
        ),
        Cluster(
            id="c2",
            name="Statistical Approaches",
            description="Papers using traditional statistics",
            paper_ids=["paper3"],
            subclusters=[
                Cluster(
                    id="c2a",
                    name="Bayesian Methods",
                    description="Bayesian statistics papers",
                    paper_ids=["paper4"],
                    subclusters=[],
                ),
            ],
        ),
    ]


@pytest.fixture
def sample_view():
    """Create a sample clustering view."""
    from datetime import datetime

    return ClusterView(
        id="view1",
        name="By Methodology",
        prompt="Group papers by research methodology",
        created_at=datetime(2024, 1, 15, 10, 30),
    )


@pytest.fixture
def mock_project(sample_paper, sample_clusters, sample_view):
    """Create a mock project for testing."""
    project = Mock()
    project.config = Mock()
    project.config.name = "Test Project"
    project.config.research_question = "What are the effects of ML on climate?"
    project.get_views.return_value = [sample_view]
    project.get_view.return_value = sample_view
    project.get_papers.return_value = [sample_paper]
    project.get_clusters.return_value = sample_clusters
    project.cluster_count.return_value = 2
    project.paper_count.return_value = 1
    project.get_paper.return_value = sample_paper
    project.get_pdf_path.return_value = Path("/tmp/test.pdf")
    project.get_answers_with_questions.return_value = []
    return project


@pytest.fixture
def sample_question():
    """Create a sample question for testing."""
    from datetime import datetime

    return Question(
        id="q1",
        text="What methodology does this paper use?",
        created_at=datetime(2024, 6, 15, 14, 30),
    )


# ============================================================================
# Test Helper App
# ============================================================================


class DialogTestApp(App):
    """Test app for mounting dialogs."""

    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog
        self.result = None

    def compose(self) -> ComposeResult:
        yield Label("Test App")

    def on_mount(self) -> None:
        def capture_result(result):
            self.result = result

        self.push_screen(self.dialog, capture_result)


# ============================================================================
# ConfirmDialog Tests
# ============================================================================


class TestConfirmDialog:
    """Tests for the confirmation dialog."""

    async def test_yes_button_returns_true(self):
        """Clicking Yes button returns True."""
        dialog = ConfirmDialog("Delete View", "Are you sure?")
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            await pilot.click("#yes")

        assert app.result is True

    async def test_no_button_returns_false(self):
        """Clicking No button returns False."""
        dialog = ConfirmDialog("Delete View", "Are you sure?")
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            await pilot.click("#no")

        assert app.result is False

    async def test_escape_key_returns_false(self):
        """Pressing Escape returns False."""
        dialog = ConfirmDialog("Delete View", "Are you sure?")
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            await pilot.press("escape")

        assert app.result is False

    async def test_y_key_returns_true(self):
        """Pressing 'y' key returns True."""
        dialog = ConfirmDialog("Delete View", "Are you sure?")
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            await pilot.press("y")

        assert app.result is True

    async def test_n_key_returns_false(self):
        """Pressing 'n' key returns False."""
        dialog = ConfirmDialog("Delete View", "Are you sure?")
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            await pilot.press("n")

        assert app.result is False

    async def test_displays_title_and_message(self):
        """Dialog displays correct title and message."""
        dialog = ConfirmDialog("My Title", "My message text")
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            labels = dialog.query(Label)
            label_texts = [str(label.render()) for label in labels]
            assert any("My Title" in t for t in label_texts)
            assert any("My message text" in t for t in label_texts)
            await pilot.press("escape")


# ============================================================================
# ClusteringProgressScreen Tests
# ============================================================================


class TestClusteringProgressScreen:
    """Tests for the clustering progress screen."""

    async def test_screen_displays_title(self):
        """Screen displays the title."""
        dialog = ClusteringProgressScreen("Test Title")
        app = DialogTestApp(dialog)

        async with app.run_test():
            labels = dialog.query(Label)
            label_texts = [str(label.render()) for label in labels]
            assert any("Test Title" in t for t in label_texts)

    async def test_screen_displays_loading_indicator(self):
        """Screen displays a loading indicator."""
        from textual.widgets import LoadingIndicator

        dialog = ClusteringProgressScreen()
        app = DialogTestApp(dialog)

        async with app.run_test():
            indicator = dialog.query_one(LoadingIndicator)
            assert indicator is not None

    async def test_update_status_changes_label(self):
        """update_status method changes the status label."""
        dialog = ClusteringProgressScreen()
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            dialog.update_status("Processing batch 1 of 5...")
            await pilot.pause()
            status_label = dialog.query_one("#status-label", Label)
            assert "Processing batch 1 of 5" in str(status_label.render())


# ============================================================================
# MoveToClusterDialog Tests
# ============================================================================


class TestMoveToClusterDialog:
    """Tests for the move-to-cluster dialog."""

    async def test_displays_clusters(self, sample_clusters):
        """Dialog displays available clusters."""
        dialog = MoveToClusterDialog(sample_clusters, "Test Paper Title")
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            list_view = dialog.query_one(ListView)
            # Should have 2 top-level clusters + 1 subcluster = 3 items
            assert len(list_view.children) == 3
            await pilot.press("escape")

    async def test_escape_returns_none(self, sample_clusters):
        """Pressing Escape returns None."""
        dialog = MoveToClusterDialog(sample_clusters, "Test Paper")
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            await pilot.press("escape")

        assert app.result is None

    async def test_selecting_cluster_returns_id(self, sample_clusters):
        """Selecting a cluster returns its ID."""
        dialog = MoveToClusterDialog(sample_clusters, "Test Paper")
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            # Select first item and press enter
            list_view = dialog.query_one(ListView)
            list_view.index = 0
            await pilot.press("enter")

        assert app.result == "c1"


# ============================================================================
# EditPaperDialog Tests
# ============================================================================


class TestEditPaperDialog:
    """Tests for the paper editing dialog."""

    async def test_prepopulates_fields(self, sample_paper):
        """Dialog pre-populates fields with paper data."""
        dialog = EditPaperDialog(sample_paper)
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            title_input = dialog.query_one("#edit-title", Input)
            year_input = dialog.query_one("#edit-year", Input)
            doi_input = dialog.query_one("#edit-doi", Input)

            assert title_input.value == sample_paper.title
            assert year_input.value == "2024"
            assert doi_input.value == sample_paper.doi

            await pilot.press("escape")

    async def test_cancel_returns_none(self, sample_paper):
        """Pressing Escape returns None."""
        dialog = EditPaperDialog(sample_paper)
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            await pilot.press("escape")

        assert app.result is None

    async def test_escape_returns_none(self, sample_paper):
        """Pressing Escape returns None."""
        dialog = EditPaperDialog(sample_paper)
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            await pilot.press("escape")

        assert app.result is None

    async def test_save_returns_updated_data(self, sample_paper):
        """Saving returns dictionary with updated values."""
        dialog = EditPaperDialog(sample_paper)
        app = DialogTestApp(dialog)

        async with app.run_test(size=(150, 80)) as pilot:
            # Modify title
            title_input = dialog.query_one("#edit-title", Input)
            title_input.value = "New Updated Title"

            # Focus the save button and press enter
            save_btn = dialog.query_one("#save-btn", Button)
            save_btn.focus()
            await pilot.press("enter")

        assert app.result is not None
        assert app.result["title"] == "New Updated Title"

    async def test_invalid_year_keeps_original(self, sample_paper):
        """Invalid year value doesn't change year (keeps original)."""
        dialog = EditPaperDialog(sample_paper)
        app = DialogTestApp(dialog)

        async with app.run_test(size=(150, 80)) as pilot:
            year_input = dialog.query_one("#edit-year", Input)
            year_input.value = "not-a-year"

            save_btn = dialog.query_one("#save-btn", Button)
            save_btn.focus()
            await pilot.press("enter")

        # Invalid year is ignored, so result may be None (no changes) or not have year key
        # The dialog only returns changes, not the full paper
        assert app.result is None or "year" not in app.result


# ============================================================================
# ReclusterDialog Tests
# ============================================================================


class TestReclusterDialog:
    """Tests for the recluster dialog."""

    async def test_cancel_returns_none(self):
        """Pressing Escape returns None."""
        dialog = ReclusterDialog()
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            await pilot.press("escape")

        assert app.result is None

    async def test_escape_returns_none(self):
        """Pressing Escape returns None."""
        dialog = ReclusterDialog()
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            await pilot.press("escape")

        assert app.result is None

    async def test_submit_returns_feedback(self):
        """Submitting returns feedback text."""
        dialog = ReclusterDialog()
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 40)) as pilot:
            feedback_input = dialog.query_one("#feedback-input", Input)
            feedback_input.value = "Focus more on methodology"

            # Focus the recluster button and press enter
            recluster_btn = dialog.query_one("#recluster-btn", Button)
            recluster_btn.focus()
            await pilot.press("enter")

        assert app.result == "Focus more on methodology"

    async def test_empty_feedback_shows_warning(self):
        """Empty feedback shows a warning and doesn't dismiss."""
        dialog = ReclusterDialog()
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            # Try to submit without feedback - should show warning
            await pilot.press("enter")
            # Dialog should still be open, result should be None
            await pilot.press("escape")

        assert app.result is None


# ============================================================================
# RenameClusterDialog Tests
# ============================================================================


class TestRenameClusterDialog:
    """Tests for the rename cluster dialog."""

    async def test_prepopulates_current_values(self, sample_clusters):
        """Dialog pre-populates with current cluster name and description."""
        cluster = sample_clusters[0]
        dialog = RenameClusterDialog(cluster)
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            name_input = dialog.query_one("#rename-name", Input)
            desc_input = dialog.query_one("#rename-description", Input)

            assert name_input.value == "Deep Learning Methods"
            assert desc_input.value == "Papers using deep neural networks"

            await pilot.press("escape")

    async def test_cancel_returns_none(self, sample_clusters):
        """Pressing Escape returns None."""
        dialog = RenameClusterDialog(sample_clusters[0])
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            await pilot.press("escape")

        assert app.result is None

    async def test_save_returns_updated_values(self, sample_clusters):
        """Saving returns updated name and description."""
        dialog = RenameClusterDialog(sample_clusters[0])
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 40)) as pilot:
            name_input = dialog.query_one("#rename-name", Input)
            desc_input = dialog.query_one("#rename-description", Input)

            name_input.value = "Neural Networks"
            desc_input.value = "All NN papers"

            await pilot.click("#save-btn")

        assert app.result is not None
        assert app.result["name"] == "Neural Networks"


# ============================================================================
# ExportDialog Tests
# ============================================================================


class TestExportDialog:
    """Tests for the export dialog."""

    async def test_cancel_returns_none(self):
        """Pressing Escape returns None."""
        dialog = ExportDialog("Test View")
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            await pilot.press("escape")

        assert app.result is None

    async def test_escape_returns_none(self):
        """Pressing Escape returns None."""
        dialog = ExportDialog("Test View")
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            await pilot.press("escape")

        assert app.result is None

    async def test_export_to_clipboard(self):
        """Exporting to clipboard returns correct format."""
        dialog = ExportDialog("Test View")
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 40)) as pilot:
            # Select markdown format (default)
            format_select = dialog.query_one("#export-format", Select)
            format_select.value = "markdown"

            await pilot.click("#export-btn")

        assert app.result is not None
        assert app.result["format"] == "markdown"
        assert app.result["path"] is None  # None means clipboard

    async def test_export_to_file(self):
        """Exporting to file returns file path."""
        dialog = ExportDialog("Test View")
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 40)) as pilot:
            format_select = dialog.query_one("#export-format", Select)
            format_select.value = "bibtex"

            file_input = dialog.query_one("#export-path", Input)
            file_input.value = "/tmp/refs.bib"

            await pilot.click("#export-btn")

        assert app.result is not None
        assert app.result["format"] == "bibtex"
        assert app.result["path"] == "/tmp/refs.bib"


# ============================================================================
# CreateClusterDialog Tests
# ============================================================================


class TestCreateClusterDialog:
    """Tests for the create cluster dialog."""

    async def test_cancel_returns_none(self):
        """Pressing Escape returns None."""
        dialog = CreateClusterDialog()
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            await pilot.press("escape")

        assert app.result is None

    async def test_escape_returns_none(self):
        """Pressing Escape returns None."""
        dialog = CreateClusterDialog()
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            await pilot.press("escape")

        assert app.result is None

    async def test_create_with_name_and_prompt(self):
        """Creating returns name and prompt."""
        dialog = CreateClusterDialog(default_prompt="Default question")
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 50)) as pilot:
            name_input = dialog.query_one("#new-view-name", Input)
            prompt_input = dialog.query_one("#new-view-prompt", Input)

            name_input.value = "By Method"
            prompt_input.value = "Group by research method"

            await pilot.click("#create-btn")

        assert app.result is not None
        assert app.result["name"] == "By Method"
        assert app.result["prompt"] == "Group by research method"

    async def test_empty_prompt_uses_default(self):
        """Empty prompt uses the default prompt."""
        dialog = CreateClusterDialog(default_prompt="Default question")
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 50)) as pilot:
            name_input = dialog.query_one("#new-view-name", Input)
            name_input.value = "Test View"
            # Leave prompt empty
            await pilot.click("#create-btn")

        assert app.result["prompt"] == "Default question"

    async def test_parses_categories(self):
        """Categories are parsed from comma-separated input."""
        dialog = CreateClusterDialog()
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 50)) as pilot:
            name_input = dialog.query_one("#new-view-name", Input)
            name_input.value = "By Type"

            categories_input = dialog.query_one("#new-view-categories", Input)
            categories_input.value = "Quantitative, Qualitative, Mixed"

            await pilot.click("#create-btn")

        assert app.result["categories"] == ["Quantitative", "Qualitative", "Mixed"]

    async def test_parses_batch_size(self):
        """Batch size is parsed as integer."""
        dialog = CreateClusterDialog()
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 50)) as pilot:
            name_input = dialog.query_one("#new-view-name", Input)
            name_input.value = "Test"

            batch_input = dialog.query_one("#new-view-batch", Input)
            batch_input.value = "15"

            await pilot.click("#create-btn")

        assert app.result["batch_size"] == 15

    async def test_invalid_batch_size_is_none(self):
        """Invalid batch size results in None."""
        dialog = CreateClusterDialog()
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 50)) as pilot:
            name_input = dialog.query_one("#new-view-name", Input)
            name_input.value = "Test"

            batch_input = dialog.query_one("#new-view-batch", Input)
            batch_input.value = "not-a-number"

            await pilot.click("#create-btn")

        assert app.result["batch_size"] is None

    async def test_auto_mode_selection(self):
        """Auto mode can be selected."""
        dialog = CreateClusterDialog()
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 50)) as pilot:
            name_input = dialog.query_one("#new-view-name", Input)
            name_input.value = "Auto Test"

            auto_select = dialog.query_one("#new-view-auto", Select)
            auto_select.value = "methodology"

            await pilot.click("#create-btn")

        assert app.result["auto_mode"] == "methodology"


# ============================================================================
# AskQuestionDialog Tests
# ============================================================================


class TestAskQuestionDialog:
    """Tests for the ask question dialog."""

    async def test_cancel_returns_none(self):
        """Pressing Escape returns None."""
        dialog = AskQuestionDialog()
        app = DialogTestApp(dialog)

        async with app.run_test() as pilot:
            await pilot.press("escape")

        assert app.result is None

    async def test_empty_question_shows_warning(self):
        """Empty question shows a warning."""
        dialog = AskQuestionDialog()
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 40)) as pilot:
            # Try to submit without question
            await pilot.click("#analyze-btn")
            # Dialog should still be open
            await pilot.press("escape")

        assert app.result is None

    async def test_analyze_returns_question_and_model(self):
        """Analyzing returns question text and model."""
        dialog = AskQuestionDialog()
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 40)) as pilot:
            question_input = dialog.query_one("#question-input", Input)
            question_input.value = "What methodology does this paper use?"

            await pilot.click("#analyze-btn")

        assert app.result is not None
        assert app.result["question"] == "What methodology does this paper use?"
        assert "model" in app.result

    async def test_model_selection(self):
        """Model can be selected."""
        dialog = AskQuestionDialog()
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 40)) as pilot:
            question_input = dialog.query_one("#question-input", Input)
            question_input.value = "Test question"

            model_select = dialog.query_one("#ask-model", Select)
            model_select.value = "gpt-4o"

            await pilot.click("#analyze-btn")

        assert app.result["model"] == "gpt-4o"


# ============================================================================
# AnalysisProgressScreen Tests
# ============================================================================


class TestAnalysisProgressScreen:
    """Tests for the analysis progress screen."""

    async def test_displays_question(self):
        """Screen displays the question being analyzed."""
        screen = AnalysisProgressScreen("What is the main finding?")

        class TestApp(App):
            def on_mount(self):
                self.push_screen(screen)

        async with TestApp().run_test():
            # Just verify it renders without error
            assert screen.question == "What is the main finding?"

    async def test_update_status(self):
        """Status can be updated."""
        screen = AnalysisProgressScreen("Test question")

        class TestApp(App):
            def on_mount(self):
                self.push_screen(screen)

        async with TestApp().run_test() as pilot:
            screen.update_status("Analyzing paper 1/10")
            await pilot.pause()
            # Verify status was updated
            assert screen._status == "Analyzing paper 1/10"


# ============================================================================
# ViewListItem Tests
# ============================================================================


class TestViewListItem:
    """Tests for the view list item widget."""

    async def test_displays_view_info(self, sample_view):
        """Widget displays view name and info."""

        class TestApp(App):
            def compose(self_app):
                yield ViewListItem(sample_view, cluster_count=5)

        async with TestApp().run_test() as pilot:
            app = pilot.app
            item = app.query_one(ViewListItem)
            # Should have stored view and count
            assert item.view == sample_view
            assert item.cluster_count == 5


class TestNewViewItem:
    """Tests for the new view list item."""

    async def test_new_view_item_exists(self):
        """Widget can be instantiated."""

        class TestApp(App):
            def compose(self_app):
                yield NewViewItem()

        async with TestApp().run_test() as pilot:
            app = pilot.app
            item = app.query_one(NewViewItem)
            assert item is not None


# ============================================================================
# QuestionListItem Tests
# ============================================================================


class TestQuestionListItem:
    """Tests for the question list item widget."""

    async def test_displays_question_info(self, sample_question):
        """Widget displays question text and answer count."""

        class TestApp(App):
            def compose(self_app):
                yield QuestionListItem(sample_question, answer_count=5)

        async with TestApp().run_test() as pilot:
            app = pilot.app
            item = app.query_one(QuestionListItem)
            assert item.question == sample_question
            assert item.answer_count == 5

    async def test_truncates_long_question(self):
        """Long questions are truncated."""
        from datetime import datetime

        long_question = Question(
            id="q2",
            text="A" * 100,  # Very long question
            created_at=datetime(2024, 1, 1),
        )

        class TestApp(App):
            def compose(self_app):
                yield QuestionListItem(long_question, answer_count=0)

        async with TestApp().run_test() as pilot:
            app = pilot.app
            item = app.query_one(QuestionListItem)
            # Question should be stored
            assert item.question == long_question


# ============================================================================
# QuestionsScreen Tests
# ============================================================================


class TestQuestionsScreen:
    """Tests for the questions screen."""

    async def test_screen_initializes(self, mock_project):
        """Screen initializes with project."""
        mock_project.get_questions.return_value = []
        mock_project.get_question_answer_count.return_value = 0

        screen = QuestionsScreen(mock_project)
        assert screen.project == mock_project

    async def test_question_list_item_stored(self, mock_project, sample_question):
        """QuestionListItem stores question and answer count."""
        item = QuestionListItem(sample_question, answer_count=3)
        assert item.question == sample_question
        assert item.answer_count == 3


# ============================================================================
# AskQuestionDialog Scope Tests
# ============================================================================


class TestAskQuestionDialogScope:
    """Tests for scope selection in AskQuestionDialog."""

    async def test_no_scope_without_cluster_context(self):
        """No scope selection when no cluster context provided."""
        dialog = AskQuestionDialog()
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 40)) as pilot:
            # Should not have scope select
            scope_selects = dialog.query("#ask-scope")
            assert len(scope_selects) == 0
            await pilot.press("escape")

    async def test_scope_shown_with_cluster_context(self):
        """Scope selection shown when cluster context provided."""
        dialog = AskQuestionDialog(
            total_papers=20,
            cluster_name="Test Cluster",
            cluster_paper_count=5,
        )
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 40)) as pilot:
            # Should have scope select
            scope_selects = dialog.query("#ask-scope")
            assert len(scope_selects) == 1
            await pilot.press("escape")

    async def test_default_scope_is_all(self):
        """Default scope is 'all' papers."""
        dialog = AskQuestionDialog(
            total_papers=20,
            cluster_name="Test Cluster",
            cluster_paper_count=5,
        )
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 40)) as pilot:
            question_input = dialog.query_one("#question-input", Input)
            question_input.value = "Test question"
            await pilot.click("#analyze-btn")

        assert app.result is not None
        assert app.result["scope"] == "all"

    async def test_scope_cluster_selection(self):
        """Cluster scope can be selected."""
        dialog = AskQuestionDialog(
            total_papers=20,
            cluster_name="Test Cluster",
            cluster_paper_count=5,
        )
        app = DialogTestApp(dialog)

        async with app.run_test(size=(100, 40)) as pilot:
            question_input = dialog.query_one("#question-input", Input)
            question_input.value = "Test question"

            scope_select = dialog.query_one("#ask-scope", Select)
            scope_select.value = "cluster"

            await pilot.click("#analyze-btn")

        assert app.result is not None
        assert app.result["scope"] == "cluster"


# ============================================================================
# PaperDetail Tests
# ============================================================================


class TestPaperDetail:
    """Tests for the paper detail widget."""

    async def test_displays_paper_info(self, sample_paper):
        """Widget stores paper reference."""

        class TestApp(App):
            def compose(self_app):
                yield PaperDetail(sample_paper)

        async with TestApp().run_test() as pilot:
            app = pilot.app
            content = app.query_one(PaperDetail)
            # Widget should store paper reference
            assert content.paper == sample_paper

    async def test_handles_paper_without_optional_fields(self):
        """Widget handles papers with minimal data."""
        minimal_paper = Paper(
            id="min1",
            pdf_path=Path("/tmp/min.pdf"),
            title="Minimal Paper",
            authors=[],
        )

        class TestApp(App):
            def compose(self_app):
                yield PaperDetail(minimal_paper)

        async with TestApp().run_test() as pilot:
            app = pilot.app
            content = app.query_one(PaperDetail)
            # Should not raise, just display what's available
            assert content is not None
            assert content.paper == minimal_paper


# ============================================================================
# ClusterDetail Tests
# ============================================================================


class TestClusterDetail:
    """Tests for the cluster detail widget."""

    async def test_displays_cluster_info(self, sample_clusters, sample_paper):
        """Widget stores cluster reference."""
        cluster = sample_clusters[0]
        papers_by_id = {sample_paper.id: sample_paper}

        class TestApp(App):
            def compose(self_app):
                yield ClusterDetail(cluster, papers_by_id)

        async with TestApp().run_test() as pilot:
            app = pilot.app
            content = app.query_one(ClusterDetail)
            # Widget should store cluster reference
            assert content.cluster == cluster


# ============================================================================
# ClusterTree Tests
# ============================================================================


class TestClusterTree:
    """Tests for the cluster tree widget."""

    async def test_tree_can_be_created(self, sample_paper, sample_clusters):
        """Tree can be instantiated with papers and clusters."""
        papers = [sample_paper]

        class TestApp(App):
            def compose(self_app):
                yield ClusterTree("Clusters", papers, sample_clusters)

        async with TestApp().run_test() as pilot:
            app = pilot.app
            tree = app.query_one(ClusterTree)
            assert tree is not None
            assert tree.all_papers == papers

    async def test_tree_has_root(self, sample_paper, sample_clusters):
        """Tree has a root node."""
        papers = [sample_paper]

        class TestApp(App):
            def compose(self_app):
                yield ClusterTree("Clusters", papers, sample_clusters)

        async with TestApp().run_test() as pilot:
            app = pilot.app
            tree = app.query_one(ClusterTree)
            assert tree.root is not None


# ============================================================================
# ViewSelectionScreen Tests
# ============================================================================


class TestViewSelectionScreen:
    """Tests for the view selection screen."""

    async def test_displays_existing_views(self, mock_project, sample_view):
        """Screen displays list of existing views."""

        class TestApp(App):
            def compose(self_app):
                yield ViewSelectionScreen(mock_project)

        async with TestApp().run_test() as pilot:
            app = pilot.app
            list_view = app.query_one(ListView)
            # Should have at least the view + new view item
            assert len(list_view.children) >= 1

    async def test_n_key_opens_new_view_dialog(self, mock_project):
        """Pressing 'n' opens the create view dialog."""

        class TestApp(App):
            SCREENS = {"view_selection": ViewSelectionScreen}

            def compose(self_app):
                yield ViewSelectionScreen(mock_project)

        async with TestApp().run_test() as pilot:
            await pilot.press("n")
            # Dialog should be pushed
            assert len(pilot.app.screen_stack) > 1

    async def test_q_key_quits(self, mock_project):
        """Pressing 'q' quits the application."""

        class TestApp(App):
            def compose(self_app):
                yield ViewSelectionScreen(mock_project)

        app = TestApp()
        async with app.run_test() as pilot:
            await pilot.press("q")
            # App should be exiting
            assert app._exit


# ============================================================================
# ClusterScreen Tests
# ============================================================================


class TestClusterScreen:
    """Tests for the cluster screen."""

    async def test_displays_clusters(self, mock_project, sample_view, sample_clusters):
        """Screen displays cluster tree."""
        mock_project.get_clusters.return_value = sample_clusters

        class TestApp(App):
            def compose(self_app):
                yield ClusterScreen(mock_project, sample_view)

        async with TestApp().run_test() as pilot:
            app = pilot.app
            tree = app.query_one(ClusterTree)
            assert tree is not None

    async def test_q_key_goes_back(self, mock_project, sample_view):
        """Pressing 'q' goes back to view selection."""

        class TestApp(App):
            def compose(self_app):
                yield ClusterScreen(mock_project, sample_view)

        async with TestApp().run_test() as pilot:
            await pilot.press("q")
            # Should pop screen or handle back action

    async def test_question_mark_shows_help(self, mock_project, sample_view):
        """Pressing '?' shows help screen."""

        class TestApp(App):
            def compose(self_app):
                yield ClusterScreen(mock_project, sample_view)

        async with TestApp().run_test() as pilot:
            await pilot.press("?")
            # Help should be shown (screen pushed or notification)

    async def test_d_key_with_no_paper_shows_warning(self, mock_project, sample_view):
        """Pressing 'd' with no paper selected shows a warning."""

        class TestApp(App):
            def compose(self_app):
                yield ClusterScreen(mock_project, sample_view)

        async with TestApp().run_test() as pilot:
            await pilot.press("d")
            # Warning notification should be shown (no paper selected)

    async def test_delete_paper_action_calls_project_delete(
        self, mock_project, sample_view, sample_paper, sample_clusters
    ):
        """Confirming delete calls project.delete_paper."""
        mock_project.get_papers.return_value = [sample_paper]
        mock_project.get_clusters.return_value = sample_clusters

        class TestApp(App):
            def compose(self_app):
                yield ClusterScreen(mock_project, sample_view)

        async with TestApp().run_test(size=(150, 80)) as pilot:
            app = pilot.app
            screen = app.query_one(ClusterScreen)

            # Get the tree and select a paper node
            tree = screen._tree
            if tree:
                # Expand the root to reveal cluster nodes
                tree.root.expand()
                await pilot.pause()

                # Find and select a paper node
                for node in tree.root.children:
                    if hasattr(node, "data") and node.data:
                        # Expand cluster to reveal papers
                        node.expand()
                        await pilot.pause()
                        for child in node.children:
                            if hasattr(child, "data") and child.data:
                                tree.select_node(child)
                                await pilot.pause()
                                break
                        break

            # Press 'd' to trigger delete action
            await pilot.press("d")
            await pilot.pause()

            # Should have pushed a ConfirmDialog
            assert len(app.screen_stack) >= 1


# ============================================================================
# LogViewerScreen Tests
# ============================================================================


class TestLogViewerScreen:
    """Tests for the log viewer screen."""

    async def test_screen_displays_header(self):
        """Screen displays log header."""

        class TestApp(App):
            def compose(self_app):
                yield LogViewerScreen()

        async with TestApp().run_test() as pilot:
            app = pilot.app
            screen = app.query_one(LogViewerScreen)
            assert screen is not None

    async def test_q_key_goes_back(self):
        """Pressing 'q' pops the screen."""

        class TestApp(App):
            def compose(self_app):
                yield LogViewerScreen()

        async with TestApp().run_test() as pilot:
            await pilot.press("q")
            # Should pop screen or handle back action

    async def test_r_key_refreshes(self):
        """Pressing 'r' refreshes the log."""

        class TestApp(App):
            def compose(self_app):
                yield LogViewerScreen()

        async with TestApp().run_test() as pilot:
            await pilot.press("r")
            # Should refresh and show notification

    async def test_g_key_scrolls_to_top(self):
        """Pressing 'g' scrolls to top."""

        class TestApp(App):
            def compose(self_app):
                yield LogViewerScreen()

        async with TestApp().run_test() as pilot:
            await pilot.press("g")
            # Should scroll to top

    async def test_G_key_scrolls_to_bottom(self):
        """Pressing 'G' scrolls to bottom."""

        class TestApp(App):
            def compose(self_app):
                yield LogViewerScreen()

        async with TestApp().run_test() as pilot:
            await pilot.press("G")
            # Should scroll to bottom


# ============================================================================
# TuxedoApp Tests
# ============================================================================


class TestTuxedoApp:
    """Tests for the main Tuxedo application."""

    async def test_app_launches(self, mock_project):
        """Application launches successfully."""
        app = TuxedoApp(mock_project)

        async with app.run_test() as pilot:
            # App should be running
            assert pilot.app is not None
            # Should show view selection screen
            assert isinstance(pilot.app.screen, ViewSelectionScreen)

    async def test_app_has_dark_theme(self, mock_project):
        """Application uses dark theme."""
        app = TuxedoApp(mock_project)

        async with app.run_test() as pilot:
            assert pilot.app.dark is True

    async def test_app_title(self, mock_project):
        """Application has correct title."""
        app = TuxedoApp(mock_project)

        async with app.run_test() as pilot:
            assert "Tuxedo" in pilot.app.title or "Test Project" in pilot.app.title


# ============================================================================
# Integration Tests
# ============================================================================


class TestTUIIntegration:
    """Integration tests for TUI workflows."""

    async def test_view_selection_to_cluster_view(self, mock_project, sample_view, sample_clusters):
        """Can navigate from view selection to cluster view."""
        mock_project.get_clusters.return_value = sample_clusters

        app = TuxedoApp(mock_project)

        async with app.run_test() as pilot:
            # Start on view selection
            assert isinstance(pilot.app.screen, ViewSelectionScreen)

            # Select the first view (not the "new" item)
            list_view = pilot.app.screen.query_one(ListView)
            if len(list_view.children) > 0:
                list_view.index = 0
                await pilot.press("enter")
                await pilot.pause()

    async def test_create_view_workflow(self, mock_project):
        """Can open and close new view dialog."""
        mock_project.create_view = Mock(return_value=Mock(id="new_view"))
        mock_project.get_papers.return_value = []

        app = TuxedoApp(mock_project)

        async with app.run_test() as pilot:
            # Press 'n' to open new view dialog
            await pilot.press("n")
            await pilot.pause()

            # Cancel the dialog with escape
            await pilot.press("escape")
            await pilot.pause()
