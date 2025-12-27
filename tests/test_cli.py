"""Tests for the CLI commands."""

import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from click.testing import CliRunner

from tuxedo.cli import main
from tuxedo.models import Author, Cluster, Paper
from tuxedo.grobid import GrobidConnectionError, ProcessingResult


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project structure."""
    # Create source PDFs directory
    source_dir = tmp_path / "papers"
    source_dir.mkdir()
    (source_dir / "paper1.pdf").write_bytes(b"%PDF-1.4 test1")
    (source_dir / "paper2.pdf").write_bytes(b"%PDF-1.4 test2")
    (source_dir / "paper3.pdf").write_bytes(b"%PDF-1.4 test3")

    return tmp_path, source_dir


@pytest.fixture
def initialized_project(temp_project, runner):
    """Create an initialized project."""
    tmp_path, source_dir = temp_project

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            main,
            ["init", str(source_dir), "-q", "What is the effect of X on Y?"],
        )
        assert result.exit_code == 0

    return tmp_path


@pytest.fixture
def sample_paper():
    """Create a sample paper."""
    return Paper(
        id="test123",
        pdf_path=Path("/tmp/test.pdf"),
        title="Test Paper Title",
        authors=[Author(name="John Doe")],
        abstract="This is the abstract.",
        year=2024,
        doi="10.1234/test",
    )


@pytest.fixture
def sample_tei_xml():
    """Sample TEI XML response."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Test Paper</title></titleStmt>
      <publicationStmt><publisher/></publicationStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author><persName><forename>John</forename><surname>Doe</surname></persName></author>
          </analytic>
          <monogr><title level="j">Test Journal</title><imprint><date when="2024"/></imprint></monogr>
          <idno type="DOI">10.1234/test</idno>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc><abstract><p>Test abstract.</p></abstract></profileDesc>
  </teiHeader>
  <text><body><div><head>Introduction</head><p>Intro text.</p></div></body></text>
</TEI>"""


# ============================================================================
# Init Command Tests
# ============================================================================


class TestInitCommand:
    """Tests for the init command."""

    def test_init_creates_project(self, runner, temp_project):
        """Init creates a new project with PDFs."""
        tmp_path, source_dir = temp_project

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                main,
                ["init", str(source_dir), "-q", "What is the effect of X on Y?"],
            )

        assert result.exit_code == 0
        assert "Project" in result.output
        assert "created" in result.output
        assert (tmp_path / "tuxedo.toml").exists()
        assert (tmp_path / "papers").exists()

    def test_init_with_custom_output(self, runner, temp_project):
        """Init can specify custom output directory."""
        tmp_path, source_dir = temp_project
        output_dir = tmp_path / "my-project"

        result = runner.invoke(
            main,
            [
                "init",
                str(source_dir),
                "-q",
                "Research question",
                "-o",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert (output_dir / "tuxedo.toml").exists()

    def test_init_with_custom_name(self, runner, temp_project):
        """Init can specify custom project name."""
        tmp_path, source_dir = temp_project

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                main,
                ["init", str(source_dir), "-q", "Question", "--name", "My Project"],
            )

        assert result.exit_code == 0
        assert "My Project" in result.output

    def test_init_fails_without_pdfs(self, runner, tmp_path):
        """Init fails if source directory has no PDFs."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = runner.invoke(
            main,
            ["init", str(empty_dir), "-q", "Question"],
        )

        assert result.exit_code != 0
        assert "No PDF files found" in result.output

    def test_init_prompts_for_overwrite(self, runner, temp_project):
        """Init prompts before overwriting existing project."""
        tmp_path, source_dir = temp_project

        # Create first project
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(main, ["init", str(source_dir), "-q", "First question"])

            # Try to create again - should prompt
            result = runner.invoke(
                main,
                ["init", str(source_dir), "-q", "Second question"],
                input="n\n",  # Say no to overwrite
            )

        assert result.exit_code != 0  # Aborted


# ============================================================================
# Process Command Tests
# ============================================================================


class TestProcessCommand:
    """Tests for the process command."""

    def test_process_requires_project(self, runner, tmp_path):
        """Process fails without an initialized project."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["process"])

        assert result.exit_code != 0
        assert "No project found" in result.output

    def test_process_checks_grobid_connection(self, runner, initialized_project):
        """Process checks Grobid connection first."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            with patch("tuxedo.cli.GrobidClient") as mock_client_class:
                mock_client = MagicMock()
                mock_client.__enter__ = Mock(return_value=mock_client)
                mock_client.__exit__ = Mock(return_value=False)
                mock_client.check_connection.side_effect = GrobidConnectionError(
                    "http://localhost:8070"
                )
                mock_client_class.return_value = mock_client

                result = runner.invoke(main, ["process"])

        assert result.exit_code != 0
        assert "Grobid" in result.output or "connection" in result.output.lower()

    def test_process_with_workers_option(self, runner, initialized_project, sample_tei_xml):
        """Process accepts --workers option."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            with patch("tuxedo.cli.GrobidClient") as mock_client_class:
                mock_client = MagicMock()
                mock_client.__enter__ = Mock(return_value=mock_client)
                mock_client.__exit__ = Mock(return_value=False)
                mock_client.check_connection.return_value = None
                mock_client.process_pdf_with_result.return_value = ProcessingResult(
                    pdf_path=Path("/test.pdf"),
                    paper=Paper(
                        id="test",
                        pdf_path=Path("/test.pdf"),
                        title="Test",
                        authors=[],
                    ),
                )
                mock_client_class.return_value = mock_client

                result = runner.invoke(main, ["process", "-w", "4"])

        # Should not error on the option
        assert "-w" not in result.output or "error" not in result.output.lower()

    def test_process_single_pdf(self, runner, initialized_project):
        """Process can re-process a single PDF."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            pdf_path = initialized_project / "papers" / "paper1.pdf"

            with patch("tuxedo.cli.GrobidClient") as mock_client_class:
                mock_client = MagicMock()
                mock_client.__enter__ = Mock(return_value=mock_client)
                mock_client.__exit__ = Mock(return_value=False)
                mock_client.check_connection.return_value = None
                mock_client.process_pdf_with_result.return_value = ProcessingResult(
                    pdf_path=pdf_path,
                    paper=Paper(
                        id="test",
                        pdf_path=pdf_path,
                        title="Test",
                        authors=[],
                    ),
                )
                mock_client_class.return_value = mock_client

                result = runner.invoke(main, ["process", str(pdf_path)])

        assert "Re-processing" in result.output


# ============================================================================
# Cluster Command Tests
# ============================================================================


class TestClusterCommand:
    """Tests for the cluster command."""

    def test_cluster_requires_project(self, runner, tmp_path):
        """Cluster fails without an initialized project."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["cluster"])

        assert result.exit_code != 0
        assert "No project found" in result.output

    def test_cluster_requires_papers(self, runner, initialized_project):
        """Cluster fails if no papers are processed."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            result = runner.invoke(main, ["cluster"])

        assert result.exit_code != 0
        assert "No papers" in result.output

    def test_cluster_with_custom_name(self, runner, initialized_project, sample_paper):
        """Cluster accepts custom view name."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            # Add a paper first
            from tuxedo.project import Project

            project = Project.load()
            project.add_paper(sample_paper)

            with patch("tuxedo.cli.PaperClusterer") as mock_clusterer_class:
                mock_clusterer = Mock()
                mock_clusterer.cluster_papers.return_value = (
                    [
                        Cluster(
                            id="c1",
                            name="Test Cluster",
                            description="Test",
                            paper_ids=[sample_paper.id],
                            subclusters=[],
                        )
                    ],
                    {sample_paper.id: 85},
                )
                mock_clusterer_class.return_value = mock_clusterer

                result = runner.invoke(main, ["cluster", "-n", "My Custom View"])

        assert result.exit_code == 0
        assert "My Custom View" in result.output

    def test_cluster_auto_mode(self, runner, initialized_project, sample_paper):
        """Cluster accepts --auto flag."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            from tuxedo.project import Project

            project = Project.load()
            project.add_paper(sample_paper)

            with patch("tuxedo.cli.PaperClusterer") as mock_clusterer_class:
                mock_clusterer = Mock()
                mock_clusterer.cluster_papers.return_value = (
                    [
                        Cluster(
                            id="c1",
                            name="Auto Cluster",
                            description="Auto discovered",
                            paper_ids=[sample_paper.id],
                            subclusters=[],
                        )
                    ],
                    {sample_paper.id: 70},
                )
                mock_clusterer_class.return_value = mock_clusterer

                result = runner.invoke(main, ["cluster", "--auto", "themes"])

        assert result.exit_code == 0

    def test_cluster_with_categories(self, runner, initialized_project, sample_paper):
        """Cluster accepts --categories flag."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            from tuxedo.project import Project

            project = Project.load()
            project.add_paper(sample_paper)

            with patch("tuxedo.cli.PaperClusterer") as mock_clusterer_class:
                mock_clusterer = Mock()
                mock_clusterer.cluster_papers.return_value = (
                    [
                        Cluster(
                            id="c1",
                            name="Quantitative",
                            description="",
                            paper_ids=[sample_paper.id],
                            subclusters=[],
                        )
                    ],
                    {sample_paper.id: 90},
                )
                mock_clusterer_class.return_value = mock_clusterer

                result = runner.invoke(main, ["cluster", "-c", "Quantitative, Qualitative, Mixed"])

        assert result.exit_code == 0


# ============================================================================
# Views Command Tests
# ============================================================================


class TestViewsCommand:
    """Tests for the views command."""

    def test_views_requires_project(self, runner, tmp_path):
        """Views shows error without an initialized project."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["views"])

        # Command returns 0 but prints error message
        assert "No project found" in result.output

    def test_views_shows_empty_message(self, runner, initialized_project):
        """Views shows message when no views exist."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            result = runner.invoke(main, ["views"])

        assert result.exit_code == 0
        assert "No clustering views" in result.output

    def test_views_lists_existing_views(self, runner, initialized_project, sample_paper):
        """Views lists existing clustering views."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            from tuxedo.project import Project

            project = Project.load()
            project.add_paper(sample_paper)
            view = project.create_view("Test View", "Test prompt")
            project.save_clusters(
                view.id,
                [
                    Cluster(
                        id="c1",
                        name="Cluster",
                        description="",
                        paper_ids=[sample_paper.id],
                        subclusters=[],
                    )
                ],
            )

            result = runner.invoke(main, ["views"])

        assert result.exit_code == 0
        assert "Test View" in result.output


# ============================================================================
# Export Command Tests
# ============================================================================


class TestExportCommand:
    """Tests for the export command."""

    def test_export_requires_project(self, runner, tmp_path):
        """Export fails without an initialized project."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["export", "view123"])

        assert result.exit_code != 0
        assert "No project found" in result.output

    def test_export_requires_valid_view(self, runner, initialized_project):
        """Export fails with invalid view ID."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            result = runner.invoke(main, ["export", "nonexistent"])

        assert result.exit_code != 0
        assert "not found" in result.output

    def test_export_markdown(self, runner, initialized_project, sample_paper):
        """Export generates markdown output."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            from tuxedo.project import Project

            project = Project.load()
            project.add_paper(sample_paper)
            view = project.create_view("Export Test", "Test prompt")
            project.save_clusters(
                view.id,
                [
                    Cluster(
                        id="c1",
                        name="Test Cluster",
                        description="Description",
                        paper_ids=[sample_paper.id],
                        subclusters=[],
                    )
                ],
            )

            result = runner.invoke(main, ["export", view.id, "-f", "markdown"])

        assert result.exit_code == 0
        assert "Test Cluster" in result.output
        assert sample_paper.title in result.output

    def test_export_bibtex(self, runner, initialized_project, sample_paper):
        """Export generates BibTeX output."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            from tuxedo.project import Project

            project = Project.load()
            project.add_paper(sample_paper)
            view = project.create_view("BibTeX Test", "Test")
            project.save_clusters(
                view.id,
                [
                    Cluster(
                        id="c1",
                        name="Cluster",
                        description="",
                        paper_ids=[sample_paper.id],
                        subclusters=[],
                    )
                ],
            )

            result = runner.invoke(main, ["export", view.id, "-f", "bibtex"])

        assert result.exit_code == 0
        assert "@" in result.output  # BibTeX entry
        assert "title" in result.output

    def test_export_to_file(self, runner, initialized_project, sample_paper):
        """Export writes to file."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            from tuxedo.project import Project

            project = Project.load()
            project.add_paper(sample_paper)
            view = project.create_view("File Test", "Test")
            project.save_clusters(
                view.id,
                [
                    Cluster(
                        id="c1",
                        name="Cluster",
                        description="",
                        paper_ids=[sample_paper.id],
                        subclusters=[],
                    )
                ],
            )

            output_file = initialized_project / "output.md"
            result = runner.invoke(
                main, ["export", view.id, "-f", "markdown", "-o", str(output_file)]
            )

        assert result.exit_code == 0
        assert output_file.exists()
        assert sample_paper.title in output_file.read_text()

    def test_export_csv(self, runner, initialized_project, sample_paper):
        """Export generates CSV output."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            from tuxedo.project import Project

            project = Project.load()
            project.add_paper(sample_paper)
            view = project.create_view("CSV Test", "Test")
            project.save_clusters(
                view.id,
                [
                    Cluster(
                        id="c1",
                        name="Cluster",
                        description="",
                        paper_ids=[sample_paper.id],
                        subclusters=[],
                    )
                ],
            )

            result = runner.invoke(main, ["export", view.id, "-f", "csv"])

        assert result.exit_code == 0
        assert "Cluster" in result.output
        assert "," in result.output  # CSV format

    def test_export_json(self, runner, initialized_project, sample_paper):
        """Export generates JSON output."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            from tuxedo.project import Project

            project = Project.load()
            project.add_paper(sample_paper)
            view = project.create_view("JSON Test", "Test")
            project.save_clusters(
                view.id,
                [
                    Cluster(
                        id="c1",
                        name="Cluster",
                        description="",
                        paper_ids=[sample_paper.id],
                        subclusters=[],
                    )
                ],
            )

            result = runner.invoke(main, ["export", view.id, "-f", "json"])

        assert result.exit_code == 0
        # Should be valid JSON
        data = json.loads(result.output)
        assert "clusters" in data

    def test_export_ris(self, runner, initialized_project, sample_paper):
        """Export generates RIS output."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            from tuxedo.project import Project

            project = Project.load()
            project.add_paper(sample_paper)
            view = project.create_view("RIS Test", "Test")
            project.save_clusters(
                view.id,
                [
                    Cluster(
                        id="c1",
                        name="Cluster",
                        description="",
                        paper_ids=[sample_paper.id],
                        subclusters=[],
                    )
                ],
            )

            result = runner.invoke(main, ["export", view.id, "-f", "ris"])

        assert result.exit_code == 0
        assert "TY  -" in result.output  # RIS format
        assert "TI  -" in result.output


# ============================================================================
# Status Command Tests
# ============================================================================


class TestStatusCommand:
    """Tests for the status command."""

    def test_status_requires_project(self, runner, tmp_path):
        """Status fails without an initialized project."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["status"])

        assert result.exit_code == 0  # status doesn't abort, just shows message
        assert "No project found" in result.output

    def test_status_shows_project_info(self, runner, initialized_project):
        """Status shows project information."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        assert "Research Question" in result.output
        assert "PDFs" in result.output


# ============================================================================
# Papers Command Tests
# ============================================================================


class TestPapersCommand:
    """Tests for the papers command."""

    def test_papers_requires_project(self, runner, tmp_path):
        """Papers fails without an initialized project."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["papers"])

        assert result.exit_code == 0  # papers doesn't abort
        assert "No project found" in result.output

    def test_papers_shows_empty_message(self, runner, initialized_project):
        """Papers shows message when no papers processed."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            result = runner.invoke(main, ["papers"])

        assert result.exit_code == 0
        assert "No papers processed" in result.output

    def test_papers_lists_processed_papers(self, runner, initialized_project, sample_paper):
        """Papers lists processed papers."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            from tuxedo.project import Project

            project = Project.load()
            project.add_paper(sample_paper)

            result = runner.invoke(main, ["papers"])

        assert result.exit_code == 0
        assert sample_paper.title in result.output or "Test Paper" in result.output


# ============================================================================
# Delete Commands Tests
# ============================================================================


class TestDeleteCommands:
    """Tests for delete-paper and delete-view commands."""

    def test_delete_paper_requires_project(self, runner, tmp_path):
        """Delete-paper fails without project."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["delete-paper", "paper123"])

        assert result.exit_code != 0
        assert "No project found" in result.output

    def test_delete_paper_requires_valid_id(self, runner, initialized_project):
        """Delete-paper fails with invalid paper ID."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            result = runner.invoke(main, ["delete-paper", "nonexistent"])

        assert result.exit_code != 0
        assert "not found" in result.output

    def test_delete_paper_prompts_for_confirmation(self, runner, initialized_project, sample_paper):
        """Delete-paper prompts for confirmation."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            from tuxedo.project import Project

            project = Project.load()
            project.add_paper(sample_paper)

            result = runner.invoke(main, ["delete-paper", sample_paper.id], input="n\n")

        assert result.exit_code != 0  # Aborted

    def test_delete_paper_with_force(self, runner, initialized_project, sample_paper):
        """Delete-paper with --force skips confirmation."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            from tuxedo.project import Project

            project = Project.load()
            project.add_paper(sample_paper)

            result = runner.invoke(main, ["delete-paper", sample_paper.id, "-f"])

        assert result.exit_code == 0
        assert "Deleted" in result.output

    def test_delete_view_requires_project(self, runner, tmp_path):
        """Delete-view fails without project."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["delete-view", "view123"])

        assert result.exit_code != 0
        assert "No project found" in result.output

    def test_delete_view_with_force(self, runner, initialized_project, sample_paper):
        """Delete-view with --force deletes view."""
        with runner.isolated_filesystem(temp_dir=initialized_project):
            from tuxedo.project import Project

            project = Project.load()
            project.add_paper(sample_paper)
            view = project.create_view("To Delete", "Test")

            result = runner.invoke(main, ["delete-view", view.id, "-f"])

        assert result.exit_code == 0
        assert "Deleted" in result.output


# ============================================================================
# Completion Command Tests
# ============================================================================


class TestCompletionCommand:
    """Tests for the completion command."""

    def test_completion_shows_instructions(self, runner):
        """Completion without args shows setup instructions."""
        result = runner.invoke(main, ["completion"])

        assert result.exit_code == 0
        assert "Shell Completion" in result.output
        assert "bash" in result.output.lower()
        assert "zsh" in result.output.lower()

    def test_completion_bash(self, runner):
        """Completion generates bash script."""
        result = runner.invoke(main, ["completion", "bash"])

        # Should either output script or show error
        assert result.exit_code == 0

    def test_completion_zsh(self, runner):
        """Completion generates zsh script."""
        result = runner.invoke(main, ["completion", "zsh"])

        assert result.exit_code == 0

    def test_completion_fish(self, runner):
        """Completion generates fish script."""
        result = runner.invoke(main, ["completion", "fish"])

        assert result.exit_code == 0


# ============================================================================
# Main Group Tests
# ============================================================================


class TestMainGroup:
    """Tests for the main CLI group."""

    def test_help(self, runner):
        """--help shows usage information."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Tuxedo" in result.output
        assert "init" in result.output
        assert "process" in result.output
        assert "cluster" in result.output

    def test_version(self, runner):
        """--version shows version information."""
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        # Should show some version info

    def test_unknown_command(self, runner):
        """Unknown command shows error."""
        result = runner.invoke(main, ["unknown-command"])

        assert result.exit_code != 0
