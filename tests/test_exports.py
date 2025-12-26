"""Tests for export functionality."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from tuxedo.models import Author, Cluster, ClusterView, Paper

# Import the export functions directly
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tuxedo.cli import (
    _export_json,
    _export_markdown,
    _export_bibtex,
    _export_latex,
    _export_csv,
    _export_ris,
)


@pytest.fixture
def sample_view():
    """Create a sample cluster view."""
    return ClusterView(
        id="view1",
        name="Test View",
        prompt="Group papers by methodology",
        created_at=datetime(2024, 1, 15, 10, 30),
    )


@pytest.fixture
def sample_papers():
    """Create sample papers."""
    return [
        Paper(
            id="paper1",
            pdf_path=Path("paper1.pdf"),
            title="Machine Learning for Healthcare",
            authors=[Author(name="Smith, John"), Author(name="Doe, Jane")],
            abstract="This paper explores ML applications in healthcare.",
            year=2024,
            doi="10.1234/ml.healthcare",
            journal="Journal of ML",
            volume="15",
            number="3",
            pages="100-120",
            keywords=["machine learning", "healthcare"],
        ),
        Paper(
            id="paper2",
            pdf_path=Path("paper2.pdf"),
            title="Deep Learning Survey",
            authors=[Author(name="Johnson, Bob")],
            abstract="A comprehensive survey of deep learning.",
            year=2023,
            booktitle="Conference on AI",
            publisher="ACM",
        ),
        Paper(
            id="paper3",
            pdf_path=Path("paper3.pdf"),
            title="Neural Networks",
            authors=[Author(name="Williams, Alice")],
            year=2022,
            arxiv_id="2201.12345",
        ),
    ]


@pytest.fixture
def sample_clusters():
    """Create sample clusters."""
    return [
        Cluster(
            id="c1",
            name="Healthcare Applications",
            description="Papers about healthcare",
            paper_ids=["paper1"],
            subclusters=[],
        ),
        Cluster(
            id="c2",
            name="Surveys & Theory",
            description="Survey papers",
            paper_ids=["paper2", "paper3"],
            subclusters=[],
        ),
    ]


@pytest.fixture
def papers_by_id(sample_papers):
    """Create papers lookup dict."""
    return {p.id: p for p in sample_papers}


class TestExportJson:
    """Tests for JSON export."""

    def test_export_json_structure(self, sample_view, sample_clusters, papers_by_id):
        """JSON export has correct structure."""
        result = _export_json(sample_view, sample_clusters, papers_by_id)
        data = json.loads(result)

        assert "view" in data
        assert "clusters" in data
        assert data["view"]["id"] == "view1"
        assert data["view"]["name"] == "Test View"
        assert len(data["clusters"]) == 2

    def test_export_json_papers(self, sample_view, sample_clusters, papers_by_id):
        """JSON export includes paper details."""
        result = _export_json(sample_view, sample_clusters, papers_by_id)
        data = json.loads(result)

        cluster1 = data["clusters"][0]
        assert len(cluster1["papers"]) == 1
        assert cluster1["papers"][0]["title"] == "Machine Learning for Healthcare"
        assert cluster1["papers"][0]["year"] == 2024


class TestExportMarkdown:
    """Tests for Markdown export."""

    def test_export_markdown_header(self, sample_view, sample_clusters, papers_by_id):
        """Markdown export has correct header."""
        result = _export_markdown(sample_view, sample_clusters, papers_by_id)

        assert "# Test View" in result
        assert "> Group papers by methodology" in result

    def test_export_markdown_clusters(self, sample_view, sample_clusters, papers_by_id):
        """Markdown export includes clusters."""
        result = _export_markdown(sample_view, sample_clusters, papers_by_id)

        assert "## Healthcare Applications" in result
        assert "## Surveys & Theory" in result

    def test_export_markdown_papers(self, sample_view, sample_clusters, papers_by_id):
        """Markdown export includes papers."""
        result = _export_markdown(sample_view, sample_clusters, papers_by_id)

        assert "Machine Learning for Healthcare" in result
        assert "Deep Learning Survey" in result


class TestExportBibtex:
    """Tests for BibTeX export."""

    def test_export_bibtex_journal(self, sample_view, sample_clusters, papers_by_id):
        """BibTeX export handles journal articles."""
        result = _export_bibtex(sample_view, sample_clusters, papers_by_id)

        assert "@article{" in result
        assert "title = {Machine Learning for Healthcare}" in result
        assert "journal = {Journal of ML}" in result
        assert "year = {2024}" in result
        assert "doi = {10.1234/ml.healthcare}" in result

    def test_export_bibtex_conference(self, sample_view, sample_clusters, papers_by_id):
        """BibTeX export handles conference papers."""
        result = _export_bibtex(sample_view, sample_clusters, papers_by_id)

        assert "@inproceedings{" in result
        assert "booktitle = {Conference on AI}" in result

    def test_export_bibtex_with_abstract(self, sample_view, sample_clusters, papers_by_id):
        """BibTeX export can include abstracts."""
        result = _export_bibtex(sample_view, sample_clusters, papers_by_id, include_abstract=True)

        assert "abstract = {This paper explores ML applications in healthcare.}" in result

    def test_export_bibtex_without_abstract(self, sample_view, sample_clusters, papers_by_id):
        """BibTeX export excludes abstracts by default."""
        result = _export_bibtex(sample_view, sample_clusters, papers_by_id, include_abstract=False)

        assert "abstract = {This paper explores" not in result


class TestExportLatex:
    """Tests for LaTeX export."""

    def test_export_latex_structure(self, sample_view, sample_clusters, papers_by_id):
        """LaTeX export has correct structure."""
        result = _export_latex(sample_view, sample_clusters, papers_by_id)

        assert r"\documentclass{article}" in result
        assert r"\begin{document}" in result
        assert r"\end{document}" in result
        assert r"\bibliography{references}" in result

    def test_export_latex_sections(self, sample_view, sample_clusters, papers_by_id):
        """LaTeX export creates sections for clusters."""
        result = _export_latex(sample_view, sample_clusters, papers_by_id)

        assert r"\section{Healthcare Applications}" in result
        assert r"\section{Surveys \& Theory}" in result  # & should be escaped

    def test_export_latex_citations(self, sample_view, sample_clusters, papers_by_id):
        """LaTeX export includes citations."""
        result = _export_latex(sample_view, sample_clusters, papers_by_id)

        assert r"\citep{" in result


class TestExportCsv:
    """Tests for CSV export."""

    def test_export_csv_header(self, sample_view, sample_clusters, papers_by_id):
        """CSV export has correct header."""
        result = _export_csv(sample_view, sample_clusters, papers_by_id)
        lines = result.strip().split("\n")

        assert "Cluster,Paper ID,Title,Authors,Year,DOI,Journal,Abstract" in lines[0]

    def test_export_csv_rows(self, sample_view, sample_clusters, papers_by_id):
        """CSV export has correct data rows."""
        result = _export_csv(sample_view, sample_clusters, papers_by_id)
        lines = result.strip().split("\n")

        # Header + 3 papers = 4 lines
        assert len(lines) == 4

    def test_export_csv_cluster_path(self, sample_view, sample_clusters, papers_by_id):
        """CSV export includes cluster names."""
        result = _export_csv(sample_view, sample_clusters, papers_by_id)

        assert "Healthcare Applications" in result
        assert "Surveys & Theory" in result


class TestExportRis:
    """Tests for RIS export."""

    def test_export_ris_journal(self, sample_view, sample_clusters, papers_by_id):
        """RIS export handles journal articles."""
        result = _export_ris(sample_view, sample_clusters, papers_by_id)

        assert "TY  - JOUR" in result
        assert "TI  - Machine Learning for Healthcare" in result
        assert "JO  - Journal of ML" in result
        assert "PY  - 2024" in result

    def test_export_ris_conference(self, sample_view, sample_clusters, papers_by_id):
        """RIS export handles conference papers."""
        result = _export_ris(sample_view, sample_clusters, papers_by_id)

        assert "TY  - CONF" in result
        assert "T2  - Conference on AI" in result

    def test_export_ris_arxiv(self, sample_view, sample_clusters, papers_by_id):
        """RIS export handles arXiv papers."""
        result = _export_ris(sample_view, sample_clusters, papers_by_id)

        assert "TY  - UNPB" in result
        assert "UR  - https://arxiv.org/abs/2201.12345" in result

    def test_export_ris_authors(self, sample_view, sample_clusters, papers_by_id):
        """RIS export includes all authors."""
        result = _export_ris(sample_view, sample_clusters, papers_by_id)

        assert "AU  - Smith, John" in result
        assert "AU  - Doe, Jane" in result

    def test_export_ris_end_records(self, sample_view, sample_clusters, papers_by_id):
        """RIS export has proper end records."""
        result = _export_ris(sample_view, sample_clusters, papers_by_id)

        # Should have 3 end records for 3 papers
        assert result.count("ER  - ") == 3

    def test_export_ris_no_duplicates(self, sample_view, papers_by_id):
        """RIS export doesn't duplicate papers in multiple clusters."""
        # Create clusters where paper1 appears in both
        clusters = [
            Cluster(id="c1", name="A", description="", paper_ids=["paper1"], subclusters=[]),
            Cluster(
                id="c2", name="B", description="", paper_ids=["paper1", "paper2"], subclusters=[]
            ),
        ]

        result = _export_ris(sample_view, clusters, papers_by_id)

        # paper1 should only appear once
        assert result.count("TI  - Machine Learning for Healthcare") == 1
