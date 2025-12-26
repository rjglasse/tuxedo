"""Integration tests for end-to-end workflows."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from tuxedo.project import Project
from tuxedo.grobid import GrobidClient
from tuxedo.clustering import PaperClusterer
from tuxedo.models import Author, Cluster, Paper


@pytest.fixture
def sample_tei_xml():
    """Sample TEI XML response from Grobid."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Test Paper Title</title>
      </titleStmt>
      <publicationStmt>
        <publisher>Test Publisher</publisher>
      </publicationStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName>
                <forename>John</forename>
                <surname>Doe</surname>
              </persName>
              <affiliation><orgName>MIT</orgName></affiliation>
            </author>
          </analytic>
          <monogr>
            <title level="j">Test Journal</title>
            <imprint>
              <date when="2024"/>
            </imprint>
          </monogr>
          <idno type="DOI">10.1234/test</idno>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>This is the abstract text.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div><head>Introduction</head><p>Intro text here.</p></div>
    </body>
  </text>
</TEI>"""


@pytest.fixture
def sample_pdf_content():
    """Minimal PDF content."""
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


@pytest.fixture
def project_with_pdfs(tmp_path, sample_pdf_content):
    """Create a project with sample PDFs."""
    # Create source PDFs
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "paper1.pdf").write_bytes(sample_pdf_content + b"1")
    (source_dir / "paper2.pdf").write_bytes(sample_pdf_content + b"2")
    (source_dir / "paper3.pdf").write_bytes(sample_pdf_content + b"3")

    # Create project
    project = Project.create(
        root=tmp_path / "myproject",
        name="Test Project",
        research_question="What are the effects of X on Y?",
        source_pdfs=source_dir,
    )

    return project


class TestProjectCreationWorkflow:
    """Test project creation and setup."""

    def test_create_project_copies_pdfs(self, project_with_pdfs):
        """Project creation copies PDFs to papers directory."""
        pdfs = project_with_pdfs.list_pdfs()
        assert len(pdfs) == 3
        assert all(p.suffix == ".pdf" for p in pdfs)

    def test_project_config_saved(self, project_with_pdfs):
        """Project config is saved correctly."""
        config = project_with_pdfs.config
        assert config.name == "Test Project"
        assert config.research_question == "What are the effects of X on Y?"

    def test_project_database_initialized(self, project_with_pdfs):
        """Database is initialized with schema."""
        assert project_with_pdfs.db_path.exists()
        assert project_with_pdfs.paper_count() == 0


class TestPdfProcessingWorkflow:
    """Test PDF processing with Grobid."""

    def test_process_pdfs_and_store(self, project_with_pdfs, sample_tei_xml, httpx_mock):
        """PDFs are processed and papers stored in database."""
        # Mock Grobid responses
        for _ in range(3):
            httpx_mock.add_response(
                url="http://localhost:8070/api/processFulltextDocument",
                content=sample_tei_xml,
                status_code=200,
            )

        # Process PDFs
        with GrobidClient("http://localhost:8070") as client:
            for pdf_path in project_with_pdfs.list_pdfs():
                paper = client.process_pdf(pdf_path)
                project_with_pdfs.add_paper(paper)

        # Verify papers stored
        assert project_with_pdfs.paper_count() == 3
        papers = project_with_pdfs.get_papers()
        assert all(p.title == "Test Paper Title" for p in papers)
        assert all(p.journal == "Test Journal" for p in papers)


class TestClusteringWorkflow:
    """Test paper clustering with LLM."""

    @pytest.fixture
    def project_with_papers(self, project_with_pdfs):
        """Project with papers already added."""
        for i, pdf_path in enumerate(project_with_pdfs.list_pdfs()):
            paper = Paper(
                id=f"paper{i + 1}",
                pdf_path=pdf_path,
                title=f"Paper {i + 1}: Machine Learning Applications",
                authors=[Author(name=f"Author {i + 1}")],
                abstract=f"Abstract for paper {i + 1}",
                year=2024,
            )
            project_with_pdfs.add_paper(paper)
        return project_with_pdfs

    def test_cluster_papers_creates_view(self, project_with_papers):
        """Clustering creates a view with clusters."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "clusters": [
                                {
                                    "name": "ML Applications",
                                    "description": "Papers about ML applications",
                                    "paper_ids": ["paper1", "paper2"],
                                },
                                {
                                    "name": "Deep Learning",
                                    "description": "Papers about DL",
                                    "paper_ids": ["paper3"],
                                },
                            ]
                        }
                    )
                )
            )
        ]

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            # Create clustering view
            view = project_with_papers.create_view(
                name="By Theme",
                prompt="Group papers by theme",
            )

            # Cluster papers
            clusterer = PaperClusterer()
            papers = project_with_papers.get_papers()
            clusters = clusterer.cluster_papers(papers, view.prompt)

            # Save clusters
            project_with_papers.save_clusters(view.id, clusters)

        # Verify view and clusters
        assert project_with_papers.view_count() == 1
        assert project_with_papers.cluster_count(view.id) == 2

        saved_clusters = project_with_papers.get_clusters(view.id)
        assert len(saved_clusters) == 2
        cluster_names = {c.name for c in saved_clusters}
        assert "ML Applications" in cluster_names
        assert "Deep Learning" in cluster_names


class TestExportWorkflow:
    """Test export functionality."""

    @pytest.fixture
    def project_with_clusters(self, project_with_pdfs):
        """Project with papers and clusters."""
        # Add papers
        papers = []
        for i, pdf_path in enumerate(project_with_pdfs.list_pdfs()):
            paper = Paper(
                id=f"paper{i + 1}",
                pdf_path=pdf_path,
                title=f"Paper {i + 1}: Research Topic",
                authors=[Author(name=f"Smith{i + 1}")],
                abstract=f"Abstract for paper {i + 1}",
                year=2024 - i,
                doi=f"10.1234/paper{i + 1}",
                journal="Test Journal" if i == 0 else None,
                booktitle="Test Conference" if i == 1 else None,
            )
            project_with_pdfs.add_paper(paper)
            papers.append(paper)

        # Create view and clusters
        view = project_with_pdfs.create_view(
            name="Test View",
            prompt="Test prompt",
        )
        clusters = [
            Cluster(
                id="c1",
                name="Cluster A",
                description="First cluster",
                paper_ids=["paper1", "paper2"],
                subclusters=[],
            ),
            Cluster(
                id="c2",
                name="Cluster B",
                description="Second cluster",
                paper_ids=["paper3"],
                subclusters=[],
            ),
        ]
        project_with_pdfs.save_clusters(view.id, clusters)

        return project_with_pdfs, view

    def test_papers_retrievable_by_cluster(self, project_with_clusters):
        """Papers can be retrieved through cluster associations."""
        project, view = project_with_clusters
        clusters = project.get_clusters(view.id)

        assert len(clusters) == 2
        assert len(clusters[0].paper_ids) == 2
        assert len(clusters[1].paper_ids) == 1

        # Verify paper data accessible
        all_papers = project.get_papers()
        paper_map = {p.id: p for p in all_papers}

        for cluster in clusters:
            for paper_id in cluster.paper_ids:
                paper = paper_map[paper_id]
                assert paper.title is not None
                assert paper.year is not None


class TestPaperEditWorkflow:
    """Test paper metadata editing."""

    def test_update_paper_metadata(self, project_with_pdfs):
        """Paper metadata can be updated."""
        # Add a paper
        paper = Paper(
            id="paper1",
            pdf_path=project_with_pdfs.list_pdfs()[0],
            title="Original Title",
            authors=[Author(name="Original Author")],
            year=2020,
        )
        project_with_pdfs.add_paper(paper)

        # Update metadata
        project_with_pdfs.update_paper(
            "paper1",
            {
                "title": "Updated Title",
                "year": 2024,
                "doi": "10.1234/new",
                "authors": [Author(name="New Author")],
            },
        )

        # Verify updates
        papers = project_with_pdfs.get_papers()
        updated = papers[0]
        assert updated.title == "Updated Title"
        assert updated.year == 2024
        assert updated.doi == "10.1234/new"
        assert updated.authors[0].name == "New Author"


class TestMultipleViewsWorkflow:
    """Test managing multiple clustering views."""

    @pytest.fixture
    def project_with_papers(self, project_with_pdfs):
        """Project with papers."""
        for i in range(3):
            paper = Paper(
                id=f"paper{i + 1}",
                pdf_path=project_with_pdfs.list_pdfs()[i],
                title=f"Paper {i + 1}",
                authors=[Author(name=f"Author {i + 1}")],
                year=2024,
            )
            project_with_pdfs.add_paper(paper)
        return project_with_pdfs

    def test_create_multiple_views(self, project_with_papers):
        """Multiple views can be created with different clusterings."""
        # Create first view
        view1 = project_with_papers.create_view("By Method", "Group by method")
        clusters1 = [
            Cluster(id="m1", name="Method A", description="", paper_ids=["paper1"], subclusters=[]),
            Cluster(
                id="m2",
                name="Method B",
                description="",
                paper_ids=["paper2", "paper3"],
                subclusters=[],
            ),
        ]
        project_with_papers.save_clusters(view1.id, clusters1)

        # Create second view with different clustering
        view2 = project_with_papers.create_view("By Year", "Group by year")
        clusters2 = [
            Cluster(
                id="y1",
                name="2024",
                description="",
                paper_ids=["paper1", "paper2", "paper3"],
                subclusters=[],
            ),
        ]
        project_with_papers.save_clusters(view2.id, clusters2)

        # Verify both views exist independently
        assert project_with_papers.view_count() == 2
        assert project_with_papers.cluster_count(view1.id) == 2
        assert project_with_papers.cluster_count(view2.id) == 1

        # Verify clusters are separate
        c1 = project_with_papers.get_clusters(view1.id)
        c2 = project_with_papers.get_clusters(view2.id)
        assert c1[0].name == "Method A"
        assert c2[0].name == "2024"

    def test_delete_view_preserves_others(self, project_with_papers):
        """Deleting a view doesn't affect other views."""
        view1 = project_with_papers.create_view("View 1", "Prompt 1")
        view2 = project_with_papers.create_view("View 2", "Prompt 2")

        clusters1 = [
            Cluster(id="c1", name="Cluster", description="", paper_ids=["paper1"], subclusters=[])
        ]
        clusters2 = [
            Cluster(id="c2", name="Cluster", description="", paper_ids=["paper1"], subclusters=[])
        ]
        project_with_papers.save_clusters(view1.id, clusters1)
        project_with_papers.save_clusters(view2.id, clusters2)

        # Delete first view
        project_with_papers.delete_view(view1.id)

        # Second view still exists
        assert project_with_papers.view_count() == 1
        assert project_with_papers.get_view(view2.id) is not None
        assert project_with_papers.cluster_count(view2.id) == 1
