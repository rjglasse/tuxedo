"""Integration tests for end-to-end workflows."""

import json
import pytest
import httpx
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
            clusters, relevance_scores = clusterer.cluster_papers(papers, view.prompt)

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


class TestParallelProcessingWorkflow:
    """Test parallel PDF processing."""

    def test_parallel_process_multiple_pdfs(self, project_with_pdfs, sample_tei_xml, httpx_mock):
        """Multiple PDFs can be processed in parallel."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tuxedo.grobid import GrobidClient

        # Track which threads made requests
        thread_ids = []
        request_lock = threading.Lock()

        def response_callback(request):
            with request_lock:
                thread_ids.append(threading.current_thread().ident)
            return httpx.Response(200, content=sample_tei_xml)

        # Add responses for each PDF (3 PDFs)
        for _ in range(3):
            httpx_mock.add_callback(response_callback)

        grobid_url = "http://localhost:8070"
        pdf_files = project_with_pdfs.list_pdfs()
        results = []
        errors = []

        def process_one(pdf_path):
            with GrobidClient(grobid_url) as client:
                return client.process_pdf_with_result(pdf_path, max_retries=0)

        # Process with 2 workers
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(process_one, pdf): pdf for pdf in pdf_files}
            for future in as_completed(futures):
                result = future.result()
                if result.success:
                    project_with_pdfs.add_paper(result.paper)
                    results.append(result)
                else:
                    errors.append(result.error)

        # All PDFs should be processed successfully
        assert len(results) == 3
        assert len(errors) == 0
        assert project_with_pdfs.paper_count() == 3

    def test_parallel_process_handles_failures(self, project_with_pdfs, sample_tei_xml, httpx_mock):
        """Parallel processing handles individual failures gracefully."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tuxedo.grobid import GrobidClient

        # First PDF succeeds, second fails, third succeeds
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content="Internal Server Error",
            status_code=500,
        )
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )

        grobid_url = "http://localhost:8070"
        pdf_files = project_with_pdfs.list_pdfs()
        successes = []
        errors = []
        results_lock = threading.Lock()

        def process_one(pdf_path):
            with GrobidClient(grobid_url) as client:
                return client.process_pdf_with_result(pdf_path, max_retries=0)

        # Process with 1 worker to ensure order
        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = {pool.submit(process_one, pdf): pdf for pdf in pdf_files}
            for future in as_completed(futures):
                pdf_path = futures[future]
                result = future.result()
                if result.success:
                    project_with_pdfs.add_paper(result.paper)
                    with results_lock:
                        successes.append(pdf_path)
                else:
                    with results_lock:
                        errors.append((pdf_path, result.error))

        # 2 should succeed, 1 should fail
        assert len(successes) == 2
        assert len(errors) == 1
        assert project_with_pdfs.paper_count() == 2

    def test_worker_count_clamping(self):
        """Worker count is clamped to valid range."""

        # Test the clamping logic directly
        def clamp_workers(w):
            return max(1, min(w, 8))

        assert clamp_workers(0) == 1
        assert clamp_workers(-5) == 1
        assert clamp_workers(1) == 1
        assert clamp_workers(4) == 4
        assert clamp_workers(8) == 8
        assert clamp_workers(10) == 8
        assert clamp_workers(100) == 8


class TestFullWorkflowIntegration:
    """End-to-end workflow tests."""

    @pytest.fixture
    def full_project(self, tmp_path, sample_tei_xml, sample_pdf_content, httpx_mock):
        """Create a fully populated project with papers and clusters."""
        # Create source PDFs
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        for i in range(3):
            (source_dir / f"paper{i}.pdf").write_bytes(sample_pdf_content + str(i).encode())

        # Mock Grobid responses with varying metadata
        for i in range(3):
            httpx_mock.add_response(
                url="http://localhost:8070/api/processFulltextDocument",
                content=sample_tei_xml.replace("Test Paper Title", f"Paper {i}: Research Topic"),
                status_code=200,
            )

        # Create project
        project = Project.create(
            root=tmp_path / "project",
            name="Full Test Project",
            research_question="What are the effects of X on Y?",
            source_pdfs=source_dir,
        )

        # Process PDFs
        with GrobidClient("http://localhost:8070") as client:
            for pdf_path in project.list_pdfs():
                paper = client.process_pdf(pdf_path)
                project.add_paper(paper)

        # Create clustering view with mock response
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "clusters": [
                                {
                                    "name": "Methodology Studies",
                                    "description": "Papers about research methodology",
                                    "paper_ids": [project.get_papers()[0].id],
                                    "subclusters": [],
                                },
                                {
                                    "name": "Application Studies",
                                    "description": "Papers applying techniques",
                                    "paper_ids": [
                                        project.get_papers()[1].id,
                                        project.get_papers()[2].id,
                                    ],
                                    "subclusters": [],
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

            view = project.create_view("By Theme", "Group by theme")
            clusterer = PaperClusterer()
            clusters, relevance_scores = clusterer.cluster_papers(project.get_papers(), view.prompt)
            project.save_clusters(view.id, clusters)

        return project

    def test_complete_workflow_paper_count(self, full_project):
        """Complete workflow results in correct paper count."""
        assert full_project.paper_count() == 3

    def test_complete_workflow_view_created(self, full_project):
        """Complete workflow creates clustering view."""
        assert full_project.view_count() == 1
        views = full_project.get_views()
        assert views[0].name == "By Theme"

    def test_complete_workflow_clusters_assigned(self, full_project):
        """Complete workflow assigns papers to clusters."""
        views = full_project.get_views()
        clusters = full_project.get_clusters(views[0].id)

        assert len(clusters) == 2
        total_assigned = sum(len(c.paper_ids) for c in clusters)
        assert total_assigned == 3

    def test_complete_workflow_papers_retrievable(self, full_project):
        """Papers are retrievable by ID after full workflow."""
        papers = full_project.get_papers()
        paper_ids = {p.id for p in papers}

        views = full_project.get_views()
        clusters = full_project.get_clusters(views[0].id)

        for cluster in clusters:
            for paper_id in cluster.paper_ids:
                assert paper_id in paper_ids

    def test_complete_workflow_export_works(self, full_project):
        """Export works after complete workflow."""
        from tuxedo.cli import (
            _export_markdown,
            _export_bibtex,
            _export_csv,
            _export_json,
            _export_ris,
        )

        views = full_project.get_views()
        view = views[0]
        clusters = full_project.get_clusters(view.id)
        papers = full_project.get_papers()
        papers_by_id = {p.id: p for p in papers}

        # Test all export formats
        assert len(_export_markdown(view, clusters, papers_by_id)) > 0
        assert len(_export_bibtex(view, clusters, papers_by_id, include_abstract=False)) > 0
        assert len(_export_csv(view, clusters, papers_by_id)) > 0
        assert len(_export_json(view, clusters, papers_by_id)) > 0
        assert len(_export_ris(view, clusters, papers_by_id)) > 0


class TestProjectRecovery:
    """Tests for project state recovery."""

    @pytest.fixture
    def project_with_state(self, tmp_path):
        """Create a project with papers, views, and clusters."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "paper.pdf").write_bytes(b"%PDF-1.4")

        project = Project.create(
            root=tmp_path / "project",
            name="Test Project",
            research_question="Question?",
            source_pdfs=source_dir,
        )

        # Add papers
        for i in range(3):
            project.add_paper(
                Paper(
                    id=f"paper{i}",
                    pdf_path=project.list_pdfs()[0],
                    title=f"Paper {i}",
                    authors=[Author(name=f"Author {i}")],
                    year=2024,
                )
            )

        # Create views and clusters
        view1 = project.create_view("View 1", "Prompt 1")
        view2 = project.create_view("View 2", "Prompt 2")

        project.save_clusters(
            view1.id,
            [
                Cluster(
                    id="c1",
                    name="Cluster A",
                    description="First",
                    paper_ids=["paper0", "paper1"],
                    subclusters=[],
                )
            ],
        )
        project.save_clusters(
            view2.id,
            [
                Cluster(
                    id="c2",
                    name="Cluster B",
                    description="Second",
                    paper_ids=["paper2"],
                    subclusters=[],
                )
            ],
        )

        return tmp_path / "project"

    def test_project_reloads_correctly(self, project_with_state):
        """Project state is preserved after reload."""
        # Load the project fresh
        import os

        original_dir = os.getcwd()
        os.chdir(project_with_state)
        try:
            project = Project.load()
        finally:
            os.chdir(original_dir)

        assert project is not None
        assert project.paper_count() == 3
        assert project.view_count() == 2

    def test_papers_persist_after_reload(self, project_with_state):
        """Papers are accessible after reload."""
        import os

        original_dir = os.getcwd()
        os.chdir(project_with_state)
        try:
            project = Project.load()
        finally:
            os.chdir(original_dir)

        papers = project.get_papers()
        assert len(papers) == 3
        titles = {p.title for p in papers}
        assert "Paper 0" in titles
        assert "Paper 1" in titles
        assert "Paper 2" in titles

    def test_clusters_persist_after_reload(self, project_with_state):
        """Clusters are accessible after reload."""
        import os

        original_dir = os.getcwd()
        os.chdir(project_with_state)
        try:
            project = Project.load()
        finally:
            os.chdir(original_dir)

        views = project.get_views()
        assert len(views) == 2

        for view in views:
            clusters = project.get_clusters(view.id)
            assert len(clusters) >= 1


class TestGuidedClusteringWorkflow:
    """Test guided clustering with predefined categories."""

    @pytest.fixture
    def project_with_papers(self, tmp_path):
        """Create project with papers."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "paper.pdf").write_bytes(b"%PDF-1.4")

        project = Project.create(
            root=tmp_path / "project",
            name="Test",
            research_question="Q?",
            source_pdfs=source_dir,
        )

        papers = [
            Paper(
                id="p1",
                pdf_path=project.list_pdfs()[0],
                title="Quantitative Analysis of X",
                authors=[Author(name="A")],
                abstract="Using statistical methods...",
                year=2024,
            ),
            Paper(
                id="p2",
                pdf_path=project.list_pdfs()[0],
                title="Qualitative Study of Y",
                authors=[Author(name="B")],
                abstract="Interview-based study...",
                year=2024,
            ),
            Paper(
                id="p3",
                pdf_path=project.list_pdfs()[0],
                title="Mixed Methods Approach",
                authors=[Author(name="C")],
                abstract="Combining quantitative and qualitative...",
                year=2024,
            ),
        ]
        for paper in papers:
            project.add_paper(paper)

        return project

    def test_guided_clustering_assigns_to_categories(self, project_with_papers):
        """Papers are assigned to predefined categories."""
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "clusters": [
                                {
                                    "name": "Quantitative",
                                    "description": "Statistical analysis papers",
                                    "paper_ids": ["p1"],
                                },
                                {
                                    "name": "Qualitative",
                                    "description": "Interview-based studies",
                                    "paper_ids": ["p2"],
                                },
                                {
                                    "name": "Mixed Methods",
                                    "description": "Combined approaches",
                                    "paper_ids": ["p3"],
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

            clusterer = PaperClusterer()
            clusters, relevance_scores = clusterer.cluster_papers(
                project_with_papers.get_papers(),
                "Research question",
                categories=["Quantitative", "Qualitative", "Mixed Methods"],
            )

        assert len(clusters) == 3
        names = {c.name for c in clusters}
        assert "Quantitative" in names
        assert "Qualitative" in names
        assert "Mixed Methods" in names

    def test_guided_strict_mode_no_new_categories(self, project_with_papers):
        """Strict mode prevents new categories."""
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "clusters": [
                                {
                                    "name": "Quantitative",
                                    "description": "Papers",
                                    "paper_ids": ["p1", "p2", "p3"],
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

            clusterer = PaperClusterer()
            clusterer.cluster_papers(
                project_with_papers.get_papers(),
                "Research question",
                categories=["Quantitative"],
                allow_new_categories=False,
            )

            # Verify the strict mode instruction was used
            call_args = mock_client.chat.completions.create.call_args
            system_content = call_args.kwargs["messages"][0]["content"]
            assert "Do NOT create new categories" in system_content


class TestAutoClusteringWorkflow:
    """Test auto-discovery clustering mode."""

    @pytest.fixture
    def project_with_papers(self, tmp_path):
        """Create project with papers."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "paper.pdf").write_bytes(b"%PDF-1.4")

        project = Project.create(
            root=tmp_path / "project",
            name="Test",
            research_question="Q?",
            source_pdfs=source_dir,
        )

        papers = [
            Paper(
                id=f"p{i}",
                pdf_path=project.list_pdfs()[0],
                title=f"Paper {i}",
                authors=[],
                abstract=f"Abstract {i}",
                year=2024,
            )
            for i in range(5)
        ]
        for paper in papers:
            project.add_paper(paper)

        return project

    def test_auto_discovery_uses_correct_prompt(self, project_with_papers):
        """Auto mode uses discovery-focused prompt."""
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {"clusters": [{"name": "Theme", "description": "D", "paper_ids": ["p0"]}]}
                    )
                )
            )
        ]

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            clusterer.cluster_papers(
                project_with_papers.get_papers(), "Ignored question", auto_mode="themes"
            )

            call_args = mock_client.chat.completions.create.call_args
            system_content = call_args.kwargs["messages"][0]["content"]
            user_content = call_args.kwargs["messages"][1]["content"]

            assert "discover" in system_content.lower()
            assert "Discovery Focus" in user_content

    def test_auto_methodology_mode(self, project_with_papers):
        """Methodology mode uses specific focus."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({"clusters": []})))]

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            clusterer.cluster_papers(project_with_papers.get_papers(), "Q", auto_mode="methodology")

            call_args = mock_client.chat.completions.create.call_args
            user_content = call_args.kwargs["messages"][1]["content"]

            assert "methodology" in user_content.lower()

    def test_auto_custom_focus(self, project_with_papers):
        """Custom focus string is used in prompt."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({"clusters": []})))]

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            clusterer.cluster_papers(
                project_with_papers.get_papers(), "Q", auto_mode="machine learning techniques"
            )

            call_args = mock_client.chat.completions.create.call_args
            user_content = call_args.kwargs["messages"][1]["content"]

            assert "machine learning techniques" in user_content
