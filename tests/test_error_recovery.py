"""Tests for error recovery and edge cases."""

import json
import pytest
import httpx
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import MagicMock, patch

from openai import APIConnectionError, RateLimitError

from tuxedo.clustering import PaperClusterer
from tuxedo.grobid import GrobidClient, GrobidConnectionError
from tuxedo.models import Author, Cluster, Paper
from tuxedo.project import Project


# ============================================================================
# Clustering Error Recovery Tests
# ============================================================================


class TestClusteringAPIErrors:
    """Test clustering behavior when OpenAI API fails."""

    @pytest.fixture
    def sample_papers(self):
        """Sample papers for testing."""
        return [
            Paper(
                id="p1",
                pdf_path=Path("paper1.pdf"),
                title="Deep Learning for NLP",
                authors=[Author(name="Alice Smith")],
                abstract="A paper about deep learning.",
                year=2024,
            ),
            Paper(
                id="p2",
                pdf_path=Path("paper2.pdf"),
                title="Reinforcement Learning",
                authors=[Author(name="Bob Jones")],
                abstract="A survey of RL methods.",
                year=2023,
            ),
        ]

    def test_connection_error_propagates(self, sample_papers):
        """API connection errors are raised to caller."""
        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = APIConnectionError(
                request=MagicMock()
            )
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()

            with pytest.raises(APIConnectionError):
                clusterer.cluster_papers(sample_papers, "research question")

    def test_rate_limit_error_propagates(self, sample_papers):
        """Rate limit errors are raised to caller."""
        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_client.chat.completions.create.side_effect = RateLimitError(
                message="Rate limit exceeded",
                response=mock_response,
                body={"error": {"message": "Rate limit exceeded"}},
            )
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()

            with pytest.raises(RateLimitError):
                clusterer.cluster_papers(sample_papers, "research question")

    def test_invalid_json_response_raises(self, sample_papers):
        """Invalid JSON in API response raises error."""
        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "not valid json {"
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()

            with pytest.raises(json.JSONDecodeError):
                clusterer.cluster_papers(sample_papers, "research question")

    def test_missing_clusters_key_returns_empty(self, sample_papers):
        """Response without 'clusters' key returns empty list."""
        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps({"themes": []})
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            result = clusterer.cluster_papers(sample_papers, "research question")

            assert result == []

    def test_empty_clusters_returns_empty(self, sample_papers):
        """Empty clusters array returns empty list."""
        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps({"clusters": []})
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            result = clusterer.cluster_papers(sample_papers, "research question")

            assert result == []

    def test_malformed_cluster_gets_defaults(self, sample_papers):
        """Malformed cluster entries get default values."""
        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            # Cluster with missing required fields
            mock_response.choices[0].message.content = json.dumps(
                {"clusters": [{"weird_field": "value"}]}
            )
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            result = clusterer.cluster_papers(sample_papers, "research question")

            assert len(result) == 1
            assert result[0].name == "Unnamed"
            assert result[0].description == ""
            assert result[0].paper_ids == []


class TestClusteringBatchErrors:
    """Test batch clustering error handling."""

    @pytest.fixture
    def sample_papers(self):
        """Create enough papers for batch processing."""
        return [
            Paper(
                id=f"p{i}",
                pdf_path=Path(f"paper{i}.pdf"),
                title=f"Paper {i}",
                authors=[Author(name=f"Author {i}")],
                abstract=f"Abstract {i}",
                year=2024,
            )
            for i in range(5)
        ]

    def test_batch_error_in_first_batch_propagates(self, sample_papers):
        """Error in first batch propagates to caller."""
        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = APIConnectionError(
                request=MagicMock()
            )
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()

            with pytest.raises(APIConnectionError):
                clusterer.cluster_papers(sample_papers, "question", batch_size=2)

    def test_batch_error_in_subsequent_batch_propagates(self, sample_papers):
        """Error in subsequent batch propagates to caller."""
        # First batch succeeds, second fails
        first_response = MagicMock()
        first_response.choices = [MagicMock()]
        first_response.choices[0].message.content = json.dumps(
            {"clusters": [{"name": "Theme A", "description": "First", "paper_ids": ["p0", "p1"]}]}
        )

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [
                first_response,
                APIConnectionError(request=MagicMock()),
            ]
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()

            with pytest.raises(APIConnectionError):
                clusterer.cluster_papers(sample_papers, "question", batch_size=2)

    def test_batch_progress_callback_on_error(self, sample_papers):
        """Progress callback is called before error occurs."""
        progress_calls = []

        def callback(batch_num, total, message):
            progress_calls.append((batch_num, total))

        first_response = MagicMock()
        first_response.choices = [MagicMock()]
        first_response.choices[0].message.content = json.dumps(
            {"clusters": [{"name": "Theme", "description": "Desc", "paper_ids": ["p0", "p1"]}]}
        )

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [
                first_response,
                APIConnectionError(request=MagicMock()),
            ]
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()

            with pytest.raises(APIConnectionError):
                clusterer.cluster_papers(
                    sample_papers, "question", batch_size=2, progress_callback=callback
                )

        # First batch callback should have been called
        assert len(progress_calls) >= 1
        assert progress_calls[0] == (1, 3)  # First of 3 batches


class TestReclusterErrorHandling:
    """Test recluster error handling."""

    @pytest.fixture
    def sample_papers(self):
        return [
            Paper(
                id="p1",
                pdf_path=Path("paper1.pdf"),
                title="Test Paper",
                authors=[],
                abstract="Abstract",
            )
        ]

    @pytest.fixture
    def current_clusters(self):
        return [
            Cluster(id="c1", name="Theme A", description="Desc", paper_ids=["p1"], subclusters=[])
        ]

    def test_recluster_api_error_propagates(self, sample_papers, current_clusters):
        """API errors during recluster propagate to caller."""
        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = APIConnectionError(
                request=MagicMock()
            )
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()

            with pytest.raises(APIConnectionError):
                clusterer.recluster(sample_papers, "question", "feedback", current_clusters)

    def test_recluster_invalid_json_raises(self, sample_papers, current_clusters):
        """Invalid JSON during recluster raises error."""
        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "invalid json"
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()

            with pytest.raises(json.JSONDecodeError):
                clusterer.recluster(sample_papers, "question", "feedback", current_clusters)


# ============================================================================
# Parallel Processing Edge Cases
# ============================================================================


class TestParallelProcessingEdgeCases:
    """Test edge cases in parallel PDF processing."""

    @pytest.fixture
    def sample_tei_xml(self):
        """Sample TEI XML response from Grobid."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Test Paper</title></titleStmt>
      <publicationStmt><publisher>Test</publisher></publicationStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author><persName><forename>John</forename><surname>Doe</surname></persName></author>
          </analytic>
          <monogr><imprint><date when="2024"/></imprint></monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc><abstract><p>Abstract text.</p></abstract></profileDesc>
  </teiHeader>
  <text><body><div><head>Introduction</head><p>Text.</p></div></body></text>
</TEI>"""

    @pytest.fixture
    def project_with_pdfs(self, tmp_path):
        """Create a project with sample PDFs."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        for i in range(5):
            (source_dir / f"paper{i}.pdf").write_bytes(b"%PDF-1.4 test " + str(i).encode())

        project = Project.create(
            root=tmp_path / "project",
            name="Test Project",
            research_question="Question?",
            source_pdfs=source_dir,
        )
        return project

    def test_all_pdfs_fail(self, project_with_pdfs, httpx_mock):
        """All PDFs failing is handled gracefully."""
        # All requests return 500
        for _ in range(5):
            httpx_mock.add_response(
                url="http://localhost:8070/api/processFulltextDocument",
                content="Internal Server Error",
                status_code=500,
            )

        grobid_url = "http://localhost:8070"
        pdf_files = project_with_pdfs.list_pdfs()
        successes = []
        errors = []

        def process_one(pdf_path):
            with GrobidClient(grobid_url) as client:
                return client.process_pdf_with_result(pdf_path, max_retries=0)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(process_one, pdf): pdf for pdf in pdf_files}
            for future in as_completed(futures):
                result = future.result()
                if result.success:
                    successes.append(result)
                else:
                    errors.append(result)

        assert len(successes) == 0
        assert len(errors) == 5
        assert project_with_pdfs.paper_count() == 0

    def test_connection_lost_mid_processing(self, project_with_pdfs, sample_tei_xml, httpx_mock):
        """Connection lost mid-processing stops remaining work."""
        # First 2 succeed, then connection error
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )
        # Simulate connection error
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))

        grobid_url = "http://localhost:8070"
        pdf_files = project_with_pdfs.list_pdfs()
        connection_error = threading.Event()
        successes = []
        errors = []
        results_lock = threading.Lock()

        def process_one(pdf_path):
            with GrobidClient(grobid_url) as client:
                return client.process_pdf_with_result(pdf_path, max_retries=0)

        # Process sequentially to ensure order
        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = {pool.submit(process_one, pdf): pdf for pdf in pdf_files}
            for future in as_completed(futures):
                if connection_error.is_set():
                    break

                result = future.result()
                if result.success:
                    with results_lock:
                        successes.append(result)
                elif isinstance(result.error, GrobidConnectionError):
                    connection_error.set()
                else:
                    with results_lock:
                        errors.append(result)

        # At least 2 should succeed before connection lost
        assert len(successes) >= 2

    def test_mixed_success_and_failure_rates(self, project_with_pdfs, sample_tei_xml, httpx_mock):
        """Various success/failure combinations are tracked correctly."""
        # Alternating success and failure
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content="Error",
            status_code=503,
        )
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content="Error",
            status_code=400,
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

        def process_one(pdf_path):
            with GrobidClient(grobid_url) as client:
                return client.process_pdf_with_result(pdf_path, max_retries=0)

        # Process sequentially to match response order
        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = {pool.submit(process_one, pdf): pdf for pdf in pdf_files}
            for future in as_completed(futures):
                result = future.result()
                if result.success:
                    project_with_pdfs.add_paper(result.paper)
                    successes.append(result)
                else:
                    errors.append(result)

        assert len(successes) == 3
        assert len(errors) == 2
        assert project_with_pdfs.paper_count() == 3

    def test_timeout_handling(self, project_with_pdfs, httpx_mock):
        """Request timeouts are handled as errors."""
        # All requests timeout
        for _ in range(5):
            httpx_mock.add_exception(httpx.ReadTimeout("Read timed out"))

        grobid_url = "http://localhost:8070"
        pdf_files = project_with_pdfs.list_pdfs()
        successes = []
        errors = []

        def process_one(pdf_path):
            with GrobidClient(grobid_url) as client:
                return client.process_pdf_with_result(pdf_path, max_retries=0)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(process_one, pdf): pdf for pdf in pdf_files}
            for future in as_completed(futures):
                result = future.result()
                if result.success:
                    successes.append(result)
                else:
                    errors.append(result)

        assert len(successes) == 0
        assert len(errors) == 5
        # All should be connection errors
        assert all(isinstance(e.error, GrobidConnectionError) for e in errors)

    def test_retry_success_tracked(self, project_with_pdfs, sample_tei_xml, httpx_mock):
        """Retries that eventually succeed are tracked."""
        # First request fails, retry succeeds (for each PDF)
        for _ in range(5):
            httpx_mock.add_response(
                url="http://localhost:8070/api/processFulltextDocument",
                content="Error",
                status_code=503,
            )
            httpx_mock.add_response(
                url="http://localhost:8070/api/processFulltextDocument",
                content=sample_tei_xml,
                status_code=200,
            )

        grobid_url = "http://localhost:8070"
        pdf_files = project_with_pdfs.list_pdfs()
        successes = []
        retry_successes = []

        def process_one(pdf_path):
            with GrobidClient(grobid_url) as client:
                return client.process_pdf_with_result(pdf_path, max_retries=1)

        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = {pool.submit(process_one, pdf): pdf for pdf in pdf_files}
            for future in as_completed(futures):
                result = future.result()
                if result.success:
                    successes.append(result)
                    if result.retried:
                        retry_successes.append(result)

        assert len(successes) == 5
        assert len(retry_successes) == 5
        assert all(r.attempts == 2 for r in retry_successes)

    def test_empty_pdf_list(self, project_with_pdfs):
        """Empty PDF list completes without errors."""
        pdf_files = []
        successes = []
        errors = []

        def process_one(pdf_path):
            with GrobidClient("http://localhost:8070") as client:
                return client.process_pdf_with_result(pdf_path, max_retries=0)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(process_one, pdf): pdf for pdf in pdf_files}
            for future in as_completed(futures):
                result = future.result()
                if result.success:
                    successes.append(result)
                else:
                    errors.append(result)

        assert len(successes) == 0
        assert len(errors) == 0

    def test_single_pdf_uses_thread_pool_correctly(
        self, project_with_pdfs, sample_tei_xml, httpx_mock
    ):
        """Single PDF still works with thread pool."""
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )

        grobid_url = "http://localhost:8070"
        pdf_files = project_with_pdfs.list_pdfs()[:1]  # Just one PDF
        successes = []

        def process_one(pdf_path):
            with GrobidClient(grobid_url) as client:
                return client.process_pdf_with_result(pdf_path, max_retries=0)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(process_one, pdf): pdf for pdf in pdf_files}
            for future in as_completed(futures):
                result = future.result()
                if result.success:
                    successes.append(result)

        assert len(successes) == 1


class TestDatabaseConcurrency:
    """Test database behavior under concurrent writes."""

    @pytest.fixture
    def project(self, tmp_path):
        """Create a project."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "test.pdf").write_bytes(b"%PDF-1.4")

        return Project.create(
            root=tmp_path / "project",
            name="Test",
            research_question="Q?",
            source_pdfs=source_dir,
        )

    def test_concurrent_paper_adds(self, project):
        """Multiple concurrent paper additions don't corrupt database."""
        papers = [
            Paper(
                id=f"paper{i}",
                pdf_path=Path(f"paper{i}.pdf"),
                title=f"Paper {i}",
                authors=[Author(name=f"Author {i}")],
                year=2024,
            )
            for i in range(20)
        ]

        errors = []
        results_lock = threading.Lock()

        def add_paper(paper):
            try:
                project.add_paper(paper)
                return True
            except Exception as e:
                with results_lock:
                    errors.append(str(e))
                return False

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(add_paper, paper) for paper in papers]
            results = [f.result() for f in futures]

        assert all(results)
        assert len(errors) == 0
        assert project.paper_count() == 20

    def test_concurrent_paper_adds_different_ids(self, project):
        """Papers with different IDs can be added concurrently."""
        import uuid

        papers = [
            Paper(
                id=str(uuid.uuid4())[:8],
                pdf_path=Path(f"paper{i}.pdf"),
                title=f"Paper {i}",
                authors=[],
                year=2024,
            )
            for i in range(10)
        ]

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(project.add_paper, papers))

        assert project.paper_count() == 10

    def test_concurrent_view_and_cluster_operations(self, project):
        """Concurrent view and cluster operations work correctly."""
        # Add some papers first
        for i in range(5):
            project.add_paper(
                Paper(
                    id=f"p{i}",
                    pdf_path=Path(f"paper{i}.pdf"),
                    title=f"Paper {i}",
                    authors=[],
                    year=2024,
                )
            )

        def create_view_and_clusters(view_num):
            view = project.create_view(f"View {view_num}", f"Prompt {view_num}")
            clusters = [
                Cluster(
                    id=f"c{view_num}",
                    name=f"Cluster {view_num}",
                    description="",
                    paper_ids=["p0", "p1"],
                    subclusters=[],
                )
            ]
            project.save_clusters(view.id, clusters)
            return view.id

        with ThreadPoolExecutor(max_workers=3) as pool:
            view_ids = list(pool.map(create_view_and_clusters, range(5)))

        assert project.view_count() == 5
        for view_id in view_ids:
            assert project.cluster_count(view_id) == 1
