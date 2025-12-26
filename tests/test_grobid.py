"""Tests for the Grobid client."""

from pathlib import Path

import httpx
import pytest

from tuxedo.grobid import (
    GrobidClient,
    GrobidConnectionError,
    GrobidError,
    GrobidParsingError,
    GrobidProcessingError,
    ProcessingResult,
)


class TestGrobidErrors:
    """Tests for Grobid exception classes."""

    def test_grobid_error_is_exception(self):
        """GrobidError should be a base exception."""
        error = GrobidError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"

    def test_grobid_connection_error(self):
        """GrobidConnectionError should include URL and cause."""
        error = GrobidConnectionError("http://localhost:8070")
        assert "http://localhost:8070" in str(error)
        assert error.url == "http://localhost:8070"
        assert error.cause is None

    def test_grobid_connection_error_with_cause(self):
        """GrobidConnectionError should include cause in message."""
        cause = Exception("Connection refused")
        error = GrobidConnectionError("http://localhost:8070", cause=cause)
        assert "http://localhost:8070" in str(error)
        assert "Connection refused" in str(error)
        assert error.cause is cause

    def test_grobid_processing_error_with_status_code(self):
        """GrobidProcessingError should include status code."""
        error = GrobidProcessingError(
            Path("test.pdf"),
            status_code=500,
            response_body="Internal Server Error",
        )
        assert "test.pdf" in str(error)
        assert "500" in str(error)
        assert "Internal Server Error" in str(error)
        assert error.status_code == 500

    def test_grobid_processing_error_truncates_long_response(self):
        """GrobidProcessingError should truncate long response bodies."""
        long_body = "x" * 300
        error = GrobidProcessingError(
            Path("test.pdf"),
            status_code=500,
            response_body=long_body,
        )
        assert len(str(error)) < 350  # Should be truncated
        assert "..." in str(error)

    def test_grobid_processing_error_with_cause(self):
        """GrobidProcessingError should handle cause without status code."""
        cause = OSError("File not found")
        error = GrobidProcessingError(Path("test.pdf"), cause=cause)
        assert "test.pdf" in str(error)
        assert "File not found" in str(error)

    def test_grobid_parsing_error(self):
        """GrobidParsingError should include PDF path and cause."""
        cause = Exception("Invalid XML")
        error = GrobidParsingError(Path("test.pdf"), cause=cause)
        assert "test.pdf" in str(error)
        assert "Invalid XML" in str(error)
        assert error.pdf_path == Path("test.pdf")


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_success_result(self):
        """ProcessingResult with paper should be successful."""
        from tuxedo.models import Paper

        paper = Paper(id="abc123", pdf_path=Path("test.pdf"), title="Test Paper")
        result = ProcessingResult(pdf_path=Path("test.pdf"), paper=paper)
        assert result.success is True
        assert result.paper == paper
        assert result.error is None

    def test_error_result(self):
        """ProcessingResult with error should not be successful."""
        error = GrobidProcessingError(Path("test.pdf"), status_code=500)
        result = ProcessingResult(pdf_path=Path("test.pdf"), error=error)
        assert result.success is False
        assert result.paper is None
        assert result.error == error


class TestGrobidClientConnection:
    """Tests for GrobidClient connection methods."""

    def test_is_alive_success(self, httpx_mock):
        """is_alive should return True when Grobid responds with 200."""
        httpx_mock.add_response(url="http://localhost:8070/api/isalive", status_code=200)

        with GrobidClient("http://localhost:8070") as client:
            assert client.is_alive() is True

    def test_is_alive_failure(self, httpx_mock):
        """is_alive should return False when Grobid responds with non-200."""
        httpx_mock.add_response(url="http://localhost:8070/api/isalive", status_code=503)

        with GrobidClient("http://localhost:8070") as client:
            assert client.is_alive() is False

    def test_is_alive_connection_error(self, httpx_mock):
        """is_alive should return False when connection fails."""
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))

        with GrobidClient("http://localhost:8070") as client:
            assert client.is_alive() is False

    def test_check_connection_success(self, httpx_mock):
        """check_connection should not raise when Grobid is available."""
        httpx_mock.add_response(url="http://localhost:8070/api/isalive", status_code=200)

        with GrobidClient("http://localhost:8070") as client:
            client.check_connection()  # Should not raise

    def test_check_connection_failure_status(self, httpx_mock):
        """check_connection should raise GrobidConnectionError on non-200."""
        httpx_mock.add_response(url="http://localhost:8070/api/isalive", status_code=503)

        with GrobidClient("http://localhost:8070") as client:
            with pytest.raises(GrobidConnectionError) as exc_info:
                client.check_connection()
            assert "503" in str(exc_info.value)

    def test_check_connection_network_error(self, httpx_mock):
        """check_connection should raise GrobidConnectionError on network error."""
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))

        with GrobidClient("http://localhost:8070") as client:
            with pytest.raises(GrobidConnectionError) as exc_info:
                client.check_connection()
            assert "Connection refused" in str(exc_info.value)


class TestGrobidClientProcessPdf:
    """Tests for GrobidClient.process_pdf method."""

    def test_process_pdf_success(self, httpx_mock, tmp_pdf, sample_tei_xml):
        """process_pdf should return Paper on success."""
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )

        with GrobidClient("http://localhost:8070") as client:
            paper = client.process_pdf(tmp_pdf)

        assert paper.title == "Deep Learning for Natural Language Processing: A Survey"
        assert len(paper.authors) == 2
        assert paper.authors[0].name == "John Smith"
        assert paper.authors[0].affiliation == "MIT"
        assert paper.authors[1].name == "Jane Doe"
        assert paper.year == 2024
        assert paper.doi == "10.1234/example.2024.001"
        assert "deep learning" in paper.keywords
        assert paper.abstract is not None
        assert "comprehensive survey" in paper.abstract

    def test_process_pdf_http_error(self, httpx_mock, tmp_pdf):
        """process_pdf should raise GrobidProcessingError on HTTP error."""
        # Mock responses for initial attempt + 2 retries (3 total)
        for _ in range(3):
            httpx_mock.add_response(
                url="http://localhost:8070/api/processFulltextDocument",
                content="Internal Server Error",
                status_code=500,
            )

        with GrobidClient("http://localhost:8070") as client:
            with pytest.raises(GrobidProcessingError) as exc_info:
                client.process_pdf(tmp_pdf)
            assert exc_info.value.status_code == 500
            assert "Internal Server Error" in str(exc_info.value)

    def test_process_pdf_connection_error(self, httpx_mock, tmp_pdf):
        """process_pdf should raise GrobidConnectionError on network error."""
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))

        with GrobidClient("http://localhost:8070") as client:
            with pytest.raises(GrobidConnectionError):
                client.process_pdf(tmp_pdf)

    def test_process_pdf_invalid_xml(self, httpx_mock, tmp_pdf):
        """process_pdf should raise GrobidParsingError on invalid XML."""
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content="<invalid xml",
            status_code=200,
        )

        with GrobidClient("http://localhost:8070") as client:
            with pytest.raises(GrobidParsingError):
                client.process_pdf(tmp_pdf)

    def test_process_pdf_file_not_found(self):
        """process_pdf should raise GrobidProcessingError for missing file."""
        with GrobidClient("http://localhost:8070") as client:
            with pytest.raises(GrobidProcessingError) as exc_info:
                client.process_pdf(Path("/nonexistent/file.pdf"))
            assert exc_info.value.cause is not None

    def test_process_pdf_generates_stable_id(self, httpx_mock, tmp_pdf, sample_tei_xml):
        """process_pdf should generate the same ID for the same content."""
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )

        with GrobidClient("http://localhost:8070") as client:
            paper1 = client.process_pdf(tmp_pdf)

        # Reset mock and process again
        httpx_mock.reset()
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )

        with GrobidClient("http://localhost:8070") as client:
            paper2 = client.process_pdf(tmp_pdf)

        assert paper1.id == paper2.id


class TestGrobidClientTeiParsing:
    """Tests for TEI XML parsing."""

    def test_parse_minimal_tei(self, httpx_mock, tmp_pdf, minimal_tei_xml):
        """Should handle minimal TEI XML with just a title."""
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=minimal_tei_xml,
            status_code=200,
        )

        with GrobidClient("http://localhost:8070") as client:
            paper = client.process_pdf(tmp_pdf)

        assert paper.title == "A Short Paper"
        assert paper.authors == []
        assert paper.abstract is None
        assert paper.year is None
        assert paper.doi is None
        assert paper.keywords == []
        assert paper.sections == {}

    def test_parse_tei_no_title_uses_filename(self, httpx_mock, tmp_pdf, tei_xml_no_title):
        """Should fall back to filename when title is empty."""
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=tei_xml_no_title,
            status_code=200,
        )

        with GrobidClient("http://localhost:8070") as client:
            paper = client.process_pdf(tmp_pdf)

        assert paper.title == "test_paper"  # Filename without .pdf

    def test_parse_sections(self, httpx_mock, tmp_pdf, sample_tei_xml):
        """Should extract sections with their content."""
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )

        with GrobidClient("http://localhost:8070") as client:
            paper = client.process_pdf(tmp_pdf)

        assert "introduction" in paper.sections
        assert "methods" in paper.sections
        assert "conclusion" in paper.sections
        assert "transformed by deep learning" in paper.sections["introduction"]

    def test_parse_abstract_multiple_paragraphs(self, httpx_mock, tmp_pdf, sample_tei_xml):
        """Should concatenate multiple abstract paragraphs."""
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )

        with GrobidClient("http://localhost:8070") as client:
            paper = client.process_pdf(tmp_pdf)

        assert "comprehensive survey" in paper.abstract
        assert "transformer architectures" in paper.abstract


class TestGrobidClientProcessDirectory:
    """Tests for GrobidClient.process_directory method."""

    def test_process_directory_all_success(self, httpx_mock, tmp_path, sample_tei_xml):
        """process_directory should return successful results for all PDFs."""
        # Create multiple PDFs
        for name in ["paper1.pdf", "paper2.pdf", "paper3.pdf"]:
            (tmp_path / name).write_bytes(b"%PDF-1.4 content " + name.encode())

        # Mock responses for each request
        for _ in range(3):
            httpx_mock.add_response(
                url="http://localhost:8070/api/processFulltextDocument",
                content=sample_tei_xml,
                status_code=200,
            )

        with GrobidClient("http://localhost:8070") as client:
            results = client.process_directory(tmp_path)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert all(r.paper is not None for r in results)

    def test_process_directory_mixed_results(self, httpx_mock, tmp_path, sample_tei_xml):
        """process_directory should handle mixed success/failure."""
        # Create PDFs
        (tmp_path / "good.pdf").write_bytes(b"%PDF-1.4 good content")
        (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 bad content")

        # First call succeeds, second fails (plus retries)
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )
        # Add 500 responses for all retry attempts (initial + 2 retries)
        for _ in range(3):
            httpx_mock.add_response(
                url="http://localhost:8070/api/processFulltextDocument",
                content="Error",
                status_code=500,
            )

        with GrobidClient("http://localhost:8070") as client:
            results = client.process_directory(tmp_path)

        assert len(results) == 2
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        assert len(successes) == 1
        assert len(failures) == 1
        assert failures[0].error is not None

    def test_process_directory_empty(self, tmp_path):
        """process_directory should return empty list for empty directory."""
        with GrobidClient("http://localhost:8070") as client:
            results = client.process_directory(tmp_path)

        assert results == []

    def test_process_directory_papers_convenience_method(
        self, httpx_mock, tmp_path, sample_tei_xml
    ):
        """process_directory_papers should return only successful papers."""
        # Create PDFs
        (tmp_path / "good.pdf").write_bytes(b"%PDF-1.4 good content")
        (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 bad content")

        # First succeeds, second fails (plus retries)
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )
        # Add 500 responses for all retry attempts (initial + 2 retries)
        for _ in range(3):
            httpx_mock.add_response(
                url="http://localhost:8070/api/processFulltextDocument",
                content="Error",
                status_code=500,
            )

        with GrobidClient("http://localhost:8070") as client:
            papers = client.process_directory_papers(tmp_path)

        assert len(papers) == 1
        assert papers[0].title == "Deep Learning for Natural Language Processing: A Survey"


class TestGrobidClientContextManager:
    """Tests for GrobidClient context manager."""

    def test_context_manager_closes_client(self):
        """Context manager should close HTTP client on exit."""
        client = GrobidClient("http://localhost:8070")
        assert not client.client.is_closed

        with client:
            pass

        assert client.client.is_closed

    def test_url_trailing_slash_stripped(self):
        """URL trailing slash should be stripped."""
        client = GrobidClient("http://localhost:8070/")
        assert client.base_url == "http://localhost:8070"


class TestGrobidClientRetry:
    """Tests for GrobidClient retry logic."""

    def test_retry_success_on_second_attempt(self, httpx_mock, tmp_pdf, sample_tei_xml):
        """process_pdf should succeed after retry."""
        # First attempt fails, second succeeds
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content="Server Error",
            status_code=500,
        )
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )

        with GrobidClient("http://localhost:8070") as client:
            paper = client.process_pdf(tmp_pdf, max_retries=2)

        assert paper.title == "Deep Learning for Natural Language Processing: A Survey"

    def test_no_retry_on_client_error(self, httpx_mock, tmp_pdf):
        """process_pdf should not retry on 4xx errors (except 400)."""
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content="Not Found",
            status_code=404,
        )

        with GrobidClient("http://localhost:8070") as client:
            with pytest.raises(GrobidProcessingError) as exc_info:
                client.process_pdf(tmp_pdf, max_retries=2)
            assert exc_info.value.status_code == 404

    def test_retry_on_400_error(self, httpx_mock, tmp_pdf, sample_tei_xml):
        """process_pdf should retry on 400 errors (possible parameter issue)."""
        # First attempt fails with 400, retry with different params succeeds
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content="Bad Request",
            status_code=400,
        )
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )

        with GrobidClient("http://localhost:8070") as client:
            paper = client.process_pdf(tmp_pdf, max_retries=2)

        assert paper.title == "Deep Learning for Natural Language Processing: A Survey"

    def test_max_retries_zero_no_retry(self, httpx_mock, tmp_pdf):
        """process_pdf should not retry when max_retries is 0."""
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content="Server Error",
            status_code=500,
        )

        with GrobidClient("http://localhost:8070") as client:
            with pytest.raises(GrobidProcessingError) as exc_info:
                client.process_pdf(tmp_pdf, max_retries=0)
            assert exc_info.value.status_code == 500

    def test_process_pdf_with_result_tracks_retry(self, httpx_mock, tmp_pdf, sample_tei_xml):
        """process_pdf_with_result should track retry attempts."""
        # First attempt fails, second succeeds
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content="Server Error",
            status_code=500,
        )
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )

        with GrobidClient("http://localhost:8070") as client:
            result = client.process_pdf_with_result(tmp_pdf, max_retries=2)

        assert result.success
        assert result.attempts == 2
        assert result.retried is True

    def test_process_pdf_with_result_first_attempt_success(self, httpx_mock, tmp_pdf, sample_tei_xml):
        """process_pdf_with_result should show no retry when first attempt succeeds."""
        httpx_mock.add_response(
            url="http://localhost:8070/api/processFulltextDocument",
            content=sample_tei_xml,
            status_code=200,
        )

        with GrobidClient("http://localhost:8070") as client:
            result = client.process_pdf_with_result(tmp_pdf, max_retries=2)

        assert result.success
        assert result.attempts == 1
        assert result.retried is False
