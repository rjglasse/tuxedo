"""Tests for the analysis module."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tuxedo.analysis import PaperAnalyzer, ANALYSIS_SYSTEM_PROMPT
from tuxedo.models import Author, Paper, PaperAnswer


@pytest.fixture
def sample_paper():
    """Create a sample paper for testing."""
    return Paper(
        id="p1",
        pdf_path=Path("paper1.pdf"),
        title="Deep Learning for Natural Language Processing",
        authors=[Author(name="Alice Smith")],
        abstract="This paper explores deep learning methods for NLP tasks.",
        year=2024,
        keywords=["deep learning", "NLP", "transformers"],
        sections={
            "Introduction": "We introduce a new approach...",
            "Methods": "Our methodology involves transformer architectures...",
            "Conclusion": "We demonstrated significant improvements...",
        },
    )


@pytest.fixture
def sample_paper_minimal():
    """Create a minimal paper without sections."""
    return Paper(
        id="p2",
        pdf_path=Path("paper2.pdf"),
        title="Minimal Paper",
        abstract=None,
    )


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response factory."""

    def _make_response(answer: str, confidence: str = "high", needs_more: bool = False):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "answer": answer,
            "confidence": confidence,
            "needs_more_context": needs_more,
        })
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        return mock_response

    return _make_response


class TestPaperAnalyzerInit:
    """Tests for PaperAnalyzer initialization."""

    def test_default_model(self):
        """Default model is gpt-4o-mini."""
        with patch("tuxedo.analysis.OpenAI"):
            analyzer = PaperAnalyzer(api_key="test-key")
            assert analyzer.model == "gpt-4o-mini"

    def test_custom_model(self):
        """Custom model can be specified."""
        with patch("tuxedo.analysis.OpenAI"):
            analyzer = PaperAnalyzer(api_key="test-key", model="gpt-4o")
            assert analyzer.model == "gpt-4o"

    def test_uses_openai_client(self):
        """OpenAI client is initialized."""
        with patch("tuxedo.analysis.OpenAI") as mock_openai:
            PaperAnalyzer(api_key="test-key")
            mock_openai.assert_called_once_with(api_key="test-key")


class TestGetSectionContent:
    """Tests for _get_section_content method."""

    def test_matches_exact_section_name(self, sample_paper):
        """Exact section name matches."""
        with patch("tuxedo.analysis.OpenAI"):
            analyzer = PaperAnalyzer()
            result = analyzer._get_section_content(sample_paper, ["Methods"])

        assert "Methods" in result
        assert "transformer architectures" in result["Methods"]

    def test_matches_case_insensitive(self, sample_paper):
        """Section matching is case insensitive."""
        with patch("tuxedo.analysis.OpenAI"):
            analyzer = PaperAnalyzer()
            result = analyzer._get_section_content(sample_paper, ["methods"])

        assert "Methods" in result

    def test_matches_substring(self, sample_paper):
        """Substring patterns match."""
        with patch("tuxedo.analysis.OpenAI"):
            analyzer = PaperAnalyzer()
            result = analyzer._get_section_content(sample_paper, ["conclu"])

        assert "Conclusion" in result

    def test_no_match_returns_empty(self, sample_paper):
        """Non-matching pattern returns empty dict."""
        with patch("tuxedo.analysis.OpenAI"):
            analyzer = PaperAnalyzer()
            result = analyzer._get_section_content(sample_paper, ["nonexistent"])

        assert result == {}

    def test_multiple_patterns(self, sample_paper):
        """Multiple patterns can match different sections."""
        with patch("tuxedo.analysis.OpenAI"):
            analyzer = PaperAnalyzer()
            result = analyzer._get_section_content(sample_paper, ["intro", "conclu"])

        assert "Introduction" in result
        assert "Conclusion" in result

    def test_truncates_long_content(self):
        """Long sections are truncated to 3000 chars."""
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title="Test",
            sections={"Methods": "x" * 5000},
        )
        with patch("tuxedo.analysis.OpenAI"):
            analyzer = PaperAnalyzer()
            result = analyzer._get_section_content(paper, ["method"])

        assert len(result["Methods"]) == 3000


class TestBuildPrompt:
    """Tests for _build_prompt method."""

    def test_includes_question(self):
        """Prompt includes the question."""
        with patch("tuxedo.analysis.OpenAI"):
            analyzer = PaperAnalyzer()
            result = analyzer._build_prompt("What methods are used?", ["Title: Test"])

        assert "What methods are used?" in result

    def test_includes_all_content_parts(self):
        """All content parts are included."""
        with patch("tuxedo.analysis.OpenAI"):
            analyzer = PaperAnalyzer()
            parts = ["Title: Test Paper", "Abstract: This is abstract", "Keywords: ml, ai"]
            result = analyzer._build_prompt("Question?", parts)

        assert "Title: Test Paper" in result
        assert "Abstract: This is abstract" in result
        assert "Keywords: ml, ai" in result


class TestAnalyzePaper:
    """Tests for analyze_paper method."""

    def test_stage1_high_confidence_returns_immediately(
        self, sample_paper, mock_openai_response
    ):
        """High confidence stage 1 answer returns without further stages."""
        response = mock_openai_response(
            "Deep learning methods are used for NLP.",
            confidence="high",
            needs_more=False,
        )

        with patch("tuxedo.analysis.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            analyzer = PaperAnalyzer()
            result = analyzer.analyze_paper(sample_paper, "What methods?", "q1")

        # Only one API call (stage 1)
        assert mock_client.chat.completions.create.call_count == 1
        assert result.answer == "Deep learning methods are used for NLP."
        assert result.confidence == "high"
        assert "title" in result.sections_used
        assert "abstract" in result.sections_used

    def test_stage2_triggered_when_needs_more_context(
        self, sample_paper, mock_openai_response
    ):
        """Stage 2 is triggered when stage 1 needs more context."""
        # Stage 1: needs more context
        response1 = mock_openai_response("Unclear", confidence="low", needs_more=True)
        # Stage 2: confident answer
        response2 = mock_openai_response(
            "Transformers are used.", confidence="high", needs_more=False
        )

        with patch("tuxedo.analysis.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [response1, response2]
            mock_openai.return_value = mock_client

            analyzer = PaperAnalyzer()
            result = analyzer.analyze_paper(sample_paper, "What methods?", "q1")

        # Two API calls (stage 1 + stage 2)
        assert mock_client.chat.completions.create.call_count == 2
        assert result.answer == "Transformers are used."
        assert "conclusion" in result.sections_used

    def test_stage3_triggered_when_still_needs_context(
        self, sample_paper, mock_openai_response
    ):
        """Stage 3 is triggered when stage 2 still needs context."""
        responses = [
            mock_openai_response("Unclear", confidence="low", needs_more=True),
            mock_openai_response("Still unclear", confidence="low", needs_more=True),
            mock_openai_response("Method details found", confidence="high", needs_more=False),
        ]

        with patch("tuxedo.analysis.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = responses
            mock_openai.return_value = mock_client

            analyzer = PaperAnalyzer()
            result = analyzer.analyze_paper(sample_paper, "What methods?", "q1")

        # Three API calls (all stages)
        assert mock_client.chat.completions.create.call_count == 3
        assert result.answer == "Method details found"
        assert "methods" in result.sections_used

    def test_max_stages_limits_calls(self, sample_paper, mock_openai_response):
        """max_stages parameter limits the number of stages."""
        response = mock_openai_response("Unclear", confidence="low", needs_more=True)

        with patch("tuxedo.analysis.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            analyzer = PaperAnalyzer()
            result = analyzer.analyze_paper(sample_paper, "What?", "q1", max_stages=1)

        # Only one call despite needs_more_context
        assert mock_client.chat.completions.create.call_count == 1

    def test_paper_without_abstract(self, sample_paper_minimal, mock_openai_response):
        """Papers without abstract still work."""
        response = mock_openai_response("Answer", confidence="medium", needs_more=False)

        with patch("tuxedo.analysis.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            analyzer = PaperAnalyzer()
            result = analyzer.analyze_paper(sample_paper_minimal, "Question?", "q1")

        assert result.answer == "Answer"
        assert "title" in result.sections_used
        assert "abstract" not in result.sections_used

    def test_returns_paper_answer_with_correct_ids(
        self, sample_paper, mock_openai_response
    ):
        """PaperAnswer has correct question_id and paper_id."""
        response = mock_openai_response("Answer", confidence="high", needs_more=False)

        with patch("tuxedo.analysis.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            analyzer = PaperAnalyzer()
            result = analyzer.analyze_paper(sample_paper, "Question?", "question-123")

        assert result.question_id == "question-123"
        assert result.paper_id == "p1"
        assert result.id is not None  # Generated UUID

    def test_uses_correct_system_prompt(self, sample_paper, mock_openai_response):
        """API is called with the correct system prompt."""
        response = mock_openai_response("Answer", confidence="high", needs_more=False)

        with patch("tuxedo.analysis.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            analyzer = PaperAnalyzer()
            analyzer.analyze_paper(sample_paper, "Question?", "q1")

            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == ANALYSIS_SYSTEM_PROMPT


class TestAnalyzePapers:
    """Tests for analyze_papers method."""

    def test_empty_papers_returns_empty(self):
        """Empty paper list returns empty list."""
        with patch("tuxedo.analysis.OpenAI"):
            analyzer = PaperAnalyzer()
            result = analyzer.analyze_papers([], "Question?", "q1")

        assert result == []

    def test_analyzes_all_papers(self, sample_paper, mock_openai_response):
        """All papers are analyzed."""
        papers = [
            sample_paper,
            Paper(id="p2", pdf_path=Path("p2.pdf"), title="Paper 2"),
            Paper(id="p3", pdf_path=Path("p3.pdf"), title="Paper 3"),
        ]
        response = mock_openai_response("Answer", confidence="high", needs_more=False)

        with patch("tuxedo.analysis.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            analyzer = PaperAnalyzer()
            results = analyzer.analyze_papers(papers, "Question?", "q1")

        assert len(results) == 3
        assert all(isinstance(r, PaperAnswer) for r in results)

    def test_progress_callback_called(self, sample_paper, mock_openai_response):
        """Progress callback is called for each paper."""
        papers = [sample_paper, sample_paper]
        response = mock_openai_response("Answer", confidence="high", needs_more=False)
        progress_calls = []

        def callback(current, total, message):
            progress_calls.append((current, total, message))

        with patch("tuxedo.analysis.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            analyzer = PaperAnalyzer()
            analyzer.analyze_papers(papers, "Question?", "q1", progress_callback=callback)

        assert len(progress_calls) == 2
        assert progress_calls[0][0] == 1
        assert progress_calls[0][1] == 2
        assert progress_calls[1][0] == 2

    def test_handles_individual_paper_failure(self, sample_paper, mock_openai_response):
        """Failed papers get fallback answers, others continue."""
        papers = [
            sample_paper,
            Paper(id="p2", pdf_path=Path("p2.pdf"), title="Paper 2"),
        ]
        # First succeeds, second fails
        response = mock_openai_response("Answer", confidence="high", needs_more=False)

        with patch("tuxedo.analysis.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [
                response,
                Exception("API Error"),
            ]
            mock_openai.return_value = mock_client

            analyzer = PaperAnalyzer()
            results = analyzer.analyze_papers(papers, "Question?", "q1")

        assert len(results) == 2
        assert results[0].answer == "Answer"
        assert "Analysis failed" in results[1].answer
        assert results[1].confidence == "low"

    def test_all_answers_have_correct_question_id(
        self, sample_paper, mock_openai_response
    ):
        """All answers have the same question_id."""
        papers = [sample_paper, sample_paper]
        response = mock_openai_response("Answer", confidence="high", needs_more=False)

        with patch("tuxedo.analysis.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            analyzer = PaperAnalyzer()
            results = analyzer.analyze_papers(papers, "Question?", "shared-question-id")

        assert all(r.question_id == "shared-question-id" for r in results)


class TestCallApi:
    """Tests for _call_api method."""

    def test_uses_json_response_format(self, mock_openai_response):
        """API is called with JSON response format."""
        response = mock_openai_response("Answer", confidence="high", needs_more=False)

        with patch("tuxedo.analysis.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            analyzer = PaperAnalyzer()
            analyzer._call_api("system", "user")

            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["response_format"] == {"type": "json_object"}

    def test_uses_low_temperature(self, mock_openai_response):
        """API is called with low temperature for consistent results."""
        response = mock_openai_response("Answer", confidence="high", needs_more=False)

        with patch("tuxedo.analysis.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            analyzer = PaperAnalyzer()
            analyzer._call_api("system", "user")

            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["temperature"] == 0.3

    def test_propagates_api_errors(self):
        """API errors are propagated to caller."""
        with patch("tuxedo.analysis.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            analyzer = PaperAnalyzer()

            with pytest.raises(Exception, match="API Error"):
                analyzer._call_api("system", "user")
