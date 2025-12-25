"""Tests for the clustering module."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tuxedo.clustering import PaperClusterer, CLUSTER_SYSTEM_PROMPT, BATCH_CLUSTER_PROMPT
from tuxedo.models import Author, Cluster, Paper


@pytest.fixture
def sample_papers():
    """Create sample papers for testing."""
    return [
        Paper(
            id="p1",
            pdf_path=Path("paper1.pdf"),
            title="Deep Learning for NLP",
            authors=[Author(name="Alice Smith")],
            abstract="A paper about deep learning applications in NLP.",
            year=2024,
            keywords=["deep learning", "NLP"],
            sections={"Methods": "We used transformers..."},
        ),
        Paper(
            id="p2",
            pdf_path=Path("paper2.pdf"),
            title="Reinforcement Learning Survey",
            authors=[Author(name="Bob Jones")],
            abstract="A comprehensive survey of RL methods.",
            year=2023,
            keywords=["reinforcement learning"],
        ),
        Paper(
            id="p3",
            pdf_path=Path("paper3.pdf"),
            title="Computer Vision with CNNs",
            authors=[Author(name="Carol White")],
            abstract="CNN architectures for image classification.",
            year=2024,
            keywords=["computer vision", "CNN"],
            sections={"Methods": "We used ResNet..."},
        ),
    ]


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response."""
    def _make_response(clusters_json):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(clusters_json)
        return mock_response
    return _make_response


class TestPaperClustererInit:
    """Tests for PaperClusterer initialization."""

    def test_default_model(self):
        """Default model is gpt-5.2."""
        with patch("tuxedo.clustering.OpenAI"):
            clusterer = PaperClusterer(api_key="test-key")
            assert clusterer.model == "gpt-5.2"

    def test_custom_model(self):
        """Custom model can be specified."""
        with patch("tuxedo.clustering.OpenAI"):
            clusterer = PaperClusterer(api_key="test-key", model="gpt-4o")
            assert clusterer.model == "gpt-4o"

    def test_uses_openai_client(self):
        """OpenAI client is initialized."""
        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            PaperClusterer(api_key="test-key")
            mock_openai.assert_called_once_with(api_key="test-key")


class TestBuildPaperSummaries:
    """Tests for _build_paper_summaries method."""

    def test_basic_summary(self, sample_papers):
        """Basic paper summaries include required fields."""
        with patch("tuxedo.clustering.OpenAI"):
            clusterer = PaperClusterer()
            summaries = clusterer._build_paper_summaries(sample_papers[:1])

        assert len(summaries) == 1
        s = summaries[0]
        assert s["id"] == "p1"
        assert s["title"] == "Deep Learning for NLP"
        assert s["year"] == 2024
        assert s["abstract"] == "A paper about deep learning applications in NLP."
        assert s["keywords"] == ["deep learning", "NLP"]

    def test_summary_without_abstract(self):
        """Papers without abstract get placeholder."""
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title="Test",
            abstract=None,
        )
        with patch("tuxedo.clustering.OpenAI"):
            clusterer = PaperClusterer()
            summaries = clusterer._build_paper_summaries([paper])

        assert summaries[0]["abstract"] == "No abstract available"

    def test_summary_limits_keywords(self):
        """Keywords are limited to 5."""
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title="Test",
            keywords=["a", "b", "c", "d", "e", "f", "g"],
        )
        with patch("tuxedo.clustering.OpenAI"):
            clusterer = PaperClusterer()
            summaries = clusterer._build_paper_summaries([paper])

        assert len(summaries[0]["keywords"]) == 5

    def test_include_sections_matching(self, sample_papers):
        """Sections matching patterns are included."""
        with patch("tuxedo.clustering.OpenAI"):
            clusterer = PaperClusterer()
            summaries = clusterer._build_paper_summaries(
                sample_papers[:1],
                include_sections=["method"]
            )

        assert "sections" in summaries[0]
        assert "Methods" in summaries[0]["sections"]

    def test_include_sections_case_insensitive(self, sample_papers):
        """Section matching is case insensitive."""
        with patch("tuxedo.clustering.OpenAI"):
            clusterer = PaperClusterer()
            summaries = clusterer._build_paper_summaries(
                sample_papers[:1],
                include_sections=["METHODS"]
            )

        assert "sections" in summaries[0]

    def test_include_sections_no_match(self, sample_papers):
        """Papers without matching sections don't include sections field."""
        with patch("tuxedo.clustering.OpenAI"):
            clusterer = PaperClusterer()
            # paper2 has no sections
            summaries = clusterer._build_paper_summaries(
                [sample_papers[1]],
                include_sections=["method"]
            )

        assert "sections" not in summaries[0]

    def test_section_content_truncated(self):
        """Long section content is truncated."""
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title="Test",
            sections={"Methods": "x" * 5000},
        )
        with patch("tuxedo.clustering.OpenAI"):
            clusterer = PaperClusterer()
            summaries = clusterer._build_paper_summaries(
                [paper],
                include_sections=["method"]
            )

        assert len(summaries[0]["sections"]["Methods"]) == 2000


class TestClusterPapers:
    """Tests for cluster_papers method."""

    def test_empty_papers_returns_empty(self):
        """Empty paper list returns empty clusters."""
        with patch("tuxedo.clustering.OpenAI"):
            clusterer = PaperClusterer()
            result = clusterer.cluster_papers([], "research question")

        assert result == []

    def test_calls_openai_with_correct_prompt(self, sample_papers, mock_openai_response):
        """OpenAI is called with correct system prompt and paper data."""
        response = mock_openai_response({"clusters": []})

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            clusterer.cluster_papers(sample_papers[:1], "What are ML trends?")

            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]

            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == CLUSTER_SYSTEM_PROMPT
            assert messages[1]["role"] == "user"
            assert "What are ML trends?" in messages[1]["content"]
            assert "Deep Learning for NLP" in messages[1]["content"]

    def test_parses_cluster_response(self, sample_papers, mock_openai_response):
        """Cluster response is correctly parsed."""
        response_data = {
            "clusters": [
                {
                    "name": "Deep Learning",
                    "description": "Papers about DL",
                    "paper_ids": ["p1"],
                    "subclusters": [
                        {
                            "name": "NLP Applications",
                            "description": "DL for NLP",
                            "paper_ids": ["p2"],
                        }
                    ],
                },
                {
                    "name": "Computer Vision",
                    "description": "CV papers",
                    "paper_ids": ["p3"],
                    "subclusters": [],
                },
            ]
        }
        response = mock_openai_response(response_data)

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            clusters = clusterer.cluster_papers(sample_papers, "question")

        assert len(clusters) == 2
        assert clusters[0].name == "Deep Learning"
        assert clusters[0].description == "Papers about DL"
        assert clusters[0].paper_ids == ["p1"]
        assert len(clusters[0].subclusters) == 1
        assert clusters[0].subclusters[0].name == "NLP Applications"
        assert clusters[1].name == "Computer Vision"

    def test_uses_json_response_format(self, sample_papers, mock_openai_response):
        """Request uses JSON response format."""
        response = mock_openai_response({"clusters": []})

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            clusterer.cluster_papers(sample_papers[:1], "question")

            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["response_format"] == {"type": "json_object"}


class TestParseCluster:
    """Tests for _parse_clusters method."""

    def test_parse_empty_list(self):
        """Empty cluster list returns empty."""
        with patch("tuxedo.clustering.OpenAI"):
            clusterer = PaperClusterer()
            result = clusterer._parse_clusters([])

        assert result == []

    def test_parse_cluster_generates_ids(self):
        """Cluster IDs are generated."""
        raw = [{"name": "Test", "description": "Desc", "paper_ids": ["p1"]}]

        with patch("tuxedo.clustering.OpenAI"):
            clusterer = PaperClusterer()
            result = clusterer._parse_clusters(raw)

        assert result[0].id is not None
        assert len(result[0].id) == 8

    def test_parse_cluster_handles_missing_fields(self):
        """Missing fields get defaults."""
        raw = [{}]

        with patch("tuxedo.clustering.OpenAI"):
            clusterer = PaperClusterer()
            result = clusterer._parse_clusters(raw)

        assert result[0].name == "Unnamed"
        assert result[0].description == ""
        assert result[0].paper_ids == []
        assert result[0].subclusters == []


class TestBatchClustering:
    """Tests for batch/iterative clustering."""

    def test_batch_not_triggered_below_threshold(self, sample_papers, mock_openai_response):
        """Batch mode not used when papers <= batch_size."""
        response = mock_openai_response({"clusters": []})

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            clusterer.cluster_papers(sample_papers, "question", batch_size=5)

            # Should only be called once (not in batch mode)
            assert mock_client.chat.completions.create.call_count == 1

    def test_batch_triggered_above_threshold(self, sample_papers, mock_openai_response):
        """Batch mode used when papers > batch_size."""
        response = mock_openai_response({
            "clusters": [{"name": "Theme", "description": "Desc", "paper_ids": ["p1"]}]
        })

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            clusterer.cluster_papers(sample_papers, "question", batch_size=1)

            # Should be called 3 times (one per paper)
            assert mock_client.chat.completions.create.call_count == 3

    def test_batch_first_uses_standard_prompt(self, sample_papers, mock_openai_response):
        """First batch uses standard clustering prompt."""
        response = mock_openai_response({
            "clusters": [{"name": "Theme", "description": "Desc", "paper_ids": ["p1"]}]
        })

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            clusterer.cluster_papers(sample_papers[:2], "question", batch_size=1)

            first_call = mock_client.chat.completions.create.call_args_list[0]
            first_system = first_call.kwargs["messages"][0]["content"]
            assert first_system == CLUSTER_SYSTEM_PROMPT

    def test_batch_subsequent_uses_batch_prompt(self, sample_papers, mock_openai_response):
        """Subsequent batches use batch clustering prompt."""
        response = mock_openai_response({
            "clusters": [{"name": "Theme", "description": "Desc", "paper_ids": ["p1"]}]
        })

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            clusterer.cluster_papers(sample_papers[:2], "question", batch_size=1)

            second_call = mock_client.chat.completions.create.call_args_list[1]
            second_system = second_call.kwargs["messages"][0]["content"]
            assert second_system == BATCH_CLUSTER_PROMPT

    def test_batch_subsequent_includes_existing_themes(self, sample_papers, mock_openai_response):
        """Subsequent batches include existing theme information."""
        response = mock_openai_response({
            "clusters": [{"name": "ML Theme", "description": "Machine learning papers", "paper_ids": ["p1"]}]
        })

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            clusterer.cluster_papers(sample_papers[:2], "question", batch_size=1)

            second_call = mock_client.chat.completions.create.call_args_list[1]
            second_user = second_call.kwargs["messages"][1]["content"]
            assert "ML Theme" in second_user
            assert "Machine learning papers" in second_user

    def test_batch_progress_callback(self, sample_papers, mock_openai_response):
        """Progress callback is called for each batch."""
        response = mock_openai_response({
            "clusters": [{"name": "Theme", "description": "Desc", "paper_ids": ["p1"]}]
        })

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            progress_calls = []

            def callback(batch_num, total, message):
                progress_calls.append((batch_num, total, message))

            clusterer = PaperClusterer()
            clusterer.cluster_papers(
                sample_papers,
                "question",
                batch_size=1,
                progress_callback=callback
            )

        assert len(progress_calls) == 3
        assert progress_calls[0][0] == 1  # First batch
        assert progress_calls[0][1] == 3  # Total batches
        assert progress_calls[2][0] == 3  # Last batch

    def test_batch_merges_results(self, sample_papers, mock_openai_response):
        """Batch results are merged correctly."""
        # Different responses for each batch
        responses = [
            mock_openai_response({
                "clusters": [{"name": "Theme A", "description": "First", "paper_ids": ["p1"]}]
            }),
            mock_openai_response({
                "clusters": [{"name": "Theme A", "description": "First", "paper_ids": ["p2"]}]
            }),
            mock_openai_response({
                "clusters": [
                    {"name": "Theme A", "description": "First", "paper_ids": []},
                    {"name": "Theme B", "description": "New theme", "paper_ids": ["p3"]},
                ]
            }),
        ]

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = responses
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            clusters = clusterer.cluster_papers(sample_papers, "question", batch_size=1)

        # Should have 2 themes: Theme A with p1,p2 and Theme B with p3
        assert len(clusters) == 2
        theme_a = next(c for c in clusters if c.name == "Theme A")
        theme_b = next(c for c in clusters if c.name == "Theme B")
        assert set(theme_a.paper_ids) == {"p1", "p2"}
        assert theme_b.paper_ids == ["p3"]


class TestRecluster:
    """Tests for recluster method."""

    def test_recluster_includes_feedback(self, sample_papers, mock_openai_response):
        """Recluster includes user feedback in prompt."""
        response = mock_openai_response({"clusters": []})

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            current = [Cluster(id="c1", name="Old", description="", paper_ids=["p1"], subclusters=[])]
            clusterer.recluster(
                sample_papers[:1],
                "question",
                "Please split into more categories",
                current
            )

            call_args = mock_client.chat.completions.create.call_args
            user_content = call_args.kwargs["messages"][1]["content"]
            assert "Please split into more categories" in user_content

    def test_recluster_includes_current_structure(self, sample_papers, mock_openai_response):
        """Recluster includes current cluster structure."""
        response = mock_openai_response({"clusters": []})

        with patch("tuxedo.clustering.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            clusterer = PaperClusterer()
            current = [
                Cluster(
                    id="c1",
                    name="Existing Theme",
                    description="Theme desc",
                    paper_ids=["p1"],
                    subclusters=[]
                )
            ]
            clusterer.recluster(sample_papers[:1], "question", "feedback", current)

            call_args = mock_client.chat.completions.create.call_args
            user_content = call_args.kwargs["messages"][1]["content"]
            assert "Existing Theme" in user_content
            assert "Theme desc" in user_content

    def test_recluster_empty_papers(self):
        """Recluster with empty papers returns empty."""
        with patch("tuxedo.clustering.OpenAI"):
            clusterer = PaperClusterer()
            result = clusterer.recluster([], "question", "feedback", [])

        assert result == []


class TestClustersToDict:
    """Tests for _clusters_to_dict method."""

    def test_converts_flat_clusters(self):
        """Flat clusters are converted to dicts."""
        clusters = [
            Cluster(id="c1", name="A", description="Desc A", paper_ids=["p1"], subclusters=[]),
            Cluster(id="c2", name="B", description="Desc B", paper_ids=["p2"], subclusters=[]),
        ]

        with patch("tuxedo.clustering.OpenAI"):
            clusterer = PaperClusterer()
            result = clusterer._clusters_to_dict(clusters)

        assert len(result) == 2
        assert result[0]["name"] == "A"
        assert result[0]["description"] == "Desc A"
        assert result[0]["paper_ids"] == ["p1"]
        assert result[0]["subclusters"] == []

    def test_converts_nested_clusters(self):
        """Nested clusters are converted recursively."""
        clusters = [
            Cluster(
                id="c1",
                name="Parent",
                description="Parent desc",
                paper_ids=["p1"],
                subclusters=[
                    Cluster(id="c1a", name="Child", description="Child desc", paper_ids=["p2"], subclusters=[])
                ]
            ),
        ]

        with patch("tuxedo.clustering.OpenAI"):
            clusterer = PaperClusterer()
            result = clusterer._clusters_to_dict(clusters)

        assert len(result[0]["subclusters"]) == 1
        assert result[0]["subclusters"][0]["name"] == "Child"
