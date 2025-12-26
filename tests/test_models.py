"""Tests for the models module."""

import pytest
from datetime import datetime
from pathlib import Path

from tuxedo.models import Author, Paper, Cluster, ClusterView


@pytest.fixture
def sample_paper():
    """Create a sample paper for testing."""
    return Paper(
        id="paper1",
        pdf_path=Path("test.pdf"),
        title="Deep Learning for Natural Language Processing",
        authors=[
            Author(name="John Smith", affiliation="MIT"),
            Author(name="Jane Doe", affiliation="Stanford"),
        ],
        abstract="A comprehensive survey of deep learning methods.",
        year=2024,
        doi="10.1234/test.2024",
        journal="Journal of AI Research",
        volume="15",
        number="3",
        pages="1--25",
    )


class TestAuthor:
    """Tests for Author model."""

    def test_author_with_affiliation(self):
        """Author can have affiliation."""
        author = Author(name="John Doe", affiliation="MIT")
        assert author.name == "John Doe"
        assert author.affiliation == "MIT"

    def test_author_without_affiliation(self):
        """Author affiliation is optional."""
        author = Author(name="Jane Doe")
        assert author.name == "Jane Doe"
        assert author.affiliation is None


class TestPaperDisplayTitle:
    """Tests for Paper.display_title property."""

    def test_short_title_unchanged(self):
        """Short titles are not truncated."""
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title="Short Title",
        )
        assert paper.display_title == "Short Title"

    def test_long_title_truncated(self):
        """Titles over 60 characters are truncated."""
        long_title = "A" * 100
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title=long_title,
        )
        assert len(paper.display_title) == 60
        assert paper.display_title.endswith("...")

    def test_exactly_60_chars_not_truncated(self):
        """Title of exactly 60 characters is not truncated."""
        title_60 = "A" * 60
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title=title_60,
        )
        assert paper.display_title == title_60
        assert "..." not in paper.display_title


class TestPaperBibtexType:
    """Tests for Paper.bibtex_type property."""

    def test_article_when_journal(self):
        """Papers with journal are articles."""
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title="Test",
            journal="Nature",
        )
        assert paper.bibtex_type == "article"

    def test_inproceedings_when_booktitle(self):
        """Papers with booktitle are inproceedings."""
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title="Test",
            booktitle="Proceedings of ICML 2024",
        )
        assert paper.bibtex_type == "inproceedings"

    def test_misc_when_arxiv(self):
        """Papers with only arxiv_id are misc."""
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title="Test",
            arxiv_id="2401.12345",
        )
        assert paper.bibtex_type == "misc"

    def test_misc_when_nothing(self):
        """Papers with no venue info are misc."""
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title="Test",
        )
        assert paper.bibtex_type == "misc"

    def test_journal_takes_precedence(self):
        """Journal takes precedence over booktitle."""
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title="Test",
            journal="Nature",
            booktitle="Some Conference",
        )
        assert paper.bibtex_type == "article"


class TestPaperCitationKey:
    """Tests for Paper.citation_key property."""

    def test_basic_citation_key(self, sample_paper):
        """Citation key uses surname + year + first significant word."""
        # "Smith" + "2024" + "deep"
        assert sample_paper.citation_key == "smith2024deep"

    def test_citation_key_skips_stopwords(self):
        """Citation key skips stopwords in title."""
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title="The Art of Programming",
            authors=[Author(name="John Doe")],
            year=2023,
        )
        # Skips "The", uses "art"
        assert paper.citation_key == "doe2023art"

    def test_citation_key_no_authors(self):
        """Citation key handles missing authors."""
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title="Some Paper",
            year=2024,
        )
        assert paper.citation_key == "unknown2024some"

    def test_citation_key_no_year(self):
        """Citation key handles missing year."""
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title="Some Paper",
            authors=[Author(name="Alice")],
        )
        assert paper.citation_key == "alicenodatesome"

    def test_citation_key_special_characters_removed(self):
        """Citation key removes special characters from surname."""
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title="Test Paper",
            authors=[Author(name="José García-López")],
            year=2024,
        )
        # Special chars removed: "garcalopez" (hyphens and accents removed)
        assert "2024" in paper.citation_key
        assert paper.citation_key.startswith("garc")

    def test_citation_key_all_stopwords_title(self):
        """Citation key handles title with all stopwords."""
        paper = Paper(
            id="p1",
            pdf_path=Path("test.pdf"),
            title="The A An",
            authors=[Author(name="Bob")],
            year=2024,
        )
        # Falls through to "untitled" since all words are stopwords
        assert paper.citation_key == "bob2024untitled"


class TestCluster:
    """Tests for Cluster model."""

    def test_cluster_creation(self):
        """Clusters can be created with required fields."""
        cluster = Cluster(
            id="c1",
            name="Machine Learning",
            description="Papers about ML",
            paper_ids=["p1", "p2"],
        )
        assert cluster.id == "c1"
        assert cluster.name == "Machine Learning"
        assert len(cluster.paper_ids) == 2

    def test_cluster_with_subclusters(self):
        """Clusters can have nested subclusters."""
        subcluster = Cluster(
            id="c2",
            name="Deep Learning",
            description="DL subset",
            paper_ids=["p3"],
        )
        cluster = Cluster(
            id="c1",
            name="Machine Learning",
            description="ML papers",
            paper_ids=["p1", "p2"],
            subclusters=[subcluster],
        )
        assert len(cluster.subclusters) == 1
        assert cluster.subclusters[0].name == "Deep Learning"


class TestClusterView:
    """Tests for ClusterView model."""

    def test_cluster_view_creation(self):
        """ClusterView can be created."""
        view = ClusterView(
            id="v1",
            name="By Method",
            prompt="Group papers by methodology",
        )
        assert view.id == "v1"
        assert view.name == "By Method"
        assert view.prompt == "Group papers by methodology"

    def test_cluster_view_default_timestamp(self):
        """ClusterView gets default created_at timestamp."""
        before = datetime.now()
        view = ClusterView(
            id="v1",
            name="Test",
            prompt="Test prompt",
        )
        after = datetime.now()

        assert before <= view.created_at <= after
