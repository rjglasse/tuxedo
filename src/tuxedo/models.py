"""Data models for Tuxedo."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class Author(BaseModel):
    """Paper author."""

    name: str
    affiliation: str | None = None


class Paper(BaseModel):
    """Extracted paper metadata and content."""

    id: str
    pdf_path: Path
    title: str
    authors: list[Author] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = None
    doi: str | None = None
    sections: dict[str, str] = Field(default_factory=dict)  # section_name -> content
    keywords: list[str] = Field(default_factory=list)

    # Additional bibliographic fields for BibTeX
    journal: str | None = None  # Journal name for articles
    booktitle: str | None = None  # Conference/proceedings name
    publisher: str | None = None  # Publisher name
    volume: str | None = None  # Journal volume
    number: str | None = None  # Journal issue number
    pages: str | None = None  # Page range (e.g., "1-15")
    arxiv_id: str | None = None  # arXiv identifier
    url: str | None = None  # URL if available

    # Relevance score against research question (0-100)
    relevance_score: int | None = None

    @property
    def display_title(self) -> str:
        """Short title for display."""
        if len(self.title) > 60:
            return self.title[:57] + "..."
        return self.title

    @property
    def bibtex_type(self) -> str:
        """Determine the appropriate BibTeX entry type."""
        if self.journal:
            return "article"
        elif self.booktitle:
            return "inproceedings"
        elif self.arxiv_id:
            return "misc"
        else:
            return "misc"

    @property
    def citation_key(self) -> str:
        """Generate a citation key (author + year + first significant word)."""
        import re

        # Get first author's last name
        if self.authors:
            name = self.authors[0].name
            # Take last word as surname (simple heuristic)
            parts = name.split()
            surname = parts[-1] if parts else "unknown"
            surname = re.sub(r"[^a-zA-Z]", "", surname).lower()
        else:
            surname = "unknown"

        # Get year
        year = str(self.year) if self.year else "nodate"

        # Get first significant word from title
        stopwords = {"a", "an", "the", "on", "in", "of", "for", "and", "to", "with"}
        words = re.findall(r"[a-zA-Z]+", self.title.lower())
        first_word = "untitled"
        for word in words:
            if word not in stopwords:
                first_word = word
                break

        return f"{surname}{year}{first_word}"


class Cluster(BaseModel):
    """A thematic cluster of papers."""

    id: str
    name: str
    description: str
    paper_ids: list[str] = Field(default_factory=list)
    subclusters: list["Cluster"] = Field(default_factory=list)


class ClusterView(BaseModel):
    """A named clustering view with its prompt."""

    id: str
    name: str
    prompt: str
    created_at: datetime = Field(default_factory=datetime.now)


class Question(BaseModel):
    """A question to be analyzed across papers."""

    id: str
    text: str
    created_at: datetime = Field(default_factory=datetime.now)


class PaperAnswer(BaseModel):
    """An answer to a question for a specific paper."""

    id: str
    question_id: str
    paper_id: str
    answer: str
    sections_used: list[str] = Field(default_factory=list)
    confidence: str | None = None  # "high", "medium", "low"
    created_at: datetime = Field(default_factory=datetime.now)
