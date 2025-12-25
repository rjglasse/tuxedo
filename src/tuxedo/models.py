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

    @property
    def display_title(self) -> str:
        """Short title for display."""
        if len(self.title) > 60:
            return self.title[:57] + "..."
        return self.title


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
