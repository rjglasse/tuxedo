"""Tuxedo - A library for organizing systematic literature review papers using LLMs.

This package provides both a programmatic API and command-line tools for:
- Processing PDF papers using Grobid
- Clustering papers using LLM-based analysis
- Analyzing papers by asking questions
- Managing literature review projects

Basic Usage:
    >>> from tuxedo import Project, PaperClusterer, PaperAnalyzer
    >>> 
    >>> # Load an existing project
    >>> project = Project.load()
    >>> 
    >>> # Cluster papers
    >>> clusterer = PaperClusterer(model="gpt-4o")
    >>> papers = project.get_papers()
    >>> clusters, scores = clusterer.cluster_papers(papers, "What are ML trends?")
    >>> 
    >>> # Analyze papers
    >>> analyzer = PaperAnalyzer(model="gpt-4o-mini")
    >>> answers = analyzer.analyze_papers(papers, "What methodology was used?")

API Classes:
    Project: Manage a literature review project
    PaperClusterer: Cluster papers by themes
    PaperAnalyzer: Answer questions about papers
    GrobidClient: Extract metadata from PDFs
    Database: Store and query paper data
    
Models:
    Paper: Paper metadata and content
    Cluster: Hierarchical paper grouping
    ClusterView: A clustering perspective
    Question: Analysis question
    PaperAnswer: Answer to a question about a paper
    Author: Paper author information
"""

__version__ = "0.1.0"

# Import main classes for public API
from tuxedo.project import Project, ProjectConfig
from tuxedo.clustering import PaperClusterer
from tuxedo.analysis import PaperAnalyzer
from tuxedo.grobid import GrobidClient
from tuxedo.database import Database
from tuxedo.models import (
    Paper,
    Cluster,
    ClusterView,
    Question,
    PaperAnswer,
    Author,
)

__all__ = [
    # Project management
    "Project",
    "ProjectConfig",
    # Analysis and clustering
    "PaperClusterer",
    "PaperAnalyzer",
    # Data processing
    "GrobidClient",
    "Database",
    # Data models
    "Paper",
    "Cluster",
    "ClusterView",
    "Question",
    "PaperAnswer",
    "Author",
]
