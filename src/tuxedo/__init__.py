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


def register_commands(parent_app):
    """Register tuxedo commands with a parent Typer app.

    This function is called by the scholar CLI to add tuxedo as a subcommand.
    Usage: scholar tuxedo <command>

    Tuxedo uses Click internally, so we create a minimal Typer wrapper that
    invokes the Click group directly, preserving all Click functionality.

    Args:
        parent_app: The parent Typer application to register commands with.
    """
    import sys

    import typer

    from tuxedo.cli import main as tuxedo_click_group

    def run_tuxedo():
        """Run tuxedo with arguments from sys.argv."""
        # Find 'tuxedo' in argv and get everything after it
        try:
            tuxedo_idx = sys.argv.index("tuxedo")
            click_args = sys.argv[tuxedo_idx + 1 :]
        except ValueError:
            click_args = []

        # Default to --help if no args
        if not click_args:
            click_args = ["--help"]

        # Temporarily modify sys.argv so Click sees the right program name
        original_argv = sys.argv
        sys.argv = ["scholar tuxedo"] + click_args

        try:
            # Invoke the Click group - it will parse sys.argv
            tuxedo_click_group()
        except SystemExit:
            pass  # Click uses SystemExit for normal termination
        finally:
            sys.argv = original_argv

    # Register a command (not a sub-app) that delegates to Click
    @parent_app.command(
        name="tuxedo",
        help="Organize literature review papers with LLMs.",
        context_settings={
            "allow_extra_args": True,
            "ignore_unknown_options": True,
            "allow_interspersed_args": False,
        },
        add_help_option=False,
    )
    def tuxedo_command(ctx: typer.Context):
        """Tuxedo - Organize literature review papers with LLMs.

        Run 'scholar tuxedo --help' for available commands.
        """
        run_tuxedo()
