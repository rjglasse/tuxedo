"""Test that the API is properly exposed through __init__.py"""

import pytest


def test_api_imports():
    """Test that all public API classes can be imported."""
    from tuxedo import (
        Project,
        ProjectConfig,
        PaperClusterer,
        PaperAnalyzer,
        GrobidClient,
        Database,
        Paper,
        Cluster,
        ClusterView,
        Question,
        PaperAnswer,
        Author,
    )
    
    # Verify classes are imported correctly
    assert Project is not None
    assert ProjectConfig is not None
    assert PaperClusterer is not None
    assert PaperAnalyzer is not None
    assert GrobidClient is not None
    assert Database is not None
    assert Paper is not None
    assert Cluster is not None
    assert ClusterView is not None
    assert Question is not None
    assert PaperAnswer is not None
    assert Author is not None


def test_api_all_exports():
    """Test that __all__ contains all expected exports."""
    import tuxedo
    
    expected_exports = [
        "Project",
        "ProjectConfig",
        "PaperClusterer",
        "PaperAnalyzer",
        "GrobidClient",
        "Database",
        "Paper",
        "Cluster",
        "ClusterView",
        "Question",
        "PaperAnswer",
        "Author",
    ]
    
    for name in expected_exports:
        assert name in tuxedo.__all__, f"{name} not in __all__"
        assert hasattr(tuxedo, name), f"{name} not accessible as tuxedo.{name}"


def test_project_api_methods():
    """Test that Project class has expected API methods."""
    from tuxedo import Project
    
    # Check static/class methods
    assert hasattr(Project, "create")
    assert hasattr(Project, "load")
    
    # Check instance methods (these would be tested on instances)
    assert callable(getattr(Project, "create", None))
    assert callable(getattr(Project, "load", None))


def test_clusterer_api():
    """Test that PaperClusterer has expected methods."""
    from tuxedo import PaperClusterer
    
    # Can be instantiated with an API key
    clusterer = PaperClusterer(api_key="test-key", model="gpt-4o-mini")
    
    # Has expected methods
    assert hasattr(clusterer, "cluster_papers")
    assert hasattr(clusterer, "recluster")
    assert callable(clusterer.cluster_papers)
    assert callable(clusterer.recluster)


def test_analyzer_api():
    """Test that PaperAnalyzer has expected methods."""
    from tuxedo import PaperAnalyzer
    
    # Can be instantiated with an API key
    analyzer = PaperAnalyzer(api_key="test-key", model="gpt-4o-mini")
    
    # Has expected methods
    assert hasattr(analyzer, "analyze_papers")
    assert hasattr(analyzer, "analyze_paper")
    assert callable(analyzer.analyze_papers)
    assert callable(analyzer.analyze_paper)


def test_grobid_client_api():
    """Test that GrobidClient has expected methods."""
    from tuxedo import GrobidClient
    
    # Can be instantiated
    client = GrobidClient("http://localhost:8070")
    
    # Has expected methods
    assert hasattr(client, "process_pdf")
    assert hasattr(client, "process_pdf_with_result")
    assert callable(client.process_pdf)
    assert callable(client.process_pdf_with_result)


def test_model_classes_are_pydantic():
    """Test that model classes are Pydantic models."""
    from tuxedo import Paper, Cluster, ClusterView, Question, PaperAnswer, Author
    from pydantic import BaseModel
    
    # All models should be Pydantic BaseModel subclasses
    assert issubclass(Paper, BaseModel)
    assert issubclass(Cluster, BaseModel)
    assert issubclass(ClusterView, BaseModel)
    assert issubclass(Question, BaseModel)
    assert issubclass(PaperAnswer, BaseModel)
    assert issubclass(Author, BaseModel)


def test_version_exposed():
    """Test that version is exposed."""
    import tuxedo
    
    assert hasattr(tuxedo, "__version__")
    assert isinstance(tuxedo.__version__, str)
    assert tuxedo.__version__ == "0.1.0"


def test_module_docstring():
    """Test that module has proper docstring."""
    import tuxedo
    
    assert tuxedo.__doc__ is not None
    assert len(tuxedo.__doc__) > 100
    assert "Python API" in tuxedo.__doc__ or "programmatic API" in tuxedo.__doc__
