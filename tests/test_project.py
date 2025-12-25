"""Tests for the project module."""

import pytest
from datetime import datetime
from pathlib import Path

from tuxedo.project import Project, ProjectConfig, CONFIG_FILE, PAPERS_DIR, DATA_DIR
from tuxedo.models import Author, Cluster, ClusterView, Paper


@pytest.fixture
def sample_pdf_content():
    """Minimal PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


@pytest.fixture
def project_dir(tmp_path, sample_pdf_content):
    """Create a project directory with sample PDFs."""
    papers_dir = tmp_path / "source_papers"
    papers_dir.mkdir()

    # Create sample PDFs
    (papers_dir / "paper1.pdf").write_bytes(sample_pdf_content)
    (papers_dir / "paper2.pdf").write_bytes(sample_pdf_content)

    return tmp_path, papers_dir


@pytest.fixture
def created_project(project_dir):
    """Create a project for testing."""
    root, source = project_dir
    project = Project.create(
        root=root / "myproject",
        name="Test Project",
        research_question="What are ML trends?",
        source_pdfs=source,
    )
    return project


@pytest.fixture
def sample_paper():
    """Create a sample paper."""
    return Paper(
        id="paper1",
        pdf_path=Path("paper1.pdf"),
        title="Test Paper",
        authors=[Author(name="John Doe")],
        abstract="Test abstract",
        year=2024,
    )


@pytest.fixture
def sample_view():
    """Create a sample cluster view."""
    return ClusterView(
        id="view1",
        name="Test View",
        prompt="Test question",
        created_at=datetime(2024, 1, 15),
    )


class TestProjectConfig:
    """Tests for ProjectConfig."""

    def test_to_toml(self):
        """Config serializes to TOML format."""
        config = ProjectConfig(
            name="My Project",
            research_question="What are the trends?",
            grobid_url="http://localhost:8070",
        )
        toml = config.to_toml()

        assert 'name = "My Project"' in toml
        assert 'research_question = """What are the trends?"""' in toml
        assert 'url = "http://localhost:8070"' in toml

    def test_from_toml(self):
        """Config deserializes from TOML format."""
        toml_content = '''
[project]
name = "Test Project"
research_question = """What are ML trends?"""

[grobid]
url = "http://grobid:8070"
'''
        config = ProjectConfig.from_toml(toml_content)

        assert config.name == "Test Project"
        assert config.research_question == "What are ML trends?"
        assert config.grobid_url == "http://grobid:8070"

    def test_from_toml_default_grobid_url(self):
        """Config uses default grobid URL if not specified."""
        toml_content = '''
[project]
name = "Test"
research_question = """Q?"""
'''
        config = ProjectConfig.from_toml(toml_content)
        assert config.grobid_url == "http://localhost:8070"


class TestProjectCreate:
    """Tests for Project.create()."""

    def test_creates_directory_structure(self, project_dir):
        """Project.create() creates required directories."""
        root, source = project_dir
        project_root = root / "newproject"

        Project.create(
            root=project_root,
            name="Test",
            research_question="Q?",
            source_pdfs=source,
        )

        assert (project_root / CONFIG_FILE).exists()
        assert (project_root / PAPERS_DIR).exists()
        assert (project_root / DATA_DIR).exists()

    def test_copies_pdfs(self, project_dir):
        """Project.create() copies PDFs from source directory."""
        root, source = project_dir
        project_root = root / "newproject"

        project = Project.create(
            root=project_root,
            name="Test",
            research_question="Q?",
            source_pdfs=source,
        )

        pdfs = list(project.papers_dir.glob("*.pdf"))
        assert len(pdfs) == 2
        assert (project.papers_dir / "paper1.pdf").exists()
        assert (project.papers_dir / "paper2.pdf").exists()

    def test_writes_config(self, project_dir):
        """Project.create() writes config file."""
        root, source = project_dir
        project_root = root / "newproject"

        Project.create(
            root=project_root,
            name="My Project",
            research_question="What are trends?",
            source_pdfs=source,
            grobid_url="http://custom:8070",
        )

        config_content = (project_root / CONFIG_FILE).read_text()
        assert "My Project" in config_content
        assert "What are trends?" in config_content
        assert "http://custom:8070" in config_content

    def test_initializes_database(self, project_dir):
        """Project.create() initializes the database."""
        root, source = project_dir
        project_root = root / "newproject"

        project = Project.create(
            root=project_root,
            name="Test",
            research_question="Q?",
            source_pdfs=source,
        )

        assert project.db_path.exists()

    def test_does_not_overwrite_existing_pdfs(self, project_dir, sample_pdf_content):
        """Existing PDFs in papers dir are not overwritten."""
        root, source = project_dir
        project_root = root / "newproject"

        # Create project first time
        Project.create(
            root=project_root,
            name="Test",
            research_question="Q?",
            source_pdfs=source,
        )

        # Modify an existing PDF
        existing_pdf = project_root / PAPERS_DIR / "paper1.pdf"
        original_size = existing_pdf.stat().st_size
        existing_pdf.write_bytes(b"modified content")

        # Create project again with same source
        Project.create(
            root=project_root,
            name="Test",
            research_question="Q?",
            source_pdfs=source,
        )

        # PDF should not be overwritten
        assert existing_pdf.read_bytes() == b"modified content"


class TestProjectLoad:
    """Tests for Project.load()."""

    def test_load_from_path(self, created_project):
        """Project can be loaded from explicit path."""
        loaded = Project.load(created_project.root)

        assert loaded is not None
        assert loaded.root == created_project.root
        assert loaded.config.name == "Test Project"

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Loading from nonexistent path returns None."""
        result = Project.load(tmp_path / "nonexistent")
        assert result is None

    def test_load_searches_up_from_cwd(self, created_project, monkeypatch):
        """Project.load() searches up from cwd when no path given."""
        # Change to a subdirectory of the project
        subdir = created_project.root / "subdir" / "deep"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)

        loaded = Project.load()

        assert loaded is not None
        assert loaded.root == created_project.root

    def test_load_returns_none_when_not_found(self, tmp_path, monkeypatch):
        """Project.load() returns None when no project found."""
        monkeypatch.chdir(tmp_path)
        result = Project.load()
        assert result is None


class TestProjectProperties:
    """Tests for Project property accessors."""

    def test_config_path(self, created_project):
        """config_path returns correct path."""
        assert created_project.config_path == created_project.root / CONFIG_FILE

    def test_papers_dir(self, created_project):
        """papers_dir returns correct path."""
        assert created_project.papers_dir == created_project.root / PAPERS_DIR

    def test_data_dir(self, created_project):
        """data_dir returns correct path."""
        assert created_project.data_dir == created_project.root / DATA_DIR

    def test_db_path(self, created_project):
        """db_path returns correct path."""
        assert created_project.db_path == created_project.data_dir / "tuxedo.db"

    def test_config_lazy_loaded(self, created_project):
        """Config is lazily loaded."""
        # Access config
        config = created_project.config

        assert config.name == "Test Project"
        assert config.research_question == "What are ML trends?"

    def test_exists(self, created_project, tmp_path):
        """exists() returns correct value."""
        assert created_project.exists() is True

        nonexistent = Project(tmp_path / "nope")
        assert nonexistent.exists() is False


class TestProjectPaperMethods:
    """Tests for Project paper operations."""

    def test_list_pdfs(self, created_project):
        """list_pdfs returns all PDFs in papers directory."""
        pdfs = created_project.list_pdfs()

        assert len(pdfs) == 2
        names = {p.name for p in pdfs}
        assert "paper1.pdf" in names
        assert "paper2.pdf" in names

    def test_add_paper(self, created_project, sample_paper):
        """Papers can be added to the project."""
        created_project.add_paper(sample_paper)

        assert created_project.paper_count() == 1

    def test_get_papers(self, created_project, sample_paper):
        """Papers can be retrieved from the project."""
        created_project.add_paper(sample_paper)
        papers = created_project.get_papers()

        assert len(papers) == 1
        assert papers[0].id == "paper1"
        assert papers[0].title == "Test Paper"

    def test_paper_count(self, created_project, sample_paper):
        """paper_count returns correct count."""
        assert created_project.paper_count() == 0

        created_project.add_paper(sample_paper)
        assert created_project.paper_count() == 1

    def test_get_pdf_path(self, created_project, sample_paper):
        """get_pdf_path returns full path to paper's PDF."""
        path = created_project.get_pdf_path(sample_paper)

        assert path == created_project.papers_dir / "paper1.pdf"


class TestProjectViewMethods:
    """Tests for Project cluster view operations."""

    def test_create_view(self, created_project):
        """Views can be created."""
        view = created_project.create_view("Test View", "What are trends?")

        assert view.name == "Test View"
        assert view.prompt == "What are trends?"
        assert view.id is not None

    def test_get_views(self, created_project):
        """Views can be retrieved."""
        created_project.create_view("View 1", "Q1")
        created_project.create_view("View 2", "Q2")

        views = created_project.get_views()
        assert len(views) == 2

    def test_get_view(self, created_project):
        """Single view can be retrieved by ID."""
        view = created_project.create_view("Test", "Q")
        retrieved = created_project.get_view(view.id)

        assert retrieved is not None
        assert retrieved.name == "Test"

    def test_get_view_not_found(self, created_project):
        """Getting nonexistent view returns None."""
        assert created_project.get_view("nonexistent") is None

    def test_delete_view(self, created_project):
        """Views can be deleted."""
        view = created_project.create_view("Test", "Q")
        created_project.delete_view(view.id)

        assert created_project.get_view(view.id) is None

    def test_view_count(self, created_project):
        """view_count returns correct count."""
        assert created_project.view_count() == 0

        created_project.create_view("V1", "Q1")
        assert created_project.view_count() == 1

        created_project.create_view("V2", "Q2")
        assert created_project.view_count() == 2


class TestProjectClusterMethods:
    """Tests for Project cluster operations."""

    def test_save_clusters(self, created_project):
        """Clusters can be saved for a view."""
        view = created_project.create_view("Test", "Q")
        clusters = [
            Cluster(id="c1", name="Theme A", description="", paper_ids=[], subclusters=[]),
            Cluster(id="c2", name="Theme B", description="", paper_ids=[], subclusters=[]),
        ]

        created_project.save_clusters(view.id, clusters)
        assert created_project.cluster_count(view.id) == 2

    def test_get_clusters(self, created_project):
        """Clusters can be retrieved for a view."""
        view = created_project.create_view("Test", "Q")
        clusters = [
            Cluster(
                id="c1",
                name="Theme A",
                description="First theme",
                paper_ids=["p1"],
                subclusters=[
                    Cluster(id="c1a", name="Subtheme", description="", paper_ids=["p2"], subclusters=[])
                ],
            ),
        ]
        created_project.save_clusters(view.id, clusters)

        retrieved = created_project.get_clusters(view.id)

        assert len(retrieved) == 1
        assert retrieved[0].name == "Theme A"
        assert retrieved[0].paper_ids == ["p1"]
        assert len(retrieved[0].subclusters) == 1

    def test_cluster_count(self, created_project):
        """cluster_count returns correct count."""
        view = created_project.create_view("Test", "Q")
        assert created_project.cluster_count(view.id) == 0

        clusters = [
            Cluster(id="c1", name="A", description="", paper_ids=[], subclusters=[
                Cluster(id="c1a", name="A1", description="", paper_ids=[], subclusters=[])
            ]),
            Cluster(id="c2", name="B", description="", paper_ids=[], subclusters=[]),
        ]
        created_project.save_clusters(view.id, clusters)

        # Only counts top-level
        assert created_project.cluster_count(view.id) == 2

    def test_move_paper_to_cluster(self, created_project, sample_paper):
        """Papers can be moved between clusters."""
        created_project.add_paper(sample_paper)
        view = created_project.create_view("Test", "Q")
        clusters = [
            Cluster(id="c1", name="A", description="", paper_ids=["paper1"], subclusters=[]),
            Cluster(id="c2", name="B", description="", paper_ids=[], subclusters=[]),
        ]
        created_project.save_clusters(view.id, clusters)

        created_project.move_paper_to_cluster(view.id, "paper1", "c2")

        updated = created_project.get_clusters(view.id)
        assert "paper1" not in updated[0].paper_ids
        assert "paper1" in updated[1].paper_ids
