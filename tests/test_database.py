"""Tests for the database module."""

import pytest
from datetime import datetime
from pathlib import Path

from tuxedo.database import Database
from tuxedo.models import Author, Cluster, ClusterView, Paper


@pytest.fixture
def db(tmp_path):
    """Create a temporary database."""
    db_path = tmp_path / "test.db"
    return Database(db_path)


@pytest.fixture
def sample_paper():
    """Create a sample paper for testing."""
    return Paper(
        id="paper1",
        pdf_path=Path("test.pdf"),
        title="Test Paper Title",
        authors=[
            Author(name="John Doe", affiliation="MIT"),
            Author(name="Jane Smith"),
        ],
        abstract="This is a test abstract.",
        year=2024,
        doi="10.1234/test",
        sections={"Introduction": "Intro text", "Methods": "Methods text"},
        keywords=["testing", "python"],
        journal="Test Journal",
        volume="1",
        number="2",
        pages="1-10",
    )


@pytest.fixture
def sample_paper2():
    """Create a second sample paper."""
    return Paper(
        id="paper2",
        pdf_path=Path("test2.pdf"),
        title="Another Test Paper",
        authors=[Author(name="Bob Wilson")],
        abstract="Another abstract.",
        year=2023,
    )


@pytest.fixture
def sample_view():
    """Create a sample cluster view."""
    return ClusterView(
        id="view1",
        name="Test View",
        prompt="Test research question",
        created_at=datetime(2024, 1, 15, 10, 30),
    )


@pytest.fixture
def sample_clusters():
    """Create sample clusters."""
    return [
        Cluster(
            id="cluster1",
            name="Theme A",
            description="First theme",
            paper_ids=["paper1"],
            subclusters=[
                Cluster(
                    id="cluster1a",
                    name="Subtheme A1",
                    description="Sub theme",
                    paper_ids=["paper2"],
                    subclusters=[],
                )
            ],
        ),
        Cluster(
            id="cluster2",
            name="Theme B",
            description="Second theme",
            paper_ids=["paper3"],
            subclusters=[],
        ),
    ]


class TestDatabaseSchema:
    """Tests for database initialization and schema."""

    def test_creates_database_file(self, tmp_path):
        """Database file is created on initialization."""
        db_path = tmp_path / "subdir" / "test.db"
        db = Database(db_path)
        assert db_path.exists()

    def test_creates_tables(self, db):
        """All required tables are created."""
        with db._connect() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {row[0] for row in tables}

        assert "papers" in table_names
        assert "cluster_views" in table_names
        assert "clusters" in table_names
        assert "cluster_papers" in table_names

    def test_migration_adds_biblio_columns(self, tmp_path):
        """Bibliographic columns are added via migration."""
        db_path = tmp_path / "test.db"
        db = Database(db_path)

        with db._connect() as conn:
            columns = conn.execute("PRAGMA table_info(papers)").fetchall()
            column_names = {row[1] for row in columns}

        assert "journal" in column_names
        assert "booktitle" in column_names
        assert "publisher" in column_names
        assert "volume" in column_names
        assert "arxiv_id" in column_names


class TestPaperOperations:
    """Tests for paper CRUD operations."""

    def test_add_paper(self, db, sample_paper):
        """Papers can be added to the database."""
        db.add_paper(sample_paper)
        assert db.paper_count() == 1

    def test_add_paper_replaces_existing(self, db, sample_paper):
        """Adding a paper with same ID replaces the existing one."""
        db.add_paper(sample_paper)

        # Modify and re-add
        sample_paper.title = "Updated Title"
        db.add_paper(sample_paper)

        assert db.paper_count() == 1
        paper = db.get_paper("paper1")
        assert paper.title == "Updated Title"

    def test_get_paper(self, db, sample_paper):
        """Papers can be retrieved by ID."""
        db.add_paper(sample_paper)
        paper = db.get_paper("paper1")

        assert paper is not None
        assert paper.id == "paper1"
        assert paper.title == "Test Paper Title"
        assert len(paper.authors) == 2
        assert paper.authors[0].name == "John Doe"
        assert paper.authors[0].affiliation == "MIT"
        assert paper.abstract == "This is a test abstract."
        assert paper.year == 2024
        assert paper.doi == "10.1234/test"
        assert paper.sections == {"Introduction": "Intro text", "Methods": "Methods text"}
        assert paper.keywords == ["testing", "python"]
        assert paper.journal == "Test Journal"

    def test_get_paper_not_found(self, db):
        """Getting a non-existent paper returns None."""
        assert db.get_paper("nonexistent") is None

    def test_get_all_papers(self, db, sample_paper, sample_paper2):
        """All papers can be retrieved."""
        db.add_paper(sample_paper)
        db.add_paper(sample_paper2)

        papers = db.get_all_papers()
        assert len(papers) == 2
        # Ordered by year DESC, title
        assert papers[0].id == "paper1"  # 2024
        assert papers[1].id == "paper2"  # 2023

    def test_paper_count(self, db, sample_paper, sample_paper2):
        """Paper count is accurate."""
        assert db.paper_count() == 0
        db.add_paper(sample_paper)
        assert db.paper_count() == 1
        db.add_paper(sample_paper2)
        assert db.paper_count() == 2

    def test_delete_paper(self, db, sample_paper):
        """Papers can be deleted."""
        db.add_paper(sample_paper)
        assert db.paper_count() == 1

        db.delete_paper("paper1")
        assert db.paper_count() == 0
        assert db.get_paper("paper1") is None

    def test_delete_paper_removes_cluster_associations(self, db, sample_paper, sample_view):
        """Deleting a paper removes it from cluster associations."""
        db.add_paper(sample_paper)
        db.create_view(sample_view)

        cluster = Cluster(
            id="c1",
            name="Test",
            description="",
            paper_ids=["paper1"],
            subclusters=[],
        )
        db.save_clusters("view1", [cluster])

        db.delete_paper("paper1")

        # Paper should be removed from cluster_papers
        with db._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM cluster_papers WHERE paper_id = ?",
                ("paper1",)
            ).fetchone()[0]
        assert count == 0


class TestClusterViewOperations:
    """Tests for cluster view CRUD operations."""

    def test_create_view(self, db, sample_view):
        """Views can be created."""
        db.create_view(sample_view)
        assert db.view_count() == 1

    def test_get_view(self, db, sample_view):
        """Views can be retrieved by ID."""
        db.create_view(sample_view)
        view = db.get_view("view1")

        assert view is not None
        assert view.id == "view1"
        assert view.name == "Test View"
        assert view.prompt == "Test research question"
        assert view.created_at == datetime(2024, 1, 15, 10, 30)

    def test_get_view_not_found(self, db):
        """Getting a non-existent view returns None."""
        assert db.get_view("nonexistent") is None

    def test_get_all_views(self, db):
        """All views can be retrieved."""
        view1 = ClusterView(id="v1", name="View 1", prompt="Q1")
        view2 = ClusterView(id="v2", name="View 2", prompt="Q2")

        db.create_view(view1)
        db.create_view(view2)

        views = db.get_all_views()
        assert len(views) == 2

    def test_delete_view(self, db, sample_view, sample_clusters):
        """Deleting a view removes its clusters."""
        db.create_view(sample_view)
        db.save_clusters("view1", sample_clusters)

        db.delete_view("view1")

        assert db.view_count() == 0
        assert db.get_view("view1") is None
        assert db.cluster_count("view1") == 0


class TestClusterOperations:
    """Tests for cluster CRUD operations."""

    def test_save_clusters(self, db, sample_view, sample_clusters):
        """Clusters can be saved for a view."""
        db.create_view(sample_view)
        db.save_clusters("view1", sample_clusters)

        assert db.cluster_count("view1") == 2

    def test_save_clusters_replaces_existing(self, db, sample_view, sample_clusters):
        """Saving clusters replaces existing ones for the view."""
        db.create_view(sample_view)
        db.save_clusters("view1", sample_clusters)

        new_clusters = [
            Cluster(id="new1", name="New Theme", description="", paper_ids=[], subclusters=[])
        ]
        db.save_clusters("view1", new_clusters)

        assert db.cluster_count("view1") == 1

    def test_get_clusters(self, db, sample_view, sample_clusters):
        """Clusters can be retrieved for a view."""
        db.create_view(sample_view)
        db.save_clusters("view1", sample_clusters)

        clusters = db.get_clusters("view1")

        assert len(clusters) == 2
        assert clusters[0].name == "Theme A"
        assert clusters[0].paper_ids == ["paper1"]
        assert len(clusters[0].subclusters) == 1
        assert clusters[0].subclusters[0].name == "Subtheme A1"
        assert clusters[0].subclusters[0].paper_ids == ["paper2"]
        assert clusters[1].name == "Theme B"

    def test_get_clusters_empty(self, db, sample_view):
        """Getting clusters for a view with none returns empty list."""
        db.create_view(sample_view)
        assert db.get_clusters("view1") == []

    def test_cluster_count(self, db, sample_view, sample_clusters):
        """Cluster count only counts top-level clusters."""
        db.create_view(sample_view)
        db.save_clusters("view1", sample_clusters)

        # Only top-level clusters (not subclusters)
        assert db.cluster_count("view1") == 2
        assert db.cluster_count() == 2  # All views

    def test_move_paper_to_cluster(self, db, sample_view, sample_paper, sample_paper2):
        """Papers can be moved between clusters."""
        db.add_paper(sample_paper)
        db.add_paper(sample_paper2)
        db.create_view(sample_view)

        clusters = [
            Cluster(id="c1", name="Cluster 1", description="", paper_ids=["paper1"], subclusters=[]),
            Cluster(id="c2", name="Cluster 2", description="", paper_ids=["paper2"], subclusters=[]),
        ]
        db.save_clusters("view1", clusters)

        # Move paper1 from c1 to c2
        db.move_paper_to_cluster("view1", "paper1", "c2")

        updated = db.get_clusters("view1")
        assert "paper1" not in updated[0].paper_ids
        assert "paper1" in updated[1].paper_ids


class TestDatabaseTransactions:
    """Tests for transaction handling."""

    def test_rollback_on_error(self, db, sample_view):
        """Transactions are rolled back on error."""
        db.create_view(sample_view)
        initial_count = db.view_count()

        # This should fail and rollback
        try:
            with db._connect() as conn:
                conn.execute(
                    "INSERT INTO cluster_views (id, name, prompt) VALUES (?, ?, ?)",
                    ("v2", "Test", "Q"),
                )
                # Force an error
                raise ValueError("Test error")
        except ValueError:
            pass

        # Count should be unchanged
        assert db.view_count() == initial_count
