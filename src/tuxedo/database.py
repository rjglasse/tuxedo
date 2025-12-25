"""SQLite database layer for Tuxedo."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from tuxedo.models import Author, Cluster, ClusterView, Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    pdf_filename TEXT NOT NULL,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,  -- JSON array
    abstract TEXT,
    year INTEGER,
    doi TEXT,
    sections TEXT NOT NULL,  -- JSON object
    keywords TEXT NOT NULL,  -- JSON array
    -- Bibliographic fields for BibTeX
    journal TEXT,
    booktitle TEXT,
    publisher TEXT,
    volume TEXT,
    number TEXT,
    pages TEXT,
    arxiv_id TEXT,
    url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cluster_views (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clusters (
    id TEXT PRIMARY KEY,
    view_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    parent_id TEXT,  -- NULL for top-level clusters
    position INTEGER DEFAULT 0,  -- ordering within parent
    FOREIGN KEY (view_id) REFERENCES cluster_views(id),
    FOREIGN KEY (parent_id) REFERENCES clusters(id)
);

CREATE TABLE IF NOT EXISTS cluster_papers (
    cluster_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    position INTEGER DEFAULT 0,
    PRIMARY KEY (cluster_id, paper_id),
    FOREIGN KEY (cluster_id) REFERENCES clusters(id),
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE INDEX IF NOT EXISTS idx_clusters_view ON clusters(view_id);
CREATE INDEX IF NOT EXISTS idx_clusters_parent ON clusters(parent_id);
CREATE INDEX IF NOT EXISTS idx_cluster_papers_cluster ON cluster_papers(cluster_id);
"""

# Migration to add cluster_views support
MIGRATION_V2 = """
-- Check if cluster_views exists, if not we need to migrate
CREATE TABLE IF NOT EXISTS cluster_views (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add view_id column if it doesn't exist (SQLite doesn't have IF NOT EXISTS for columns)
-- We handle this in Python by checking the schema
"""

# Migration to add bibliographic fields for BibTeX export
BIBLIO_COLUMNS = [
    ("journal", "TEXT"),
    ("booktitle", "TEXT"),
    ("publisher", "TEXT"),
    ("volume", "TEXT"),
    ("number", "TEXT"),
    ("pages", "TEXT"),
    ("arxiv_id", "TEXT"),
    ("url", "TEXT"),
]


class Database:
    """SQLite database for storing paper and cluster data."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._connect() as conn:
            # Check if we need to migrate (old schema without cluster_views)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='cluster_views'"
            )
            has_views_table = cursor.fetchone() is not None

            if not has_views_table:
                # Check if clusters table exists (old schema)
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='clusters'"
                )
                has_old_clusters = cursor.fetchone() is not None

                if has_old_clusters:
                    # Migrate: drop old tables and recreate
                    conn.executescript("""
                        DROP TABLE IF EXISTS cluster_papers;
                        DROP TABLE IF EXISTS clusters;
                    """)

            conn.executescript(SCHEMA)

            # Migration: add bibliographic columns if they don't exist
            cursor = conn.execute("PRAGMA table_info(papers)")
            existing_columns = {row[1] for row in cursor.fetchall()}

            for col_name, col_type in BIBLIO_COLUMNS:
                if col_name not in existing_columns:
                    conn.execute(f"ALTER TABLE papers ADD COLUMN {col_name} {col_type}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # Paper operations

    def add_paper(self, paper: Paper) -> None:
        """Add or update a paper."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO papers
                (id, pdf_filename, title, authors, abstract, year, doi, sections, keywords,
                 journal, booktitle, publisher, volume, number, pages, arxiv_id, url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper.id,
                    paper.pdf_path.name,
                    paper.title,
                    json.dumps([a.model_dump() for a in paper.authors]),
                    paper.abstract,
                    paper.year,
                    paper.doi,
                    json.dumps(paper.sections),
                    json.dumps(paper.keywords),
                    paper.journal,
                    paper.booktitle,
                    paper.publisher,
                    paper.volume,
                    paper.number,
                    paper.pages,
                    paper.arxiv_id,
                    paper.url,
                ),
            )

    def get_paper(self, paper_id: str) -> Paper | None:
        """Get a paper by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE id = ?", (paper_id,)
            ).fetchone()
            if row:
                return self._row_to_paper(row)
            return None

    def get_all_papers(self) -> list[Paper]:
        """Get all papers."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM papers ORDER BY year DESC, title").fetchall()
            return [self._row_to_paper(row) for row in rows]

    def paper_count(self) -> int:
        """Get the number of papers."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def delete_paper(self, paper_id: str) -> None:
        """Delete a paper."""
        with self._connect() as conn:
            conn.execute("DELETE FROM cluster_papers WHERE paper_id = ?", (paper_id,))
            conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))

    def _row_to_paper(self, row: sqlite3.Row) -> Paper:
        """Convert a database row to a Paper model."""
        return Paper(
            id=row["id"],
            pdf_path=Path(row["pdf_filename"]),  # Just filename, resolve from project
            title=row["title"],
            authors=[Author(**a) for a in json.loads(row["authors"])],
            abstract=row["abstract"],
            year=row["year"],
            doi=row["doi"],
            sections=json.loads(row["sections"]),
            keywords=json.loads(row["keywords"]),
            journal=row["journal"],
            booktitle=row["booktitle"],
            publisher=row["publisher"],
            volume=row["volume"],
            number=row["number"],
            pages=row["pages"],
            arxiv_id=row["arxiv_id"],
            url=row["url"],
        )

    # Cluster View operations

    def create_view(self, view: ClusterView) -> None:
        """Create a new cluster view."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cluster_views (id, name, prompt, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (view.id, view.name, view.prompt, view.created_at.isoformat()),
            )

    def get_view(self, view_id: str) -> ClusterView | None:
        """Get a cluster view by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cluster_views WHERE id = ?", (view_id,)
            ).fetchone()
            if row:
                return self._row_to_view(row)
            return None

    def get_all_views(self) -> list[ClusterView]:
        """Get all cluster views."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM cluster_views ORDER BY created_at DESC"
            ).fetchall()
            return [self._row_to_view(row) for row in rows]

    def delete_view(self, view_id: str) -> None:
        """Delete a cluster view and its clusters."""
        with self._connect() as conn:
            # Get all cluster IDs for this view
            cluster_ids = conn.execute(
                "SELECT id FROM clusters WHERE view_id = ?", (view_id,)
            ).fetchall()

            # Delete cluster papers
            for (cluster_id,) in cluster_ids:
                conn.execute("DELETE FROM cluster_papers WHERE cluster_id = ?", (cluster_id,))

            # Delete clusters
            conn.execute("DELETE FROM clusters WHERE view_id = ?", (view_id,))

            # Delete view
            conn.execute("DELETE FROM cluster_views WHERE id = ?", (view_id,))

    def _row_to_view(self, row: sqlite3.Row) -> ClusterView:
        """Convert a database row to a ClusterView model."""
        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return ClusterView(
            id=row["id"],
            name=row["name"],
            prompt=row["prompt"],
            created_at=created_at,
        )

    # Cluster operations

    def save_clusters(self, view_id: str, clusters: list[Cluster]) -> None:
        """Save clusters for a view (replaces existing for that view)."""
        with self._connect() as conn:
            # Get existing cluster IDs for this view
            existing = conn.execute(
                "SELECT id FROM clusters WHERE view_id = ?", (view_id,)
            ).fetchall()

            # Delete existing cluster papers and clusters for this view
            for (cluster_id,) in existing:
                conn.execute("DELETE FROM cluster_papers WHERE cluster_id = ?", (cluster_id,))
            conn.execute("DELETE FROM clusters WHERE view_id = ?", (view_id,))

            # Insert new clusters
            self._insert_clusters(conn, view_id, clusters, parent_id=None)

    def _insert_clusters(
        self,
        conn: sqlite3.Connection,
        view_id: str,
        clusters: list[Cluster],
        parent_id: str | None,
    ) -> None:
        """Recursively insert clusters."""
        for position, cluster in enumerate(clusters):
            conn.execute(
                """
                INSERT INTO clusters (id, view_id, name, description, parent_id, position)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cluster.id, view_id, cluster.name, cluster.description, parent_id, position),
            )

            # Insert paper associations
            for pos, paper_id in enumerate(cluster.paper_ids):
                conn.execute(
                    """
                    INSERT INTO cluster_papers (cluster_id, paper_id, position)
                    VALUES (?, ?, ?)
                    """,
                    (cluster.id, paper_id, pos),
                )

            # Recurse for subclusters
            if cluster.subclusters:
                self._insert_clusters(conn, view_id, cluster.subclusters, parent_id=cluster.id)

    def get_clusters(self, view_id: str) -> list[Cluster]:
        """Get all clusters for a view as a hierarchical structure."""
        with self._connect() as conn:
            # Get all clusters for this view
            cluster_rows = conn.execute(
                "SELECT * FROM clusters WHERE view_id = ? ORDER BY parent_id NULLS FIRST, position",
                (view_id,),
            ).fetchall()

            if not cluster_rows:
                return []

            # Get cluster IDs
            cluster_ids = [row["id"] for row in cluster_rows]
            placeholders = ",".join("?" * len(cluster_ids))

            # Get all paper associations for these clusters
            paper_rows = conn.execute(
                f"SELECT * FROM cluster_papers WHERE cluster_id IN ({placeholders}) ORDER BY cluster_id, position",
                cluster_ids,
            ).fetchall()

            # Build paper_ids map
            cluster_papers: dict[str, list[str]] = {}
            for row in paper_rows:
                cluster_id = row["cluster_id"]
                if cluster_id not in cluster_papers:
                    cluster_papers[cluster_id] = []
                cluster_papers[cluster_id].append(row["paper_id"])

            # Build cluster objects
            clusters_by_id: dict[str, Cluster] = {}
            for row in cluster_rows:
                cluster = Cluster(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"] or "",
                    paper_ids=cluster_papers.get(row["id"], []),
                    subclusters=[],
                )
                clusters_by_id[cluster.id] = cluster

            # Build hierarchy
            top_level: list[Cluster] = []
            for row in cluster_rows:
                cluster = clusters_by_id[row["id"]]
                parent_id = row["parent_id"]
                if parent_id is None:
                    top_level.append(cluster)
                else:
                    parent = clusters_by_id.get(parent_id)
                    if parent:
                        parent.subclusters.append(cluster)

            return top_level

    def cluster_count(self, view_id: str | None = None) -> int:
        """Get the number of top-level clusters, optionally for a specific view."""
        with self._connect() as conn:
            if view_id:
                return conn.execute(
                    "SELECT COUNT(*) FROM clusters WHERE view_id = ? AND parent_id IS NULL",
                    (view_id,),
                ).fetchone()[0]
            return conn.execute(
                "SELECT COUNT(*) FROM clusters WHERE parent_id IS NULL"
            ).fetchone()[0]

    def view_count(self) -> int:
        """Get the number of cluster views."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM cluster_views").fetchone()[0]
