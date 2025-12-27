"""SQLite database layer for Tuxedo."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from tuxedo.models import Author, Cluster, ClusterView, Paper, PaperAnswer, Question

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

CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_answers (
    id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    answer TEXT NOT NULL,
    sections_used TEXT NOT NULL,  -- JSON array
    confidence TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (question_id) REFERENCES questions(id),
    FOREIGN KEY (paper_id) REFERENCES papers(id),
    UNIQUE(question_id, paper_id)
);

CREATE INDEX IF NOT EXISTS idx_paper_answers_question ON paper_answers(question_id);
CREATE INDEX IF NOT EXISTS idx_paper_answers_paper ON paper_answers(paper_id);
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
    ("relevance_score", "INTEGER"),
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
                 journal, booktitle, publisher, volume, number, pages, arxiv_id, url,
                 relevance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    paper.relevance_score,
                ),
            )

    def get_paper(self, paper_id: str) -> Paper | None:
        """Get a paper by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
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

    def update_paper(self, paper_id: str, updates: dict) -> None:
        """Update specific fields of a paper.

        Args:
            paper_id: The ID of the paper to update
            updates: Dictionary of field names to new values.
                    Supported fields: title, authors, abstract, year, doi,
                    journal, booktitle, publisher, volume, number, pages,
                    arxiv_id, url, keywords, relevance_score
        """
        if not updates:
            return

        # Map of field names to their SQL column and whether they need JSON serialization
        field_map = {
            "title": ("title", False),
            "authors": ("authors", "authors"),  # Special handling for Author objects
            "abstract": ("abstract", False),
            "year": ("year", False),
            "doi": ("doi", False),
            "journal": ("journal", False),
            "booktitle": ("booktitle", False),
            "publisher": ("publisher", False),
            "volume": ("volume", False),
            "number": ("number", False),
            "pages": ("pages", False),
            "arxiv_id": ("arxiv_id", False),
            "url": ("url", False),
            "keywords": ("keywords", "json"),
            "relevance_score": ("relevance_score", False),
        }

        # Build UPDATE query dynamically
        set_clauses = []
        values = []

        for field, value in updates.items():
            if field not in field_map:
                continue

            col_name, serialization = field_map[field]
            set_clauses.append(f"{col_name} = ?")

            if serialization == "authors":
                # Convert Author objects or dicts to JSON
                if value and isinstance(value[0], Author):
                    values.append(json.dumps([a.model_dump() for a in value]))
                else:
                    values.append(json.dumps(value))
            elif serialization == "json":
                values.append(json.dumps(value))
            else:
                values.append(value)

        if not set_clauses:
            return

        values.append(paper_id)
        query = f"UPDATE papers SET {', '.join(set_clauses)} WHERE id = ?"

        with self._connect() as conn:
            conn.execute(query, values)

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
            relevance_score=row["relevance_score"],
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
            row = conn.execute("SELECT * FROM cluster_views WHERE id = ?", (view_id,)).fetchone()
            if row:
                return self._row_to_view(row)
            return None

    def get_all_views(self) -> list[ClusterView]:
        """Get all cluster views."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM cluster_views ORDER BY created_at DESC").fetchall()
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

    def rename_view(self, view_id: str, name: str) -> None:
        """Rename a cluster view."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE cluster_views SET name = ? WHERE id = ?",
                (name, view_id),
            )

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
            return conn.execute("SELECT COUNT(*) FROM clusters WHERE parent_id IS NULL").fetchone()[
                0
            ]

    def view_count(self) -> int:
        """Get the number of cluster views."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM cluster_views").fetchone()[0]

    def move_paper_to_cluster(self, view_id: str, paper_id: str, target_cluster_id: str) -> None:
        """Move a paper to a different cluster within a view."""
        with self._connect() as conn:
            # Get all cluster IDs for this view
            cluster_ids = [
                row[0]
                for row in conn.execute(
                    "SELECT id FROM clusters WHERE view_id = ?", (view_id,)
                ).fetchall()
            ]

            # Remove paper from all clusters in this view
            placeholders = ",".join("?" * len(cluster_ids))
            conn.execute(
                f"DELETE FROM cluster_papers WHERE paper_id = ? AND cluster_id IN ({placeholders})",
                [paper_id] + cluster_ids,
            )

            # Add paper to target cluster
            conn.execute(
                "INSERT INTO cluster_papers (cluster_id, paper_id, position) VALUES (?, ?, 0)",
                (target_cluster_id, paper_id),
            )

    def rename_cluster(
        self, cluster_id: str, new_name: str, new_description: str | None = None
    ) -> None:
        """Rename a cluster and optionally update its description."""
        with self._connect() as conn:
            if new_description is not None:
                conn.execute(
                    "UPDATE clusters SET name = ?, description = ? WHERE id = ?",
                    (new_name, new_description, cluster_id),
                )
            else:
                conn.execute(
                    "UPDATE clusters SET name = ? WHERE id = ?",
                    (new_name, cluster_id),
                )

    # Question operations

    def add_question(self, question: Question) -> None:
        """Add a question."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO questions (id, text, created_at)
                VALUES (?, ?, ?)
                """,
                (question.id, question.text, question.created_at.isoformat()),
            )

    def get_question(self, question_id: str) -> Question | None:
        """Get a question by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM questions WHERE id = ?", (question_id,)).fetchone()
            if row:
                return self._row_to_question(row)
            return None

    def get_all_questions(self) -> list[Question]:
        """Get all questions."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM questions ORDER BY created_at DESC").fetchall()
            return [self._row_to_question(row) for row in rows]

    def delete_question(self, question_id: str) -> None:
        """Delete a question and its answers."""
        with self._connect() as conn:
            conn.execute("DELETE FROM paper_answers WHERE question_id = ?", (question_id,))
            conn.execute("DELETE FROM questions WHERE id = ?", (question_id,))

    def _row_to_question(self, row: sqlite3.Row) -> Question:
        """Convert a database row to a Question model."""
        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return Question(
            id=row["id"],
            text=row["text"],
            created_at=created_at,
        )

    # Paper Answer operations

    def add_paper_answer(self, answer: PaperAnswer) -> None:
        """Add or update a paper answer."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_answers
                (id, question_id, paper_id, answer, sections_used, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    answer.id,
                    answer.question_id,
                    answer.paper_id,
                    answer.answer,
                    json.dumps(answer.sections_used),
                    answer.confidence,
                    answer.created_at.isoformat(),
                ),
            )

    def get_answers_for_paper(self, paper_id: str) -> list[PaperAnswer]:
        """Get all answers for a paper."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_answers WHERE paper_id = ? ORDER BY created_at DESC",
                (paper_id,),
            ).fetchall()
            return [self._row_to_paper_answer(row) for row in rows]

    def get_answers_for_question(self, question_id: str) -> list[PaperAnswer]:
        """Get all answers for a question."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_answers WHERE question_id = ? ORDER BY created_at DESC",
                (question_id,),
            ).fetchall()
            return [self._row_to_paper_answer(row) for row in rows]

    def get_answer_count_for_question(self, question_id: str) -> int:
        """Get the number of answers for a question."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM paper_answers WHERE question_id = ?",
                (question_id,),
            ).fetchone()[0]

    def _row_to_paper_answer(self, row: sqlite3.Row) -> PaperAnswer:
        """Convert a database row to a PaperAnswer model."""
        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return PaperAnswer(
            id=row["id"],
            question_id=row["question_id"],
            paper_id=row["paper_id"],
            answer=row["answer"],
            sections_used=json.loads(row["sections_used"]),
            confidence=row["confidence"],
            created_at=created_at,
        )
