"""Project management for Tuxedo."""

import shutil
import tomllib
import uuid
from pathlib import Path

from tuxedo.database import Database
from tuxedo.models import Cluster, ClusterView, Paper, PaperAnswer, Question

CONFIG_FILE = "tuxedo.toml"
PAPERS_DIR = "papers"
DATA_DIR = "data"
DB_FILE = "tuxedo.db"


class ProjectConfig:
    """Project configuration."""

    def __init__(
        self,
        name: str,
        research_question: str,
        grobid_url: str = "http://localhost:8070",
    ):
        self.name = name
        self.research_question = research_question
        self.grobid_url = grobid_url

    def to_toml(self) -> str:
        """Serialize to TOML string."""
        return f'''[project]
name = "{self.name}"
research_question = """{self.research_question}"""

[grobid]
url = "{self.grobid_url}"
'''

    @classmethod
    def from_toml(cls, content: str) -> "ProjectConfig":
        """Parse from TOML string."""
        data = tomllib.loads(content)
        return cls(
            name=data["project"]["name"],
            research_question=data["project"]["research_question"],
            grobid_url=data.get("grobid", {}).get("url", "http://localhost:8070"),
        )


class Project:
    """A Tuxedo literature review project."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._config: ProjectConfig | None = None
        self._db: Database | None = None

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_FILE

    @property
    def papers_dir(self) -> Path:
        return self.root / PAPERS_DIR

    @property
    def data_dir(self) -> Path:
        return self.root / DATA_DIR

    @property
    def db_path(self) -> Path:
        return self.data_dir / DB_FILE

    @property
    def config(self) -> ProjectConfig:
        """Load project configuration."""
        if self._config is None:
            self._config = ProjectConfig.from_toml(self.config_path.read_text())
        return self._config

    @property
    def db(self) -> Database:
        """Get database connection."""
        if self._db is None:
            self._db = Database(self.db_path)
        return self._db

    def exists(self) -> bool:
        """Check if project exists at this location."""
        return self.config_path.exists()

    @classmethod
    def create(
        cls,
        root: Path,
        name: str,
        research_question: str,
        source_pdfs: Path,
        grobid_url: str = "http://localhost:8070",
    ) -> "Project":
        """Create a new project, copying PDFs from source directory."""
        root = root.resolve()

        # Create directory structure
        root.mkdir(parents=True, exist_ok=True)
        (root / PAPERS_DIR).mkdir(exist_ok=True)
        (root / DATA_DIR).mkdir(exist_ok=True)

        # Write config
        config = ProjectConfig(
            name=name,
            research_question=research_question,
            grobid_url=grobid_url,
        )
        (root / CONFIG_FILE).write_text(config.to_toml())

        # Copy PDFs
        pdf_files = list(source_pdfs.glob("*.pdf"))
        for pdf_path in pdf_files:
            dest = root / PAPERS_DIR / pdf_path.name
            if not dest.exists():
                shutil.copy2(pdf_path, dest)

        # Initialize database
        project = cls(root)
        _ = project.db  # Trigger schema creation

        return project

    @classmethod
    def load(cls, path: Path | None = None) -> "Project | None":
        """Load project from path, or search up from cwd."""
        if path is not None:
            project = cls(path)
            return project if project.exists() else None

        # Search up from cwd
        current = Path.cwd()
        while current != current.parent:
            project = cls(current)
            if project.exists():
                return project
            current = current.parent

        return None

    # Paper methods

    def get_pdf_path(self, paper: Paper) -> Path:
        """Get full path to a paper's PDF."""
        return self.papers_dir / paper.pdf_path.name

    def list_pdfs(self) -> list[Path]:
        """List all PDFs in the papers directory."""
        return sorted(self.papers_dir.glob("*.pdf"))

    def add_paper(self, paper: Paper) -> None:
        """Add a paper to the database."""
        self.db.add_paper(paper)

    def get_papers(self) -> list[Paper]:
        """Get all papers."""
        return self.db.get_all_papers()

    def paper_count(self) -> int:
        """Get number of processed papers."""
        return self.db.paper_count()

    def update_paper(self, paper_id: str, updates: dict) -> None:
        """Update paper metadata.

        Args:
            paper_id: The ID of the paper to update
            updates: Dictionary of field names to new values
        """
        self.db.update_paper(paper_id, updates)

    def get_paper(self, paper_id: str) -> Paper | None:
        """Get a paper by ID."""
        return self.db.get_paper(paper_id)

    def delete_paper(self, paper_id: str) -> None:
        """Delete a paper from the database."""
        self.db.delete_paper(paper_id)

    # Cluster View methods

    def create_view(self, name: str, prompt: str) -> ClusterView:
        """Create a new cluster view."""
        view = ClusterView(
            id=str(uuid.uuid4())[:8],
            name=name,
            prompt=prompt,
        )
        self.db.create_view(view)
        return view

    def get_views(self) -> list[ClusterView]:
        """Get all cluster views."""
        return self.db.get_all_views()

    def get_view(self, view_id: str) -> ClusterView | None:
        """Get a cluster view by ID."""
        return self.db.get_view(view_id)

    def delete_view(self, view_id: str) -> None:
        """Delete a cluster view."""
        self.db.delete_view(view_id)

    def rename_view(self, view_id: str, name: str) -> None:
        """Rename a cluster view."""
        self.db.rename_view(view_id, name)

    def view_count(self) -> int:
        """Get number of cluster views."""
        return self.db.view_count()

    # Cluster methods

    def save_clusters(self, view_id: str, clusters: list[Cluster]) -> None:
        """Save clusters for a view."""
        self.db.save_clusters(view_id, clusters)

    def get_clusters(self, view_id: str) -> list[Cluster]:
        """Get clusters for a view."""
        return self.db.get_clusters(view_id)

    def cluster_count(self, view_id: str | None = None) -> int:
        """Get number of top-level clusters."""
        return self.db.cluster_count(view_id)

    def move_paper_to_cluster(self, view_id: str, paper_id: str, target_cluster_id: str) -> None:
        """Move a paper to a different cluster."""
        self.db.move_paper_to_cluster(view_id, paper_id, target_cluster_id)

    def rename_cluster(
        self, cluster_id: str, new_name: str, new_description: str | None = None
    ) -> None:
        """Rename a cluster and optionally update its description."""
        self.db.rename_cluster(cluster_id, new_name, new_description)

    # Question methods

    def create_question(self, text: str) -> Question:
        """Create a new question."""
        question = Question(
            id=str(uuid.uuid4())[:8],
            text=text,
        )
        self.db.add_question(question)
        return question

    def get_questions(self) -> list[Question]:
        """Get all questions."""
        return self.db.get_all_questions()

    def get_question(self, question_id: str) -> Question | None:
        """Get a question by ID."""
        return self.db.get_question(question_id)

    def delete_question(self, question_id: str) -> None:
        """Delete a question and its answers."""
        self.db.delete_question(question_id)

    # Paper Answer methods

    def save_answer(self, answer: PaperAnswer) -> None:
        """Save a paper answer."""
        self.db.add_paper_answer(answer)

    def get_answers_for_paper(self, paper_id: str) -> list[PaperAnswer]:
        """Get all answers for a paper."""
        return self.db.get_answers_for_paper(paper_id)

    def get_answers_with_questions(self, paper_id: str) -> list[tuple[Question, PaperAnswer]]:
        """Get all answers for a paper with their questions."""
        answers = self.db.get_answers_for_paper(paper_id)
        result = []
        for answer in answers:
            question = self.db.get_question(answer.question_id)
            if question:
                result.append((question, answer))
        return result

    def get_question_answer_count(self, question_id: str) -> int:
        """Get the number of answers for a question."""
        return self.db.get_answer_count_for_question(question_id)
