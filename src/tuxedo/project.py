"""Project management for Tuxedo."""

import shutil
import tomllib
import uuid
from pathlib import Path

from tuxedo.database import Database
from tuxedo.models import Cluster, ClusterView, Paper

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
