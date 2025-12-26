"""CLI interface for Tuxedo."""

from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from tuxedo.clustering import PaperClusterer
from tuxedo.grobid import (
    GrobidClient,
    GrobidConnectionError,
    GrobidError,
    GrobidParsingError,
    GrobidProcessingError,
)
from tuxedo.project import Project
from tuxedo.tui import run_tui

console = Console()


@click.group()
@click.version_option()
def main():
    """Tuxedo - Organize literature review papers with LLMs."""
    pass


@main.command()
@click.argument("source_pdfs", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-q", "--question", prompt="Research question", help="The research question guiding the review")
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None, help="Project directory (defaults to parent of SOURCE_PDFS)")
@click.option("--grobid-url", default="http://localhost:8070", help="Grobid service URL")
@click.option("--name", default=None, help="Project name (defaults to directory name)")
def init(source_pdfs: Path, question: str, output: Path | None, grobid_url: str, name: str | None):
    """Initialize a new literature review project.

    SOURCE_PDFS is the directory containing PDF files to import.
    """
    project_dir = output or source_pdfs.parent
    project_name = name or project_dir.name

    # Check for existing project
    if (project_dir / "tuxedo.toml").exists():
        if not click.confirm("Project already exists. Overwrite?"):
            raise click.Abort()

    # Count PDFs
    pdf_files = list(source_pdfs.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[red]No PDF files found in {source_pdfs}[/red]")
        raise click.Abort()

    console.print(f"Creating project in {project_dir}...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Copying {len(pdf_files)} PDFs...", total=None)
        project = Project.create(
            root=project_dir,
            name=project_name,
            research_question=question,
            source_pdfs=source_pdfs,
            grobid_url=grobid_url,
        )

    console.print(f"\n[green]Project '{project_name}' created[/green]")
    console.print(f"  Copied {len(pdf_files)} PDF files to {project.papers_dir}")
    console.print(f"  Research question: {question[:60]}...")
    console.print("\n[dim]Next steps:[/dim]")
    console.print("  1. Ensure Grobid is running at", grobid_url)
    console.print("  2. Run [bold]tuxedo process[/bold] to extract paper content")
    console.print("  3. Run [bold]tuxedo cluster[/bold] to organize papers")
    console.print("  4. Run [bold]tuxedo view[/bold] to explore the structure")


@main.command()
@click.option("--max-retries", "-r", default=2, type=int, help="Maximum retry attempts per PDF (default: 2)")
def process(max_retries: int):
    """Process PDFs using Grobid to extract content.

    Automatically retries failed PDFs with different Grobid configurations.
    Use 'tuxedo view' to manually repair papers that fail after all retries.
    """
    project = Project.load()
    if not project:
        console.print("[red]No project found. Run 'tuxedo init' first.[/red]")
        raise click.Abort()

    # Check Grobid connectivity
    with GrobidClient(project.config.grobid_url) as client:
        try:
            client.check_connection()
        except GrobidConnectionError as e:
            console.print(f"[red]{e}[/red]")
            console.print("[dim]Make sure Grobid is running. You can start it with:[/dim]")
            console.print("  docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0")
            raise click.Abort()

        pdf_files = project.list_pdfs()
        console.print(f"Processing {len(pdf_files)} PDFs (max {max_retries} retries per file)...")

        success_count = 0
        retry_success_count = 0
        errors: list[tuple[Path, GrobidError, int]] = []  # (path, error, attempts)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing...", total=len(pdf_files))

            for pdf_path in pdf_files:
                progress.update(task, description=f"{pdf_path.name[:30]}...")

                result = client.process_pdf_with_result(pdf_path, max_retries=max_retries)

                if result.success:
                    project.add_paper(result.paper)
                    success_count += 1
                    if result.retried:
                        retry_success_count += 1
                elif isinstance(result.error, GrobidConnectionError):
                    # Connection lost mid-processing
                    console.print("\n[red]Lost connection to Grobid service[/red]")
                    raise click.Abort()
                else:
                    errors.append((pdf_path, result.error, result.attempts))

                progress.advance(task)

        # Summary
        if retry_success_count > 0:
            console.print(f"\n[green]Processed {success_count}/{len(pdf_files)} papers[/green] ({retry_success_count} succeeded on retry)")
        else:
            console.print(f"\n[green]Processed {success_count}/{len(pdf_files)} papers[/green]")

        # Show errors if any
        if errors:
            console.print(f"\n[yellow]{len(errors)} paper(s) failed after all retries:[/yellow]")
            for pdf_path, error, attempts in errors:
                attempt_info = f"({attempts} attempts)" if attempts > 1 else ""
                if isinstance(error, GrobidProcessingError) and error.status_code:
                    console.print(f"  [dim]•[/dim] {pdf_path.name}: HTTP {error.status_code} {attempt_info}")
                elif isinstance(error, GrobidParsingError):
                    console.print(f"  [dim]•[/dim] {pdf_path.name}: Invalid response {attempt_info}")
                else:
                    console.print(f"  [dim]•[/dim] {pdf_path.name}: {error} {attempt_info}")
            console.print("\n[dim]Use 'tuxedo view' and press 'e' to manually edit paper metadata[/dim]")

        if success_count > 0:
            console.print("\n[dim]Run 'tuxedo cluster' to organize papers[/dim]")


@main.command()
@click.option("--name", "-n", default=None, help="Name for this clustering view")
@click.option("--prompt", "-p", default=None, help="Custom prompt for clustering (defaults to research question)")
@click.option("--model", default="gpt-5.2", help="OpenAI model to use")
@click.option(
    "--include-sections", "-s",
    default=None,
    help="Comma-separated section patterns to include (e.g., 'method,methodology')",
)
@click.option(
    "--batch-size", "-b",
    default=None,
    type=int,
    help="Process papers in batches of this size to handle token limits (e.g., 10)",
)
def cluster(name: str | None, prompt: str | None, model: str, include_sections: str | None, batch_size: int | None):
    """Cluster papers using LLM.

    Creates a new clustering view. You can have multiple views with different
    prompts to organize papers in different ways.

    Use --batch-size for large paper sets to avoid token limits. Themes are
    developed incrementally: first batch establishes themes, subsequent batches
    add papers to existing themes or create new ones as needed.
    """
    project = Project.load()
    if not project:
        console.print("[red]No project found. Run 'tuxedo init' first.[/red]")
        raise click.Abort()

    papers = project.get_papers()
    if not papers:
        console.print("[red]No papers processed. Run 'tuxedo process' first.[/red]")
        raise click.Abort()

    # Validate papers for clustering
    papers_without_abstract = [p for p in papers if not p.abstract]
    papers_without_title = [p for p in papers if not p.title or p.title == p.pdf_path.stem]

    if papers_without_title:
        console.print(f"[yellow]Warning: {len(papers_without_title)} paper(s) have no extracted title[/yellow]")
        for p in papers_without_title[:3]:
            console.print(f"  [dim]• {p.pdf_path.name}[/dim]")
        if len(papers_without_title) > 3:
            console.print(f"  [dim]... and {len(papers_without_title) - 3} more[/dim]")

    if papers_without_abstract:
        console.print(f"[yellow]Warning: {len(papers_without_abstract)} paper(s) have no abstract[/yellow]")
        for p in papers_without_abstract[:3]:
            console.print(f"  [dim]• {p.display_title}[/dim]")
        if len(papers_without_abstract) > 3:
            console.print(f"  [dim]... and {len(papers_without_abstract) - 3} more[/dim]")
        console.print("[dim]Papers without abstracts may be poorly categorized.[/dim]")

    if papers_without_title or papers_without_abstract:
        console.print()  # Add spacing before continuing

    # Default name and prompt
    if not name:
        existing_views = project.get_views()
        if not existing_views:
            name = "Research Question"
        else:
            name = f"View {len(existing_views) + 1}"

    if not prompt:
        prompt = project.config.research_question

    # Parse section patterns
    section_patterns = None
    if include_sections:
        section_patterns = [s.strip() for s in include_sections.split(",") if s.strip()]

    console.print(f"Creating view '[bold]{name}[/bold]'...")
    console.print(f"Clustering {len(papers)} papers using {model}...")
    if section_patterns:
        console.print(f"Including sections matching: {', '.join(section_patterns)}")
    if batch_size:
        num_batches = (len(papers) + batch_size - 1) // batch_size
        console.print(f"Using batch mode: {batch_size} papers per batch ({num_batches} batches)")

    # Create the view
    view = project.create_view(name=name, prompt=prompt)

    clusterer = PaperClusterer(model=model)

    if batch_size and len(papers) > batch_size:
        # Batch mode with progress updates
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            num_batches = (len(papers) + batch_size - 1) // batch_size
            task = progress.add_task("Processing batches...", total=num_batches)

            def progress_callback(batch_num: int, total: int, message: str) -> None:
                progress.update(task, completed=batch_num - 1, description=message)

            clusters = clusterer.cluster_papers(
                papers, prompt,
                include_sections=section_patterns,
                batch_size=batch_size,
                progress_callback=progress_callback,
            )
            progress.update(task, completed=num_batches)
    else:
        # Standard single-pass mode
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            progress.add_task(f"Analyzing {len(papers)} papers with {model}...", total=None)
            clusters = clusterer.cluster_papers(papers, prompt, include_sections=section_patterns)

    project.save_clusters(view.id, clusters)

    # Display summary
    console.print(f"\n[green]Created view '{name}' with {len(clusters)} clusters[/green]\n")

    for c in clusters:
        paper_count = len(c.paper_ids)
        for sub in c.subclusters:
            paper_count += len(sub.paper_ids)
        console.print(f"  [bold]{c.name}[/bold] ({paper_count} papers)")
        console.print(f"     [dim]{c.description}[/dim]")
        for sub in c.subclusters:
            console.print(f"       -> {sub.name} ({len(sub.paper_ids)} papers)")

    console.print("\n[dim]Run 'tuxedo view' to explore interactively[/dim]")


@main.command()
def views():
    """List all clustering views."""
    project = Project.load()
    if not project:
        console.print("[red]No project found. Run 'tuxedo init' first.[/red]")
        return

    view_list = project.get_views()
    if not view_list:
        console.print("[yellow]No clustering views yet. Run 'tuxedo cluster' first.[/yellow]")
        return

    table = Table(title="Clustering Views")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Prompt", max_width=50)
    table.add_column("Clusters", justify="right")
    table.add_column("Created")

    for v in view_list:
        prompt_display = v.prompt[:47] + "..." if len(v.prompt) > 50 else v.prompt
        cluster_count = project.cluster_count(v.id)
        table.add_row(
            v.id,
            v.name,
            prompt_display,
            str(cluster_count),
            v.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


@main.command("delete-view")
@click.argument("view_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
def delete_view(view_id: str, force: bool):
    """Delete a clustering view."""
    project = Project.load()
    if not project:
        console.print("[red]No project found. Run 'tuxedo init' first.[/red]")
        raise click.Abort()

    view = project.get_view(view_id)
    if not view:
        console.print(f"[red]View '{view_id}' not found.[/red]")
        raise click.Abort()

    if not force:
        if not click.confirm(f"Delete view '{view.name}'?"):
            raise click.Abort()

    project.delete_view(view_id)
    console.print(f"[green]Deleted view '{view.name}'[/green]")


@main.command()
@click.argument("view_id")
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(["json", "markdown", "md", "bibtex", "bib", "latex", "tex"]),
    default="markdown",
    help="Output format (default: markdown)",
)
@click.option("-o", "--output", type=click.Path(path_type=Path), help="Output file (default: stdout)")
def export(view_id: str, output_format: str, output: Path | None):
    """Export a clustering view to file.

    VIEW_ID is the ID of the view to export (use 'tuxedo views' to list).

    Formats:
      - markdown/md: Hierarchical outline for writing
      - json: Structured data for processing
      - bibtex/bib: Bibliography file for LaTeX
      - latex/tex: LaTeX skeleton with sections and citations
    """
    import json

    project = Project.load()
    if not project:
        console.print("[red]No project found. Run 'tuxedo init' first.[/red]")
        raise click.Abort()

    view = project.get_view(view_id)
    if not view:
        console.print(f"[red]View '{view_id}' not found.[/red]")
        console.print("[dim]Use 'tuxedo views' to list available views.[/dim]")
        raise click.Abort()

    clusters = project.get_clusters(view_id)
    papers_by_id = {p.id: p for p in project.get_papers()}

    if output_format == "json":
        result = _export_json(view, clusters, papers_by_id)
    elif output_format in ("bibtex", "bib"):
        result = _export_bibtex(view, clusters, papers_by_id)
    elif output_format in ("latex", "tex"):
        result = _export_latex(view, clusters, papers_by_id)
    else:  # markdown or md
        result = _export_markdown(view, clusters, papers_by_id)

    if output:
        output.write_text(result)
        console.print(f"[green]Exported to {output}[/green]")
    else:
        console.print(result)


def _export_json(view, clusters, papers_by_id) -> str:
    """Export view to JSON format."""
    import json

    def cluster_to_dict(cluster):
        papers = [
            {
                "id": pid,
                "title": papers_by_id[pid].title if pid in papers_by_id else "Unknown",
                "authors": [a.name for a in papers_by_id[pid].authors] if pid in papers_by_id else [],
                "year": papers_by_id[pid].year if pid in papers_by_id else None,
            }
            for pid in cluster.paper_ids
        ]
        return {
            "name": cluster.name,
            "description": cluster.description,
            "papers": papers,
            "subclusters": [cluster_to_dict(sub) for sub in cluster.subclusters],
        }

    data = {
        "view": {
            "id": view.id,
            "name": view.name,
            "prompt": view.prompt,
            "created_at": view.created_at.isoformat(),
        },
        "clusters": [cluster_to_dict(c) for c in clusters],
    }
    return json.dumps(data, indent=2)


def _export_markdown(view, clusters, papers_by_id) -> str:
    """Export view to Markdown format."""
    lines = [
        f"# {view.name}",
        "",
        f"> {view.prompt}",
        "",
    ]

    def render_cluster(cluster, level=2):
        header = "#" * level
        lines.append(f"{header} {cluster.name}")
        lines.append("")
        if cluster.description:
            lines.append(f"*{cluster.description}*")
            lines.append("")

        if cluster.paper_ids:
            for pid in cluster.paper_ids:
                if pid in papers_by_id:
                    paper = papers_by_id[pid]
                    authors = ", ".join(a.name for a in paper.authors[:2])
                    if len(paper.authors) > 2:
                        authors += " et al."
                    year_str = f" ({paper.year})" if paper.year else ""
                    lines.append(f"- **{paper.title}**{year_str}")
                    if authors:
                        lines.append(f"  - {authors}")
                else:
                    lines.append(f"- _{pid}_ (not found)")
            lines.append("")

        for sub in cluster.subclusters:
            render_cluster(sub, level + 1)

    for cluster in clusters:
        render_cluster(cluster)

    return "\n".join(lines)


def _export_bibtex(view, clusters, papers_by_id) -> str:
    """Export view to BibTeX format."""
    import re

    def doi_to_key(doi: str) -> str:
        """Convert DOI to a valid BibTeX key."""
        key = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        key = re.sub(r"[^a-zA-Z0-9]", "_", key)
        return f"doi_{key}"

    def escape_bibtex(text: str) -> str:
        """Escape special characters for BibTeX."""
        if not text:
            return ""
        # Escape special LaTeX characters
        replacements = [
            ("&", r"\&"),
            ("%", r"\%"),
            ("$", r"\$"),
            ("#", r"\#"),
            ("_", r"\_"),
            ("{", r"\{"),
            ("}", r"\}"),
            ("~", r"\textasciitilde{}"),
            ("^", r"\textasciicircum{}"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    def format_authors(authors) -> str:
        """Format authors for BibTeX (Last, First and Last, First)."""
        if not authors:
            return ""
        formatted = []
        for author in authors:
            parts = author.name.split()
            if len(parts) >= 2:
                # Assume last word is surname
                surname = parts[-1]
                forenames = " ".join(parts[:-1])
                formatted.append(f"{surname}, {forenames}")
            else:
                formatted.append(author.name)
        return " and ".join(formatted)

    def paper_to_bibtex(paper) -> str:
        """Convert a Paper to a BibTeX entry."""
        entry_type = paper.bibtex_type
        key = doi_to_key(paper.doi) if paper.doi else paper.citation_key

        # Build fields
        fields = []
        fields.append(f"  title = {{{escape_bibtex(paper.title)}}}")

        if paper.authors:
            fields.append(f"  author = {{{format_authors(paper.authors)}}}")

        if paper.year:
            fields.append(f"  year = {{{paper.year}}}")

        if paper.journal:
            fields.append(f"  journal = {{{escape_bibtex(paper.journal)}}}")

        if paper.booktitle:
            fields.append(f"  booktitle = {{{escape_bibtex(paper.booktitle)}}}")

        if paper.publisher:
            fields.append(f"  publisher = {{{escape_bibtex(paper.publisher)}}}")

        if paper.volume:
            fields.append(f"  volume = {{{paper.volume}}}")

        if paper.number:
            fields.append(f"  number = {{{paper.number}}}")

        if paper.pages:
            fields.append(f"  pages = {{{paper.pages}}}")

        if paper.doi:
            fields.append(f"  doi = {{{paper.doi}}}")

        if paper.url:
            fields.append(f"  url = {{{paper.url}}}")

        if paper.arxiv_id:
            fields.append(f"  eprint = {{{paper.arxiv_id}}}")
            fields.append(f"  archiveprefix = {{arXiv}}")

        if paper.abstract:
            # Truncate very long abstracts
            abstract = paper.abstract[:2000] if len(paper.abstract) > 2000 else paper.abstract
            fields.append(f"  abstract = {{{escape_bibtex(abstract)}}}")

        if paper.keywords:
            fields.append(f"  keywords = {{{', '.join(paper.keywords)}}}")

        fields_str = ",\n".join(fields)
        return f"@{entry_type}{{{key},\n{fields_str}\n}}"

    # Collect all unique papers from clusters
    all_paper_ids = set()

    def collect_papers(cluster):
        for pid in cluster.paper_ids:
            all_paper_ids.add(pid)
        for sub in cluster.subclusters:
            collect_papers(sub)

    for cluster in clusters:
        collect_papers(cluster)

    # Generate BibTeX entries
    entries = []
    entries.append(f"% BibTeX export from Tuxedo")
    entries.append(f"% View: {view.name}")
    entries.append(f"% Generated: {view.created_at.strftime('%Y-%m-%d')}")
    entries.append(f"% Papers: {len(all_paper_ids)}")
    entries.append("")

    for pid in sorted(all_paper_ids):
        if pid in papers_by_id:
            paper = papers_by_id[pid]
            entries.append(paper_to_bibtex(paper))
            entries.append("")

    return "\n".join(entries)


def _export_latex(view, clusters, papers_by_id) -> str:
    """Export view to LaTeX skeleton format."""
    import re

    def doi_to_key(doi: str) -> str:
        """Convert DOI to a valid BibTeX key."""
        # Remove DOI prefix and clean for use as key
        key = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        key = re.sub(r"[^a-zA-Z0-9]", "_", key)
        return f"doi_{key}"

    def paper_cite_key(paper) -> str:
        """Get citation key for a paper, preferring DOI-based keys."""
        if paper.doi:
            return doi_to_key(paper.doi)
        return paper.citation_key

    def escape_latex(text: str) -> str:
        """Escape special LaTeX characters."""
        if not text:
            return ""
        replacements = [
            ("\\", r"\textbackslash{}"),
            ("&", r"\&"),
            ("%", r"\%"),
            ("$", r"\$"),
            ("#", r"\#"),
            ("_", r"\_"),
            ("{", r"\{"),
            ("}", r"\}"),
            ("~", r"\textasciitilde{}"),
            ("^", r"\textasciicircum{}"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    lines = [
        "% LaTeX skeleton generated by Tuxedo",
        f"% View: {view.name}",
        f"% Research Question: {view.prompt[:80]}...",
        "%",
        "% Usage: Include this in your document and compile with your .bib file",
        "% Generate .bib with: tuxedo export <view_id> -f bibtex -o references.bib",
        "",
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{natbib}",
        "",
        r"\begin{document}",
        "",
        f"\\title{{{escape_latex(view.name)}}}",
        r"\maketitle",
        "",
    ]

    def render_cluster(cluster, level=0):
        # Determine section command based on level
        if level == 0:
            cmd = "section"
        elif level == 1:
            cmd = "subsection"
        else:
            cmd = "subsubsection"

        lines.append(f"\\{cmd}{{{escape_latex(cluster.name)}}}")
        lines.append("")

        if cluster.description:
            lines.append(f"% {cluster.description}")
            lines.append("")

        # Add placeholder text with citations
        if cluster.paper_ids:
            cite_keys = []
            for pid in cluster.paper_ids:
                if pid in papers_by_id:
                    paper = papers_by_id[pid]
                    cite_keys.append(paper_cite_key(paper))

            if cite_keys:
                lines.append("% TODO: Write synthesis of the following papers:")
                for pid in cluster.paper_ids:
                    if pid in papers_by_id:
                        paper = papers_by_id[pid]
                        lines.append(f"%   - {paper.title[:70]}")
                lines.append("")
                lines.append(f"\\citep{{{', '.join(cite_keys)}}}")
                lines.append("")

        # Recurse for subclusters
        for sub in cluster.subclusters:
            render_cluster(sub, level + 1)

    for cluster in clusters:
        render_cluster(cluster)

    lines.extend([
        "",
        r"\bibliographystyle{plainnat}",
        r"\bibliography{references}",
        "",
        r"\end{document}",
    ])

    return "\n".join(lines)


@main.command()
def view():
    """Open interactive TUI to explore the literature structure."""
    project = Project.load()
    if not project:
        console.print("[red]No project found. Run 'tuxedo init' first.[/red]")
        raise click.Abort()

    if project.paper_count() == 0:
        console.print("[yellow]No papers processed yet.[/yellow]")
        if not click.confirm("Open empty project?"):
            raise click.Abort()

    run_tui(project)


@main.command()
def status():
    """Show project status."""
    project = Project.load()
    if not project:
        console.print("[red]No project found. Run 'tuxedo init' first.[/red]")
        return

    table = Table(title=f"Project: {project.config.name}")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    rq = project.config.research_question
    table.add_row("Location", str(project.root))
    table.add_row("Research Question", rq[:80] + "..." if len(rq) > 80 else rq)
    table.add_row("Grobid URL", project.config.grobid_url)
    table.add_row("PDFs in papers/", str(len(project.list_pdfs())))
    table.add_row("Papers Processed", str(project.paper_count()))
    table.add_row("Clustering Views", str(project.view_count()))

    console.print(table)


@main.command()
def papers():
    """List all processed papers."""
    project = Project.load()
    if not project:
        console.print("[red]No project found. Run 'tuxedo init' first.[/red]")
        return

    paper_list = project.get_papers()
    if not paper_list:
        console.print("[yellow]No papers processed. Run 'tuxedo process' first.[/yellow]")
        return

    table = Table(title=f"Papers ({len(paper_list)})")
    table.add_column("#", style="dim")
    table.add_column("Title")
    table.add_column("Authors", style="dim")
    table.add_column("Year", justify="right")

    for i, paper in enumerate(paper_list, 1):
        authors = ", ".join(a.name for a in paper.authors[:2])
        if len(paper.authors) > 2:
            authors += " et al."
        table.add_row(
            str(i),
            paper.display_title,
            authors or "-",
            str(paper.year) if paper.year else "-",
        )

    console.print(table)


if __name__ == "__main__":
    main()
