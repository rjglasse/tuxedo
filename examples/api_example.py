#!/usr/bin/env python3
"""Example demonstrating Tuxedo's Python API.

This example shows how to use Tuxedo programmatically without the CLI.
"""

from pathlib import Path
from tuxedo import (
    Project,
    PaperClusterer,
    PaperAnalyzer,
    GrobidClient,
)


def main():
    """Demonstrate the Tuxedo API."""
    
    # Example 1: Create a new project
    print("Example 1: Creating a new project")
    print("-" * 50)
    
    project_root = Path("example_review")
    source_pdfs = Path("~/papers")  # Your PDF directory
    
    # Note: This would actually create the project in production
    # project = Project.create(
    #     root=project_root,
    #     name="Example Literature Review",
    #     research_question="What are the trends in machine learning?",
    #     source_pdfs=source_pdfs,
    # )
    print("✓ Project.create() creates a new review project\n")
    
    # Example 2: Process PDFs
    print("Example 2: Processing PDFs with Grobid")
    print("-" * 50)
    
    # Note: Requires Grobid running on localhost:8070
    # with GrobidClient("http://localhost:8070") as client:
    #     for pdf in project.list_pdfs():
    #         result = client.process_pdf_with_result(pdf, max_retries=2)
    #         if result.success:
    #             project.add_paper(result.paper)
    #         else:
    #             print(f"Failed to process {pdf.name}: {result.error}")
    print("✓ GrobidClient extracts metadata from PDFs")
    print("✓ Project.add_paper() stores extracted papers\n")
    
    # Example 3: Load an existing project
    print("Example 3: Loading an existing project")
    print("-" * 50)
    
    # Load from current directory or specified path
    # project = Project.load()  # From current directory
    # project = Project.load(Path("my_review"))  # From specific path
    print("✓ Project.load() loads an existing project\n")
    
    # Example 4: Cluster papers
    print("Example 4: Clustering papers by themes")
    print("-" * 50)
    
    # Get papers from project
    # papers = project.get_papers()
    
    # Create clusterer and cluster papers
    # clusterer = PaperClusterer(model="gpt-4o")
    # clusters, relevance_scores = clusterer.cluster_papers(
    #     papers=papers,
    #     research_question="What are the trends in machine learning?",
    #     batch_size=10,  # Process in batches for large collections
    # )
    
    # Save to a view
    # view = project.create_view(
    #     name="Main Themes",
    #     prompt="ML trends clustering"
    # )
    # project.save_clusters(view.id, clusters)
    
    # Update relevance scores
    # for paper_id, score in relevance_scores.items():
    #     project.update_paper(paper_id, {"relevance_score": score})
    
    print("✓ PaperClusterer organizes papers into themes")
    print("✓ Supports batch processing for large collections")
    print("✓ Returns clusters and relevance scores\n")
    
    # Example 5: Analyze papers with questions
    print("Example 5: Asking questions about papers")
    print("-" * 50)
    
    # Create a question
    # question = project.create_question("What methodology does this paper use?")
    
    # Analyze papers
    # analyzer = PaperAnalyzer(model="gpt-4o-mini")
    # answers = analyzer.analyze_papers(
    #     papers=papers,
    #     question=question.text,
    #     question_id=question.id,
    # )
    
    # Save answers
    # for answer in answers:
    #     project.save_answer(answer)
    
    print("✓ PaperAnalyzer answers questions for each paper")
    print("✓ Questions and answers are stored in the project\n")
    
    # Example 6: Export results
    print("Example 6: Exporting to various formats")
    print("-" * 50)
    
    # Get data for export
    # views = project.get_views()
    # clusters = project.get_clusters(views[0].id)
    # papers_by_id = {p.id: p for p in project.get_papers()}
    
    # Export to BibTeX
    # from tuxedo.cli import _export_bibtex
    # bibtex = _export_bibtex(views[0], clusters, papers_by_id)
    # Path("references.bib").write_text(bibtex)
    
    # Export to other formats
    # from tuxedo.cli import _export_markdown, _export_csv, _export_latex
    # markdown = _export_markdown(views[0], clusters, papers_by_id)
    # csv_data = _export_csv(views[0], clusters, papers_by_id)
    # latex = _export_latex(views[0], clusters, papers_by_id)
    
    print("✓ Export functions available for:")
    print("  - BibTeX (for LaTeX)")
    print("  - Markdown (for writing)")
    print("  - CSV (for spreadsheets)")
    print("  - LaTeX (document skeleton)")
    print("  - JSON (structured data)")
    print("  - RIS (reference managers)\n")
    
    # Example 7: Working with the data
    print("Example 7: Accessing project data")
    print("-" * 50)
    
    # Access papers
    # papers = project.get_papers()
    # for paper in papers:
    #     print(f"Title: {paper.title}")
    #     print(f"Authors: {', '.join(a.name for a in paper.authors)}")
    #     print(f"Year: {paper.year}")
    #     print(f"DOI: {paper.doi}")
    #     print(f"Abstract: {paper.abstract[:100]}...")
    #     print()
    
    # Access views and clusters
    # views = project.get_views()
    # for view in views:
    #     clusters = project.get_clusters(view.id)
    #     print(f"View: {view.name}")
    #     for cluster in clusters:
    #         print(f"  Cluster: {cluster.name} ({len(cluster.paper_ids)} papers)")
    
    # Access questions and answers
    # questions = project.get_questions()
    # for question in questions:
    #     answers = project.get_answers_with_questions(paper_id)
    #     print(f"Q: {question.text}")
    #     for q, answer in answers:
    #         print(f"  A: {answer.answer}")
    
    print("✓ Full programmatic access to:")
    print("  - Papers with metadata")
    print("  - Clustering views")
    print("  - Questions and answers")
    print("  - All project data\n")
    
    print("=" * 50)
    print("For more details, see the README and source code.")
    print("=" * 50)


if __name__ == "__main__":
    main()
