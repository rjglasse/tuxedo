# Tuxedo

A simple app for organizing systematic literature review papers using LLMs.

<img width="1904" height="824" alt="Screenshot 2025-12-26 at 00 27 37" src="https://github.com/user-attachments/assets/11bfd720-88f2-4d35-bcae-c249accdc6ed" />

## How it works

Start with papers from a systematic literature review:

1. **Initialize** — Create a project with your research question
2. **Process** — Extract metadata and content from PDFs via Grobid
3. **Cluster** — LLM organizes papers into thematic groups based on your question, auto-discovered themes, or predefined categories
4. **Analyze** — Ask questions across all papers (e.g., "What methodology?" or "What are the key findings?") with progressive content extraction
5. **Refine** — Move papers between clusters, rename themes, create alternative views
6. **Export** — Generate LaTeX skeletons, BibTeX, CSV, RIS, or Markdown

## Quick Start

```bash
# Install
uv sync

# Start Grobid (PDF extraction)
docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0

# Set OpenAI key
export OPENAI_API_KEY=sk-...

# Initialize project
uv run tuxedo init ~/papers -q "What are the effects of X on Y?"

# Process PDFs → Cluster → View
uv run tuxedo process
uv run tuxedo cluster
uv run tuxedo view
```

## Parallel Processing

Speed up PDF extraction with multiple workers:

```bash
# Process with 4 parallel workers (up to 4x faster)
uv run tuxedo process -w 4

# Maximum 8 workers
uv run tuxedo process --workers 8
```

Each worker creates its own connection to Grobid. Adjust based on your Grobid server capacity.

## Commands

| Command | Description |
|---------|-------------|
| `init <pdfs> -q "..."` | Create project from PDF folder |
| `process [-w N]` | Extract metadata via Grobid (parallel with N workers) |
| `process <file.pdf>` | Re-process a single PDF |
| `cluster` | Cluster papers with LLM |
| `view` | Interactive TUI |
| `views` | List clustering views |
| `export <view_id> -f FORMAT` | Export (bibtex, csv, ris, markdown, json, latex) |
| `export-questions` | Export questions & answers to CSV |
| `papers` | List all papers |
| `delete-paper <id>` | Remove a paper from the project |
| `delete-view <id>` | Remove a clustering view |
| `status` | Show project info |
| `completion [shell]` | Generate shell completion script |

## Clustering Modes

Tuxedo supports three clustering modes to fit different workflows:

### 1. Research Question (Default)

Uses your project's research question to organize papers:

```bash
uv run tuxedo cluster
uv run tuxedo cluster -p "Focus on methodology used"
```

### 2. Auto-Discovery

Let the AI discover themes without a predefined question:

```bash
# Discover main themes
uv run tuxedo cluster --auto

# Specific discovery modes
uv run tuxedo cluster --auto methodology   # Group by research methods
uv run tuxedo cluster --auto domain        # Group by application area
uv run tuxedo cluster --auto temporal      # Group by temporal evolution
uv run tuxedo cluster --auto findings      # Group by key findings

# Custom focus
uv run tuxedo cluster --auto "machine learning techniques"
```

### 3. Guided Clustering

Provide your own categories - the AI assigns papers to them:

```bash
# Simple category list
uv run tuxedo cluster --categories "Quantitative, Qualitative, Mixed Methods"

# Strict mode - no new categories allowed
uv run tuxedo cluster -c "ML, NLP, Vision" --strict

# From a structure file (YAML or JSON)
uv run tuxedo cluster --structure taxonomy.yaml
uv run tuxedo cluster -S categories.json --strict
```

#### Structure File Format

YAML:
```yaml
clusters:
  - name: Quantitative Methods
    description: Papers using statistical analysis, experiments, surveys
  - name: Qualitative Methods
    description: Interviews, case studies, ethnography
  - name: Mixed Methods
    description: Combining quantitative and qualitative approaches
```

JSON:
```json
{
  "clusters": [
    {"name": "Quantitative", "description": "Statistical analysis"},
    {"name": "Qualitative", "description": "Interviews, case studies"},
    {"name": "Theoretical"}
  ]
}
```

## Multiple Views

Create different clusterings with custom prompts:

```bash
uv run tuxedo cluster -n "By Method" -p "Group by research methodology"
uv run tuxedo cluster -n "By Year" -p "Group chronologically by publication era"
uv run tuxedo cluster -n "Auto Themes" --auto
uv run tuxedo cluster -n "My Categories" -c "Theory, Practice, Case Studies"
```

## Large Paper Sets

For collections with many papers, use batch processing to avoid token limits:

```bash
# Process papers in batches of 10
uv run tuxedo cluster --batch-size 10

# Include specific paper sections for better clustering
uv run tuxedo cluster -s "method,methodology,approach"
```

Themes are developed incrementally: the first batch establishes themes, subsequent batches add papers to existing themes or create new ones as needed.

## TUI Keybindings

### View Selection Screen
| Key | Action |
|-----|--------|
| `n` | New clustering view |
| `r` | Rename view |
| `d` | Delete view |
| `Q` | View all questions |
| `L` | View logs |
| `q` | Quit |

### Cluster View Screen
| Key | Action |
|-----|--------|
| `/` | Search papers |
| `o` | Open web (DOI or Google Scholar) |
| `p` | Open local PDF |
| `m` | Move paper to different cluster |
| `E` | Edit paper metadata |
| `d` | Delete paper from project |
| `r` | Rename selected cluster |
| `R` | Recluster with feedback |
| `a` | Ask question (analyze papers) |
| `Q` | View all questions |
| `x` | Export view |
| `e` / `c` | Expand / Collapse all |
| `L` | View logs |
| `?` | Show help |
| `q` | Back |

## Export

Export from CLI:
```bash
# BibTeX for LaTeX
uv run tuxedo export abc123 -f bibtex -o refs.bib

# BibTeX with abstracts included
uv run tuxedo export abc123 -f bibtex --abstract -o refs.bib

# LaTeX skeleton with sections and citations
uv run tuxedo export abc123 -f latex -o review.tex

# Markdown outline for writing
uv run tuxedo export abc123 -f markdown -o outline.md

# CSV for spreadsheets
uv run tuxedo export abc123 -f csv -o papers.csv

# RIS for reference managers (EndNote, Zotero, Mendeley)
uv run tuxedo export abc123 -f ris -o papers.ris

# JSON for custom processing
uv run tuxedo export abc123 -f json -o data.json
```

Or press `x` in the TUI to export the current view directly.

## Shell Completion

Enable tab completion for commands and options:

```bash
# Bash
eval "$(tuxedo completion bash)"

# Zsh
eval "$(tuxedo completion zsh)"

# Fish
tuxedo completion fish | source
```

Add to your shell config file (`.bashrc`, `.zshrc`, etc.) for persistence.

## Project Structure

```
my-review/
├── tuxedo.toml      # Config
├── papers/          # PDFs
└── data/
    └── tuxedo.db    # Paper data & clusters
```

## Requirements

- Python 3.11+
- [Grobid](https://github.com/kermitt2/grobid) for PDF extraction
- OpenAI API key for clustering
