# Tuxedo

A simple app for organizing systematic literature review papers using LLMs.

<img width="1904" height="824" alt="Screenshot 2025-12-26 at 00 27 37" src="https://github.com/user-attachments/assets/11bfd720-88f2-4d35-bcae-c249accdc6ed" />

## How it works

We begin with a collection of papers that have been gathered through a data-driven or snowballing systematic literature review.
- Create a project with a guiding research question
- Process all the PDFs to extract metadata and content
- Cluster papers based on the research question
- View proposed clusters and optionally recategorise papers
- Create new cluster views (e.g. type of experiement, method, etc)
- Export to LaTeX / BiBTeX

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

## Commands

| Command | Description |
|---------|-------------|
| `init <pdfs> -q "..."` | Create project from PDF folder |
| `process` | Extract metadata via Grobid |
| `cluster` | Cluster papers with LLM |
| `view` | Interactive TUI |
| `views` | List clustering views |
| `export <view_id> -f bibtex` | Export to BibTeX/Markdown/JSON |
| `papers` | List all papers |
| `status` | Show project info |

## TUI Keybindings

### View Selection Screen
| Key | Action |
|-----|--------|
| `n` | New clustering view |
| `d` | Delete view |
| `q` | Quit |

### Cluster View Screen
| Key | Action |
|-----|--------|
| `/` | Search papers |
| `o` | Open web (DOI or Google Scholar) |
| `p` | Open local PDF |
| `m` | Move paper to different cluster |
| `E` | Edit paper metadata |
| `e` / `c` | Expand / Collapse all |
| `?` | Show help |
| `q` | Back |

## Multiple Views

Create different clusterings with custom prompts:

```bash
uv run tuxedo cluster -n "By Method" -p "Group by research methodology"
uv run tuxedo cluster -n "By Year" -p "Group chronologically by publication era"
```

## Export

```bash
# BibTeX for LaTeX
uv run tuxedo export abc123 -f bibtex -o refs.bib

# Markdown outline for writing
uv run tuxedo export abc123 -f markdown -o outline.md
```

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
