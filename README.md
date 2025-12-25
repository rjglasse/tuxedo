# Tuxedo

TUI app for organizing systematic literature review papers using LLMs.

## Installation

```bash
uv sync
```

## Usage

```bash
# Initialize a project (copies PDFs to project folder)
uv run tuxedo init ~/Downloads/papers -q "What are the effects of X on Y?"

# Or create in a specific directory
uv run tuxedo init ~/Downloads/papers -o ./my-review -q "Research question here"

# Process PDFs with Grobid
uv run tuxedo process

# Cluster papers with LLM
uv run tuxedo cluster

# View interactive structure
uv run tuxedo view

# Check project status
uv run tuxedo status
```

## Project Structure

After `init`, your project will have this structure:

```
my-review/
├── tuxedo.toml      # Project config (name, research question, settings)
├── papers/          # PDFs copied here
│   ├── paper1.pdf
│   └── paper2.pdf
└── data/
    └── tuxedo.db    # SQLite database with extracted paper data & clusters
```

## Requirements

- **Grobid**: PDF extraction service
  ```bash
  docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0
  ```

- **OpenAI API key**: For clustering
  ```bash
  export OPENAI_API_KEY=sk-...
  ```
