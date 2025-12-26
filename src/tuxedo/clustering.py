"""LLM-based paper clustering using OpenAI."""

import json
import uuid
from typing import Callable

from openai import OpenAI

from tuxedo.models import Cluster, Paper

CLUSTER_SYSTEM_PROMPT = """You are a systematic literature review assistant. Your task is to organize academic papers into a hierarchical structure suitable for writing a literature review.

Given a research question and paper abstracts, create a logical hierarchical structure that:
1. Groups papers by theme, methodology, or findings
2. Creates meaningful cluster names that could become section headings
3. Provides brief descriptions explaining each cluster's focus
4. Organizes subclusters where appropriate (max 2 levels deep)

Respond ONLY with valid JSON in this exact format:
{
  "clusters": [
    {
      "name": "Cluster Name",
      "description": "Brief description of what papers in this cluster share",
      "paper_ids": ["id1", "id2"],
      "subclusters": [
        {
          "name": "Subcluster Name",
          "description": "Description",
          "paper_ids": ["id3"]
        }
      ]
    }
  ]
}

Important:
- Every paper must appear in exactly one cluster or subcluster
- Cluster names should be suitable as section headings
- Aim for 3-7 top-level clusters depending on paper count
- Only create subclusters if there's meaningful differentiation"""

BATCH_CLUSTER_PROMPT = """You are a systematic literature review assistant. Your task is to assign new papers to existing themes OR create new themes if papers don't fit.

You are given:
1. A research question
2. Existing themes with their descriptions
3. New papers to organize

For each paper, either:
- Assign it to an existing theme that fits well
- Create a new theme if it represents a genuinely different topic

Respond ONLY with valid JSON in this exact format:
{
  "clusters": [
    {
      "name": "Theme Name",
      "description": "Brief description of the theme",
      "paper_ids": ["id1", "id2"],
      "subclusters": []
    }
  ]
}

Important:
- Include ALL existing themes, even if no new papers were added to them
- paper_ids should ONLY contain papers from the new batch (not previous papers)
- You may create new themes if papers don't fit existing ones
- You may refine theme names/descriptions as patterns become clearer
- Keep theme names suitable as section headings"""


class PaperClusterer:
    """Cluster papers using OpenAI."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-5.2"):
        self.client = OpenAI(api_key=api_key)  # Uses OPENAI_API_KEY env var if None
        self.model = model

    def cluster_papers(
        self,
        papers: list[Paper],
        research_question: str,
        include_sections: list[str] | None = None,
        batch_size: int | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[Cluster]:
        """Cluster papers based on research question.

        Args:
            papers: List of papers to cluster
            research_question: The research question/prompt for clustering
            include_sections: Optional list of section name patterns to include
                             (e.g., ["method", "methodology"] to include method sections)
            batch_size: If set, process papers in batches to handle token limits.
                       Themes are developed incrementally across batches.
            progress_callback: Optional callback(batch_num, total_batches, message) for progress
        """
        if not papers:
            return []

        # If batch_size is set and we have more papers than the batch size, use iterative mode
        if batch_size and len(papers) > batch_size:
            return self._cluster_papers_iterative(
                papers, research_question, include_sections, batch_size, progress_callback
            )

        # Standard single-pass clustering
        paper_summaries = self._build_paper_summaries(papers, include_sections)

        user_prompt = f"""Research Question: {research_question}

Papers to organize:
{json.dumps(paper_summaries, indent=2)}

Create a hierarchical structure for these {len(papers)} papers that would support writing a literature review addressing the research question."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CLUSTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        result = json.loads(response.choices[0].message.content)
        return self._parse_clusters(result.get("clusters", []))

    def _build_paper_summaries(
        self, papers: list[Paper], include_sections: list[str] | None = None
    ) -> list[dict]:
        """Build paper summaries for the prompt."""
        paper_summaries = []
        for paper in papers:
            summary = {
                "id": paper.id,
                "title": paper.title,
                "year": paper.year,
                "abstract": paper.abstract or "No abstract available",
                "keywords": paper.keywords[:5] if paper.keywords else [],
            }

            # Include matching sections if requested
            if include_sections and paper.sections:
                matched_sections = {}
                for section_name, content in paper.sections.items():
                    section_lower = section_name.lower()
                    for pattern in include_sections:
                        if pattern.lower() in section_lower:
                            # Truncate long sections to avoid token limits
                            matched_sections[section_name] = content[:2000]
                            break
                if matched_sections:
                    summary["sections"] = matched_sections

            paper_summaries.append(summary)
        return paper_summaries

    def _cluster_papers_iterative(
        self,
        papers: list[Paper],
        research_question: str,
        include_sections: list[str] | None,
        batch_size: int,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> list[Cluster]:
        """Cluster papers iteratively in batches.

        First batch establishes initial themes, subsequent batches add papers
        to existing themes or create new ones as needed.
        """
        # Split papers into batches
        batches = [papers[i : i + batch_size] for i in range(0, len(papers), batch_size)]
        total_batches = len(batches)

        # Track accumulated paper_ids per theme (by name)
        theme_papers: dict[str, list[str]] = {}
        theme_descriptions: dict[str, str] = {}

        for batch_num, batch in enumerate(batches, 1):
            if progress_callback:
                progress_callback(
                    batch_num, total_batches, f"Processing batch {batch_num}/{total_batches}"
                )

            paper_summaries = self._build_paper_summaries(batch, include_sections)

            if batch_num == 1:
                # First batch: use standard clustering to establish themes
                user_prompt = f"""Research Question: {research_question}

Papers to organize:
{json.dumps(paper_summaries, indent=2)}

Create a hierarchical structure for these {len(batch)} papers that would support writing a literature review addressing the research question."""

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": CLUSTER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )

                result = json.loads(response.choices[0].message.content)
                clusters = self._parse_clusters(result.get("clusters", []))

                # Initialize theme tracking from first batch
                for cluster in clusters:
                    theme_papers[cluster.name] = list(cluster.paper_ids)
                    theme_descriptions[cluster.name] = cluster.description
                    for sub in cluster.subclusters:
                        sub_name = f"{cluster.name} > {sub.name}"
                        theme_papers[sub_name] = list(sub.paper_ids)
                        theme_descriptions[sub_name] = sub.description

            else:
                # Subsequent batches: assign to existing themes or create new ones
                existing_themes = [
                    {"name": name, "description": desc} for name, desc in theme_descriptions.items()
                ]

                user_prompt = f"""Research Question: {research_question}

Existing themes:
{json.dumps(existing_themes, indent=2)}

New papers to organize:
{json.dumps(paper_summaries, indent=2)}

Assign each paper to the most appropriate existing theme, or create new themes if papers don't fit existing ones."""

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": BATCH_CLUSTER_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )

                result = json.loads(response.choices[0].message.content)
                batch_clusters = result.get("clusters", [])

                # Merge batch results into accumulated themes
                for cluster in batch_clusters:
                    name = cluster.get("name", "")
                    desc = cluster.get("description", "")
                    pids = cluster.get("paper_ids", [])

                    if name in theme_papers:
                        # Add to existing theme
                        theme_papers[name].extend(pids)
                        # Update description if refined
                        if desc:
                            theme_descriptions[name] = desc
                    else:
                        # New theme created
                        theme_papers[name] = list(pids)
                        theme_descriptions[name] = desc

        # Build final cluster structure
        final_clusters = []
        for name, pids in theme_papers.items():
            if " > " in name:
                # This is a subcluster, will be handled by parent
                continue
            final_clusters.append(
                Cluster(
                    id=str(uuid.uuid4())[:8],
                    name=name,
                    description=theme_descriptions.get(name, ""),
                    paper_ids=pids,
                    subclusters=[],
                )
            )

        # Add subclusters
        for name, pids in theme_papers.items():
            if " > " not in name:
                continue
            parent_name, sub_name = name.split(" > ", 1)
            for cluster in final_clusters:
                if cluster.name == parent_name:
                    cluster.subclusters.append(
                        Cluster(
                            id=str(uuid.uuid4())[:8],
                            name=sub_name,
                            description=theme_descriptions.get(name, ""),
                            paper_ids=pids,
                            subclusters=[],
                        )
                    )
                    break

        return final_clusters

    def _parse_clusters(self, raw_clusters: list[dict]) -> list[Cluster]:
        """Parse raw cluster dicts into Cluster models."""
        clusters = []
        for raw in raw_clusters:
            subclusters = []
            for sub in raw.get("subclusters", []):
                subclusters.append(
                    Cluster(
                        id=str(uuid.uuid4())[:8],
                        name=sub.get("name", "Unnamed"),
                        description=sub.get("description", ""),
                        paper_ids=sub.get("paper_ids", []),
                        subclusters=[],
                    )
                )

            clusters.append(
                Cluster(
                    id=str(uuid.uuid4())[:8],
                    name=raw.get("name", "Unnamed"),
                    description=raw.get("description", ""),
                    paper_ids=raw.get("paper_ids", []),
                    subclusters=subclusters,
                )
            )
        return clusters

    def recluster(
        self,
        papers: list[Paper],
        research_question: str,
        feedback: str,
        current_clusters: list[Cluster],
    ) -> list[Cluster]:
        """Recluster papers with user feedback."""
        if not papers:
            return []

        paper_summaries = []
        for paper in papers:
            summary = {
                "id": paper.id,
                "title": paper.title,
                "abstract": paper.abstract or "No abstract available",
            }
            paper_summaries.append(summary)

        current_structure = self._clusters_to_dict(current_clusters)

        user_prompt = f"""Research Question: {research_question}

Current structure:
{json.dumps(current_structure, indent=2)}

Papers:
{json.dumps(paper_summaries, indent=2)}

User feedback: {feedback}

Please reorganize the papers based on this feedback while maintaining the overall goal of supporting the literature review."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CLUSTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        result = json.loads(response.choices[0].message.content)
        return self._parse_clusters(result.get("clusters", []))

    def _clusters_to_dict(self, clusters: list[Cluster]) -> list[dict]:
        """Convert clusters to dict for serialization."""
        result = []
        for c in clusters:
            result.append(
                {
                    "name": c.name,
                    "description": c.description,
                    "paper_ids": c.paper_ids,
                    "subclusters": self._clusters_to_dict(c.subclusters),
                }
            )
        return result
