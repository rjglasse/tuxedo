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

AUTO_CLUSTER_SYSTEM_PROMPT = """You are a systematic literature review assistant. Your task is to analyze a collection of academic papers and discover the natural themes, topics, and patterns that emerge from them.

Without a predetermined research question, analyze the papers and:
1. Identify the main themes, topics, or research areas represented
2. Group papers by shared concepts, methodologies, or findings
3. Create meaningful cluster names that capture the essence of each group
4. Provide descriptions explaining what unifies papers in each cluster
5. Organize subclusters where appropriate (max 2 levels deep)

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
- Only create subclusters if there's meaningful differentiation
- Focus on discovering what themes naturally emerge from the papers"""

# Predefined auto-discovery modes
AUTO_DISCOVERY_PROMPTS = {
    "themes": "Discover the main themes and topics in these papers",
    "methodology": "Group papers by their research methodology (qualitative, quantitative, mixed methods, theoretical, etc.)",
    "domain": "Group papers by their application domain or field of study",
    "temporal": "Group papers by how ideas and approaches evolved over time",
    "findings": "Group papers by their key findings or conclusions",
}

GUIDED_CLUSTER_SYSTEM_PROMPT = """You are a systematic literature review assistant. Your task is to organize academic papers into researcher-specified categories.

You are given a list of categories that the researcher wants to use. Your job is to:
1. Assign each paper to the most appropriate category
2. Provide descriptions for each category based on the papers assigned
3. {new_category_instruction}

Respond ONLY with valid JSON in this exact format:
{{
  "clusters": [
    {{
      "name": "Category Name",
      "description": "Brief description based on papers in this category",
      "paper_ids": ["id1", "id2"],
      "subclusters": []
    }}
  ]
}}

Important:
- Every paper must appear in exactly one cluster
- Use the exact category names provided by the researcher
- {new_category_rule}
- If a paper could fit multiple categories, choose the best fit
- Category descriptions should reflect the actual papers assigned"""

GUIDED_ALLOW_NEW = "If papers don't fit any provided category well, you may create new categories"
GUIDED_STRICT = "Do NOT create new categories - assign every paper to one of the provided categories, even if the fit isn't perfect"

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
        auto_mode: str | None = None,
        categories: list[str | dict] | None = None,
        allow_new_categories: bool = True,
    ) -> list[Cluster]:
        """Cluster papers based on research question, auto-discovery, or guided categories.

        Args:
            papers: List of papers to cluster
            research_question: The research question/prompt for clustering
            include_sections: Optional list of section name patterns to include
                             (e.g., ["method", "methodology"] to include method sections)
            batch_size: If set, process papers in batches to handle token limits.
                       Themes are developed incrementally across batches.
            progress_callback: Optional callback(batch_num, total_batches, message) for progress
            auto_mode: If set, use auto-discovery mode. Values: "themes", "methodology",
                      "domain", "temporal", "findings", or any custom focus string.
            categories: If set, use guided clustering with these predefined categories.
                       Can be a list of strings or dicts with "name" and optional "description".
            allow_new_categories: If True (default), AI can create new categories for
                                 papers that don't fit. If False, strict assignment only.
        """
        if not papers:
            return []

        # If batch_size is set and we have more papers than the batch size, use iterative mode
        if batch_size and len(papers) > batch_size:
            return self._cluster_papers_iterative(
                papers,
                research_question,
                include_sections,
                batch_size,
                progress_callback,
                auto_mode=auto_mode,
                categories=categories,
                allow_new_categories=allow_new_categories,
            )

        # Standard single-pass clustering
        paper_summaries = self._build_paper_summaries(papers, include_sections)

        # Choose system prompt and user prompt based on mode
        if categories:
            # Guided clustering with predefined categories
            if allow_new_categories:
                new_instruction = GUIDED_ALLOW_NEW
                new_rule = "You may create additional categories if needed"
            else:
                new_instruction = GUIDED_STRICT
                new_rule = "Only use the provided categories"

            system_prompt = GUIDED_CLUSTER_SYSTEM_PROMPT.format(
                new_category_instruction=new_instruction,
                new_category_rule=new_rule,
            )
            # Handle both simple strings and structured dicts
            categories_list = self._format_categories(categories)
            user_prompt = f"""Predefined Categories:
{categories_list}

Papers to organize:
{json.dumps(paper_summaries, indent=2)}

Assign each of these {len(papers)} papers to the most appropriate category."""
        elif auto_mode:
            system_prompt = AUTO_CLUSTER_SYSTEM_PROMPT
            # Get the discovery focus
            focus = AUTO_DISCOVERY_PROMPTS.get(auto_mode, auto_mode)
            user_prompt = f"""Discovery Focus: {focus}

Papers to analyze:
{json.dumps(paper_summaries, indent=2)}

Analyze these {len(papers)} papers and discover the natural themes and groupings that emerge. Create a hierarchical structure that captures the main topics and patterns you identify."""
        else:
            system_prompt = CLUSTER_SYSTEM_PROMPT
            user_prompt = f"""Research Question: {research_question}

Papers to organize:
{json.dumps(paper_summaries, indent=2)}

Create a hierarchical structure for these {len(papers)} papers that would support writing a literature review addressing the research question."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
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

    def _format_categories(self, categories: list[str | dict]) -> str:
        """Format categories for the prompt.

        Handles both simple strings and structured dicts with name/description.
        """
        lines = []
        for cat in categories:
            if isinstance(cat, str):
                lines.append(f"- {cat}")
            elif isinstance(cat, dict):
                name = cat.get("name", "")
                desc = cat.get("description", "")
                if desc:
                    lines.append(f"- {name}: {desc}")
                else:
                    lines.append(f"- {name}")
        return "\n".join(lines)

    def _get_category_names(self, categories: list[str | dict]) -> list[str]:
        """Extract category names from mixed list."""
        names = []
        for cat in categories:
            if isinstance(cat, str):
                names.append(cat)
            elif isinstance(cat, dict):
                names.append(cat.get("name", ""))
        return [n for n in names if n]

    def _cluster_papers_iterative(
        self,
        papers: list[Paper],
        research_question: str,
        include_sections: list[str] | None,
        batch_size: int,
        progress_callback: Callable[[int, int, str], None] | None,
        auto_mode: str | None = None,
        categories: list[str | dict] | None = None,
        allow_new_categories: bool = True,
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

        # For guided mode, initialize themes from categories
        if categories:
            for cat in categories:
                if isinstance(cat, str):
                    theme_papers[cat] = []
                    theme_descriptions[cat] = ""
                elif isinstance(cat, dict):
                    name = cat.get("name", "")
                    if name:
                        theme_papers[name] = []
                        theme_descriptions[name] = cat.get("description", "")

        for batch_num, batch in enumerate(batches, 1):
            if progress_callback:
                progress_callback(
                    batch_num, total_batches, f"Processing batch {batch_num}/{total_batches}"
                )

            paper_summaries = self._build_paper_summaries(batch, include_sections)

            if batch_num == 1 and not categories:
                # First batch without categories: use standard clustering to establish themes
                if auto_mode:
                    system_prompt = AUTO_CLUSTER_SYSTEM_PROMPT
                    focus = AUTO_DISCOVERY_PROMPTS.get(auto_mode, auto_mode)
                    user_prompt = f"""Discovery Focus: {focus}

Papers to analyze:
{json.dumps(paper_summaries, indent=2)}

Analyze these {len(batch)} papers and discover the natural themes and groupings that emerge. Create a hierarchical structure that captures the main topics and patterns you identify."""
                else:
                    system_prompt = CLUSTER_SYSTEM_PROMPT
                    user_prompt = f"""Research Question: {research_question}

Papers to organize:
{json.dumps(paper_summaries, indent=2)}

Create a hierarchical structure for these {len(batch)} papers that would support writing a literature review addressing the research question."""

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
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

            elif categories:
                # Guided mode: assign papers to predefined categories
                if allow_new_categories:
                    new_instruction = GUIDED_ALLOW_NEW
                    new_rule = "You may create additional categories if needed"
                else:
                    new_instruction = GUIDED_STRICT
                    new_rule = "Only use the provided categories"

                system_prompt = GUIDED_CLUSTER_SYSTEM_PROMPT.format(
                    new_category_instruction=new_instruction,
                    new_category_rule=new_rule,
                )
                existing_themes = [
                    {"name": name, "description": desc}
                    for name, desc in theme_descriptions.items()
                    if desc  # Only include categories that have descriptions
                ]
                categories_list = self._format_categories(categories)

                if existing_themes:
                    user_prompt = f"""Predefined Categories:
{categories_list}

Current category descriptions (from previous papers):
{json.dumps(existing_themes, indent=2)}

New papers to organize:
{json.dumps(paper_summaries, indent=2)}

Assign each paper to the most appropriate category."""
                else:
                    user_prompt = f"""Predefined Categories:
{categories_list}

Papers to organize:
{json.dumps(paper_summaries, indent=2)}

Assign each paper to the most appropriate category."""

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
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
                    elif allow_new_categories or not categories:
                        # New theme created (only if allowed)
                        theme_papers[name] = list(pids)
                        theme_descriptions[name] = desc

            else:
                # Subsequent batches in normal mode: assign to existing themes
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
