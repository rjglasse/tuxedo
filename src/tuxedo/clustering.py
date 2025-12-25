"""LLM-based paper clustering using OpenAI."""

import json
import uuid

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
    ) -> list[Cluster]:
        """Cluster papers based on research question.

        Args:
            papers: List of papers to cluster
            research_question: The research question/prompt for clustering
            include_sections: Optional list of section name patterns to include
                             (e.g., ["method", "methodology"] to include method sections)
        """
        if not papers:
            return []

        # Build paper summaries for the prompt
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
