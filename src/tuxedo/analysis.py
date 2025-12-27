"""LLM-based paper question analysis using OpenAI."""

import json
import time
import uuid
from typing import Callable

from openai import OpenAI

from tuxedo.logging import get_logger
from tuxedo.models import Paper, PaperAnswer

ANALYSIS_SYSTEM_PROMPT = """You are a systematic literature review assistant. Your task is to answer a specific question about an academic paper based on the provided content.

Provide a concise 1-2 sentence answer to the question. Be specific and factual based only on the content provided.

Respond ONLY with valid JSON in this exact format:
{
  "answer": "Your 1-2 sentence answer here",
  "confidence": "high" | "medium" | "low",
  "needs_more_context": true | false
}

Confidence levels:
- "high": The answer is clearly stated in the provided content
- "medium": The answer can be reasonably inferred from the content
- "low": The answer is uncertain or the content doesn't address the question well

Set "needs_more_context" to true if:
- The current content doesn't adequately address the question
- Additional paper sections might provide a better answer
- You're uncertain and more context would help

Set "needs_more_context" to false if:
- The answer is clear from the provided content
- You have high confidence in the answer
- Additional context is unlikely to change the answer significantly"""


class PaperAnalyzer:
    """Analyze papers to answer questions using OpenAI."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)  # Uses OPENAI_API_KEY env var if None
        self.model = model
        self.log = get_logger("analysis")

    def _call_api(self, system_prompt: str, user_prompt: str, context: str = "") -> dict:
        """Make an API call with logging.

        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt
            context: Optional context for logging

        Returns:
            Parsed JSON response as dict
        """
        ctx = f" ({context})" if context else ""
        self.log.info(f"Sending OpenAI API request{ctx}, model={self.model}")
        self.log.debug(f"Prompt length: {len(user_prompt)} chars")
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            elapsed = time.time() - start_time
            self.log.info(f"OpenAI API response received{ctx} in {elapsed:.2f}s")

            if response.usage:
                self.log.info(
                    f"Token usage: prompt={response.usage.prompt_tokens}, "
                    f"completion={response.usage.completion_tokens}, "
                    f"total={response.usage.total_tokens}"
                )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            self.log.error(
                f"OpenAI API call failed{ctx} after {elapsed:.2f}s: {type(e).__name__}: {e}"
            )
            raise

    def _get_section_content(self, paper: Paper, patterns: list[str]) -> dict[str, str]:
        """Get sections matching any of the patterns.

        Args:
            paper: The paper to extract sections from
            patterns: List of patterns to match (case-insensitive, substring)

        Returns:
            Dict of section_name -> content for matching sections
        """
        matched = {}
        for section_name, content in paper.sections.items():
            section_lower = section_name.lower()
            for pattern in patterns:
                if pattern.lower() in section_lower:
                    # Truncate long sections
                    matched[section_name] = content[:3000]
                    break
        return matched

    def analyze_paper(
        self,
        paper: Paper,
        question: str,
        question_id: str,
        max_stages: int = 3,
    ) -> PaperAnswer:
        """Analyze a paper to answer a question using progressive content extraction.

        Stage 1: title + abstract
        Stage 2: + conclusion (if stage 1 unclear)
        Stage 3: + method (if still unclear)

        Args:
            paper: The paper to analyze
            question: The question to answer
            question_id: ID of the question (for the PaperAnswer)
            max_stages: Maximum number of stages to try (1-3)

        Returns:
            PaperAnswer with the answer and metadata
        """
        self.log.info(f"Analyzing paper '{paper.title[:50]}...' for question")
        sections_used = []

        # Stage 1: Title + Abstract
        content_parts = [f"Title: {paper.title}"]
        sections_used.append("title")

        if paper.abstract:
            content_parts.append(f"Abstract: {paper.abstract}")
            sections_used.append("abstract")

        if paper.keywords:
            content_parts.append(f"Keywords: {', '.join(paper.keywords[:10])}")

        user_prompt = self._build_prompt(question, content_parts)
        result = self._call_api(
            ANALYSIS_SYSTEM_PROMPT,
            user_prompt,
            f"stage 1: {paper.id}",
        )

        # Check if more context needed and we have more stages
        if result.get("needs_more_context", False) and max_stages >= 2:
            # Stage 2: Add conclusion
            conclusion_sections = self._get_section_content(
                paper, ["conclusion", "summary", "discussion"]
            )
            if conclusion_sections:
                for name, content in conclusion_sections.items():
                    content_parts.append(f"{name}: {content}")
                    sections_used.append(name.lower())

                user_prompt = self._build_prompt(question, content_parts)
                result = self._call_api(
                    ANALYSIS_SYSTEM_PROMPT,
                    user_prompt,
                    f"stage 2: {paper.id}",
                )

        # Check if still needs more context
        if result.get("needs_more_context", False) and max_stages >= 3:
            # Stage 3: Add methodology
            method_sections = self._get_section_content(
                paper, ["method", "approach", "design", "experiment"]
            )
            if method_sections:
                for name, content in method_sections.items():
                    if name.lower() not in sections_used:
                        content_parts.append(f"{name}: {content}")
                        sections_used.append(name.lower())

                user_prompt = self._build_prompt(question, content_parts)
                result = self._call_api(
                    ANALYSIS_SYSTEM_PROMPT,
                    user_prompt,
                    f"stage 3: {paper.id}",
                )

        return PaperAnswer(
            id=str(uuid.uuid4())[:8],
            question_id=question_id,
            paper_id=paper.id,
            answer=result.get("answer", "Unable to determine an answer."),
            sections_used=sections_used,
            confidence=result.get("confidence", "low"),
        )

    def _build_prompt(self, question: str, content_parts: list[str]) -> str:
        """Build the user prompt from question and content."""
        content = "\n\n".join(content_parts)
        return f"""Question: {question}

Paper content:
{content}

Based on the paper content above, answer the question."""

    def analyze_papers(
        self,
        papers: list[Paper],
        question: str,
        question_id: str,
        progress_callback: Callable[[int, int, str], None] | None = None,
        max_stages: int = 3,
    ) -> list[PaperAnswer]:
        """Analyze multiple papers to answer a question.

        Args:
            papers: List of papers to analyze
            question: The question to answer
            question_id: ID of the question
            progress_callback: Optional callback(current, total, message) for progress
            max_stages: Maximum number of stages per paper (1-3)

        Returns:
            List of PaperAnswer objects
        """
        if not papers:
            return []

        self.log.info(f"Analyzing {len(papers)} papers for question: {question[:50]}...")
        answers = []

        for i, paper in enumerate(papers, 1):
            if progress_callback:
                progress_callback(
                    i,
                    len(papers),
                    f"Analyzing paper {i}/{len(papers)}: {paper.display_title}",
                )

            try:
                answer = self.analyze_paper(paper, question, question_id, max_stages)
                answers.append(answer)
            except Exception as e:
                self.log.error(f"Failed to analyze paper {paper.id}: {e}")
                # Create a fallback answer for failed papers
                answers.append(
                    PaperAnswer(
                        id=str(uuid.uuid4())[:8],
                        question_id=question_id,
                        paper_id=paper.id,
                        answer=f"Analysis failed: {e}",
                        sections_used=[],
                        confidence="low",
                    )
                )

        self.log.info(f"Completed analysis of {len(answers)} papers")
        return answers
