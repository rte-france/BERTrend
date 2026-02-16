#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import asyncio
import os
import time
from datetime import datetime

from agents import Runner, function_tool
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from loguru import logger
from pydantic import BaseModel, Field

from bertrend.bertrend_apps.data_provider.data_provider import DataProvider
from bertrend.bertrend_apps.data_provider.utils import wait
from bertrend.llm_utils.agent_utils import BaseAgentFactory, run_config_no_tracing

load_dotenv(override=True)

DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1-mini")
DEFAULT_MAX_SUB_QUERIES = 5
DEFAULT_SEARCH_DELAY = 3.0  # seconds between DuckDuckGo requests to avoid rate limiting


# ----------------------------------------------------------------------------
# Custom web search (to ensure it works both with openAI and
# and openAI compatible (ex. LiteLLM) environments)
# ----------------------------------------------------------------------------
@function_tool
def news_search(query: str, max_results: int = 10, timelimit: str = None) -> list[dict]:
    """Search for recent news articles.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
        timelimit: Time limit for results (e.g., 'd' for day, 'w' for week, 'm' for month).

    Returns:
        List of article dicts with title, url, date, source, body
    """
    try:
        time.sleep(DEFAULT_SEARCH_DELAY)
        with DDGS() as ddgs:
            kwargs = {"keywords": query, "max_results": max_results}
            if timelimit:
                kwargs["timelimit"] = timelimit
            news_results = list(ddgs.news(**kwargs))

        if not news_results:
            return []

        # Return structured data instead of formatted text
        return [
            {
                "title": article.get("title", "No title"),
                "url": article.get("url", ""),
                "date": article.get("date", ""),
                "source": article.get("source", ""),
                "body": article.get("body", ""),
            }
            for article in news_results
        ]
    except Exception as e:
        logger.error(f"News search error: {str(e)}")
        return []


# ---------------------------------------------------------------------------
# Pydantic models for structured agent outputs
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Output of the planning step: a list of targeted sub-questions."""

    sub_queries: list[str] = Field(
        description="List of specific, targeted sub-questions to research"
    )


class SubQueryResult(BaseModel):
    """Output of a single research sub-query."""

    sub_query: str = Field(description="The sub-question that was researched")
    findings: str = Field(
        description="Detailed findings from web research on this sub-question"
    )
    source_urls: list[str] = Field(
        default_factory=list,
        description="URLs of sources consulted for this sub-query",
    )


class ResearchReport(BaseModel):
    """Output of the final synthesis step."""

    title: str = Field(description="A clear, descriptive title for the research report")
    text: str = Field(
        description="The full synthesized research report with structured sections"
    )
    summary: str = Field(description="A concise 2-3 sentence summary of key findings")
    source_urls: list[str] = Field(
        default_factory=list,
        description="All source URLs consulted across all sub-queries",
    )
    timestamp: str = Field(description="The current date in YYYY-MM-DD HH:MM:SS format")


# ---------------------------------------------------------------------------
# System prompts for each step
# ---------------------------------------------------------------------------

PLAN_PROMPT = """You are a research planning expert. Given a topic and date range, break it down into specific, targeted sub-questions that together will provide comprehensive coverage of the topic.

Each sub-question should:
- Be specific enough to yield focused search results
- Cover a different angle or aspect of the topic
- Be answerable through web research

Decide how many sub-questions are needed based on the complexity and breadth of the topic. Use fewer sub-questions for narrow or simple topics and more for broad or complex ones. You must return between 1 and {max_sub_queries} sub-questions (inclusive)."""

RESEARCH_PROMPT = """You are an Advanced Web Research Analyst. Your task is to investigate a specific user query, synthesize findings from high-quality sources, and provide a curated list of references.

**Search Constraints & Source Selection:**
1.  **Accessibility First:** Do NOT cite sources that are behind strict paywalls, require user registration, or are likely to be broken (404).
2.  **Quality Control:** Prioritize primary sources, official documentation, academic institutions, and reputable journalism. Strictly exclude SEO-spam, content farms, clickbait, and low-quality aggregators.
3.  **Information Utility (No Redundancy):** Only include a new URL if it provides *unique* value (e.g., a distinct statistic, a counter-argument, or a primary account) not found in the other sources. Do not list multiple sources that merely regurgitate the exact same AP wire or press release.

**Research Guidelines:**
- **Scope:** Focus on information from the date range: {after} to {before}.
- **Depth:** Look for specific data points, hard statistics, exact dates, and key stakeholder names.
- **Nuance:** actively seek out and highlight conflicting information or distinct viewpoints between sources.

**Output Format:**
1.  **Executive Summary:** A concise answer to the query.
2.  **Curated Source List:**
    - Format: [Title](URL)
    - *Requirement:* For each URL you consulted, add a one-line note explaining *what unique information* this specific link contributes (e.g., "Contains the 2024 financial table" or "Provides the opposing legal argument").

"""


SYNTHESIZE_PROMPT = """You are a senior research analyst. Your task is to synthesize multiple research findings into a single, comprehensive, well-structured report.

Instructions:
- Combine all sub-query findings into a coherent narrative
- Organize by themes, not by sub-query
- Include specific data points and statistics from the findings
- Note areas of consensus and any conflicting information
- Write in a professional, analytical tone
- Structure the report with clear sections
- The timestamp should be: {timestamp}
{language_instruction}"""


class DeepResearchProvider(DataProvider):
    """Data provider that uses an agent-based approach to perform
    multi-step deep web research and return individual news articles
    with their URLs, as well as a synthesized research report.

    The research process follows three steps:
    1. PLAN — Break the query into targeted sub-questions
    2. RESEARCH — Search the web for each sub-question independently (in parallel)
    3. SYNTHESIZE — Combine all findings into a coherent report

    Each individual article discovered during research is returned as a
    separate entry, preserving all source URLs.
    """

    def __init__(
        self,
        model: str = None,
        max_sub_queries: int = DEFAULT_MAX_SUB_QUERIES,
        parallel_research: bool = True,
    ):
        # Do NOT call super().__init__() — we don't need Goose3 article parser
        # since text is generated by the agent, not scraped from URLs.
        self.model = model or DEFAULT_MODEL
        self.max_sub_queries = max_sub_queries
        self.parallel_research = parallel_research
        self._factory = BaseAgentFactory(model_name=self.model)

    # ------------------------------------------------------------------
    # Step 1: PLAN
    # ------------------------------------------------------------------

    def _plan(
        self, query: str, after: str, before: str, language: str = None
    ) -> list[str]:
        """Break the research query into targeted sub-questions."""
        agent = self._factory.create_agent(
            name="research_planner",
            instructions=PLAN_PROMPT.format(max_sub_queries=self.max_sub_queries),
            output_type=ResearchPlan,
        )

        prompt = (
            f"Topic: {query}\n"
            f"Date range: {after} to {before}\n"
            f"Current date: {datetime.now().strftime('%Y-%m-%d')}"
        )
        if language:
            prompt += f"\nGenerate sub-questions in {language}."

        result = Runner.run_sync(agent, input=prompt, run_config=run_config_no_tracing)
        plan: ResearchPlan = result.final_output
        return plan.sub_queries

    # ------------------------------------------------------------------
    # Step 2: RESEARCH each sub-question
    # ------------------------------------------------------------------

    def _research_sub_query(
        self, sub_query: str, after: str, before: str
    ) -> SubQueryResult:
        """Search the web for a single sub-question and return findings."""
        agent = self._factory.create_agent(
            name="web_researcher",
            instructions=RESEARCH_PROMPT.format(after=after, before=before),
            tools=[news_search],
            output_type=SubQueryResult,
        )

        result = Runner.run_sync(
            agent, input=sub_query, run_config=run_config_no_tracing
        )
        return result.final_output

    async def _research_sub_query_async(
        self, sub_query: str, after: str, before: str
    ) -> SubQueryResult:
        """Async version of _research_sub_query for parallel execution."""
        agent = self._factory.create_agent(
            name="web_researcher",
            instructions=RESEARCH_PROMPT.format(after=after, before=before),
            tools=[news_search],
            output_type=SubQueryResult,
        )

        result = await Runner.run(
            agent, input=sub_query, run_config=run_config_no_tracing
        )
        return result.final_output

    async def _research_all_async(
        self, sub_queries: list[str], after: str, before: str
    ) -> list[SubQueryResult]:
        """Research all sub-queries concurrently and return successful results."""
        tasks = [
            self._research_sub_query_async(sq, after, before) for sq in sub_queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        findings: list[SubQueryResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"  → Sub-query {i + 1} failed: {result} (skipping)")
            else:
                n_sources = len(result.source_urls)
                logger.info(
                    f"  → Sub-query {i + 1}/{len(sub_queries)} complete: "
                    f"{n_sources} source(s)"
                )
                findings.append(result)
        return findings

    # ------------------------------------------------------------------
    # Step 3: SYNTHESIZE all findings
    # ------------------------------------------------------------------

    def _synthesize(
        self, query: str, findings: list[SubQueryResult], language: str = None
    ) -> ResearchReport:
        """Combine all sub-query findings into a single coherent report."""
        language_instruction = f"Write the report in {language}." if language else ""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        agent = self._factory.create_agent(
            name="research_synthesizer",
            instructions=SYNTHESIZE_PROMPT.format(
                timestamp=timestamp,
                language_instruction=language_instruction,
            ),
            output_type=ResearchReport,
        )

        # Build the input with all findings
        findings_text = f"Original research topic: {query}\n\n"
        for i, f in enumerate(findings, 1):
            findings_text += f"--- Sub-query {i}: {f.sub_query} ---\n"
            findings_text += f"{f.findings}\n"
            if f.source_urls:
                findings_text += f"Sources: {', '.join(f.source_urls)}\n"
            findings_text += "\n"

        result = Runner.run_sync(
            agent, input=findings_text, run_config=run_config_no_tracing
        )
        return result.final_output

    # ------------------------------------------------------------------
    # Main entry point (DataProvider interface)
    # ------------------------------------------------------------------

    @wait(2)
    def get_articles(
        self,
        query: str,
        after: str,
        before: str,
        max_results: int,
        language: str = None,
    ) -> list[dict]:
        """Perform multi-step deep web research and return individual articles.

        The process follows three steps:
        1. PLAN — Break the query into targeted sub-questions
        2. RESEARCH — Search the web for each sub-question (in parallel)
        3. SYNTHESIZE — Combine all findings into a coherent report

        Each individual article discovered during research is returned as a
        separate entry. The synthesized report is also included as the first
        entry in the returned list.

        Parameters
        ----------
        query : str
            Keywords or topic to research.
        after : str
            Start date formatted as YYYY-MM-DD.
        before : str
            End date formatted as YYYY-MM-DD.
        max_results : int
            Maximum number of articles to return.
        language : str, optional
            Language hint for the report output.

        Returns
        -------
        list[dict]
            A list of article dicts with keys: title, text, summary, url,
            link, source_urls, timestamp.
        """
        logger.info(
            f"DeepResearchProvider: starting multi-step research for '{query}' "
            f"(date range: {after} to {before}, max_results: {max_results})"
        )
        articles = []
        try:
            result = self._run_research_pipeline(query, after, before, language)
            if result is not None:
                articles = result
        except Exception as e:
            logger.error(
                f"DeepResearchProvider: error in research pipeline "
                f"for query '{query}': {e}"
            )

        # Respect max_results
        articles = articles[:max_results]

        logger.info(
            f"DeepResearchProvider: completed with {len(articles)} article(s) "
            f"for query '{query}'"
        )
        return articles

    def _run_research_pipeline(
        self, query: str, after: str, before: str, language: str = None
    ) -> list[dict] | None:
        """Execute the full PLAN → RESEARCH → SYNTHESIZE pipeline.

        Returns a list of article dicts: one synthesized report followed by
        individual articles discovered during research.
        """

        # Step 1: PLAN
        logger.info("[PLAN] Breaking query into sub-questions...")
        sub_queries = self._plan(query, after, before, language)
        for j, sq in enumerate(sub_queries, 1):
            logger.info(f"  → Sub-question {j}/{len(sub_queries)}: {sq}")

        # Step 2: RESEARCH each sub-question
        use_async = self.parallel_research
        if use_async:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # Already inside an async context — fall back to sequential
                    use_async = False
            except RuntimeError:
                pass

        if use_async:
            logger.info(
                f"[RESEARCH] Searching {len(sub_queries)} sub-queries in parallel..."
            )
            findings = asyncio.run(self._research_all_async(sub_queries, after, before))
        else:
            logger.info(
                f"[RESEARCH] Searching {len(sub_queries)} sub-queries sequentially..."
            )
            findings: list[SubQueryResult] = []
            for j, sq in enumerate(sub_queries, 1):
                logger.info(f'[RESEARCH {j}/{len(sub_queries)}] Searching: "{sq}"')
                try:
                    sub_result = self._research_sub_query(sq, after, before)
                    findings.append(sub_result)
                    n_sources = len(sub_result.source_urls)
                    logger.info(f"  → Found {n_sources} source(s)")
                except Exception as e:
                    logger.warning(f"  → Sub-query {j} failed: {e} (skipping)")
                    continue

        if not findings:
            logger.error("DeepResearchProvider: all sub-queries failed, no findings")
            return None

        # Step 3: SYNTHESIZE
        logger.info(
            f"[SYNTHESIZE] Combining {len(findings)} sub-query results into final report..."
        )
        report = self._synthesize(query, findings, language)

        # Build the result list: synthesized report + individual articles
        return self._build_articles(report, findings)

    def _build_articles(
        self, report: ResearchReport, findings: list[SubQueryResult]
    ) -> list[dict]:
        """Build a list of article dicts from the report and individual findings.

        The synthesized report is the first entry, followed by individual
        articles (one per unique source URL found across all sub-queries).
        """
        articles = []

        # Add the synthesized report as the first article
        report_article = self._parse_entry(report)
        if report_article is not None:
            articles.append(report_article)

        # Collect individual articles from sub-query findings
        seen_urls = set()
        for finding in findings:
            for url in finding.source_urls:
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    articles.append(
                        {
                            "title": finding.sub_query,
                            "text": finding.findings,
                            "summary": finding.findings[:200],
                            "url": url,
                            "link": url,
                            "source_urls": finding.source_urls,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )

        return articles

    def _parse_entry(self, entry: ResearchReport) -> dict | None:
        """Convert a ResearchReport to the standard article dict format.

        Parameters
        ----------
        entry : ResearchReport
            The structured research report from the agent.

        Returns
        -------
        dict | None
            Article dict with keys: title, text, summary, url, link,
            source_urls, timestamp.
            Returns None if the entry is invalid.
        """
        try:
            primary_url = entry.source_urls[0] if entry.source_urls else ""
            timestamp = entry.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            return {
                "title": entry.title,
                "text": entry.text,
                "summary": entry.summary,
                "url": primary_url,
                "link": primary_url,
                "source_urls": entry.source_urls,
                "timestamp": timestamp,
            }
        except Exception as e:
            logger.error(f"DeepResearchProvider: error parsing report: {e}")
            return None
