#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import asyncio
import os
import time
from datetime import datetime

from agents import Runner, function_tool
from ddgs import DDGS
from dotenv import load_dotenv
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
def _make_news_search_tool(
    collected_articles: list[dict],
    language_code: str = "us-en",
):
    """Create a news_search function tool that also records every article found.

    Parameters
    ----------
    collected_articles : list[dict]
        Mutable list where every raw article dict returned by DuckDuckGo
        will be appended, so that no URL is lost even if the LLM agent
        omits it from its structured output.
    """

    @function_tool
    def news_search(
        query: str,
        max_results: int = 10,
        timelimit: str = None,
    ) -> list[dict]:
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
                kwargs = {"query": query, "max_results": max_results}
                kwargs["region"] = language_code

                if timelimit:
                    kwargs["timelimit"] = timelimit
                news_results = list(ddgs.news(**kwargs))

            if not news_results:
                return []

            articles = [
                {
                    "title": article.get("title", "No title"),
                    "url": article.get("url", ""),
                    "date": article.get("date", ""),
                    "source": article.get("source", ""),
                    "body": article.get("body", ""),
                }
                for article in news_results
            ]

            # Record every article so the aggregation step never loses URLs, avoiding duplicates
            collected_articles.extend(articles)

            return articles
        except Exception as e:
            logger.error(f"News search error: {str(e)}")
            return []

    return news_search


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


class DeepResearchProvider(DataProvider):
    """Data provider that uses an agent-based approach to perform
    multi-step deep web research and return individual news articles
    with their URLs.

    The research process follows two steps:
    1. PLAN — Break the query into targeted sub-questions
    2. RESEARCH — Search the web for each sub-question independently (in parallel)

    URLs are deduplicated across all sub-queries and each unique URL is
    returned as a separate article entry.
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
        self,
        sub_query: str,
        after: str,
        before: str,
        collected_articles: list[dict] | None = None,
        language_code: str = "us-en",
    ) -> SubQueryResult:
        """Search the web for a single sub-question and return findings."""
        if collected_articles is None:
            collected_articles = []
        tool = _make_news_search_tool(collected_articles, language_code)
        agent = self._factory.create_agent(
            name="web_researcher",
            instructions=RESEARCH_PROMPT.format(after=after, before=before),
            tools=[tool],
            output_type=SubQueryResult,
        )

        result = Runner.run_sync(
            agent, input=sub_query, run_config=run_config_no_tracing
        )
        return result.final_output

    async def _research_sub_query_async(
        self,
        sub_query: str,
        after: str,
        before: str,
        collected_articles: list[dict] | None = None,
        language_code: str = "us-en",
    ) -> SubQueryResult:
        """Async version of _research_sub_query for parallel execution."""
        if collected_articles is None:
            collected_articles = []
        tool = _make_news_search_tool(collected_articles, language_code)
        agent = self._factory.create_agent(
            name="web_researcher",
            instructions=RESEARCH_PROMPT.format(after=after, before=before),
            tools=[tool],
            output_type=SubQueryResult,
        )

        result = await Runner.run(
            agent, input=sub_query, run_config=run_config_no_tracing
        )
        return result.final_output

    async def _research_all_async(
        self,
        sub_queries: list[str],
        after: str,
        before: str,
        collected_articles: list[dict] | None = None,
        language_code: str = "us-en",
    ) -> list[SubQueryResult]:
        """Research all sub-queries concurrently and return successful results."""
        if collected_articles is None:
            collected_articles = []
        tasks = [
            self._research_sub_query_async(
                sq, after, before, collected_articles, language_code
            )
            for sq in sub_queries
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

        The process follows two steps:
        1. PLAN — Break the query into targeted sub-questions
        2. RESEARCH — Search the web for each sub-question (in parallel)

        URLs are deduplicated across all sub-queries and each unique URL is
        returned as a separate article entry.

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
            Language hint (unused, kept for interface compatibility).

        Returns
        -------
        list[dict]
            A list of article dicts with keys: title, text, summary, url,
            link, timestamp.
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
        """Execute the full PLAN → RESEARCH → AGGREGATE pipeline.

        Returns a list of article dicts with deduplicated URLs from all
        sub-query findings.
        """
        language_code = "fr-fr" if language == "fr" else "us-en"

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

        # collected_articles accumulates every raw article dict returned by
        # the news_search tool, so that no URL is lost even if the LLM agent
        # omits it from its structured SubQueryResult.source_urls.
        collected_articles: list[dict] = []

        if use_async:
            logger.info(
                f"[RESEARCH] Searching {len(sub_queries)} sub-queries in parallel..."
            )
            findings = asyncio.run(
                self._research_all_async(
                    sub_queries, after, before, collected_articles, language_code
                )
            )
        else:
            logger.info(
                f"[RESEARCH] Searching {len(sub_queries)} sub-queries sequentially..."
            )
            findings: list[SubQueryResult] = []
            for j, sq in enumerate(sub_queries, 1):
                logger.info(f'[RESEARCH {j}/{len(sub_queries)}] Searching: "{sq}"')
                try:
                    sub_result = self._research_sub_query(
                        sq, after, before, collected_articles, language_code
                    )
                    findings.append(sub_result)
                    n_sources = len(sub_result.source_urls)
                    logger.info(f"  → Found {n_sources} source(s)")
                except Exception as e:
                    logger.warning(f"  → Sub-query {j} failed: {e} (skipping)")
                    continue

        if not findings and not collected_articles:
            logger.error("DeepResearchProvider: all sub-queries failed, no findings")
            return None

        # Step 3: AGGREGATE deduplicated URLs
        logger.info(
            f"[AGGREGATE] Deduplicating URLs from {len(findings)} sub-query results "
            f"and {len(collected_articles)} raw search result(s)..."
        )
        return self._aggregate_articles(findings, collected_articles)

    def _parse_entry(self, entry: dict) -> dict | None:
        """Not used — aggregation is handled by _aggregate_articles."""
        return entry

    def _aggregate_articles(
        self,
        findings: list[SubQueryResult],
        collected_articles: list[dict] | None = None,
    ) -> list[dict]:
        """Aggregate and deduplicate URLs from all sub-query findings and
        raw search results.

        Each unique URL becomes a separate article entry. The raw
        ``collected_articles`` (captured directly from the news_search tool)
        are the primary source of truth so that no URL is lost even when the
        LLM agent omits it from its structured output.

        Parameters
        ----------
        findings : list[SubQueryResult]
            Results from all sub-query research steps.
        collected_articles : list[dict] | None
            Raw article dicts captured directly from the news_search tool.

        Returns
        -------
        list[dict]
            A list of article dicts with keys: title, text, summary, url,
            link, timestamp.
        """
        articles = []
        seen_urls: set[str] = set()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # First, add every article captured directly from the search tool
        # (this is the authoritative source — no URL can be lost here).
        for raw in collected_articles or []:
            url = raw.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                # title = raw.get("title", "")
                # body = raw.get("body", "")
                text, title = self._get_text(url=url)
                if text:
                    articles.append(
                        {
                            "title": title,
                            "text": text,
                            "summary": text[:200] if text else "",
                            "url": url,
                            "link": url,
                            "timestamp": timestamp,
                        }
                    )

        # Then, add any URLs from the agent's structured output that the
        # search tool might not have returned (e.g. URLs the agent found
        # in page content).
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
                            "timestamp": timestamp,
                        }
                    )

        return articles
