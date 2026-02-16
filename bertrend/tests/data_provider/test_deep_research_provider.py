#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bertrend.bertrend_apps.data_provider.deep_research_provider import (
    DeepResearchProvider,
    ResearchPlan,
    SubQueryResult,
)

MODULE = "bertrend.bertrend_apps.data_provider.deep_research_provider"


def _make_plan(sub_queries: list[str] = None) -> ResearchPlan:
    """Helper to build a ResearchPlan."""
    return ResearchPlan(
        sub_queries=sub_queries
        or ["Sub-question 1?", "Sub-question 2?", "Sub-question 3?"]
    )


def _make_sub_result(
    sub_query: str = "Q?", findings: str = "Some findings."
) -> SubQueryResult:
    """Helper to build a SubQueryResult."""
    return SubQueryResult(
        sub_query=sub_query,
        findings=findings,
        source_urls=["https://example.com/src1"],
    )


def _make_runner_result(output) -> MagicMock:
    """Wrap any Pydantic model in a mock Runner result."""
    result = MagicMock()
    result.final_output = output
    return result


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestDeepResearchProviderInit:
    @patch(f"{MODULE}.BaseAgentFactory")
    @patch(f"{MODULE}.DEFAULT_MODEL", "gpt-4.1-mini")
    def test_init_default(self, mock_factory):
        """Verify default model and max_sub_queries."""
        provider = DeepResearchProvider(parallel_research=False)
        assert provider.model == "gpt-4.1-mini"
        assert provider.max_sub_queries == 5
        mock_factory.assert_called_once_with(model_name="gpt-4.1-mini")

    @patch(f"{MODULE}.BaseAgentFactory")
    def test_init_custom(self, mock_factory):
        """Verify custom params."""
        provider = DeepResearchProvider(model="gpt-4o", max_sub_queries=3)
        assert provider.model == "gpt-4o"
        assert provider.max_sub_queries == 3
        mock_factory.assert_called_once_with(model_name="gpt-4o")

    @patch(f"{MODULE}.BaseAgentFactory")
    def test_factory_cached(self, mock_factory):
        """Verify factory is created once and cached."""
        provider = DeepResearchProvider(parallel_research=False)
        assert provider._factory is mock_factory.return_value


# ---------------------------------------------------------------------------
# Individual pipeline steps
# ---------------------------------------------------------------------------


class TestPlanStep:
    @patch(f"{MODULE}.Runner.run_sync")
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_plan_returns_sub_queries(self, mock_factory, mock_run_sync):
        """Verify _plan returns a list of sub-queries from the planner agent."""
        plan = _make_plan(["Q1?", "Q2?"])
        mock_run_sync.return_value = _make_runner_result(plan)

        provider = DeepResearchProvider(max_sub_queries=2, parallel_research=False)
        result = provider._plan("test topic", "2025-01-01", "2025-01-31")

        assert result == ["Q1?", "Q2?"]
        mock_run_sync.assert_called_once()

    @patch(f"{MODULE}.Runner.run_sync")
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_plan_includes_language(self, mock_factory, mock_run_sync):
        """Verify language is passed in the prompt when provided."""
        plan = _make_plan(["Q1?"])
        mock_run_sync.return_value = _make_runner_result(plan)

        provider = DeepResearchProvider(max_sub_queries=1, parallel_research=False)
        provider._plan("topic", "2025-01-01", "2025-01-31", language="fr")

        call_kwargs = mock_run_sync.call_args
        prompt = call_kwargs[1].get("input") or call_kwargs[0][1]
        assert "fr" in prompt


class TestResearchStep:
    @patch(f"{MODULE}.Runner.run_sync")
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_research_sub_query(self, mock_factory, mock_run_sync):
        """Verify _research_sub_query returns a SubQueryResult."""
        sub_result = _make_sub_result("What is X?", "X is a thing.")
        mock_run_sync.return_value = _make_runner_result(sub_result)

        provider = DeepResearchProvider(parallel_research=False)
        result = provider._research_sub_query("What is X?", "2025-01-01", "2025-01-31")

        assert result.sub_query == "What is X?"
        assert result.findings == "X is a thing."
        assert len(result.source_urls) == 1


# ---------------------------------------------------------------------------
# Full pipeline via get_articles
# ---------------------------------------------------------------------------


class TestGetArticles:
    @patch("bertrend.bertrend_apps.data_provider.utils.time.sleep")
    @patch(f"{MODULE}.Runner.run_sync")
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_get_articles_success(self, mock_factory, mock_run_sync, _mock_sleep):
        """Full pipeline: plan → research → aggregate. Verify article dicts."""
        plan = _make_plan(["Q1?", "Q2?"])
        sub1 = _make_sub_result("Q1?", "Finding 1")
        sub2 = _make_sub_result("Q2?", "Finding 2")

        # Calls: 1=plan, 2=research Q1, 3=research Q2 (no synthesize)
        mock_run_sync.side_effect = [
            _make_runner_result(plan),
            _make_runner_result(sub1),
            _make_runner_result(sub2),
        ]

        provider = DeepResearchProvider(max_sub_queries=2, parallel_research=False)
        articles = provider.get_articles(
            query="test", after="2025-01-01", before="2025-01-31", max_results=10
        )

        # Both sub-queries have the same URL, so only 1 deduplicated article
        assert len(articles) >= 1
        assert all("url" in a for a in articles)
        assert all("title" in a for a in articles)
        assert mock_run_sync.call_count == 3

    @patch("bertrend.bertrend_apps.data_provider.utils.time.sleep")
    @patch(f"{MODULE}.Runner.run_sync")
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_get_articles_returns_individual_articles(
        self, mock_factory, mock_run_sync, _mock_sleep
    ):
        """Verify individual articles from sub-queries are returned with their URLs."""
        plan = _make_plan(["Q1?"])
        sub1 = SubQueryResult(
            sub_query="Q1?",
            findings="Finding about Q1.",
            source_urls=["https://a.com/1", "https://b.com/2"],
        )

        mock_run_sync.side_effect = [
            _make_runner_result(plan),
            _make_runner_result(sub1),
        ]

        provider = DeepResearchProvider(max_sub_queries=1, parallel_research=False)
        articles = provider.get_articles(
            query="test", after="2025-01-01", before="2025-01-31", max_results=10
        )

        # 2 individual articles (unique URLs)
        assert len(articles) == 2
        urls = [a["url"] for a in articles]
        assert "https://a.com/1" in urls
        assert "https://b.com/2" in urls

    @patch("bertrend.bertrend_apps.data_provider.utils.time.sleep")
    @patch(f"{MODULE}.Runner.run_sync")
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_get_articles_respects_max_results(
        self, mock_factory, mock_run_sync, _mock_sleep
    ):
        """Verify max_results limits the number of returned articles."""
        plan = _make_plan(["Q1?"])
        sub1 = SubQueryResult(
            sub_query="Q1?",
            findings="Finding.",
            source_urls=["https://a.com/1", "https://b.com/2", "https://c.com/3"],
        )

        mock_run_sync.side_effect = [
            _make_runner_result(plan),
            _make_runner_result(sub1),
        ]

        provider = DeepResearchProvider(max_sub_queries=1, parallel_research=False)
        articles = provider.get_articles(
            query="test", after="2025-01-01", before="2025-01-31", max_results=2
        )

        assert len(articles) == 2

    @patch("bertrend.bertrend_apps.data_provider.utils.time.sleep")
    @patch(f"{MODULE}.Runner.run_sync")
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_get_articles_deduplicates_urls(
        self, mock_factory, mock_run_sync, _mock_sleep
    ):
        """Verify duplicate URLs across sub-queries are deduplicated."""
        plan = _make_plan(["Q1?", "Q2?"])
        sub1 = SubQueryResult(
            sub_query="Q1?",
            findings="Finding 1.",
            source_urls=["https://shared.com/1", "https://a.com/2"],
        )
        sub2 = SubQueryResult(
            sub_query="Q2?",
            findings="Finding 2.",
            source_urls=["https://shared.com/1", "https://b.com/3"],
        )

        mock_run_sync.side_effect = [
            _make_runner_result(plan),
            _make_runner_result(sub1),
            _make_runner_result(sub2),
        ]

        provider = DeepResearchProvider(max_sub_queries=2, parallel_research=False)
        articles = provider.get_articles(
            query="test", after="2025-01-01", before="2025-01-31", max_results=20
        )

        # 3 unique URLs: shared.com/1, a.com/2, b.com/3
        all_urls = [a["url"] for a in articles]
        assert all_urls.count("https://shared.com/1") == 1
        assert len(articles) == 3

    @patch("bertrend.bertrend_apps.data_provider.utils.time.sleep")
    @patch(f"{MODULE}.Runner.run_sync")
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_get_articles_plan_failure(self, mock_factory, mock_run_sync, _mock_sleep):
        """If the plan step fails, get_articles returns empty list."""
        mock_run_sync.side_effect = RuntimeError("Plan failed")

        provider = DeepResearchProvider(parallel_research=False)
        articles = provider.get_articles(
            query="test", after="2025-01-01", before="2025-01-31", max_results=1
        )

        assert articles == []

    @patch("bertrend.bertrend_apps.data_provider.utils.time.sleep")
    @patch(f"{MODULE}.Runner.run_sync")
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_get_articles_partial_research_failure(
        self, mock_factory, mock_run_sync, _mock_sleep
    ):
        """If some sub-queries fail, aggregation still runs with available findings."""
        plan = _make_plan(["Q1?", "Q2?", "Q3?"])
        sub1 = _make_sub_result("Q1?", "Finding 1")
        sub3 = _make_sub_result("Q3?", "Finding 3")

        # Q2 research fails
        mock_run_sync.side_effect = [
            _make_runner_result(plan),
            _make_runner_result(sub1),
            RuntimeError("Q2 search failed"),
            _make_runner_result(sub3),
        ]

        provider = DeepResearchProvider(max_sub_queries=3, parallel_research=False)
        articles = provider.get_articles(
            query="test", after="2025-01-01", before="2025-01-31", max_results=10
        )

        # sub1 and sub3 each have 1 URL (same default), so 1 deduplicated article
        assert len(articles) >= 1
        assert articles[0]["url"] == "https://example.com/src1"

    @patch("bertrend.bertrend_apps.data_provider.utils.time.sleep")
    @patch(f"{MODULE}.Runner.run_sync")
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_get_articles_all_research_fails(
        self, mock_factory, mock_run_sync, _mock_sleep
    ):
        """If ALL sub-queries fail, pipeline returns None (no article)."""
        plan = _make_plan(["Q1?", "Q2?"])

        mock_run_sync.side_effect = [
            _make_runner_result(plan),
            RuntimeError("Q1 failed"),
            RuntimeError("Q2 failed"),
        ]

        provider = DeepResearchProvider(max_sub_queries=2, parallel_research=False)
        articles = provider.get_articles(
            query="test", after="2025-01-01", before="2025-01-31", max_results=1
        )

        assert articles == []

    @patch("bertrend.bertrend_apps.data_provider.utils.time.sleep")
    @patch(f"{MODULE}.Runner.run_sync")
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_get_articles_no_source_urls_key(
        self, mock_factory, mock_run_sync, _mock_sleep
    ):
        """Articles should not contain source_urls key (only url/link)."""
        plan = _make_plan(["Q1?"])
        sub1 = _make_sub_result("Q1?", "Finding 1")

        mock_run_sync.side_effect = [
            _make_runner_result(plan),
            _make_runner_result(sub1),
        ]

        provider = DeepResearchProvider(max_sub_queries=1, parallel_research=False)
        articles = provider.get_articles(
            query="test", after="2025-01-01", before="2025-01-31", max_results=10
        )

        for article in articles:
            assert "source_urls" not in article
            assert "url" in article
            assert "link" in article
            assert "title" in article
            assert "text" in article
            assert "summary" in article
            assert "timestamp" in article


# ---------------------------------------------------------------------------
# _aggregate_articles
# ---------------------------------------------------------------------------


class TestAggregateArticles:
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_aggregate_articles_deduplicates(self, mock_factory):
        """Verify duplicate URLs across findings are deduplicated."""
        provider = DeepResearchProvider(parallel_research=False)
        findings = [
            SubQueryResult(
                sub_query="Q1?",
                findings="F1.",
                source_urls=["https://dup.com/1"],
            ),
            SubQueryResult(
                sub_query="Q2?",
                findings="F2.",
                source_urls=["https://dup.com/1"],
            ),
        ]

        articles = provider._aggregate_articles(findings)

        # 1 unique URL (deduplicated)
        assert len(articles) == 1
        assert articles[0]["url"] == "https://dup.com/1"

    @patch(f"{MODULE}.BaseAgentFactory")
    def test_aggregate_articles_multiple_urls(self, mock_factory):
        """Verify all unique URLs are returned."""
        provider = DeepResearchProvider(parallel_research=False)
        findings = [
            SubQueryResult(
                sub_query="Q1?",
                findings="Finding 1.",
                source_urls=["https://example.com/1"],
            ),
            SubQueryResult(
                sub_query="Q2?",
                findings="Finding 2.",
                source_urls=["https://example.com/2", "https://example.com/3"],
            ),
        ]

        articles = provider._aggregate_articles(findings)

        assert len(articles) == 3
        urls = {a["url"] for a in articles}
        assert urls == {
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        }

    @patch(f"{MODULE}.BaseAgentFactory")
    def test_aggregate_articles_format_from_collected(self, mock_factory):
        """Verify article format when using collected_articles (primary source)."""
        provider = DeepResearchProvider(parallel_research=False)
        collected = [
            {
                "title": "News Title",
                "url": "https://example.com/1",
                "body": "Some detailed findings here.",
                "date": "2025-01-15",
                "source": "example.com",
            },
        ]

        articles = provider._aggregate_articles([], collected_articles=collected)

        assert len(articles) == 1
        article = articles[0]
        assert article["title"] == "News Title"
        assert article["text"] == "Some detailed findings here."
        assert article["summary"] == "Some detailed findings here."
        assert article["url"] == "https://example.com/1"
        assert article["link"] == "https://example.com/1"
        assert "timestamp" in article
        assert "source_urls" not in article

    @patch(f"{MODULE}.BaseAgentFactory")
    def test_aggregate_articles_format_from_findings(self, mock_factory):
        """Verify article format when using agent source_urls (fallback)."""
        provider = DeepResearchProvider(parallel_research=False)
        findings = [
            SubQueryResult(
                sub_query="Q1?",
                findings="Some detailed findings here.",
                source_urls=["https://example.com/1"],
            ),
        ]

        articles = provider._aggregate_articles(findings)

        assert len(articles) == 1
        article = articles[0]
        assert article["title"] == "Q1?"
        assert article["text"] == "Some detailed findings here."
        assert article["summary"] == "Some detailed findings here."
        assert article["url"] == "https://example.com/1"
        assert article["link"] == "https://example.com/1"
        assert "timestamp" in article
        assert "source_urls" not in article

    @patch(f"{MODULE}.BaseAgentFactory")
    def test_aggregate_articles_collected_takes_priority(self, mock_factory):
        """Collected articles (from tool) take priority over agent source_urls."""
        provider = DeepResearchProvider(parallel_research=False)
        collected = [
            {"title": "Tool Title", "url": "https://shared.com/1", "body": "Tool body."},
        ]
        findings = [
            SubQueryResult(
                sub_query="Q1?",
                findings="Agent findings.",
                source_urls=["https://shared.com/1", "https://extra.com/2"],
            ),
        ]

        articles = provider._aggregate_articles(findings, collected_articles=collected)

        # shared URL uses tool data; extra URL uses agent data
        assert len(articles) == 2
        by_url = {a["url"]: a for a in articles}
        assert by_url["https://shared.com/1"]["title"] == "Tool Title"
        assert by_url["https://extra.com/2"]["title"] == "Q1?"

    @patch(f"{MODULE}.BaseAgentFactory")
    def test_aggregate_articles_empty_findings(self, mock_factory):
        """Empty findings list returns empty articles."""
        provider = DeepResearchProvider(parallel_research=False)
        articles = provider._aggregate_articles([])
        assert articles == []
