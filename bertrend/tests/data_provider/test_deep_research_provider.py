#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bertrend.bertrend_apps.data_provider.deep_research_provider import (
    DeepResearchProvider,
    ResearchPlan,
    ResearchReport,
    SubQueryResult,
)

MODULE = "bertrend.bertrend_apps.data_provider.deep_research_provider"


def _make_report(**overrides) -> ResearchReport:
    """Helper to build a ResearchReport with sensible defaults."""
    defaults = {
        "title": "Test Report",
        "text": "Full research text with multiple paragraphs.",
        "summary": "A concise summary of findings.",
        "source_urls": ["https://example.com/1", "https://example.com/2"],
        "timestamp": "2025-01-15 10:30:00",
    }
    defaults.update(overrides)
    return ResearchReport(**defaults)


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
        """Verify default model and num_sub_queries."""
        provider = DeepResearchProvider(parallel_research=False)
        assert provider.model == "gpt-4.1-mini"
        assert provider.num_sub_queries == 5
        mock_factory.assert_called_once_with(model_name="gpt-4.1-mini")

    @patch(f"{MODULE}.BaseAgentFactory")
    def test_init_custom(self, mock_factory):
        """Verify custom params."""
        provider = DeepResearchProvider(model="gpt-4o", num_sub_queries=3)
        assert provider.model == "gpt-4o"
        assert provider.num_sub_queries == 3
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

        provider = DeepResearchProvider(num_sub_queries=2, parallel_research=False)
        result = provider._plan("test topic", "2025-01-01", "2025-01-31")

        assert result == ["Q1?", "Q2?"]
        mock_run_sync.assert_called_once()

    @patch(f"{MODULE}.Runner.run_sync")
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_plan_includes_language(self, mock_factory, mock_run_sync):
        """Verify language is passed in the prompt when provided."""
        plan = _make_plan(["Q1?"])
        mock_run_sync.return_value = _make_runner_result(plan)

        provider = DeepResearchProvider(num_sub_queries=1, parallel_research=False)
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


class TestSynthesizeStep:
    @patch(f"{MODULE}.Runner.run_sync")
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_synthesize(self, mock_factory, mock_run_sync):
        """Verify _synthesize combines findings into a ResearchReport."""
        report = _make_report()
        mock_run_sync.return_value = _make_runner_result(report)

        provider = DeepResearchProvider(parallel_research=False)
        findings = [
            _make_sub_result("Q1?", "Finding 1"),
            _make_sub_result("Q2?", "Finding 2"),
        ]
        result = provider._synthesize("test topic", findings, language="fr")

        assert result.title == "Test Report"
        # Verify the input to the synthesizer contains all findings
        call_kwargs = mock_run_sync.call_args
        prompt = call_kwargs[1].get("input") or call_kwargs[0][1]
        assert "Finding 1" in prompt
        assert "Finding 2" in prompt


# ---------------------------------------------------------------------------
# Full pipeline via get_articles
# ---------------------------------------------------------------------------


class TestGetArticles:
    @patch("bertrend.bertrend_apps.data_provider.utils.time.sleep")
    @patch(f"{MODULE}.Runner.run_sync")
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_get_articles_success(self, mock_factory, mock_run_sync, _mock_sleep):
        """Full pipeline: plan → research → synthesize. Verify article dicts."""
        plan = _make_plan(["Q1?", "Q2?"])
        sub1 = _make_sub_result("Q1?", "Finding 1")
        sub2 = _make_sub_result("Q2?", "Finding 2")
        report = _make_report()

        # Calls: 1=plan, 2=research Q1, 3=research Q2, 4=synthesize
        mock_run_sync.side_effect = [
            _make_runner_result(plan),
            _make_runner_result(sub1),
            _make_runner_result(sub2),
            _make_runner_result(report),
        ]

        provider = DeepResearchProvider(num_sub_queries=2, parallel_research=False)
        articles = provider.get_articles(
            query="test", after="2025-01-01", before="2025-01-31", max_results=10
        )

        # Should have: 1 synthesized report + 2 individual articles (1 URL each, deduplicated)
        assert len(articles) >= 1
        # First article is the synthesized report
        assert articles[0]["title"] == "Test Report"
        assert articles[0]["source_urls"] == [
            "https://example.com/1",
            "https://example.com/2",
        ]
        assert mock_run_sync.call_count == 4

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
        report = _make_report(source_urls=["https://a.com/1", "https://b.com/2"])

        mock_run_sync.side_effect = [
            _make_runner_result(plan),
            _make_runner_result(sub1),
            _make_runner_result(report),
        ]

        provider = DeepResearchProvider(num_sub_queries=1, parallel_research=False)
        articles = provider.get_articles(
            query="test", after="2025-01-01", before="2025-01-31", max_results=10
        )

        # 1 report + 2 individual articles
        assert len(articles) == 3
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
        report = _make_report()

        mock_run_sync.side_effect = [
            _make_runner_result(plan),
            _make_runner_result(sub1),
            _make_runner_result(report),
        ]

        provider = DeepResearchProvider(num_sub_queries=1, parallel_research=False)
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
        report = _make_report(source_urls=["https://shared.com/1"])

        mock_run_sync.side_effect = [
            _make_runner_result(plan),
            _make_runner_result(sub1),
            _make_runner_result(sub2),
            _make_runner_result(report),
        ]

        provider = DeepResearchProvider(num_sub_queries=2, parallel_research=False)
        articles = provider.get_articles(
            query="test", after="2025-01-01", before="2025-01-31", max_results=20
        )

        # Count how many times shared URL appears (should be only once in individual articles)
        individual_urls = [a["url"] for a in articles[1:]]  # skip report
        assert individual_urls.count("https://shared.com/1") == 1

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
        """If some sub-queries fail, synthesis still runs with available findings."""
        plan = _make_plan(["Q1?", "Q2?", "Q3?"])
        sub1 = _make_sub_result("Q1?", "Finding 1")
        sub3 = _make_sub_result("Q3?", "Finding 3")
        report = _make_report()

        # Q2 research fails
        mock_run_sync.side_effect = [
            _make_runner_result(plan),
            _make_runner_result(sub1),
            RuntimeError("Q2 search failed"),
            _make_runner_result(sub3),
            _make_runner_result(report),
        ]

        provider = DeepResearchProvider(num_sub_queries=3, parallel_research=False)
        articles = provider.get_articles(
            query="test", after="2025-01-01", before="2025-01-31", max_results=10
        )

        assert len(articles) >= 1
        assert articles[0]["title"] == "Test Report"

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

        provider = DeepResearchProvider(num_sub_queries=2, parallel_research=False)
        articles = provider.get_articles(
            query="test", after="2025-01-01", before="2025-01-31", max_results=1
        )

        assert articles == []


# ---------------------------------------------------------------------------
# _parse_entry
# ---------------------------------------------------------------------------


class TestParseEntry:
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_parse_entry_valid(self, mock_factory):
        """Verify dict format from _parse_entry includes source_urls."""
        provider = DeepResearchProvider(parallel_research=False)
        report = _make_report()
        result = provider._parse_entry(report)

        assert result is not None
        assert result["title"] == "Test Report"
        assert result["text"] == "Full research text with multiple paragraphs."
        assert result["summary"] == "A concise summary of findings."
        assert result["url"] == "https://example.com/1"
        assert result["link"] == "https://example.com/1"
        assert result["source_urls"] == [
            "https://example.com/1",
            "https://example.com/2",
        ]
        assert result["timestamp"] == "2025-01-15 10:30:00"

    @patch(f"{MODULE}.BaseAgentFactory")
    def test_parse_entry_no_urls(self, mock_factory):
        """Empty source_urls → empty url/link strings, empty source_urls list."""
        provider = DeepResearchProvider(parallel_research=False)
        report = _make_report(source_urls=[])
        result = provider._parse_entry(report)

        assert result is not None
        assert result["url"] == ""
        assert result["link"] == ""
        assert result["source_urls"] == []

    @patch(f"{MODULE}.BaseAgentFactory")
    def test_parse_entry_none_input(self, mock_factory):
        """None input → None output."""
        provider = DeepResearchProvider(parallel_research=False)
        result = provider._parse_entry(None)
        assert result is None


# ---------------------------------------------------------------------------
# _build_articles
# ---------------------------------------------------------------------------


class TestBuildArticles:
    @patch(f"{MODULE}.BaseAgentFactory")
    def test_build_articles_includes_report_and_individuals(self, mock_factory):
        """Verify _build_articles returns report + individual articles."""
        provider = DeepResearchProvider(parallel_research=False)
        report = _make_report(
            source_urls=["https://example.com/1", "https://example.com/2"]
        )
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

        articles = provider._build_articles(report, findings)

        # 1 report + 3 unique individual URLs
        assert len(articles) == 4
        assert articles[0]["title"] == "Test Report"
        individual_urls = {a["url"] for a in articles[1:]}
        assert individual_urls == {
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        }

    @patch(f"{MODULE}.BaseAgentFactory")
    def test_build_articles_deduplicates(self, mock_factory):
        """Verify duplicate URLs across findings are deduplicated."""
        provider = DeepResearchProvider(parallel_research=False)
        report = _make_report(source_urls=[])
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

        articles = provider._build_articles(report, findings)

        # 1 report + 1 unique URL (deduplicated)
        assert len(articles) == 2
        assert articles[1]["url"] == "https://dup.com/1"
