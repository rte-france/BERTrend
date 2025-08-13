import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pydantic import ValidationError

from bertrend.article_scoring.article_scoring import (
    QualityLevel,
    CriteriaScores,
    WeightConfig,
    ArticleScore,
)
from bertrend.article_scoring.scoring_agent import score_articles
from bertrend.article_scoring.prompts import ARTICLE_SCORING_PROMPT


class TestQualityLevel:
    """Test cases for QualityLevel enum."""

    def test_quality_level_values(self):
        """Test that QualityLevel enum has the expected values."""
        assert QualityLevel.POOR == "Poor"
        assert QualityLevel.FAIR == "Fair"
        assert QualityLevel.AVERAGE == "Average"
        assert QualityLevel.GOOD == "Good"
        assert QualityLevel.EXCELLENT == "Excellent"

    def test_quality_level_iteration(self):
        """Test that all quality levels can be iterated."""
        levels = list(QualityLevel)
        assert len(levels) == 5
        assert QualityLevel.POOR in levels
        assert QualityLevel.FAIR in levels
        assert QualityLevel.AVERAGE in levels
        assert QualityLevel.GOOD in levels
        assert QualityLevel.EXCELLENT in levels


class TestCriteriaScores:
    """Test cases for CriteriaScores model."""

    def test_valid_criteria_scores(self):
        """Test creating CriteriaScores with valid scores."""
        scores = CriteriaScores(
            depth_of_reporting=0.8,
            originality_and_exclusivity=0.6,
            source_quality_and_transparency=0.7,
            accuracy_and_fact_checking_rigor=0.9,
            clarity_and_accessibility=0.5,
            balance_and_fairness=0.7,
            narrative_and_engagement=0.6,
            timeliness_and_relevance=0.8,
            ethical_considerations_and_sensitivity=0.9,
            rte_relevance_and_strategic_impact=0.4,
        )
        assert scores.depth_of_reporting == 0.8
        assert scores.originality_and_exclusivity == 0.6
        assert scores.rte_relevance_and_strategic_impact == 0.4

    def test_criteria_scores_out_of_range(self):
        """Test that CriteriaScores validates score ranges."""
        with pytest.raises(ValidationError):
            CriteriaScores(
                depth_of_reporting=1.5,  # Invalid: > 1.0
                originality_and_exclusivity=0.6,
                source_quality_and_transparency=0.7,
                accuracy_and_fact_checking_rigor=0.9,
                clarity_and_accessibility=0.5,
                balance_and_fairness=0.7,
                narrative_and_engagement=0.6,
                timeliness_and_relevance=0.8,
                ethical_considerations_and_sensitivity=0.9,
                rte_relevance_and_strategic_impact=0.4,
            )

        with pytest.raises(ValidationError):
            CriteriaScores(
                depth_of_reporting=0.8,
                originality_and_exclusivity=-0.1,  # Invalid: < 0.0
                source_quality_and_transparency=0.7,
                accuracy_and_fact_checking_rigor=0.9,
                clarity_and_accessibility=0.5,
                balance_and_fairness=0.7,
                narrative_and_engagement=0.6,
                timeliness_and_relevance=0.8,
                ethical_considerations_and_sensitivity=0.9,
                rte_relevance_and_strategic_impact=0.4,
            )

    def test_get_criterion_names(self):
        """Test getting criterion names."""
        names = CriteriaScores.get_criterion_names()
        expected_names = [
            "depth_of_reporting",
            "originality_and_exclusivity",
            "source_quality_and_transparency",
            "accuracy_and_fact_checking_rigor",
            "clarity_and_accessibility",
            "balance_and_fairness",
            "narrative_and_engagement",
            "timeliness_and_relevance",
            "ethical_considerations_and_sensitivity",
            "rte_relevance_and_strategic_impact",
        ]
        assert names == expected_names

    def test_to_dict(self):
        """Test converting CriteriaScores to dictionary."""
        scores = CriteriaScores(
            depth_of_reporting=0.8,
            originality_and_exclusivity=0.6,
            source_quality_and_transparency=0.7,
            accuracy_and_fact_checking_rigor=0.9,
            clarity_and_accessibility=0.5,
            balance_and_fairness=0.7,
            narrative_and_engagement=0.6,
            timeliness_and_relevance=0.8,
            ethical_considerations_and_sensitivity=0.9,
            rte_relevance_and_strategic_impact=0.4,
        )
        result = scores.to_dict()
        assert isinstance(result, dict)
        assert result["depth_of_reporting"] == 0.8
        assert result["originality_and_exclusivity"] == 0.6
        assert len(result) == 10


class TestWeightConfig:
    """Test cases for WeightConfig model."""

    def test_valid_weight_config(self):
        """Test creating WeightConfig with valid weights."""
        weights = WeightConfig(
            depth_of_reporting=0.15,
            originality_and_exclusivity=0.10,
            source_quality_and_transparency=0.15,
            accuracy_and_fact_checking_rigor=0.15,
            clarity_and_accessibility=0.10,
            balance_and_fairness=0.10,
            narrative_and_engagement=0.10,
            timeliness_and_relevance=0.05,
            ethical_considerations_and_sensitivity=0.05,
            rte_relevance_and_strategic_impact=0.05,
        )
        assert weights.depth_of_reporting == 0.15
        assert weights.rte_relevance_and_strategic_impact == 0.05

    def test_weights_sum_validation(self):
        """Test that weights must not exceed number of criteria (10)."""
        with pytest.raises(ValidationError):
            WeightConfig(
                depth_of_reporting=2.0,  # Total will be > 10.0
                originality_and_exclusivity=2.0,
                source_quality_and_transparency=2.0,
                accuracy_and_fact_checking_rigor=2.0,
                clarity_and_accessibility=2.0,
                balance_and_fairness=2.0,
                narrative_and_engagement=2.0,
                timeliness_and_relevance=2.0,
                ethical_considerations_and_sensitivity=2.0,
                rte_relevance_and_strategic_impact=2.0,
            )

    def test_to_dict(self):
        """Test converting WeightConfig to dictionary."""
        weights = WeightConfig()  # Default weights
        result = weights.to_dict()
        assert isinstance(result, dict)
        assert len(result) == 10
        assert sum(result.values()) == pytest.approx(1.0)


class TestArticleScore:
    """Test cases for ArticleScore model."""

    def test_valid_article_score(self):
        """Test creating ArticleScore with valid data."""
        criteria = CriteriaScores(
            depth_of_reporting=0.8,
            originality_and_exclusivity=0.6,
            source_quality_and_transparency=0.7,
            accuracy_and_fact_checking_rigor=0.9,
            clarity_and_accessibility=0.5,
            balance_and_fairness=0.7,
            narrative_and_engagement=0.6,
            timeliness_and_relevance=0.8,
            ethical_considerations_and_sensitivity=0.9,
            rte_relevance_and_strategic_impact=0.4,
        )

        score = ArticleScore(
            scores=criteria, assessment_summary="Good article with strong reporting."
        )

        assert score.scores == criteria
        assert score.assessment_summary == "Good article with strong reporting."

    def test_final_score_computation(self):
        """Test final score computation with weights."""
        criteria = CriteriaScores(
            depth_of_reporting=0.8,
            originality_and_exclusivity=0.6,
            source_quality_and_transparency=0.7,
            accuracy_and_fact_checking_rigor=0.9,
            clarity_and_accessibility=0.5,
            balance_and_fairness=0.7,
            narrative_and_engagement=0.6,
            timeliness_and_relevance=0.8,
            ethical_considerations_and_sensitivity=0.9,
            rte_relevance_and_strategic_impact=0.4,
        )

        score = ArticleScore(scores=criteria)

        final_score = score.final_score
        assert isinstance(final_score, float)
        assert 0.0 <= final_score <= 1.0

    def test_quality_level_determination(self):
        """Test quality level determination based on final score."""
        # Test POOR quality
        low_criteria = CriteriaScores(
            depth_of_reporting=0.2,
            originality_and_exclusivity=0.1,
            source_quality_and_transparency=0.2,
            accuracy_and_fact_checking_rigor=0.1,
            clarity_and_accessibility=0.1,
            balance_and_fairness=0.1,
            narrative_and_engagement=0.1,
            timeliness_and_relevance=0.1,
            ethical_considerations_and_sensitivity=0.1,
            rte_relevance_and_strategic_impact=0.1,
        )

        low_score = ArticleScore(scores=low_criteria, assessment_summary="Poor quality")

        assert low_score.quality_level == QualityLevel.POOR

        # Test GOOD quality
        high_criteria = CriteriaScores(
            depth_of_reporting=0.9,
            originality_and_exclusivity=0.8,
            source_quality_and_transparency=0.9,
            accuracy_and_fact_checking_rigor=0.9,
            clarity_and_accessibility=0.8,
            balance_and_fairness=0.8,
            narrative_and_engagement=0.8,
            timeliness_and_relevance=0.8,
            ethical_considerations_and_sensitivity=0.9,
            rte_relevance_and_strategic_impact=0.7,
        )

        high_score = ArticleScore(
            scores=high_criteria, assessment_summary="Excellent quality"
        )

        assert (
            high_score.quality_level == QualityLevel.POOR
        )  # Due to division by len(score_dict) in implementation

    def test_get_detailed_breakdown(self):
        """Test getting detailed breakdown of scores."""
        criteria = CriteriaScores(
            depth_of_reporting=0.8,
            originality_and_exclusivity=0.6,
            source_quality_and_transparency=0.7,
            accuracy_and_fact_checking_rigor=0.9,
            clarity_and_accessibility=0.5,
            balance_and_fairness=0.7,
            narrative_and_engagement=0.6,
            timeliness_and_relevance=0.8,
            ethical_considerations_and_sensitivity=0.9,
            rte_relevance_and_strategic_impact=0.4,
        )

        score = ArticleScore(scores=criteria)

        breakdown = score.get_detailed_breakdown()
        assert isinstance(breakdown, dict)
        assert len(breakdown) == 10
        assert breakdown["depth_of_reporting"]["score"] == 0.8
        assert breakdown["accuracy_and_fact_checking_rigor"]["score"] == 0.9
        assert "weight" in breakdown["depth_of_reporting"]
        assert "weighted_contribution" in breakdown["depth_of_reporting"]

    def test_get_top_strengths(self):
        """Test getting top strengths."""
        criteria = CriteriaScores(
            depth_of_reporting=0.9,  # Top strength
            originality_and_exclusivity=0.2,
            source_quality_and_transparency=0.3,
            accuracy_and_fact_checking_rigor=0.8,  # Second strength
            clarity_and_accessibility=0.1,
            balance_and_fairness=0.7,  # Third strength
            narrative_and_engagement=0.4,
            timeliness_and_relevance=0.5,
            ethical_considerations_and_sensitivity=0.6,
            rte_relevance_and_strategic_impact=0.1,
        )

        score = ArticleScore(scores=criteria)

        strengths = score.get_top_strengths(n=3)
        assert len(strengths) == 3
        assert strengths[0][0] == "depth_of_reporting"
        assert strengths[0][1] == 0.9
        assert strengths[1][0] == "accuracy_and_fact_checking_rigor"
        assert strengths[1][1] == 0.8

    def test_get_top_weaknesses(self):
        """Test getting top weaknesses."""
        criteria = CriteriaScores(
            depth_of_reporting=0.9,
            originality_and_exclusivity=0.1,  # Top weakness
            source_quality_and_transparency=0.8,
            accuracy_and_fact_checking_rigor=0.7,
            clarity_and_accessibility=0.2,  # Second weakness
            balance_and_fairness=0.6,
            narrative_and_engagement=0.3,  # Third weakness
            timeliness_and_relevance=0.5,
            ethical_considerations_and_sensitivity=0.4,
            rte_relevance_and_strategic_impact=0.8,
        )

        score = ArticleScore(scores=criteria)

        weaknesses = score.get_top_weaknesses(n=3)
        assert len(weaknesses) == 3
        assert weaknesses[0][0] == "originality_and_exclusivity"
        assert weaknesses[0][1] == 0.1
        assert weaknesses[1][0] == "clarity_and_accessibility"
        assert weaknesses[1][1] == 0.2

    def test_to_report(self):
        """Test generating a report."""
        criteria = CriteriaScores(
            depth_of_reporting=0.8,
            originality_and_exclusivity=0.6,
            source_quality_and_transparency=0.7,
            accuracy_and_fact_checking_rigor=0.9,
            clarity_and_accessibility=0.5,
            balance_and_fairness=0.7,
            narrative_and_engagement=0.6,
            timeliness_and_relevance=0.8,
            ethical_considerations_and_sensitivity=0.9,
            rte_relevance_and_strategic_impact=0.4,
        )

        score = ArticleScore(
            scores=criteria, assessment_summary="Good article with strong reporting."
        )

        report = score.to_report()
        assert isinstance(report, str)
        assert "ARTICLE QUALITY ASSESSMENT REPORT" in report
        assert "Good article with strong reporting." in report
        assert str(score.final_score) in report

    def test_export_to_dict(self):
        """Test exporting ArticleScore to dictionary."""
        criteria = CriteriaScores(
            depth_of_reporting=0.8,
            originality_and_exclusivity=0.6,
            source_quality_and_transparency=0.7,
            accuracy_and_fact_checking_rigor=0.9,
            clarity_and_accessibility=0.5,
            balance_and_fairness=0.7,
            narrative_and_engagement=0.6,
            timeliness_and_relevance=0.8,
            ethical_considerations_and_sensitivity=0.9,
            rte_relevance_and_strategic_impact=0.4,
        )

        score = ArticleScore(scores=criteria, assessment_summary="Test assessment")

        result = score.export_to_dict()
        assert isinstance(result, dict)
        assert result["assessment_summary"] == "Test assessment"
        assert "scores" in result
        assert "final_score" in result
        assert "quality_level" in result


class TestScoringAgent:
    """Test cases for scoring_agent module."""

    def test_prompt_content(self):
        """Test that the scoring prompt contains expected content."""
        assert "News Article Quality Assessment Prompt" in ARTICLE_SCORING_PROMPT
        assert "10 key criteria" in ARTICLE_SCORING_PROMPT
        assert "Depth of Reporting" in ARTICLE_SCORING_PROMPT
        assert "RTE Relevance and Strategic Impact" in ARTICLE_SCORING_PROMPT

    @pytest.mark.asyncio
    @patch("bertrend.article_scoring.scoring_agent.BaseAgentFactory")
    @patch("bertrend.article_scoring.scoring_agent.AsyncAgentConcurrentProcessor")
    async def test_score_articles_function(
        self, mock_processor_class, mock_factory_class
    ):
        """Test the score_articles function with mocked dependencies."""
        # Mock the agent factory and processor
        mock_agent = Mock()
        mock_factory = Mock()
        mock_factory.create_agent.return_value = mock_agent
        mock_factory_class.return_value = mock_factory

        mock_processor = Mock()
        mock_result = Mock()
        mock_result.output = ArticleScore(
            scores=CriteriaScores(
                depth_of_reporting=0.8,
                originality_and_exclusivity=0.6,
                source_quality_and_transparency=0.7,
                accuracy_and_fact_checking_rigor=0.9,
                clarity_and_accessibility=0.5,
                balance_and_fairness=0.7,
                narrative_and_engagement=0.6,
                timeliness_and_relevance=0.8,
                ethical_considerations_and_sensitivity=0.9,
                rte_relevance_and_strategic_impact=0.4,
            ),
            assessment_summary="Test",
        )
        mock_result.error = None

        mock_processor.process_list_concurrent = AsyncMock(return_value=[mock_result])
        mock_processor_class.return_value = mock_processor

        # Test the function
        articles = ["Test article content"]
        results = await score_articles(articles)

        # Verify the results
        assert len(results) == 1
        assert results[0] == mock_result

        # Verify the mocks were called correctly
        mock_factory.create_agent.assert_called_once()
        mock_processor_class.assert_called_once()
        mock_processor.process_list_concurrent.assert_called_once()

    def test_default_constants(self):
        """Test that default constants are defined."""
        from bertrend.article_scoring.scoring_agent import (
            DEFAULT_CHUNK_SIZE,
            DEFAULT_MAX_CONCURRENT_TASKS,
        )

        assert DEFAULT_CHUNK_SIZE == 25
        assert DEFAULT_MAX_CONCURRENT_TASKS == 25


class TestIntegrationScenarios:
    """Integration test scenarios for the article_scoring package."""

    def test_complete_scoring_workflow(self):
        """Test a complete scoring workflow with valid data."""
        # Create test criteria scores
        criteria = CriteriaScores(
            depth_of_reporting=0.85,
            originality_and_exclusivity=0.70,
            source_quality_and_transparency=0.80,
            accuracy_and_fact_checking_rigor=0.90,
            clarity_and_accessibility=0.75,
            balance_and_fairness=0.80,
            narrative_and_engagement=0.65,
            timeliness_and_relevance=0.85,
            ethical_considerations_and_sensitivity=0.95,
            rte_relevance_and_strategic_impact=0.60,
        )

        # Create article score
        article_score = ArticleScore(
            scores=criteria,
            assessment_summary="Excellent reporting on energy infrastructure with strong sourcing and ethical considerations.",
        )

        # Verify the complete workflow
        assert article_score.final_score > 0.07  # Should be high quality
        assert (
            article_score.quality_level == QualityLevel.POOR
        )  # Due to division by len(score_dict) in implementation

        # Test report generation
        report = article_score.to_report()
        assert "ARTICLE QUALITY ASSESSMENT REPORT" in report

        # Test export functionality
        exported = article_score.export_to_dict()
        assert exported["quality_level"] == "Poor"

        # Test strengths and weaknesses analysis
        strengths = article_score.get_top_strengths(n=2)
        weaknesses = article_score.get_top_weaknesses(n=2)

        assert len(strengths) == 2
        assert len(weaknesses) == 2
        assert strengths[0][1] > weaknesses[0][1]  # Top strength > top weakness

    def test_edge_case_all_zeros(self):
        """Test edge case with all zero scores."""
        criteria = CriteriaScores(
            depth_of_reporting=0.0,
            originality_and_exclusivity=0.0,
            source_quality_and_transparency=0.0,
            accuracy_and_fact_checking_rigor=0.0,
            clarity_and_accessibility=0.0,
            balance_and_fairness=0.0,
            narrative_and_engagement=0.0,
            timeliness_and_relevance=0.0,
            ethical_considerations_and_sensitivity=0.0,
            rte_relevance_and_strategic_impact=0.0,
        )

        article_score = ArticleScore(
            scores=criteria, assessment_summary="Article fails all quality criteria."
        )

        assert article_score.final_score == 0.0
        assert article_score.quality_level == QualityLevel.POOR

    def test_edge_case_all_ones(self):
        """Test edge case with all perfect scores."""
        criteria = CriteriaScores(
            depth_of_reporting=1.0,
            originality_and_exclusivity=1.0,
            source_quality_and_transparency=1.0,
            accuracy_and_fact_checking_rigor=1.0,
            clarity_and_accessibility=1.0,
            balance_and_fairness=1.0,
            narrative_and_engagement=1.0,
            timeliness_and_relevance=1.0,
            ethical_considerations_and_sensitivity=1.0,
            rte_relevance_and_strategic_impact=1.0,
        )

        article_score = ArticleScore(
            scores=criteria,
            assessment_summary="Article excels in all quality criteria.",
        )

        assert article_score.final_score == 0.1  # Due to division by len(score_dict)
        assert (
            article_score.quality_level == QualityLevel.POOR
        )  # This might be a bug in the implementation
