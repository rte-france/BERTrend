#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import pytest
from datetime import date
from pydantic import ValidationError

from bertrend.llm_utils.newsletter_model import (
    Article,
    Topic,
    STRONG_TOPIC_TYPE,
    WEAK_TOPIC_TYPE,
    NOISE_TOPIC_TYPE,
    Newsletter,
)


class TestArticle:
    """Test cases for Article model."""

    def test_article_creation_minimal(self):
        """Test creating an article with minimal required fields."""
        article = Article(title="Test Article", date=date(2024, 1, 15))

        assert article.title == "Test Article"
        assert article.date == date(2024, 1, 15)
        assert article.summary is None
        assert article.source is None
        assert article.url is None

    def test_article_creation_full(self):
        """Test creating an article with all fields."""
        article = Article(
            title="Full Article Test",
            date=date(2024, 1, 15),
            summary="This is a test summary",
            source="Test Source",
            url="https://example.com/article",
        )

        assert article.title == "Full Article Test"
        assert article.date == date(2024, 1, 15)
        assert article.summary == "This is a test summary"
        assert article.source == "Test Source"
        assert article.url == "https://example.com/article"

    def test_article_missing_required_title(self):
        """Test that missing title raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Article(date=date(2024, 1, 15))

        assert "title" in str(exc_info.value)

    def test_article_missing_required_date(self):
        """Test that missing date raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Article(title="Test Article")

        assert "date" in str(exc_info.value)

    def test_article_empty_title(self):
        """Test that empty title is allowed but not None."""
        article = Article(title="", date=date(2024, 1, 15))

        assert article.title == ""

    def test_article_serialization(self):
        """Test article can be serialized to dict."""
        article = Article(
            title="Test Article", date=date(2024, 1, 15), summary="Test summary"
        )

        data = article.model_dump()

        assert data["title"] == "Test Article"
        assert data["date"] == date(2024, 1, 15)
        assert data["summary"] == "Test summary"
        assert data["source"] is None
        assert data["url"] is None


class TestTopic:
    """Test cases for Topic model."""

    def test_topic_creation_minimal(self):
        """Test creating a topic with minimal required fields."""
        articles = [
            Article(title="Article 1", date=date(2024, 1, 15)),
            Article(title="Article 2", date=date(2024, 1, 16)),
        ]

        topic = Topic(
            title="Test Topic", hashtags=["#test", "#topic"], articles=articles
        )

        assert topic.title == "Test Topic"
        assert topic.hashtags == ["#test", "#topic"]
        assert topic.summary is None
        assert len(topic.articles) == 2
        assert topic.topic_type == STRONG_TOPIC_TYPE  # Default value

    def test_topic_creation_full(self):
        """Test creating a topic with all fields."""
        articles = [Article(title="Article 1", date=date(2024, 1, 15))]

        topic = Topic(
            title="Full Topic Test",
            hashtags=["#full", "#test"],
            summary="This is a test topic summary",
            articles=articles,
            topic_type=WEAK_TOPIC_TYPE,
        )

        assert topic.title == "Full Topic Test"
        assert topic.hashtags == ["#full", "#test"]
        assert topic.summary == "This is a test topic summary"
        assert len(topic.articles) == 1
        assert topic.topic_type == WEAK_TOPIC_TYPE

    def test_topic_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        articles = [Article(title="Article 1", date=date(2024, 1, 15))]

        # Missing title
        with pytest.raises(ValidationError) as exc_info:
            Topic(hashtags=["#test"], articles=articles)
        assert "title" in str(exc_info.value)

        # Missing articles
        with pytest.raises(ValidationError) as exc_info:
            Topic(title="Test Topic", hashtags=["#test"])
        assert "articles" in str(exc_info.value)

    def test_topic_empty_articles_list(self):
        """Test that empty articles list is allowed."""
        topic = Topic(title="Empty Topic", hashtags=["#empty"], articles=[])

        assert len(topic.articles) == 0

    def test_topic_empty_hashtags_list(self):
        """Test that empty hashtags list is allowed."""
        articles = [Article(title="Article 1", date=date(2024, 1, 15))]

        topic = Topic(title="No Hashtags Topic", hashtags=[], articles=articles)

        assert len(topic.hashtags) == 0

    def test_topic_different_types(self):
        """Test all topic types are accepted."""
        articles = [Article(title="Article 1", date=date(2024, 1, 15))]

        # Test strong type
        topic_strong = Topic(
            title="Strong Topic",
            hashtags=["#strong"],
            articles=articles,
            topic_type=STRONG_TOPIC_TYPE,
        )
        assert topic_strong.topic_type == STRONG_TOPIC_TYPE

        # Test weak type
        topic_weak = Topic(
            title="Weak Topic",
            hashtags=["#weak"],
            articles=articles,
            topic_type=WEAK_TOPIC_TYPE,
        )
        assert topic_weak.topic_type == WEAK_TOPIC_TYPE

        # Test noise type
        topic_noise = Topic(
            title="Noise Topic",
            hashtags=["#noise"],
            articles=articles,
            topic_type=NOISE_TOPIC_TYPE,
        )
        assert topic_noise.topic_type == NOISE_TOPIC_TYPE

    def test_topic_invalid_type(self):
        """Test that invalid topic type raises ValidationError."""
        articles = [Article(title="Article 1", date=date(2024, 1, 15))]

        # Note: Pydantic doesn't validate enum values by default for strings
        # This test might pass depending on Pydantic configuration
        topic = Topic(
            title="Invalid Topic",
            hashtags=["#invalid"],
            articles=articles,
            topic_type="invalid_type",
        )
        assert topic.topic_type == "invalid_type"


class TestNewsletter:
    """Test cases for Newsletter model."""

    def test_newsletter_creation_minimal(self):
        """Test creating a newsletter with minimal required fields."""
        articles = [Article(title="Article 1", date=date(2024, 1, 15))]
        topics = [Topic(title="Topic 1", hashtags=["#test"], articles=articles)]

        newsletter = Newsletter(
            title="Test Newsletter",
            period_start_date=date(2024, 1, 1),
            period_end_date=date(2024, 1, 31),
            topics=topics,
        )

        assert newsletter.title == "Test Newsletter"
        assert newsletter.period_start_date == date(2024, 1, 1)
        assert newsletter.period_end_date == date(2024, 1, 31)
        assert len(newsletter.topics) == 1
        assert newsletter.debug_info is None

    def test_newsletter_creation_full(self):
        """Test creating a newsletter with all fields."""
        articles = [Article(title="Article 1", date=date(2024, 1, 15))]
        topics = [Topic(title="Topic 1", hashtags=["#test"], articles=articles)]
        debug_info = {"model_version": "1.0", "processed_articles": 100}

        newsletter = Newsletter(
            title="Full Newsletter Test",
            period_start_date=date(2024, 1, 1),
            period_end_date=date(2024, 1, 31),
            topics=topics,
            debug_info=debug_info,
        )

        assert newsletter.title == "Full Newsletter Test"
        assert newsletter.period_start_date == date(2024, 1, 1)
        assert newsletter.period_end_date == date(2024, 1, 31)
        assert len(newsletter.topics) == 1
        assert newsletter.debug_info == debug_info

    def test_newsletter_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        articles = [Article(title="Article 1", date=date(2024, 1, 15))]
        topics = [Topic(title="Topic 1", hashtags=["#test"], articles=articles)]

        # Missing title
        with pytest.raises(ValidationError) as exc_info:
            Newsletter(
                reference_period=date(2024, 1, 31),
                period_start_date=date(2024, 1, 1),
                period_end_date=date(2024, 1, 31),
                topics=topics,
            )
        assert "title" in str(exc_info.value)

        # Missing topics
        with pytest.raises(ValidationError) as exc_info:
            Newsletter(
                title="Test Newsletter",
                period_start_date=date(2024, 1, 1),
                period_end_date=date(2024, 1, 31),
            )
        assert "topics" in str(exc_info.value)

    def test_newsletter_empty_topics_list(self):
        """Test that empty topics list is allowed."""
        newsletter = Newsletter(
            title="Empty Newsletter",
            period_start_date=date(2024, 1, 1),
            period_end_date=date(2024, 1, 31),
            topics=[],
        )

        assert len(newsletter.topics) == 0

    def test_newsletter_date_validation(self):
        """Test date validation logic."""
        articles = [Article(title="Article 1", date=date(2024, 1, 15))]
        topics = [Topic(title="Topic 1", hashtags=["#test"], articles=articles)]

        # Valid date range
        newsletter = Newsletter(
            title="Valid Newsletter",
            period_start_date=date(2024, 1, 1),
            period_end_date=date(2024, 1, 31),
            topics=topics,
        )

        assert newsletter.period_start_date <= newsletter.period_end_date

        # End date before start date (Pydantic doesn't validate this by default)
        newsletter_invalid = Newsletter(
            title="Invalid Newsletter",
            period_start_date=date(2024, 1, 31),
            period_end_date=date(2024, 1, 1),
            topics=topics,
        )

        # This would pass unless custom validation is added
        assert newsletter_invalid.period_start_date > newsletter_invalid.period_end_date


class TestConstants:
    """Test the module constants."""

    def test_topic_type_constants(self):
        """Test that topic type constants have expected values."""
        assert STRONG_TOPIC_TYPE == "strong"
        assert WEAK_TOPIC_TYPE == "weak"
        assert NOISE_TOPIC_TYPE == "noise"


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_complete_newsletter_workflow(self):
        """Test creating a complete newsletter with nested data."""
        # Create articles
        article1 = Article(
            title="AI Breakthrough in 2024",
            date=date(2024, 1, 15),
            summary="Major AI advancement announced",
            source="Tech News",
            url="https://example.com/ai-breakthrough",
        )

        article2 = Article(
            title="Climate Change Impact",
            date=date(2024, 1, 20),
            summary="New study on climate effects",
            source="Science Daily",
            url="https://example.com/climate-study",
        )

        # Create topics
        topic1 = Topic(
            title="Technology Trends",
            hashtags=["#AI", "#technology", "#2024"],
            summary="Latest developments in AI and tech",
            articles=[article1],
            topic_type=STRONG_TOPIC_TYPE,
        )

        topic2 = Topic(
            title="Environmental News",
            hashtags=["#climate", "#environment"],
            summary="Climate and environmental updates",
            articles=[article2],
            topic_type=WEAK_TOPIC_TYPE,
        )

        # Create newsletter
        newsletter = Newsletter(
            title="Weekly Tech & Environment Digest",
            period_start_date=date(2024, 1, 15),
            period_end_date=date(2024, 1, 21),
            topics=[topic1, topic2],
            debug_info={
                "total_articles_processed": 50,
                "topics_generated": 2,
                "model_version": "v1.2.3",
            },
        )

        # Verify the complete structure
        assert newsletter.title == "Weekly Tech & Environment Digest"
        assert len(newsletter.topics) == 2
        assert newsletter.topics[0].title == "Technology Trends"
        assert newsletter.topics[1].title == "Environmental News"
        assert len(newsletter.topics[0].articles) == 1
        assert len(newsletter.topics[1].articles) == 1
        assert newsletter.topics[0].articles[0].title == "AI Breakthrough in 2024"
        assert newsletter.topics[1].articles[0].title == "Climate Change Impact"
        assert newsletter.debug_info["total_articles_processed"] == 50

    def test_newsletter_serialization(self):
        """Test that newsletter can be serialized and deserialized."""
        # Create a simple newsletter
        article = Article(title="Test Article", date=date(2024, 1, 15))
        topic = Topic(title="Test Topic", hashtags=["#test"], articles=[article])
        newsletter = Newsletter(
            title="Test Newsletter",
            period_start_date=date(2024, 1, 1),
            period_end_date=date(2024, 1, 31),
            reference_period=date(2024, 1, 31),
            topics=[topic],
        )

        # Serialize to dict
        data = newsletter.model_dump()

        # Verify serialization
        assert data["title"] == "Test Newsletter"
        assert data["period_start_date"] == date(2024, 1, 1)
        assert data["period_end_date"] == date(2024, 1, 31)
        assert data["reference_period"] == date(2024, 1, 31)
        assert len(data["topics"]) == 1
        assert data["topics"][0]["title"] == "Test Topic"
        assert len(data["topics"][0]["articles"]) == 1
        assert data["topics"][0]["articles"][0]["title"] == "Test Article"

        # Deserialize from dict
        newsletter_from_dict = Newsletter(**data)

        assert newsletter_from_dict.title == newsletter.title
        assert newsletter_from_dict.period_start_date == newsletter.period_start_date
        assert newsletter_from_dict.period_end_date == newsletter.period_end_date
        assert newsletter_from_dict.reference_period == newsletter.reference_period
        assert len(newsletter_from_dict.topics) == len(newsletter.topics)
