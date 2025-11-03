#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tempfile

from bertrend.llm_utils.newsletter_features import (
    generate_newsletter,
    DEFAULT_TOP_N_DOCS,
    DEFAULT_TOP_N_DOCS_MODE,
    DEFAULT_SUMMARY_MODE,
    DEFAULT_TOP_N_TOPICS,
    render_newsletter_md,
    render_newsletter_html,
    render_newsletter,
)
from bertrend.llm_utils.newsletter_model import (
    Newsletter,
    Topic,
    Article,
    STRONG_TOPIC_TYPE,
)


class TestNewsletterGenerator:
    """Test cases for the newsletter generation functionality."""

    @pytest.fixture
    def mock_bertopic_model(self):
        """Mock BERTopic model."""
        model = Mock()
        topic_info = pd.DataFrame(
            {
                "Topic": [0, 1, 2],
                "Count": [100, 80, 60],
                "Representation": [
                    ["ai", "artificial", "intelligence"],
                    ["climate", "environment", "green"],
                    ["tech", "technology", "innovation"],
                ],
            }
        )
        model.get_topic_info.return_value = topic_info
        return model

    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame with document data."""
        return pd.DataFrame(
            {
                "text": [
                    "Article about AI advancement",
                    "Climate change report",
                    "Technology innovation news",
                    "Another AI article",
                    "Environmental policy update",
                ],
                "title": [
                    "AI Breakthrough 2024",
                    "Climate Report",
                    "Tech Innovation",
                    "AI Progress",
                    "Environmental Policy",
                ],
                "url": [
                    "https://example.com/ai-breakthrough",
                    "https://news.com/climate-report",
                    "https://tech.com/innovation",
                    "https://ai.com/progress",
                    "https://env.com/policy",
                ],
                "timestamp": [
                    datetime(2024, 1, 15),
                    datetime(2024, 1, 16),
                    datetime(2024, 1, 17),
                    datetime(2024, 1, 18),
                    datetime(2024, 1, 19),
                ],
            }
        )

    @pytest.fixture
    def sample_topics(self):
        """Sample topics list."""
        return [0, 1, 2, 0, 1]

    @pytest.fixture
    def mock_summarizer(self):
        """Mock summarizer class."""
        summarizer = Mock()
        summarizer.summarize_batch.return_value = [
            "Summary 1",
            "Summary 2",
            "Summary 3",
        ]
        return summarizer

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        client = Mock()
        client.generate.return_value = "Generated summary"
        return client

    @pytest.fixture
    def mock_representative_docs(self):
        """Mock representative documents function."""

        def mock_get_docs(
            topic_model, df, topics, mode, df_split, topic_number, top_n_docs
        ):
            # Return subset of dataframe based on topic
            topic_mask = [t == topic_number for t in topics]
            return df[topic_mask].head(top_n_docs)

        return mock_get_docs

    @patch(
        "bertrend.LLM_CONFIG",
        {"model": "gpt-4", "api_key": "test-key", "base_url": "test-base_url"},
    )
    @patch("bertrend.llm_utils.newsletter_features.OpenAI_Client")
    @patch("bertrend.services.summary.chatgpt_summarizer.GPTSummarizer")
    @patch("bertrend.llm_utils.newsletter_features.get_most_representative_docs")
    @patch("tldextract.extract")
    def test_generate_newsletter_basic(
        self,
        mock_tldextract,
        mock_get_docs,
        mock_summarizer_class,
        mock_openai_class,
        mock_bertopic_model,
        sample_dataframe,
        sample_topics,
        mock_summarizer,
        mock_openai_client,
    ):
        """Test basic newsletter generation."""
        # Setup mocks
        mock_summarizer_class.return_value = mock_summarizer
        mock_openai_class.return_value = mock_openai_client
        mock_tldextract.return_value.domain = "example.com"

        # Mock get_most_representative_docs to return sample data
        mock_get_docs.side_effect = lambda **kwargs: sample_dataframe.head(2)

        # Generate newsletter
        newsletter = generate_newsletter(
            topic_model=mock_bertopic_model,
            df=sample_dataframe,
            topics=sample_topics,
            top_n_topics=2,
            top_n_docs=2,
            newsletter_title="Test Newsletter",
        )

        # Assertions
        assert isinstance(newsletter, Newsletter)
        assert newsletter.title == "Test Newsletter"
        assert newsletter.period_start_date == sample_dataframe.timestamp.min().date()
        assert newsletter.period_end_date == sample_dataframe.timestamp.max().date()
        assert len(newsletter.topics) == 2

        # Verify topic structure
        for topic in newsletter.topics:
            assert isinstance(topic, Topic)
            assert topic.topic_type == STRONG_TOPIC_TYPE
            assert len(topic.articles) == 2
            for article in topic.articles:
                assert isinstance(article, Article)
                assert article.source == "example.com"

    @patch(
        "bertrend.LLM_CONFIG",
        {"model": "gpt-4", "api_key": "test-key", "base_url": "test-base_url"},
    )
    @patch("bertrend.llm_utils.newsletter_features.OpenAI_Client")
    @patch("bertrend.services.summary.chatgpt_summarizer.GPTSummarizer")
    @patch("bertrend.llm_utils.newsletter_features.get_most_representative_docs")
    @patch("tldextract.extract")
    def test_generate_newsletter_topic_summary_mode(
        self,
        mock_tldextract,
        mock_get_docs,
        mock_summarizer_class,
        mock_openai_class,
        mock_bertopic_model,
        sample_dataframe,
        sample_topics,
        mock_summarizer,
        mock_openai_client,
    ):
        """Test newsletter generation with topic summary mode."""
        # Setup mocks
        mock_summarizer_class.return_value = mock_summarizer
        mock_openai_class.return_value = mock_openai_client
        mock_tldextract.return_value.domain = "example.com"
        mock_get_docs.side_effect = lambda **kwargs: sample_dataframe.head(2)

        # Generate newsletter with topic summary mode
        newsletter = generate_newsletter(
            topic_model=mock_bertopic_model,
            df=sample_dataframe,
            topics=sample_topics,
            summary_mode="topic",
            top_n_topics=1,
        )

        # Verify OpenAI was called for topic summary
        mock_openai_client.generate.assert_called()

        # Verify topic has summary
        assert newsletter.topics[0].summary == "Generated summary"

        # Verify articles don't have individual summaries in topic mode
        for article in newsletter.topics[0].articles:
            assert article.summary is None

    @patch(
        "bertrend.LLM_CONFIG",
        {"model": "gpt-4", "api_key": "test-key", "base_url": "test-base_url"},
    )
    @patch("bertrend.llm_utils.newsletter_features.OpenAI_Client")
    @patch("bertrend.services.summary.chatgpt_summarizer.GPTSummarizer")
    @patch("bertrend.llm_utils.newsletter_features.get_most_representative_docs")
    @patch("tldextract.extract")
    def test_generate_newsletter_no_summary_mode(
        self,
        mock_tldextract,
        mock_get_docs,
        mock_summarizer_class,
        mock_openai_class,
        mock_bertopic_model,
        sample_dataframe,
        sample_topics,
        mock_summarizer,
        mock_openai_client,
    ):
        """Test newsletter generation with no summary mode."""
        # Setup mocks
        mock_summarizer_class.return_value = mock_summarizer
        mock_openai_class.return_value = mock_openai_client
        mock_tldextract.return_value.domain = "example.com"
        mock_get_docs.side_effect = lambda **kwargs: sample_dataframe.head(2)

        # Generate newsletter with no summary mode
        newsletter = generate_newsletter(
            topic_model=mock_bertopic_model,
            df=sample_dataframe,
            topics=sample_topics,
            summary_mode="none",
            top_n_topics=1,
        )

        # Verify summarizer was not called
        mock_summarizer.summarize_batch.assert_not_called()

        # Verify articles have full text instead of summaries
        for article in newsletter.topics[0].articles:
            assert article.summary in sample_dataframe["text"].values

    @patch(
        "bertrend.LLM_CONFIG",
        {"model": "gpt-4", "api_key": "test-key", "base_url": "test-base_url"},
    )
    @patch("bertrend.llm_utils.newsletter_features.OpenAI_Client")
    @patch("bertrend.services.summary.chatgpt_summarizer.GPTSummarizer")
    @patch("bertrend.llm_utils.newsletter_features.get_most_representative_docs")
    @patch("tldextract.extract")
    def test_generate_newsletter_improve_topic_description(
        self,
        mock_tldextract,
        mock_get_docs,
        mock_summarizer_class,
        mock_openai_class,
        mock_bertopic_model,
        sample_dataframe,
        sample_topics,
        mock_summarizer,
        mock_openai_client,
    ):
        """Test newsletter generation with improved topic descriptions."""
        # Setup mocks
        mock_summarizer_class.return_value = mock_summarizer
        mock_openai_class.return_value = mock_openai_client
        mock_tldextract.return_value.domain = "example.com"
        mock_get_docs.side_effect = lambda **kwargs: sample_dataframe.head(2)

        # Mock improved topic description
        mock_openai_client.generate.return_value = '"Improved Topic Title"'

        # Generate newsletter with improved topic descriptions
        newsletter = generate_newsletter(
            topic_model=mock_bertopic_model,
            df=sample_dataframe,
            topics=sample_topics,
            improve_topic_description=True,
            top_n_topics=1,
        )

        # Verify OpenAI was called for topic improvement
        mock_openai_client.generate.assert_called()

        # Verify topic title was improved (quotes removed)
        assert newsletter.topics[0].title == "Improved Topic Title"

    @patch(
        "bertrend.LLM_CONFIG",
        {"model": "gpt-4", "api_key": "test-key", "base_url": "test-base_url"},
    )
    @patch("bertrend.llm_utils.newsletter_features.OpenAI_Client")
    @patch("bertrend.services.summary.chatgpt_summarizer.GPTSummarizer")
    @patch("bertrend.llm_utils.newsletter_features.get_most_representative_docs")
    @patch("tldextract.extract")
    def test_generate_newsletter_url_extraction_error(
        self,
        mock_tldextract,
        mock_get_docs,
        mock_summarizer_class,
        mock_openai_class,
        mock_bertopic_model,
        sample_dataframe,
        sample_topics,
        mock_summarizer,
        mock_openai_client,
    ):
        """Test newsletter generation when URL extraction fails."""
        # Setup mocks
        mock_summarizer_class.return_value = mock_summarizer
        mock_openai_class.return_value = mock_openai_client
        mock_tldextract.side_effect = Exception("URL extraction failed")
        mock_get_docs.side_effect = lambda **kwargs: sample_dataframe.head(1)

        # Generate newsletter (should handle URL extraction error gracefully)
        newsletter = generate_newsletter(
            topic_model=mock_bertopic_model,
            df=sample_dataframe,
            topics=sample_topics,
            top_n_topics=1,
        )

        # Verify newsletter was created despite URL extraction error
        assert isinstance(newsletter, Newsletter)
        assert len(newsletter.topics) == 1
        assert newsletter.topics[0].articles[0].source is None

    @patch(
        "bertrend.LLM_CONFIG",
        {"model": "gpt-4", "api_key": "test-key", "base_url": "test-base_url"},
    )
    @patch("bertrend.llm_utils.newsletter_features.OpenAI_Client")
    @patch("bertrend.services.summary.chatgpt_summarizer.GPTSummarizer")
    @patch("bertrend.llm_utils.newsletter_features.get_most_representative_docs")
    def test_generate_newsletter_auto_limit_topics(
        self,
        mock_get_docs,
        mock_summarizer_class,
        mock_openai_class,
        mock_bertopic_model,
        sample_dataframe,
        sample_topics,
        mock_summarizer,
        mock_openai_client,
    ):
        """Test that top_n_topics is automatically limited by available topics."""
        # Setup mocks
        mock_summarizer_class.return_value = mock_summarizer
        mock_openai_class.return_value = mock_openai_client
        mock_get_docs.side_effect = lambda **kwargs: sample_dataframe.head(1)

        # Generate newsletter with more topics than available
        newsletter = generate_newsletter(
            topic_model=mock_bertopic_model,
            df=sample_dataframe,
            topics=sample_topics,
            top_n_topics=100,  # More than available
        )

        # Should be limited to available topics (3 in mock)
        assert len(newsletter.topics) < 100

    def test_generate_newsletter_defaults(self):
        """Test that default values are correctly set."""
        assert DEFAULT_TOP_N_TOPICS == 5
        assert DEFAULT_TOP_N_DOCS == 3
        assert DEFAULT_TOP_N_DOCS_MODE == "cluster_probability"
        assert DEFAULT_SUMMARY_MODE == "document"


class TestNewsletterRendering:
    """Test cases for newsletter rendering functionality."""

    @pytest.fixture
    def sample_newsletter(self):
        """Sample newsletter for testing."""
        articles = [
            Article(
                title="AI Breakthrough",
                url="https://example.com/ai",
                summary="AI summary",
                date=date(2024, 1, 15),
                source="example.com",
            ),
            Article(
                title="Climate News",
                url="https://news.com/climate",
                summary="Climate summary",
                date=date(2024, 1, 16),
                source="news.com",
            ),
        ]

        topics = [
            Topic(
                title="Technology",
                hashtags=["ai", "tech"],
                summary="Tech topic summary",
                articles=[articles[0]],
                topic_type=STRONG_TOPIC_TYPE,
            ),
            Topic(
                title="Environment",
                hashtags=["climate", "environment"],
                summary="Environment topic summary",
                articles=[articles[1]],
                topic_type=STRONG_TOPIC_TYPE,
            ),
        ]

        return Newsletter(
            title="Test Newsletter",
            period_start_date=date(2024, 1, 15),
            period_end_date=date(2024, 1, 20),
            topics=topics,
        )

    def test_render_newsletter_md(self, sample_newsletter):
        """Test Markdown rendering of newsletter."""
        md_content = render_newsletter_md(sample_newsletter)

        # Check main components are present
        assert "# Test Newsletter" in md_content
        assert "Period: January 15, 2024 to January 20, 2024" in md_content
        assert "## Technology" in md_content
        assert "## Environment" in md_content
        assert "### AI Breakthrough" in md_content
        assert "### Climate News" in md_content
        assert "Hashtags: #ai #tech" in md_content
        assert "Hashtags: #climate #environment" in md_content
        assert "[Link to Article](https://example.com/ai)" in md_content
        assert "Source: example.com" in md_content
        assert "AI summary" in md_content
        assert "---" in md_content  # Topic separator

    def test_render_newsletter_md_minimal(self):
        """Test Markdown rendering with minimal newsletter data."""
        newsletter = Newsletter(
            title="Minimal Newsletter",
            period_start_date=date(2024, 1, 1),
            period_end_date=date(2024, 1, 31),
            topics=[],
        )

        md_content = render_newsletter_md(newsletter)

        assert "# Minimal Newsletter" in md_content
        assert "Period: January 01, 2024 to January 31, 2024" in md_content
        assert "##" not in md_content  # No topics

    def test_render_newsletter_md_no_optional_fields(self):
        """Test Markdown rendering when optional fields are missing."""
        article = Article(
            title="Simple Article",
            date=date(2024, 1, 15),
            # No URL, summary, or source
        )

        topic = Topic(
            title="Simple Topic",
            hashtags=[],
            articles=[article],
            # No summary
        )

        newsletter = Newsletter(
            title="Simple Newsletter",
            period_start_date=date(2024, 1, 1),
            period_end_date=date(2024, 1, 31),
            topics=[topic],
        )

        md_content = render_newsletter_md(newsletter)

        assert "### Simple Article" in md_content
        assert "Date: January 15, 2024" in md_content
        assert "Hashtags:" not in md_content  # Empty hashtags
        assert "Source:" not in md_content  # No source
        assert "[Link to Article]" not in md_content  # No URL

    @patch("jinja2.Template")
    def test_render_newsletter_html(self, mock_template, sample_newsletter):
        """Test HTML rendering of newsletter."""
        # Mock template
        mock_template_instance = Mock()
        mock_template_instance.render.return_value = "<html>Test HTML</html>"
        mock_template.return_value = mock_template_instance

        # Mock file reading
        with patch(
            "builtins.open", mock_open(read_data="<html>{{ newsletter.title }}</html>")
        ):
            html_content = render_newsletter_html(
                newsletter=sample_newsletter,
                html_template=Path("test_template.html"),
                custom_css=Path("test.css"),
                language="en",
            )

        assert html_content == "<html>Test HTML</html>"

    @patch("bertrend.llm_utils.newsletter_features.render_newsletter_md")
    def test_render_newsletter_md_format(self, mock_render_md, sample_newsletter):
        """Test rendering newsletter to file in MD format."""
        mock_render_md.return_value = "# Test Newsletter\nContent"

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_newsletter.md"

            render_newsletter(
                newsletter=sample_newsletter, path=output_path, output_format="md"
            )

            # Verify file was created and contains expected content
            assert output_path.exists()
            with open(output_path, "r") as f:
                content = f.read()
            assert content == "# Test Newsletter\nContent"

    @patch("bertrend.llm_utils.newsletter_features.render_newsletter_html")
    @patch("inspect.getfile")
    def test_render_newsletter_html_format(
        self, mock_getfile, mock_render_html, sample_newsletter
    ):
        """Test rendering newsletter to file in HTML format."""
        mock_render_html.return_value = "<html>Test HTML</html>"
        mock_getfile.return_value = "/path/to/module.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_newsletter.html"

            render_newsletter(
                newsletter=sample_newsletter,
                path=output_path,
                output_format="html",
                language="en",
            )

            # Verify file was created
            assert output_path.exists()
            with open(output_path, "r") as f:
                content = f.read()
            assert content == "<html>Test HTML</html>"

            # Verify render_newsletter_html was called with correct parameters
            mock_render_html.assert_called_once()

    def test_render_newsletter_creates_directory(self, sample_newsletter):
        """Test that render_newsletter creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "subdir" / "test_newsletter.md"

            render_newsletter(
                newsletter=sample_newsletter, path=output_path, output_format="md"
            )

            # Verify directory was created
            assert output_path.parent.exists()
            assert output_path.exists()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame(columns=["text", "title", "url", "timestamp"])

        with pytest.raises(Exception):
            # This should raise an exception due to empty dataframe
            generate_newsletter(topic_model=Mock(), df=df, topics=[], top_n_topics=1)

    def test_newsletter_with_no_topics(self):
        """Test rendering newsletter with no topics."""
        newsletter = Newsletter(
            title="Empty Newsletter",
            period_start_date=date(2024, 1, 1),
            period_end_date=date(2024, 1, 31),
            topics=[],
        )

        md_content = render_newsletter_md(newsletter)

        assert "# Empty Newsletter" in md_content
        assert "Period: January 01, 2024 to January 31, 2024" in md_content
        assert "##" not in md_content  # No topic headers

    def test_article_with_invalid_url(self):
        """Test handling of articles with invalid URLs."""
        article = Article(
            title="Test Article", url="invalid-url-format", date=date(2024, 1, 15)
        )

        topic = Topic(title="Test Topic", hashtags=["test"], articles=[article])

        newsletter = Newsletter(
            title="Test Newsletter",
            period_start_date=date(2024, 1, 1),
            period_end_date=date(2024, 1, 31),
            topics=[topic],
        )

        # Should render without errors
        md_content = render_newsletter_md(newsletter)
        assert "invalid-url-format" in md_content
