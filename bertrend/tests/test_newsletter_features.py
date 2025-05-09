#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
import os
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from bertrend.llm_utils.newsletter_features import (
    generate_newsletter,
    export_md_string,
    md2html,
)


@pytest.fixture
def mock_topic_model():
    """Create a mock BERTopic model for testing."""
    mock_model = MagicMock()
    topic_info = pd.DataFrame(
        {
            "Topic": [-1, 0, 1],
            "Count": [10, 5, 3],
            "Name": ["", "", ""],
            "Representation": [
                [""],
                ["keyword1", "keyword2"],
                ["keyword3", "keyword4"],
            ],
        }
    )
    mock_model.get_topic_info.return_value = topic_info
    return mock_model


@pytest.fixture
def mock_documents_df():
    """Create a mock DataFrame with documents for testing."""
    return pd.DataFrame(
        {
            "title": ["Document 1", "Document 2", "Document 3"],
            "text": [
                "Content of document 1",
                "Content of document 2",
                "Content of document 3",
            ],
            "url": [
                "http://example1.com",
                "http://example2.com",
                "http://example3.com",
            ],
            "timestamp": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-01-03"),
            ],
        }
    )


@pytest.fixture
def mock_topics():
    """Create mock topic assignments for testing."""
    return [0, 1, 0]


@pytest.fixture
def mock_summarizer():
    """Create a mock Summarizer for testing."""
    mock_sum = MagicMock()
    mock_sum.summarize_batch.return_value = ["Summary 1", "Summary 2", "Summary 3"]
    return mock_sum


@patch("bertrend.llm_utils.newsletter_features.OpenAI_Client")
@patch("bertrend.llm_utils.newsletter_features.get_most_representative_docs")
def test_generate_newsletter_document_mode(
    mock_get_docs,
    mock_openai_client,
    mock_topic_model,
    mock_documents_df,
    mock_topics,
    mock_summarizer,
):
    """Test generate_newsletter with document summary mode."""
    # Setup
    mock_get_docs.return_value = mock_documents_df
    mock_client_instance = MagicMock()
    mock_openai_client.return_value = mock_client_instance

    # Call function
    result, date_min, date_max = generate_newsletter(
        topic_model=mock_topic_model,
        df=mock_documents_df,
        topics=mock_topics,
        top_n_topics=2,
        summarizer_class=lambda: mock_summarizer,
        summary_mode="document",
        prompt_language="en",  # Set English explicitly to ensure consistent date format
    )

    # Assertions
    assert isinstance(result, str)
    assert "# Newsletter" in result
    assert "## Sujet 1 : keyword1, keyword2" in result
    assert mock_summarizer.summarize_batch.called
    assert date_min == "Sunday 01 Jan 2023"
    assert date_max == "Tuesday 03 Jan 2023"


@patch("bertrend.llm_utils.newsletter_features.OpenAI_Client")
@patch("bertrend.llm_utils.newsletter_features.get_most_representative_docs")
def test_generate_newsletter_topic_mode(
    mock_get_docs,
    mock_openai_client,
    mock_topic_model,
    mock_documents_df,
    mock_topics,
    mock_summarizer,
):
    """Test generate_newsletter with topic summary mode."""
    # Setup
    mock_get_docs.return_value = mock_documents_df
    mock_client_instance = MagicMock()
    mock_client_instance.generate.return_value = "Topic summary text"
    mock_openai_client.return_value = mock_client_instance

    # Call function
    result, _, _ = generate_newsletter(
        topic_model=mock_topic_model,
        df=mock_documents_df,
        topics=mock_topics,
        top_n_topics=2,
        summarizer_class=lambda: mock_summarizer,
        summary_mode="topic",
        prompt_language="en",  # Set English explicitly to ensure consistent date format
    )

    # Assertions
    assert isinstance(result, str)
    assert "# Newsletter" in result
    assert "Topic summary text" in result
    assert mock_client_instance.generate.called


@patch("bertrend.llm_utils.newsletter_features.OpenAI_Client")
@patch("bertrend.llm_utils.newsletter_features.get_most_representative_docs")
def test_generate_newsletter_none_mode(
    mock_get_docs,
    mock_openai_client,
    mock_topic_model,
    mock_documents_df,
    mock_topics,
    mock_summarizer,
):
    """Test generate_newsletter with no summary mode."""
    # Setup
    mock_get_docs.return_value = mock_documents_df
    mock_client_instance = MagicMock()
    mock_openai_client.return_value = mock_client_instance

    # Call function
    result, _, _ = generate_newsletter(
        topic_model=mock_topic_model,
        df=mock_documents_df,
        topics=mock_topics,
        top_n_topics=2,
        summarizer_class=lambda: mock_summarizer,
        summary_mode="none",
        prompt_language="en",  # Set English explicitly to ensure consistent date format
    )

    # Assertions
    assert isinstance(result, str)
    assert "# Newsletter" in result
    assert "Content of document" in result
    assert not mock_summarizer.summarize_batch.called


@patch("bertrend.llm_utils.newsletter_features.OpenAI_Client")
@patch("bertrend.llm_utils.newsletter_features.get_most_representative_docs")
def test_generate_newsletter_improve_topic(
    mock_get_docs,
    mock_openai_client,
    mock_topic_model,
    mock_documents_df,
    mock_topics,
    mock_summarizer,
):
    """Test generate_newsletter with improved topic descriptions."""
    # Setup
    mock_get_docs.return_value = mock_documents_df
    mock_client_instance = MagicMock()
    mock_client_instance.generate.return_value = "Improved topic description"
    mock_openai_client.return_value = mock_client_instance

    # Call function
    result, _, _ = generate_newsletter(
        topic_model=mock_topic_model,
        df=mock_documents_df,
        topics=mock_topics,
        top_n_topics=2,
        summarizer_class=lambda: mock_summarizer,
        summary_mode="document",
        improve_topic_description=True,
        prompt_language="en",  # Set English explicitly to ensure consistent date format
    )

    # Assertions
    assert isinstance(result, str)
    assert "## Sujet 1 : Improved topic description" in result
    assert mock_client_instance.generate.called


@patch("bertrend.llm_utils.newsletter_features.OpenAI_Client")
@patch("bertrend.llm_utils.newsletter_features.get_most_representative_docs")
def test_generate_newsletter_french_language(
    mock_get_docs,
    mock_openai_client,
    mock_topic_model,
    mock_documents_df,
    mock_topics,
    mock_summarizer,
):
    """Test generate_newsletter with French language setting."""
    # Setup
    mock_get_docs.return_value = mock_documents_df
    mock_client_instance = MagicMock()
    mock_openai_client.return_value = mock_client_instance

    # Call function
    result, date_min, date_max = generate_newsletter(
        topic_model=mock_topic_model,
        df=mock_documents_df,
        topics=mock_topics,
        top_n_topics=2,
        summarizer_class=lambda: mock_summarizer,
        summary_mode="document",
        prompt_language="fr",  # Set French explicitly
    )

    # Assertions
    assert isinstance(result, str)
    assert "# Newsletter" in result
    # Check for French date format and text
    assert "du " in result  # French date range text
    # We don't check the exact date strings as they depend on the system's locale configuration
    # but we can check that they're not in English format
    assert "Sunday" not in date_min
    assert "Tuesday" not in date_max


@patch("bertrend.llm_utils.newsletter_features.OpenAI_Client")
@patch("bertrend.llm_utils.newsletter_features.get_most_representative_docs")
@patch("bertrend.llm_utils.newsletter_features.logger")
@patch("bertrend.llm_utils.newsletter_features.exit")
def test_generate_newsletter_invalid_summary_mode(
    mock_exit,
    mock_logger,
    mock_get_docs,
    mock_openai_client,
    mock_topic_model,
    mock_documents_df,
    mock_topics,
    mock_summarizer,
):
    """Test generate_newsletter with invalid summary mode."""
    # Setup
    mock_get_docs.return_value = mock_documents_df
    mock_client_instance = MagicMock()
    mock_openai_client.return_value = mock_client_instance

    # Call function
    generate_newsletter(
        topic_model=mock_topic_model,
        df=mock_documents_df,
        topics=mock_topics,
        top_n_topics=2,
        summarizer_class=lambda: mock_summarizer,
        summary_mode="invalid_mode",  # Invalid summary mode
        prompt_language="en",
    )

    # Assertions
    assert mock_logger.error.call_count >= 1  # Called at least once
    error_message = mock_logger.error.call_args_list[0][0][
        0
    ]  # Get the first call's arguments
    assert "invalid_mode is not a valid parameter" in error_message
    assert mock_exit.call_count >= 1  # Called at least once


def test_export_md_string_md_format():
    """Test exporting markdown string to a markdown file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "test_newsletter.md"
        test_content = "# Test Newsletter\n\nThis is a test."

        with patch("builtins.open", mock_open()) as mock_file:
            export_md_string(test_content, test_path, output_format="md")

            mock_file.assert_called_once_with(test_path, "w")
            mock_file().write.assert_called_once_with(test_content)


@patch("bertrend.llm_utils.newsletter_features.md2html")
def test_export_md_string_html_format(mock_md2html):
    """Test exporting markdown string to an HTML file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "test_newsletter.html"
        test_content = "# Test Newsletter\n\nThis is a test."
        mock_md2html.return_value = "<html>Converted HTML</html>"

        with patch("builtins.open", mock_open()) as mock_file:
            with patch("inspect.getfile", return_value=__file__):
                export_md_string(test_content, test_path, output_format="html")

                mock_file.assert_called_once_with(test_path, "w")
                mock_file().write.assert_called_once_with("<html>Converted HTML</html>")


def test_md2html_without_css():
    """Test converting markdown to HTML without CSS."""
    test_md = "# Test Heading\n\nTest paragraph."

    with patch(
        "markdown.markdown", return_value="<h1>Test Heading</h1><p>Test paragraph.</p>"
    ):
        result = md2html(test_md)

        assert result == "<h1>Test Heading</h1><p>Test paragraph.</p>"


def test_md2html_with_css():
    """Test converting markdown to HTML with CSS."""
    test_md = "# Test Heading\n\nTest paragraph."
    test_css = Path("test_css.css")

    with patch(
        "markdown.markdown", return_value="<h1>Test Heading</h1><p>Test paragraph.</p>"
    ):
        with patch("builtins.open", mock_open(read_data="body { color: black; }")):
            result = md2html(test_md, test_css)

            assert "<!DOCTYPE html>" in result
            assert '<style type="text/css">' in result
            assert "body { color: black; }" in result
            assert "<h1>Test Heading</h1><p>Test paragraph.</p>" in result
            assert "</body>" in result
            assert "</html>" in result
