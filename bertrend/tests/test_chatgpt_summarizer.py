#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
from unittest.mock import patch, MagicMock

from bertrend.services.summary.chatgpt_summarizer import (
    GPTSummarizer,
    keep_first_n_words,
)
from bertrend.services.summary.prompts import (
    FR_SYSTEM_SUMMARY_SENTENCES,
    EN_SYSTEM_SUMMARY_SENTENCES,
)
from bertrend.services.summarizer import DEFAULT_MAX_SENTENCES


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing."""
    mock = MagicMock()
    mock.generate.return_value = "This is a mock summary."
    return mock


@pytest.fixture
def gpt_summarizer(mock_openai_client):
    """Create a GPTSummarizer with a mocked OpenAI client."""
    with patch(
        "bertrend.services.summary.chatgpt_summarizer.OpenAI_Client",
        return_value=mock_openai_client,
    ):
        summarizer = GPTSummarizer(api_key="test_key", endpoint="test_endpoint")
        # Set the mock client directly to ensure it's used
        summarizer.api = mock_openai_client
        return summarizer


def test_initialization():
    """Test that the GPTSummarizer initializes with the correct parameters."""
    with patch(
        "bertrend.services.summary.chatgpt_summarizer.OpenAI_Client"
    ) as mock_client_class:
        # Create a summarizer with custom API key and endpoint
        GPTSummarizer(api_key="custom_key", endpoint="custom_endpoint")

        # Check that OpenAI_Client was initialized with the correct parameters
        mock_client_class.assert_called_once()
        args, kwargs = mock_client_class.call_args
        assert kwargs["api_key"] == "custom_key"
        assert kwargs["endpoint"] == "custom_endpoint"


def test_generate_summary_french(gpt_summarizer, mock_openai_client):
    """Test generate_summary with French language."""
    article_text = "Ceci est un article de test."

    result = gpt_summarizer.generate_summary(article_text)

    # Check that the API was called with the correct parameters
    mock_openai_client.generate.assert_called_once_with(
        system_prompt=FR_SYSTEM_SUMMARY_SENTENCES.format(
            num_sentences=DEFAULT_MAX_SENTENCES
        ),
        user_prompt=article_text,
    )

    # Check the result
    assert result == "This is a mock summary."


def test_generate_summary_english(gpt_summarizer, mock_openai_client):
    """Test generate_summary with English language."""
    article_text = "This is a test article."

    result = gpt_summarizer.generate_summary(article_text, prompt_language="en")

    # Check that the API was called with the correct parameters
    mock_openai_client.generate.assert_called_once_with(
        system_prompt=EN_SYSTEM_SUMMARY_SENTENCES.format(
            num_sentences=DEFAULT_MAX_SENTENCES
        ),
        user_prompt=article_text,
    )

    # Check the result
    assert result == "This is a mock summary."


def test_generate_summary_custom_params(gpt_summarizer, mock_openai_client):
    """Test generate_summary with custom parameters."""
    article_text = "This is a test article."

    result = gpt_summarizer.generate_summary(
        article_text,
        max_sentences=5,
        prompt_language="en",
        max_article_length=1000,
        model_name="gpt-4",
    )

    # Check that the API was called with the correct parameters
    mock_openai_client.generate.assert_called_once_with(
        system_prompt=EN_SYSTEM_SUMMARY_SENTENCES.format(num_sentences=5),
        user_prompt=article_text,
    )

    # Check the result
    assert result == "This is a mock summary."


def test_keep_first_n_words():
    """Test the keep_first_n_words function."""
    # Test with a short text
    short_text = "This is a short text."
    assert keep_first_n_words(short_text, 10) == short_text

    # Test with a long text
    long_text = (
        "This is a longer text with more than ten words that should be truncated."
    )
    assert (
        keep_first_n_words(long_text, 10)
        == "This is a longer text with more than ten words"
    )

    # Test with n=0
    assert keep_first_n_words(short_text, 0) == ""

    # Test with empty text
    assert keep_first_n_words("", 10) == ""


def test_long_article_truncation(gpt_summarizer, mock_openai_client):
    """Test that long articles are truncated before being sent to the API."""
    # Create a long article
    words = ["word"] * 2000
    long_article = " ".join(words)

    # Call generate_summary with a max_article_length of 1000
    gpt_summarizer.generate_summary(long_article, max_article_length=1000)

    # Get the user_prompt that was sent to the API
    args, kwargs = mock_openai_client.generate.call_args
    user_prompt = kwargs["user_prompt"]

    # Check that the user_prompt has been truncated
    assert len(user_prompt.split()) == 1000
