#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
from unittest.mock import patch, MagicMock, Mock

from bertrend.services.summary.abstractive_summarizer import (
    AbstractiveSummarizer,
    DEFAULT_ABSTRACTIVE_MODEL,
)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    mock = MagicMock()
    mock.return_value = MagicMock(
        input_ids=MagicMock(to=lambda device: MagicMock()),
        attention_mask=MagicMock(to=lambda device: MagicMock()),
    )
    mock.decode.side_effect = lambda output_id, **kwargs: f"Summary of text {output_id}"
    return mock


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    mock = MagicMock()
    mock.generate.return_value = [1, 2]  # Mock output IDs
    mock.to.return_value = mock  # Return self for chaining
    return mock


@pytest.fixture
def mock_abstractive_summarizer(mock_tokenizer, mock_model):
    """Create a mock AbstractiveSummarizer with mocked dependencies."""
    with patch(
        "bertrend.services.summary.abstractive_summarizer.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    ), patch(
        "bertrend.services.summary.abstractive_summarizer.AutoModelForSeq2SeqLM.from_pretrained",
        return_value=mock_model,
    ):
        summarizer = AbstractiveSummarizer()
        # Set the mocks directly to ensure they're used
        summarizer.tokenizer = mock_tokenizer
        summarizer.model = mock_model
        return summarizer


def test_initialization():
    """Test that the AbstractiveSummarizer initializes with the correct model."""
    with patch(
        "bertrend.services.summary.abstractive_summarizer.AutoTokenizer.from_pretrained"
    ) as mock_tokenizer, patch(
        "bertrend.services.summary.abstractive_summarizer.AutoModelForSeq2SeqLM.from_pretrained"
    ) as mock_model:

        # Mock the to() method to avoid device-related issues
        mock_model.return_value.to.return_value = mock_model.return_value

        summarizer = AbstractiveSummarizer()

        # Check that the correct model was loaded
        mock_tokenizer.assert_called_once_with(DEFAULT_ABSTRACTIVE_MODEL)
        mock_model.assert_called_once_with(DEFAULT_ABSTRACTIVE_MODEL)


def test_generate_summary(mock_abstractive_summarizer):
    """Test that generate_summary calls summarize_batch with the correct parameters."""
    with patch.object(
        mock_abstractive_summarizer, "summarize_batch", return_value=["Test summary"]
    ) as mock_summarize_batch:
        result = mock_abstractive_summarizer.generate_summary("Test article")

        # Check that summarize_batch was called with the correct parameters
        mock_summarize_batch.assert_called_once_with(["Test article"])

        # Check that the result is correct
        assert result == "Test summary"


def test_summarize_batch(mock_abstractive_summarizer, mock_tokenizer, mock_model):
    """Test the summarize_batch method."""
    # Setup test data
    article_texts = ["Article 1", "Article 2"]

    # Call the method
    result = mock_abstractive_summarizer.summarize_batch(article_texts)

    # Check that the tokenizer was called with the correct parameters
    mock_tokenizer.assert_called_once()

    # Check that the model.generate was called
    mock_model.generate.assert_called_once()

    # Check that the result has the correct length
    assert len(result) == 2

    # Check that the tokenizer.decode was called for each output ID
    assert mock_tokenizer.decode.call_count == 2


def test_whitespace_handler(mock_abstractive_summarizer):
    """Test the WHITESPACE_HANDLER function."""
    # Test with various whitespace patterns
    text_with_newlines = "This is\na test\n\nwith newlines"
    text_with_spaces = "This   is  a   test  with   spaces"

    # Apply the whitespace handler
    cleaned_newlines = mock_abstractive_summarizer.WHITESPACE_HANDLER(
        text_with_newlines
    )
    cleaned_spaces = mock_abstractive_summarizer.WHITESPACE_HANDLER(text_with_spaces)

    # Check the results
    assert cleaned_newlines == "This is a test with newlines"
    assert cleaned_spaces == "This is a test with spaces"


def test_custom_model():
    """Test initialization with a custom model."""
    custom_model = "custom/model"

    with patch(
        "bertrend.services.summary.abstractive_summarizer.AutoTokenizer.from_pretrained"
    ) as mock_tokenizer, patch(
        "bertrend.services.summary.abstractive_summarizer.AutoModelForSeq2SeqLM.from_pretrained"
    ) as mock_model:

        # Mock the to() method to avoid device-related issues
        mock_model.return_value.to.return_value = mock_model.return_value

        summarizer = AbstractiveSummarizer(model_name=custom_model)

        # Check that the correct model was loaded
        mock_tokenizer.assert_called_once_with(custom_model)
        mock_model.assert_called_once_with(custom_model)
