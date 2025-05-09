#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
from unittest.mock import patch

from bertrend.services.summarizer import (
    Summarizer,
    DEFAULT_MAX_SENTENCES,
    DEFAULT_MAX_WORDS,
    DEFAULT_SUMMARIZATION_RATIO,
)


class ConcreteSummarizer(Summarizer):
    """A concrete implementation of the Summarizer abstract class for testing purposes."""

    def generate_summary(self, article_text, **kwargs):
        """Simple implementation that returns the first sentence of the article."""
        if not article_text:
            return ""
        sentences = article_text.split(".")
        if sentences and sentences[0]:
            return sentences[0] + "."
        return ""


@pytest.fixture
def summarizer():
    """Create a concrete summarizer instance for testing."""
    return ConcreteSummarizer()


def test_summarizer_constants():
    """Test that the default constants are correctly defined."""
    assert DEFAULT_SUMMARIZATION_RATIO == 0.2
    assert DEFAULT_MAX_SENTENCES == 3
    assert DEFAULT_MAX_WORDS == 50


def test_generate_summary(summarizer):
    """Test the generate_summary method of the concrete implementation."""
    text = "This is the first sentence. This is the second sentence."
    summary = summarizer.generate_summary(text)
    assert summary == "This is the first sentence."


def test_summarize_batch(summarizer):
    """Test the default implementation of summarize_batch."""
    texts = [
        "This is article one. It has multiple sentences.",
        "This is article two. It also has multiple sentences.",
    ]

    # Mock the generate_summary method to track calls
    with patch.object(
        summarizer, "generate_summary", wraps=summarizer.generate_summary
    ) as mock_generate:
        summaries = summarizer.summarize_batch(texts)

        # Check that generate_summary was called for each text
        assert mock_generate.call_count == 2

        # Check that the correct parameters were passed
        mock_generate.assert_any_call(
            article_text=texts[0],
            max_sentences=DEFAULT_MAX_SENTENCES,
            max_words=DEFAULT_MAX_WORDS,
            max_length_ratio=DEFAULT_SUMMARIZATION_RATIO,
            prompt_language="fr",
            model_name=None,
        )

        # Check the results
        assert len(summaries) == 2
        assert summaries[0] == "This is article one."
        assert summaries[1] == "This is article two."


def test_summarize_batch_with_custom_params(summarizer):
    """Test summarize_batch with custom parameters."""
    texts = ["This is a test article."]

    with patch.object(
        summarizer, "generate_summary", wraps=summarizer.generate_summary
    ) as mock_generate:
        summarizer.summarize_batch(
            texts,
            max_sentences=5,
            max_words=100,
            max_length_ratio=0.3,
            prompt_language="en",
            model_name="custom-model",
        )

        # Check that the custom parameters were passed to generate_summary
        mock_generate.assert_called_once_with(
            article_text=texts[0],
            max_sentences=5,
            max_words=100,
            max_length_ratio=0.3,
            prompt_language="en",
            model_name="custom-model",
        )


def test_empty_input(summarizer):
    """Test behavior with empty input."""
    # Empty string
    assert summarizer.generate_summary("") == ""

    # Empty list
    assert summarizer.summarize_batch([]) == []
