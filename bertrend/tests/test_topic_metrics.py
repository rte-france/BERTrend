#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from gensim.models.coherencemodel import CoherenceModel

from bertrend.metrics.topic_metrics import (
    get_coherence_value,
    get_diversity_value,
    compute_cluster_metrics,
    proportion_unique_words,
    pairwise_jaccard_diversity,
)


@pytest.fixture
def mock_bertopic():
    """Create a mock BERTopic model for testing."""
    mock_model = MagicMock()

    # Mock the _preprocess_text method
    mock_model._preprocess_text.return_value = [
        ["word1", "word2", "word3"],
        ["word2", "word3", "word4"],
        ["word3", "word4", "word5"],
    ]

    # Mock the vectorizer_model
    mock_vectorizer = MagicMock()
    mock_analyzer = MagicMock()
    mock_analyzer.return_value = ["word1", "word2", "word3"]
    mock_vectorizer.build_analyzer.return_value = mock_analyzer
    mock_vectorizer.get_feature_names_out.return_value = [
        "word1",
        "word2",
        "word3",
        "word4",
        "word5",
    ]
    mock_model.vectorizer_model = mock_vectorizer

    # Mock the get_topic method
    mock_model.get_topic.return_value = [("word1", 0.9), ("word2", 0.8), ("word3", 0.7)]

    # Set _outliers attribute
    mock_model._outliers = 1

    return mock_model


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {
        "docs": ["This is document 1", "This is document 2", "This is document 3"],
        "topics": [0, 1, 2],
        "topic_words": [
            ["word1", "word2", "word3"],
            ["word2", "word3", "word4"],
            ["word3", "word4", "word5"],
        ],
    }


def test_proportion_unique_words():
    """Test the proportion_unique_words function."""
    topics = [
        ["word1", "word2", "word3"],
        ["word2", "word3", "word4"],
        ["word3", "word4", "word5"],
    ]

    # With top_k=3, we have 5 unique words out of 9 total words
    result = proportion_unique_words(topics, 3)
    expected = 5 / 9  # 5 unique words / (3 topics * 3 words per topic)
    assert result == expected

    # With top_k=2, we have 4 unique words out of 6 total words
    result = proportion_unique_words(topics, 2)
    expected = 4 / 6  # 4 unique words / (3 topics * 2 words per topic)
    assert result == expected


def test_pairwise_jaccard_diversity():
    """Test the pairwise_jaccard_diversity function."""
    topics = [
        ["word1", "word2", "word3"],
        ["word2", "word3", "word4"],
        ["word3", "word4", "word5"],
    ]

    # Calculate expected Jaccard distances manually
    # For topics[0] and topics[1]: JS = 1 - |{word1, word2, word3} ∩ {word2, word3, word4}| / |{word1, word2, word3} ∪ {word2, word3, word4}|
    #                             = 1 - |{word2, word3}| / |{word1, word2, word3, word4}|
    #                             = 1 - 2/4 = 0.5
    # For topics[0] and topics[2]: JS = 1 - |{word1, word2, word3} ∩ {word3, word4, word5}| / |{word1, word2, word3} ∪ {word3, word4, word5}|
    #                             = 1 - |{word3}| / |{word1, word2, word3, word4, word5}|
    #                             = 1 - 1/5 = 0.8
    # For topics[1] and topics[2]: JS = 1 - |{word2, word3, word4} ∩ {word3, word4, word5}| / |{word2, word3, word4} ∪ {word3, word4, word5}|
    #                             = 1 - |{word3, word4}| / |{word2, word3, word4, word5}|
    #                             = 1 - 2/4 = 0.5
    # Average = (0.5 + 0.8 + 0.5) / 3 = 0.6

    result = pairwise_jaccard_diversity(topics, 3)
    expected = 0.6
    assert pytest.approx(result) == expected


def test_get_coherence_value(mock_bertopic, sample_data):
    """Test the get_coherence_value function."""
    # Mock the CoherenceModel
    mock_coherence_model = MagicMock(spec=CoherenceModel)
    mock_coherence_model.get_coherence.return_value = 0.75

    with patch(
        "bertrend.metrics.topic_metrics.CoherenceModel",
        return_value=mock_coherence_model,
    ):
        result = get_coherence_value(
            mock_bertopic, sample_data["topics"], sample_data["docs"], "c_npmi"
        )

        assert result == 0.75
        # Verify CoherenceModel was called with the correct parameters
        assert mock_coherence_model.get_coherence.called


def test_get_diversity_value_puw(mock_bertopic, sample_data):
    """Test the get_diversity_value function with puw diversity score type."""
    with patch(
        "bertrend.metrics.topic_metrics.proportion_unique_words", return_value=0.6
    ) as mock_puw:
        result = get_diversity_value(
            mock_bertopic, sample_data["topics"], sample_data["docs"], "puw", 5
        )

        assert result == 0.6
        assert mock_puw.called


def test_get_diversity_value_pjd(mock_bertopic, sample_data):
    """Test the get_diversity_value function with pjd diversity score type."""
    with patch(
        "bertrend.metrics.topic_metrics.pairwise_jaccard_diversity", return_value=0.7
    ) as mock_pjd:
        result = get_diversity_value(
            mock_bertopic, sample_data["topics"], sample_data["docs"], "pjd", 5
        )

        assert result == 0.7
        assert mock_pjd.called


def test_get_diversity_value_invalid_type(mock_bertopic, sample_data):
    """Test the get_diversity_value function with an invalid diversity score type."""
    with pytest.raises(ValueError, match="Unknown diversity score type: invalid_type"):
        get_diversity_value(
            mock_bertopic, sample_data["topics"], sample_data["docs"], "invalid_type", 5
        )


def test_compute_cluster_metrics(mock_bertopic, sample_data):
    """Test the compute_cluster_metrics function."""
    with patch(
        "bertrend.metrics.topic_metrics.get_coherence_value", return_value=0.75
    ) as mock_coherence, patch(
        "bertrend.metrics.topic_metrics.get_diversity_value", return_value=0.6
    ) as mock_diversity, patch(
        "bertrend.metrics.topic_metrics.logger"
    ) as mock_logger:

        compute_cluster_metrics(
            mock_bertopic, sample_data["topics"], sample_data["docs"]
        )

        # Verify that the functions were called with the correct parameters
        mock_coherence.assert_called_once_with(
            mock_bertopic, sample_data["topics"], sample_data["docs"], "c_npmi"
        )

        mock_diversity.assert_called_once_with(
            mock_bertopic,
            sample_data["topics"],
            sample_data["docs"],
            diversity_score_type="puw",
        )

        # Verify that the logger was called with the correct messages
        mock_logger.info.assert_called_once()
        mock_logger.success.assert_called()


def test_compute_cluster_metrics_coherence_error(mock_bertopic, sample_data):
    """Test the compute_cluster_metrics function when coherence calculation raises an error."""
    with patch(
        "bertrend.metrics.topic_metrics.get_coherence_value",
        side_effect=IndexError("Test error"),
    ) as mock_coherence, patch(
        "bertrend.metrics.topic_metrics.get_diversity_value", return_value=0.6
    ) as mock_diversity, patch(
        "bertrend.metrics.topic_metrics.logger"
    ) as mock_logger:

        compute_cluster_metrics(
            mock_bertopic, sample_data["topics"], sample_data["docs"]
        )

        # Verify that the error was logged
        mock_logger.error.assert_called()


def test_compute_cluster_metrics_diversity_error(mock_bertopic, sample_data):
    """Test the compute_cluster_metrics function when diversity calculation raises an error."""
    with patch(
        "bertrend.metrics.topic_metrics.get_coherence_value", return_value=0.75
    ) as mock_coherence, patch(
        "bertrend.metrics.topic_metrics.get_diversity_value",
        side_effect=IndexError("Test error"),
    ) as mock_diversity, patch(
        "bertrend.metrics.topic_metrics.logger"
    ) as mock_logger:

        compute_cluster_metrics(
            mock_bertopic, sample_data["topics"], sample_data["docs"]
        )

        # Verify that the error was logged
        mock_logger.error.assert_called()
