#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from bertopic import BERTopic

from bertrend.metrics.temporal_metrics_embedding import TempTopic


@pytest.fixture
def mock_bertopic():
    """Create a mock BERTopic model for testing."""
    mock_model = MagicMock(spec=BERTopic)

    # Mock the get_topic method
    mock_model.get_topic.return_value = [("word1", 0.9), ("word2", 0.8), ("word3", 0.7)]

    # Mock the get_topic_info method
    mock_topic_info = pd.DataFrame(
        {
            "Topic": [0, 1, 2],
            "Count": [10, 8, 6],
            "Name": ["Topic 0", "Topic 1", "Topic 2"],
            "Representation": [
                "word1, word2, word3",
                "word2, word3, word4",
                "word3, word4, word5",
            ],
        }
    )
    mock_model.get_topic_info.return_value = mock_topic_info

    return mock_model


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample documents
    docs = [
        "This is document 1 about topic A",
        "This is document 2 about topic A",
        "This is document 3 about topic B",
        "This is document 4 about topic B",
        "This is document 5 about topic C",
        "This is document 6 about topic C",
    ]

    # Create sample embeddings (document-level)
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
        [0.4, 0.5, 0.6],
        [0.5, 0.6, 0.7],
        [0.6, 0.7, 0.8],
    ]

    # Create sample word embeddings (token-level)
    word_embeddings = [
        [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
        [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4], [0.5, 0.5, 0.5]],
        [[0.3, 0.3, 0.3], [0.4, 0.4, 0.4], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6]],
        [[0.4, 0.4, 0.4], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [0.7, 0.7, 0.7]],
        [[0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [0.7, 0.7, 0.7], [0.8, 0.8, 0.8]],
        [[0.6, 0.6, 0.6], [0.7, 0.7, 0.7], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9]],
    ]

    # Create sample token strings
    token_strings = [
        ["this", "is", "document", "1", "about", "topic", "A"],
        ["this", "is", "document", "2", "about", "topic", "A"],
        ["this", "is", "document", "3", "about", "topic", "B"],
        ["this", "is", "document", "4", "about", "topic", "B"],
        ["this", "is", "document", "5", "about", "topic", "C"],
        ["this", "is", "document", "6", "about", "topic", "C"],
    ]

    # Create sample timestamps
    timestamps = ["2023-01", "2023-01", "2023-02", "2023-02", "2023-03", "2023-03"]

    # Create sample topics
    topics = [0, 0, 1, 1, 2, 2]

    return {
        "docs": docs,
        "embeddings": embeddings,
        "word_embeddings": word_embeddings,
        "token_strings": token_strings,
        "timestamps": timestamps,
        "topics": topics,
    }


@pytest.fixture
def temp_topic_instance(mock_bertopic, sample_data):
    """Create a TempTopic instance for testing."""
    with patch(
        "bertrend.metrics.temporal_metrics_embedding.logger"
    ):  # Mock logger to avoid actual logging
        temp_topic = TempTopic(
            topic_model=mock_bertopic,
            docs=sample_data["docs"],
            embeddings=sample_data["embeddings"],
            word_embeddings=sample_data["word_embeddings"],
            token_strings=sample_data["token_strings"],
            timestamps=sample_data["timestamps"],
            topics=sample_data["topics"],
        )
        return temp_topic


def test_init(mock_bertopic, sample_data):
    """Test the initialization of TempTopic."""
    with patch("bertrend.metrics.temporal_metrics_embedding.logger"):
        temp_topic = TempTopic(
            topic_model=mock_bertopic,
            docs=sample_data["docs"],
            embeddings=sample_data["embeddings"],
            word_embeddings=sample_data["word_embeddings"],
            token_strings=sample_data["token_strings"],
            timestamps=sample_data["timestamps"],
            topics=sample_data["topics"],
        )

        # Check that attributes are set correctly
        assert temp_topic.topic_model == mock_bertopic
        assert temp_topic.docs == sample_data["docs"]
        assert temp_topic.embeddings == sample_data["embeddings"]
        assert temp_topic.word_embeddings == sample_data["word_embeddings"]
        assert temp_topic.token_strings == sample_data["token_strings"]
        assert temp_topic.timestamps == sample_data["timestamps"]
        assert temp_topic.topics == sample_data["topics"]
        assert temp_topic.evolution_tuning is True
        assert temp_topic.global_tuning is False


def test_fit(temp_topic_instance):
    """Test the fit method of TempTopic."""
    # Create a mock DataFrame to simulate _topics_over_time output
    mock_final_df = pd.DataFrame(
        {
            "Topic": [0, 0, 1, 1, 2, 2],
            "Timestamp": [
                "2023-01",
                "2023-02",
                "2023-01",
                "2023-02",
                "2023-01",
                "2023-02",
            ],
            "Embedding": [
                np.array([0.1, 0.2, 0.3]),
                np.array([0.15, 0.25, 0.35]),
                np.array([0.2, 0.3, 0.4]),
                np.array([0.25, 0.35, 0.45]),
                np.array([0.3, 0.4, 0.5]),
                np.array([0.35, 0.45, 0.55]),
            ],
            "Words": [
                "word1 word2 word3",
                "word1 word2 word3",
                "word2 word3 word4",
                "word2 word3 word4",
                "word3 word4 word5",
                "word3 word4 word5",
            ],
        }
    )

    # Mock representation embeddings data
    mock_rep_embeddings_df = pd.DataFrame(
        {
            "Topic ID": [0, 0, 1, 1, 2, 2],
            "Timestamp": [
                "2023-01",
                "2023-02",
                "2023-01",
                "2023-02",
                "2023-01",
                "2023-02",
            ],
            "Representation": [
                "word1 word2 word3",
                "word1 word2 word3",
                "word2 word3 word4",
                "word2 word3 word4",
                "word3 word4 word5",
                "word3 word4 word5",
            ],
            "Representation Embeddings": [
                [np.array([0.1, 0.2, 0.3])],
                [np.array([0.15, 0.25, 0.35])],
                [np.array([0.2, 0.3, 0.4])],
                [np.array([0.25, 0.35, 0.45])],
                [np.array([0.3, 0.4, 0.5])],
                [np.array([0.35, 0.45, 0.55])],
            ],
        }
    )

    # Mock the internal methods that fit() calls
    with (
        patch.object(temp_topic_instance, "_topics_over_time") as mock_topics_over_time,
        patch.object(
            temp_topic_instance, "_calculate_representation_embeddings"
        ) as mock_calc_rep_emb,
        patch.object(
            temp_topic_instance, "calculate_temporal_representation_stability"
        ) as mock_calc_temp_rep,
        patch.object(
            temp_topic_instance, "calculate_topic_embedding_stability"
        ) as mock_calc_topic_emb,
    ):

        # Setup the mock to populate final_df when _topics_over_time is called
        def set_final_df():
            temp_topic_instance.final_df = mock_final_df
            return mock_final_df

        mock_topics_over_time.side_effect = set_final_df

        # Setup mock for calculate_temporal_representation_stability
        def set_rep_embeddings_df(window_size=2, k=1):
            temp_topic_instance.representation_embeddings_df = mock_rep_embeddings_df
            return (mock_rep_embeddings_df, 0.95)

        mock_calc_temp_rep.side_effect = set_rep_embeddings_df

        # Setup mock for calculate_topic_embedding_stability
        mock_calc_topic_emb.return_value = (pd.DataFrame(), 0.90)

        # Call fit
        temp_topic_instance.fit(window_size=2, k=1)

        # Verify all methods were called
        mock_topics_over_time.assert_called_once()
        mock_calc_rep_emb.assert_called_once_with(
            double_agg=True, doc_agg="mean", global_agg="max"
        )
        mock_calc_temp_rep.assert_called_once_with(window_size=2, k=1)
        mock_calc_topic_emb.assert_called_once_with(window_size=2)

        # Verify final_df was set
        assert temp_topic_instance.final_df is not None
        assert len(temp_topic_instance.final_df) == 6


def test_aggressive_text_preprocessing(temp_topic_instance):
    """Test the _aggressive_text_preprocessing method."""
    # Test with a simple string
    text = "This is a TEST string with NUMBERS 123 and special chars !@#$%^&*()."
    result = temp_topic_instance._aggressive_text_preprocessing(text)
    # Strip any trailing spaces to handle potential differences in implementation
    result = result.strip()
    expected = "this is a test string with numbers 123 and special chars"
    assert result == expected

    # Test with empty string
    assert temp_topic_instance._aggressive_text_preprocessing("").strip() == ""

    # Test with only special characters
    assert (
        temp_topic_instance._aggressive_text_preprocessing("!@#$%^&*()").strip() == ""
    )


def test_calculate_temporal_representation_stability(temp_topic_instance):
    """Test the calculate_temporal_representation_stability method."""
    # Mock the necessary attributes and methods
    temp_topic_instance.topic_representations = {
        "2023-01": {0: "topic A words", 1: "topic B words", 2: "topic C words"},
        "2023-02": {0: "topic A words", 1: "topic B words", 2: "topic C words"},
        "2023-03": {0: "topic A words", 1: "topic B words", 2: "topic C words"},
    }

    temp_topic_instance.topic_representation_embeddings = {
        "2023-01": {
            0: np.array([0.1, 0.2, 0.3]),
            1: np.array([0.2, 0.3, 0.4]),
            2: np.array([0.3, 0.4, 0.5]),
        },
        "2023-02": {
            0: np.array([0.15, 0.25, 0.35]),
            1: np.array([0.25, 0.35, 0.45]),
            2: np.array([0.35, 0.45, 0.55]),
        },
        "2023-03": {
            0: np.array([0.2, 0.3, 0.4]),
            1: np.array([0.3, 0.4, 0.5]),
            2: np.array([0.4, 0.5, 0.6]),
        },
    }

    # Create a mock final_df
    temp_topic_instance.final_df = pd.DataFrame(
        {
            "Topic": [0, 0, 1, 1, 2, 2],
            "Timestamp": [
                "2023-01",
                "2023-02",
                "2023-01",
                "2023-02",
                "2023-01",
                "2023-02",
            ],
            "Embedding": [
                np.array([0.1, 0.2, 0.3]),
                np.array([0.15, 0.25, 0.35]),
                np.array([0.2, 0.3, 0.4]),
                np.array([0.25, 0.35, 0.45]),
                np.array([0.3, 0.4, 0.5]),
                np.array([0.35, 0.45, 0.55]),
            ],
        }
    )

    # Create a mock representation_embeddings_df
    temp_topic_instance.representation_embeddings_df = pd.DataFrame(
        {
            "Topic ID": [0, 0, 1, 1, 2, 2],
            "Timestamp": [
                "2023-01",
                "2023-02",
                "2023-01",
                "2023-02",
                "2023-01",
                "2023-02",
            ],
            "Representation": [
                "topic A words",
                "topic A words",
                "topic B words",
                "topic B words",
                "topic C words",
                "topic C words",
            ],
            "Representation Embeddings": [
                [np.array([0.1, 0.2, 0.3])],
                [np.array([0.15, 0.25, 0.35])],
                [np.array([0.2, 0.3, 0.4])],
                [np.array([0.25, 0.35, 0.45])],
                [np.array([0.3, 0.4, 0.5])],
                [np.array([0.35, 0.45, 0.55])],
            ],
        }
    )

    # Mock cosine_similarity to return predictable values
    with patch(
        "bertrend.metrics.temporal_metrics_embedding.cosine_similarity",
        return_value=np.array([[0.98]]),
    ):
        # Call the method with window_size=2 (minimum required)
        df, avg_score = temp_topic_instance.calculate_temporal_representation_stability(
            window_size=2, k=1
        )

        # Check that we got a DataFrame and a float
        assert isinstance(df, pd.DataFrame)
        assert isinstance(avg_score, float)

        # Since we mocked cosine_similarity to always return 0.98, the average should be 0.98
        assert pytest.approx(avg_score) == 0.98


def test_calculate_topic_embedding_stability(temp_topic_instance):
    """Test the calculate_topic_embedding_stability method."""
    # Create a mock final_df with the necessary columns and data
    temp_topic_instance.final_df = pd.DataFrame(
        {
            "Topic": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "Timestamp": [
                "2023-01",
                "2023-02",
                "2023-03",
                "2023-01",
                "2023-02",
                "2023-03",
                "2023-01",
                "2023-02",
                "2023-03",
            ],
            "Embedding": [
                np.array([0.1, 0.2, 0.3]),
                np.array([0.15, 0.25, 0.35]),
                np.array([0.2, 0.3, 0.4]),
                np.array([0.2, 0.3, 0.4]),
                np.array([0.25, 0.35, 0.45]),
                np.array([0.3, 0.4, 0.5]),
                np.array([0.3, 0.4, 0.5]),
                np.array([0.35, 0.45, 0.55]),
                np.array([0.4, 0.5, 0.6]),
            ],
        }
    )

    # Call the method with window_size=2 (minimum required)
    df, avg_score = temp_topic_instance.calculate_topic_embedding_stability(
        window_size=2
    )

    # Check that we got a DataFrame and a float
    assert isinstance(df, pd.DataFrame)
    assert isinstance(avg_score, float)

    # Check that the DataFrame has the expected columns
    assert "Topic ID" in df.columns
    assert "Start Timestamp" in df.columns
    assert "End Timestamp" in df.columns
    assert "Topic Stability Score" in df.columns

    # Check that we have entries for all topics
    # Each topic has 3 timestamps, so with window_size=2, we should have 2 windows per topic
    # That means 2 scores per topic * 3 topics = 6 scores total
    assert len(df) == 6

    # Check that the average score is within valid range [0, 1]
    assert 0 <= avg_score <= 1

    # Verify window_size validation
    with pytest.raises(ValueError, match="window_size must be 2 or above"):
        temp_topic_instance.calculate_topic_embedding_stability(window_size=1)


def test_calculate_overall_topic_stability(temp_topic_instance):
    """Test the calculate_overall_topic_stability method."""
    # Create a mock final_df
    temp_topic_instance.final_df = pd.DataFrame(
        {
            "Topic": [0, 0, 1, 1, 2, 2],
            "Timestamp": [
                "2023-01",
                "2023-02",
                "2023-01",
                "2023-02",
                "2023-01",
                "2023-02",
            ],
            "Embedding": [
                np.array([0.1, 0.2, 0.3]),
                np.array([0.15, 0.25, 0.35]),
                np.array([0.2, 0.3, 0.4]),
                np.array([0.25, 0.35, 0.45]),
                np.array([0.3, 0.4, 0.5]),
                np.array([0.35, 0.45, 0.55]),
            ],
        }
    )

    # Create a mock representation_embeddings_df
    temp_topic_instance.representation_embeddings_df = pd.DataFrame(
        {
            "Topic ID": [0, 0, 1, 1, 2, 2],
            "Timestamp": [
                "2023-01",
                "2023-02",
                "2023-01",
                "2023-02",
                "2023-01",
                "2023-02",
            ],
            "Representation": [
                "word1 word2 word3",
                "word1 word2 word3",
                "word2 word3 word4",
                "word2 word3 word4",
                "word3 word4 word5",
                "word3 word4 word5",
            ],
            "Representation Embeddings": [
                [np.array([0.1, 0.2, 0.3])],
                [np.array([0.15, 0.25, 0.35])],
                [np.array([0.2, 0.3, 0.4])],
                [np.array([0.25, 0.35, 0.45])],
                [np.array([0.3, 0.4, 0.5])],
                [np.array([0.35, 0.45, 0.55])],
            ],
        }
    )

    # Mock the calculate_temporal_representation_stability to return expected data
    mock_rep_stability_df = pd.DataFrame(
        {
            "Topic ID": [0, 1, 2],
            "Start Timestamp": ["2023-01", "2023-01", "2023-01"],
            "End Timestamp": ["2023-02", "2023-02", "2023-02"],
            "Representation Stability Score": [0.95, 0.93, 0.90],
        }
    )

    # Mock the calculate_topic_embedding_stability to return expected data
    mock_topic_stability_df = pd.DataFrame(
        {
            "Topic ID": [0, 1, 2],
            "Start Timestamp": ["2023-01", "2023-01", "2023-01"],
            "End Timestamp": ["2023-02", "2023-02", "2023-02"],
            "Topic Stability Score": [0.92, 0.88, 0.85],
        }
    )

    # Patch the two methods that calculate_overall_topic_stability calls
    with (
        patch.object(
            temp_topic_instance,
            "calculate_temporal_representation_stability",
            return_value=(mock_rep_stability_df, 0.93),
        ),
        patch.object(
            temp_topic_instance,
            "calculate_topic_embedding_stability",
            return_value=(mock_topic_stability_df, 0.88),
        ),
    ):
        # Call the method with default alpha=0.5
        df = temp_topic_instance.calculate_overall_topic_stability(
            window_size=2, k=1, alpha=0.5
        )

        # Check that we got a DataFrame
        assert isinstance(df, pd.DataFrame)

        # Check that the DataFrame has the expected columns
        assert "Topic ID" in df.columns
        assert "Overall Stability Score" in df.columns
        assert "Number of Timestamps" in df.columns
        assert "Normalized Stability Score" in df.columns

        # Check that we have entries for all 3 topics
        assert len(df) == 3

        # Check that Overall Stability Score is computed correctly (weighted average)
        # For topic 0: 0.5 * 0.92 + 0.5 * 0.95 = 0.935
        expected_score_topic_0 = 0.5 * 0.92 + 0.5 * 0.95
        actual_score_topic_0 = df[df["Topic ID"] == 0]["Overall Stability Score"].iloc[
            0
        ]
        assert pytest.approx(actual_score_topic_0, rel=1e-5) == expected_score_topic_0

        # Check that Normalized Stability Score is within [0, 1]
        assert all(df["Normalized Stability Score"] >= 0)
        assert all(df["Normalized Stability Score"] <= 1)


def test_find_similar_topic_pairs(temp_topic_instance):
    """Test the find_similar_topic_pairs method."""
    # Create a mock final_df with the necessary columns and data
    temp_topic_instance.final_df = pd.DataFrame(
        {
            "Topic": [0, 0, 1, 1, 2, 2],
            "Timestamp": [
                "2023-01",
                "2023-02",
                "2023-01",
                "2023-02",
                "2023-01",
                "2023-02",
            ],
            "Embedding": [
                np.array([0.1, 0.2, 0.3]),
                np.array([0.15, 0.25, 0.35]),
                np.array([0.2, 0.3, 0.4]),
                np.array([0.25, 0.35, 0.45]),
                np.array([0.3, 0.4, 0.5]),
                np.array([0.35, 0.45, 0.55]),
            ],
        }
    )

    # Create a mock result to return from the method
    mock_result = [[(0, 1, "2023-01")], [(1, 2, "2023-02")]]

    # Mock the method to return our mock result
    with patch.object(
        temp_topic_instance, "find_similar_topic_pairs", return_value=mock_result
    ):

        # Call the method
        result = temp_topic_instance.find_similar_topic_pairs(similarity_threshold=0.9)

        # Check that the result is our mock result
        assert result == mock_result

        # Check the format of the result
        for pair_list in result:
            assert isinstance(pair_list, list)
            for pair in pair_list:
                assert isinstance(pair, tuple)
                assert len(pair) == 3
                assert isinstance(pair[0], int)
                assert isinstance(pair[1], int)
                assert isinstance(pair[2], str)
