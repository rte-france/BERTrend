#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime
from plotly import graph_objects as go

from bertrend.topic_analysis.visualizations import (
    plot_topics_over_time,
    plot_docs_repartition_over_time,
    plot_remaining_docs_repartition_over_time,
    plot_topic_evolution,
    plot_temporal_stability_metrics,
    plot_overall_topic_stability,
)


@pytest.fixture
def sample_topics_over_time():
    """Create a sample topics_over_time DataFrame for testing."""
    return pd.DataFrame(
        {
            "Topic": [0, 0, 1, 1, 2, 2],
            "Timestamp": [
                "2023-01-01",
                "2023-02-01",
                "2023-01-01",
                "2023-02-01",
                "2023-01-01",
                "2023-02-01",
            ],
            "Frequency": [10, 15, 8, 12, 5, 7],
        }
    )


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with timestamps for testing."""
    return pd.DataFrame(
        {
            "title": ["Doc 1", "Doc 2", "Doc 3", "Doc 4", "Doc 5"],
            "text": [
                "This is document 1",
                "This is document 2",
                "This is document 3",
                "This is document 4",
                "This is document 5",
            ],
            "timestamp": pd.to_datetime(
                ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15", "2023-03-01"]
            ),
        }
    )


@pytest.fixture
def sample_remaining_df():
    """Create a sample remaining DataFrame with timestamps for testing."""
    return pd.DataFrame(
        {
            "title": ["Doc 6", "Doc 7", "Doc 8"],
            "text": ["This is document 6", "This is document 7", "This is document 8"],
            "timestamp": pd.to_datetime(["2023-03-15", "2023-04-01", "2023-04-15"]),
        }
    )


@pytest.fixture
def mock_temptopic():
    """Create a mock TempTopic instance for testing."""
    mock = MagicMock()

    # Mock final_df
    mock.final_df = pd.DataFrame(
        {
            "Topic": [0, 0, 1, 1, 2, 2],
            "Timestamp": [
                "2023-01-01 00:00:00",
                "2023-02-01 00:00:00",
                "2023-01-01 00:00:00",
                "2023-02-01 00:00:00",
                "2023-01-01 00:00:00",
                "2023-02-01 00:00:00",
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
                "word1, word2, word3",
                "word1, word2, word3",
                "word4, word5, word6",
                "word4, word5, word6",
                "word7, word8, word9",
                "word7, word8, word9",
            ],
        }
    )

    # Mock topic_stability_scores_df
    mock.topic_stability_scores_df = pd.DataFrame(
        {
            "Topic ID": [0, 0, 1, 1, 2, 2],
            "Start Timestamp": [
                "2023-01-01",
                "2023-02-01",
                "2023-01-01",
                "2023-02-01",
                "2023-01-01",
                "2023-02-01",
            ],
            "End Timestamp": [
                "2023-02-01",
                "2023-03-01",
                "2023-02-01",
                "2023-03-01",
                "2023-02-01",
                "2023-03-01",
            ],
            "Topic Stability Score": [0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
        }
    )

    # Mock representation_stability_scores_df
    mock.representation_stability_scores_df = pd.DataFrame(
        {
            "Topic ID": [0, 0, 1, 1, 2, 2],
            "Start Timestamp": [
                "2023-01-01",
                "2023-02-01",
                "2023-01-01",
                "2023-02-01",
                "2023-01-01",
                "2023-02-01",
            ],
            "End Timestamp": [
                "2023-02-01",
                "2023-03-01",
                "2023-02-01",
                "2023-03-01",
                "2023-02-01",
                "2023-03-01",
            ],
            "Start Representation": [
                "word1, word2, word3",
                "word1, word2, word3",
                "word4, word5, word6",
                "word4, word5, word6",
                "word7, word8, word9",
                "word7, word8, word9",
            ],
            "End Representation": [
                "word1, word2, word3",
                "word1, word2, word3",
                "word4, word5, word6",
                "word4, word5, word6",
                "word7, word8, word9",
                "word7, word8, word9",
            ],
            "Representation Stability Score": [0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
        }
    )

    # Mock overall_stability_df
    mock.overall_stability_df = pd.DataFrame(
        {
            "Topic ID": [0, 1, 2],
            "Number of Timestamps": [2, 2, 2],
            "Overall Stability Score": [0.9, 0.8, 0.7],
            "Normalized Stability Score": [0.9, 0.8, 0.7],
        }
    )

    # Mock calculate_overall_topic_stability method
    mock.calculate_overall_topic_stability.return_value = None

    return mock


@pytest.fixture
def mock_topic_model():
    """Create a mock BERTopic model for testing."""
    mock = MagicMock()

    # Mock visualize_topics_over_time method
    mock.visualize_topics_over_time.return_value = go.Figure()

    return mock


def test_plot_topics_over_time(sample_topics_over_time, mock_topic_model):
    """Test the plot_topics_over_time function."""
    # Call the function with test data
    fig = plot_topics_over_time(
        topics_over_time=sample_topics_over_time,
        dynamic_topics_list="0,1",
        topic_model=mock_topic_model,
        time_split="2023-02-01",
    )

    # Check that the result is a Figure
    assert isinstance(fig, go.Figure)

    # Check that the topic_model's visualize_topics_over_time method was called
    mock_topic_model.visualize_topics_over_time.assert_called_once()


def test_plot_docs_repartition_over_time(sample_df):
    """Test the plot_docs_repartition_over_time function."""
    # Call the function with test data
    fig = plot_docs_repartition_over_time(df=sample_df, freq="M")

    # Check that the result is a Figure
    assert isinstance(fig, go.Figure)


def test_plot_remaining_docs_repartition_over_time(sample_df, sample_remaining_df):
    """Test the plot_remaining_docs_repartition_over_time function."""
    # Call the function with test data
    fig = plot_remaining_docs_repartition_over_time(
        df_base=sample_df, df_remaining=sample_remaining_df, freq="M"
    )

    # Check that the result is a Figure
    assert isinstance(fig, go.Figure)


def test_plot_topic_evolution(mock_temptopic):
    """Test the plot_topic_evolution function."""
    # Mock the umap.UMAP class
    with patch("bertrend.topic_analysis.visualizations.umap.UMAP") as mock_umap:
        # Make the UMAP instance return a fixed transformation
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = np.array(
            [
                [0.1, 0.2],
                [0.15, 0.25],
                [0.2, 0.3],
                [0.25, 0.35],
                [0.3, 0.4],
                [0.35, 0.45],
            ]
        )
        mock_umap.return_value = mock_umap_instance

        # Call the function with test data
        fig = plot_topic_evolution(
            temptopic=mock_temptopic, granularity="M", topics_to_show=[0, 1]
        )

        # Check that the result is a Figure
        assert isinstance(fig, go.Figure)


def test_plot_temporal_stability_metrics(mock_temptopic):
    """Test the plot_temporal_stability_metrics function."""
    # Call the function with test data for topic_stability
    fig1 = plot_temporal_stability_metrics(
        temptopic=mock_temptopic, metric="topic_stability", topics_to_show=[0, 1]
    )

    # Check that the result is a Figure
    assert isinstance(fig1, go.Figure)

    # Call the function with test data for representation_stability
    fig2 = plot_temporal_stability_metrics(
        temptopic=mock_temptopic,
        metric="representation_stability",
        topics_to_show=[0, 1],
    )

    # Check that the result is a Figure
    assert isinstance(fig2, go.Figure)

    # Test with invalid metric
    with pytest.raises(ValueError):
        plot_temporal_stability_metrics(
            temptopic=mock_temptopic, metric="invalid_metric", topics_to_show=[0, 1]
        )


def test_plot_overall_topic_stability(mock_temptopic):
    """Test the plot_overall_topic_stability function."""
    # Call the function with test data
    fig = plot_overall_topic_stability(temptopic=mock_temptopic, topics_to_show=[0, 1])

    # Check that the result is a Figure
    assert isinstance(fig, go.Figure)

    # Test with normalize=True
    fig_normalized = plot_overall_topic_stability(
        temptopic=mock_temptopic, normalize=True, topics_to_show=[0, 1]
    )

    # Check that the result is a Figure
    assert isinstance(fig_normalized, go.Figure)
