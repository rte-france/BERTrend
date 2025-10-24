#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from plotly import graph_objects as go

from bertrend.trend_analysis.visualizations import (
    plot_num_topics,
    plot_size_outliers,
    plot_topics_for_model,
    create_topic_size_evolution_figure,
    plot_topic_size_evolution,
    plot_newly_emerged_topics,
    create_sankey_diagram_plotly,
)


@pytest.fixture
def mock_topic_models():
    """Create mock topic models for testing."""
    model1 = MagicMock()
    model1.get_topic_info.return_value = pd.DataFrame(
        {
            "Topic": [0, 1, 2, -1],
            "Count": [10, 15, 5, 3],
            "Name": ["Topic 0", "Topic 1", "Topic 2", "Outlier"],
            "Representation": [
                ["word1", "word2"],
                ["word3", "word4"],
                ["word5", "word6"],
                ["outlier"],
            ],
        }
    )

    model2 = MagicMock()
    model2.get_topic_info.return_value = pd.DataFrame(
        {
            "Topic": [0, 1, -1],
            "Count": [12, 18, 5],
            "Name": ["Topic 0", "Topic 1", "Outlier"],
            "Representation": [["word1", "word2"], ["word3", "word4"], ["outlier"]],
        }
    )

    return {pd.Timestamp("2023-01-01"): model1, pd.Timestamp("2023-02-01"): model2}


@pytest.fixture
def mock_selected_model():
    """Create a mock selected model for testing."""
    model = MagicMock()
    model.doc_info_df = pd.DataFrame(
        {
            "source": ["source1", "source1", "source2", "source2", "source3"],
            "Topic": [0, 1, 0, 1, 2],
            "document_id": ["doc1", "doc2", "doc3", "doc4", "doc5"],
            "Representation": [
                ["word1", "word2"],
                ["word3", "word4"],
                ["word1", "word2"],
                ["word3", "word4"],
                ["word5", "word6"],
            ],
        }
    )
    return model


@pytest.fixture
def mock_topic_sizes():
    """Create mock topic sizes for testing."""
    return {
        0: {
            "Timestamps": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")],
            "Popularity": [10, 15],
            "Representations": ["word1_word2", "word1_word2_word3"],
        },
        1: {
            "Timestamps": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")],
            "Popularity": [8, 12],
            "Representations": ["word3_word4", "word3_word4_word5"],
        },
    }


@pytest.fixture
def mock_all_new_topics_df():
    """Create mock all_new_topics_df for testing."""
    return pd.DataFrame(
        {
            "Timestamp": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-02-01"),
            ],
            "Topic": [0, 1, 2],
            "Count": [10, 15, 20],
            "Document_Count": [5, 8, 10],
            "Representation": [
                ["word1", "word2"],
                ["word3", "word4"],
                ["word5", "word6"],
            ],
        }
    )


@pytest.fixture
def mock_all_merge_histories_df():
    """Create mock all_merge_histories_df for testing."""
    return pd.DataFrame(
        {
            "Timestamp": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")],
            "Topic1": [0, 1],
            "Topic2": [1, 2],
            "Count1": [10, 15],
            "Count2": [15, 20],
            "Document_Count1": [5, 8],
            "Document_Count2": [8, 10],
            "Representation1": [["word1", "word2"], ["word3", "word4"]],
            "Representation2": [["word3", "word4"], ["word5", "word6"]],
            "Documents1": [["doc1", "doc2"], ["doc3", "doc4"]],
            "Documents2": [["doc3", "doc4"], ["doc5", "doc6"]],
            "Source1": [["source1"], ["source2"]],
            "Source2": [["source2"], ["source3"]],
        }
    )


def test_plot_num_topics(mock_topic_models):
    """Test that plot_num_topics returns a figure with the correct data."""
    fig = plot_num_topics(mock_topic_models)

    # Check that the result is a Figure
    assert isinstance(fig, go.Figure)

    # Check that the figure has the correct data
    assert len(fig.data) == 1
    assert fig.data[0].x[0] == pd.Timestamp("2023-01-01")
    assert fig.data[0].x[1] == pd.Timestamp("2023-02-01")
    assert fig.data[0].y[0] == 4  # Number of topics in model1
    assert fig.data[0].y[1] == 3  # Number of topics in model2


def test_plot_size_outliers(mock_topic_models):
    """Test that plot_size_outliers returns a figure with the correct data."""
    fig = plot_size_outliers(mock_topic_models)

    # Check that the result is a Figure
    assert isinstance(fig, go.Figure)

    # Check that the figure has the correct data
    assert len(fig.data) == 1
    assert fig.data[0].x[0] == pd.Timestamp("2023-01-01")
    assert fig.data[0].x[1] == pd.Timestamp("2023-02-01")
    assert fig.data[0].y[0] == 3  # Size of outlier topic in model1
    assert fig.data[0].y[1] == 5  # Size of outlier topic in model2


def test_plot_topics_for_model(mock_selected_model):
    """Test that plot_topics_for_model returns a figure with the correct data."""
    fig = plot_topics_for_model(mock_selected_model)

    # Check that the result is a Figure
    assert isinstance(fig, go.Figure)

    # Check that the figure has the correct number of traces (one per topic)
    assert len(fig.data) == 3


def test_create_topic_size_evolution_figure(mock_topic_sizes):
    """Test that create_topic_size_evolution_figure returns a figure with the correct data."""
    # Test with all topics
    fig_all = create_topic_size_evolution_figure(mock_topic_sizes)

    # Check that the result is a Figure
    assert isinstance(fig_all, go.Figure)

    # Check that the figure has the correct number of traces (one per topic)
    assert len(fig_all.data) == 2

    # Test with specific topics
    fig_specific = create_topic_size_evolution_figure(mock_topic_sizes, topic_ids=[0])

    # Check that the result is a Figure
    assert isinstance(fig_specific, go.Figure)

    # Check that the figure has the correct number of traces (one for the specified topic)
    assert len(fig_specific.data) == 1
    assert fig_specific.data[0].name.startswith("Topic 0")


def test_plot_topic_size_evolution():
    """Test that plot_topic_size_evolution returns a figure with the correct data."""
    # Create a base figure
    fig = go.Figure()

    # Call the function
    result = plot_topic_size_evolution(
        fig=fig,
        current_date=pd.Timestamp("2023-02-01"),
        window_start=pd.Timestamp("2023-01-01"),
        window_end=pd.Timestamp("2023-03-01"),
        all_popularity_values=[5, 10, 15, 20],
        q1=7,
        q3=17,
    )

    # Check that the result is a Figure
    assert isinstance(result, go.Figure)

    # Check that the figure has the correct layout
    assert result.layout.title.text == "Popularity Evolution"
    assert result.layout.xaxis.title.text == "Timestamp"
    assert result.layout.yaxis.title.text == "Popularity"

    # Check that the figure has the correct shapes (vertical line for current date and 3 horizontal regions)
    assert len(result.layout.shapes) == 4

    # First shape should be the vertical line for current date
    assert result.layout.shapes[0].type == "line"
    assert result.layout.shapes[0].x0 == pd.Timestamp("2023-02-01")
    assert result.layout.shapes[0].x1 == pd.Timestamp("2023-02-01")

    # Next three shapes should be the horizontal rectangles for noise, weak signal, and strong signal regions
    assert result.layout.shapes[1].type == "rect"  # Noise region (grey)
    assert result.layout.shapes[1].fillcolor == "rgba(128, 128, 128, 0.2)"
    assert result.layout.shapes[1].y1 == 7  # q1

    assert result.layout.shapes[2].type == "rect"  # Weak signal region (orange)
    assert result.layout.shapes[2].fillcolor == "rgba(255, 165, 0, 0.2)"
    assert result.layout.shapes[2].y0 == 7  # q1
    assert result.layout.shapes[2].y1 == 17  # q3

    assert result.layout.shapes[3].type == "rect"  # Strong signal region (green)
    assert result.layout.shapes[3].fillcolor == "rgba(0, 255, 0, 0.2)"
    assert result.layout.shapes[3].y0 == 17  # q3

    # Check that the figure has the correct annotations (current date)
    assert len(result.layout.annotations) == 1
    assert result.layout.annotations[0].x == pd.Timestamp("2023-02-01")


def test_plot_newly_emerged_topics(mock_all_new_topics_df):
    """Test that plot_newly_emerged_topics returns a figure with the correct data."""
    fig = plot_newly_emerged_topics(mock_all_new_topics_df)

    # Check that the result is a Figure
    assert isinstance(fig, go.Figure)

    # Check that the figure has the correct number of traces (one per timestamp group)
    assert (
        len(fig.data) == 2
    )  # One for 2023-01-01 (with 2 topics) and one for 2023-02-01 (with 1 topic)


@patch("bertrend.trend_analysis.visualizations._transform_merge_histories_for_sankey")
def test_create_sankey_diagram_plotly(mock_transform, mock_all_merge_histories_df):
    """Test that create_sankey_diagram_plotly returns a figure with the correct data."""
    # Setup mock transformed dataframe
    mock_transform.return_value = pd.DataFrame(
        {
            "Source": ["0_1_0", "0_2_1", "1_1_1", "1_2_2"],
            "Destination": ["1_1_0", "1_1_0", "2_1_1", "2_1_1"],
            "Representation": [
                ("word1", "word2"),
                ("word3", "word4"),
                ("word3", "word4"),
                ("word5", "word6"),
            ],
            "Timestamp": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-02-01"),
                pd.Timestamp("2023-02-01"),
            ],
            "Count": [5, 8, 8, 10],
        }
    )

    # Call the function
    fig = create_sankey_diagram_plotly(mock_all_merge_histories_df)

    # Check that the result is a Figure
    assert isinstance(fig, go.Figure)

    # Check that the figure has the correct data
    assert len(fig.data) == 1
    assert fig.data[0].type == "sankey"

    # Check that the transform function was called with the correct arguments
    mock_transform.assert_called_once_with(mock_all_merge_histories_df)

    # Test with search_term
    fig_search = create_sankey_diagram_plotly(
        mock_all_merge_histories_df, search_term="word1"
    )
    assert isinstance(fig_search, go.Figure)

    # Test with max_pairs
    fig_max = create_sankey_diagram_plotly(mock_all_merge_histories_df, max_pairs=2)
    assert isinstance(fig_max, go.Figure)
