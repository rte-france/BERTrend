#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime

from bertrend.trend_analysis.weak_signals import (
    detect_weak_signals_zeroshot,
    _apply_decay,
    _filter_data,
    _is_rising_popularity,
    _create_df,
    _create_dataframes,
    _initialize_new_topic,
    update_existing_topic,
    _apply_decay_to_inactive_topics,
    analyze_signal,
    MAXIMUM_ANALYZED_PERIODS,
)
from bertrend.trend_analysis.data_structure import TopicSummaryList, SignalAnalysis


@pytest.fixture
def mock_topic_models():
    """Create mock topic models for testing."""
    model1 = MagicMock()
    model1.topic_info_df = pd.DataFrame(
        {
            "Topic": [0, 1, 2],
            "Name": ["Topic A", "Topic B", "Topic C"],
            "Representation": [
                ["word1", "word2"],
                ["word3", "word4"],
                ["word5", "word6"],
            ],
            "Representative_Docs": [
                ["doc1", "doc2"],
                ["doc3", "doc4"],
                ["doc5", "doc6"],
            ],
            "Count": [10, 15, 5],
            "Document_Count": [5, 8, 3],
        }
    )

    model2 = MagicMock()
    model2.topic_info_df = pd.DataFrame(
        {
            "Topic": [0, 1, 3],
            "Name": ["Topic A", "Topic B", "Topic D"],
            "Representation": [
                ["word1", "word2"],
                ["word3", "word4"],
                ["word7", "word8"],
            ],
            "Representative_Docs": [
                ["doc1", "doc2"],
                ["doc3", "doc4"],
                ["doc7", "doc8"],
            ],
            "Count": [12, 18, 7],
            "Document_Count": [6, 9, 4],
        }
    )

    return {pd.Timestamp("2023-01-01"): model1, pd.Timestamp("2023-02-01"): model2}


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {
        "Timestamps": [
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-02-01"),
            pd.Timestamp("2023-03-01"),
        ],
        "Popularity": [10, 15, 20],
        "Representations": ["word1_word2", "word3_word4", "word5_word6"],
        "Representation": "word1_word2_word3",
        "Documents": [
            (pd.Timestamp("2023-01-01"), ["doc1", "doc2"]),
            (pd.Timestamp("2023-02-01"), ["doc3", "doc4"]),
            (pd.Timestamp("2023-03-01"), ["doc5", "doc6"]),
        ],
        "Sources": [
            (pd.Timestamp("2023-01-01"), ["source1"]),
            (pd.Timestamp("2023-02-01"), ["source2"]),
            (pd.Timestamp("2023-03-01"), ["source3"]),
        ],
        "Docs_Count": [5, 8, 10],
        "Paragraphs_Count": [10, 15, 20],
        "Source_Diversity": [1, 2, 3],
    }


@pytest.fixture
def mock_bertrend():
    """Create a mock BERTrend instance for testing."""
    mock = MagicMock()

    # Mock all_merge_histories_df
    mock.all_merge_histories_df = pd.DataFrame(
        {
            "Timestamp": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")],
            "Topic1": [0, 0],
            "Topic2": [1, 2],
            "Representation1": [["word1", "word2"], ["word1", "word2"]],
            "Representation2": [["word3", "word4"], ["word5", "word6"]],
            "Documents1": [["doc1", "doc2"], ["doc1", "doc2"]],
            "Documents2": [["doc3", "doc4"], ["doc5", "doc6"]],
            "Source1": [["source1"], ["source1"]],
            "Source2": [["source2"], ["source3"]],
            "Count1": [10, 10],
            "Count2": [15, 5],
            "Document_Count1": [5, 5],
            "Document_Count2": [8, 3],
        }
    )

    # Mock get_periods method
    mock.get_periods.return_value = [
        pd.Timestamp("2022-12-01"),
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-02-01"),
        pd.Timestamp("2023-03-01"),
    ]

    # Mock topic_model
    mock.topic_model = MagicMock()
    mock.topic_model.config = {"global": {"language": "English"}}

    # Mock config
    mock.config = {"granularity": 30}  # 30 days

    return mock


def test_detect_weak_signals_zeroshot(mock_topic_models):
    """Test that detect_weak_signals_zeroshot returns the correct data."""
    zeroshot_topic_list = ["Topic A", "Topic B"]
    granularity = 30  # 30 days

    result = detect_weak_signals_zeroshot(
        topic_models=mock_topic_models,
        zeroshot_topic_list=zeroshot_topic_list,
        granularity=granularity,
    )

    # Check that the result has the correct structure
    assert isinstance(result, dict)
    assert set(result.keys()) == set(zeroshot_topic_list)

    # Check that each topic has data
    for topic in zeroshot_topic_list:
        # Check that the topic has some timestamps
        assert len(result[topic]) > 0

        # Check that the data for each timestamp has the correct structure
        for timestamp in result[topic]:
            assert "Representation" in result[topic][timestamp]
            assert "Representative_Docs" in result[topic][timestamp]
            assert "Count" in result[topic][timestamp]
            assert "Document_Count" in result[topic][timestamp]


def test_apply_decay():
    """Test that _apply_decay correctly applies decay to topic popularity."""
    topic = "Topic A"
    timestamp = pd.Timestamp("2023-02-01")
    topic_last_popularity = {"Topic A": 10}
    topic_last_update = {"Topic A": pd.Timestamp("2023-01-01")}
    granularity = 30  # 30 days
    decay_factor = 0.01
    decay_power = 2

    result = _apply_decay(
        topic=topic,
        timestamp=timestamp,
        topic_last_popularity=topic_last_popularity,
        topic_last_update=topic_last_update,
        granularity=granularity,
        decay_factor=decay_factor,
        decay_power=decay_power,
    )

    # Check that the result has the correct structure
    assert isinstance(result, dict)
    assert "Representation" in result
    assert "Representative_Docs" in result
    assert "Count" in result
    assert "Document_Count" in result

    # Check that the decay was applied correctly
    periods_since_last_update = 1  # 1 month
    expected_decayed_popularity = 10 * np.exp(
        -decay_factor * (periods_since_last_update**decay_power)
    )
    assert result["Document_Count"] == expected_decayed_popularity

    # Check that the topic_last_popularity was updated
    assert topic_last_popularity[topic] == expected_decayed_popularity


def test_filter_data(sample_data):
    """Test that _filter_data correctly filters data based on window_end."""
    window_end = pd.Timestamp("2023-02-01")
    keep_documents = True

    result = _filter_data(sample_data, window_end, keep_documents)

    # Check that the result has the correct structure
    assert isinstance(result, dict)
    assert "Timestamps" in result
    assert "Popularity" in result
    assert "Representation" in result
    assert "Documents" in result
    assert "Sources" in result
    assert "Docs_Count" in result
    assert "Paragraphs_Count" in result
    assert "Source_Diversity" in result

    # Check that the data was filtered correctly
    assert len(result["Timestamps"]) == 2  # Only 2023-01-01 and 2023-02-01
    assert len(result["Popularity"]) == 2
    assert len(result["Representation"]) == 2
    assert len(result["Documents"]) == 4  # 2 documents for each of the 2 timestamps
    assert len(result["Sources"]) == 2
    assert len(result["Docs_Count"]) == 2
    assert len(result["Paragraphs_Count"]) == 2
    assert len(result["Source_Diversity"]) == 2

    # Test with keep_documents=False
    result_no_docs = _filter_data(sample_data, window_end, False)
    assert result_no_docs["Documents"] == []


def test_is_rising_popularity(sample_data):
    """Test that _is_rising_popularity correctly determines if popularity is rising."""
    filtered_data = _filter_data(sample_data, pd.Timestamp("2023-03-01"), True)
    latest_timestamp = pd.Timestamp("2023-03-01")

    result = _is_rising_popularity(filtered_data, latest_timestamp)

    # Check that the result is a boolean
    assert isinstance(result, bool)

    # Check that the result is correct (popularity is rising in the sample data)
    assert result is True


def test_create_df():
    """Test that _create_df correctly creates a DataFrame from topic data."""
    topics = [
        (
            "Topic A",
            10,
            pd.Timestamp("2023-01-01"),
            5,
            10,
            1,
            {
                "Representation": ["word1_word2"],
                "Documents": ["doc1", "doc2"],
                "Sources": [{"source1"}],
                "Source_Diversity": [1],
            },
        ),
        (
            "Topic B",
            15,
            pd.Timestamp("2023-02-01"),
            8,
            15,
            2,
            {
                "Representation": ["word3_word4"],
                "Documents": ["doc3", "doc4"],
                "Sources": [{"source2"}],
                "Source_Diversity": [1],
            },
        ),
    ]
    keep_documents = True

    result = _create_df(topics, keep_documents)

    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that the DataFrame has the correct columns
    assert "Topic" in result.columns
    assert "Representation" in result.columns
    assert "Latest_Popularity" in result.columns
    assert "Docs_Count" in result.columns
    assert "Paragraphs_Count" in result.columns
    assert "Latest_Timestamp" in result.columns
    assert "Documents" in result.columns
    assert "Sources" in result.columns
    assert "Source_Diversity" in result.columns

    # Check that the DataFrame has the correct number of rows
    assert len(result) == 2


def test_create_dataframes():
    """Test that _create_dataframes correctly creates DataFrames for each category."""
    noise_topics = [
        (
            "Topic A",
            10,
            pd.Timestamp("2023-01-01"),
            5,
            10,
            1,
            {
                "Representation": ["word1_word2"],
                "Documents": ["doc1", "doc2"],
                "Sources": [{"source1"}],
                "Source_Diversity": [1],
            },
        )
    ]

    weak_signal_topics = [
        (
            "Topic B",
            15,
            pd.Timestamp("2023-02-01"),
            8,
            15,
            2,
            {
                "Representation": ["word3_word4"],
                "Documents": ["doc3", "doc4"],
                "Sources": [{"source2"}],
                "Source_Diversity": [1],
            },
        )
    ]

    strong_signal_topics = [
        (
            "Topic C",
            20,
            pd.Timestamp("2023-03-01"),
            10,
            20,
            3,
            {
                "Representation": ["word5_word6"],
                "Documents": ["doc5", "doc6"],
                "Sources": [{"source3"}],
                "Source_Diversity": [1],
            },
        )
    ]

    keep_documents = True

    noise_df, weak_signal_df, strong_signal_df = _create_dataframes(
        noise_topics, weak_signal_topics, strong_signal_topics, keep_documents
    )

    # Check that the results are DataFrames
    assert isinstance(noise_df, pd.DataFrame)
    assert isinstance(weak_signal_df, pd.DataFrame)
    assert isinstance(strong_signal_df, pd.DataFrame)

    # Check that the DataFrames have the correct number of rows
    assert len(noise_df) == 1
    assert len(weak_signal_df) == 1
    assert len(strong_signal_df) == 1


def test_initialize_new_topic():
    """Test that _initialize_new_topic correctly initializes a new topic."""
    topic_sizes = {"Topic A": {}}
    topic_last_popularity = {}
    topic_last_update = {}
    topic = "Topic A"
    timestamp = pd.Timestamp("2023-01-01")

    # Create a mock row with specific attribute values
    row = MagicMock()
    # Configure the mock to return specific values for attributes
    row.Document_Count1 = 5
    row.Representation1 = ["word1", "word2"]
    row.Documents1 = ["doc1", "doc2"]
    row.Source1 = ["source1"]
    row.Count1 = 10

    # Configure the mock to return specific values for dictionary-style access
    row.__getitem__.side_effect = lambda key: {
        "Document_Count1": 5,
        "Representation1": ["word1", "word2"],
        "Documents1": ["doc1", "doc2"],
        "Source1": ["source1"],
        "Count1": 10,
    }[key]

    _initialize_new_topic(
        topic_sizes, topic_last_popularity, topic_last_update, topic, timestamp, row
    )

    # Check that the topic_sizes was updated correctly
    assert "Timestamps" in topic_sizes[topic]
    assert "Popularity" in topic_sizes[topic]
    assert "Representation" in topic_sizes[topic]
    assert "Documents" in topic_sizes[topic]
    assert "Sources" in topic_sizes[topic]
    assert "Docs_Count" in topic_sizes[topic]
    assert "Paragraphs_Count" in topic_sizes[topic]
    assert "Source_Diversity" in topic_sizes[topic]
    assert "Representations" in topic_sizes[topic]

    # Check that the topic_last_popularity and topic_last_update were updated
    assert topic_last_popularity[topic] == 5
    assert topic_last_update[topic] == timestamp


def test_update_existing_topic():
    """Test that update_existing_topic correctly updates an existing topic."""
    topic_sizes = {
        "Topic A": {
            "Timestamps": [pd.Timestamp("2023-01-01")],
            "Popularity": [10],
            "Representation": "word1_word2",
            "Documents": [(pd.Timestamp("2023-01-01"), ["doc1", "doc2"])],
            "Sources": [(pd.Timestamp("2023-01-01"), ["source1"])],
            "Docs_Count": [5],
            "Paragraphs_Count": [10],
            "Source_Diversity": [1],
            "Representations": ["word1_word2"],
        }
    }
    topic_last_popularity = {"Topic A": 10}
    topic_last_update = {"Topic A": pd.Timestamp("2023-01-01")}
    topic = "Topic A"
    timestamp = pd.Timestamp("2023-01-01")
    granularity = pd.Timedelta(days=30)  # 30 days

    # Create a mock row with specific attribute values
    row = MagicMock()
    # Configure the mock to return specific values for attributes
    row.Document_Count2 = 8
    row.Representation2 = ["word3", "word4"]
    row.Documents2 = ["doc3", "doc4"]
    row.Source2 = ["source2"]
    row.Count2 = 15

    # Configure the mock to return specific values for dictionary-style access
    row.__getitem__.side_effect = lambda key: {
        "Document_Count2": 8,
        "Representation2": ["word3", "word4"],
        "Documents2": ["doc3", "doc4"],
        "Source2": ["source2"],
        "Count2": 15,
    }[key]

    update_existing_topic(
        topic_sizes,
        topic_last_popularity,
        topic_last_update,
        topic,
        timestamp,
        granularity,
        row,
    )

    # Check that the topic_sizes was updated correctly
    assert len(topic_sizes[topic]["Timestamps"]) == 2
    assert len(topic_sizes[topic]["Popularity"]) == 2
    assert topic_sizes[topic]["Representation"] == "word3_word4"
    assert len(topic_sizes[topic]["Documents"]) == 2
    assert len(topic_sizes[topic]["Sources"]) == 2
    assert len(topic_sizes[topic]["Docs_Count"]) == 2
    assert len(topic_sizes[topic]["Paragraphs_Count"]) == 2
    assert len(topic_sizes[topic]["Source_Diversity"]) == 2
    assert len(topic_sizes[topic]["Representations"]) == 2

    # Check that the topic_last_popularity and topic_last_update were updated
    assert topic_last_popularity[topic] == 18  # 10 + 8
    assert topic_last_update[topic] == pd.Timestamp(
        "2023-01-31"
    )  # timestamp + granularity


def test_apply_decay_to_inactive_topics():
    """Test that _apply_decay_to_inactive_topics correctly applies decay to inactive topics."""
    topic_sizes = {
        "Topic A": {
            "Timestamps": [pd.Timestamp("2023-01-01")],
            "Popularity": [10],
            "Representation": "word1_word2",
            "Documents": [(pd.Timestamp("2023-01-01"), ["doc1", "doc2"])],
            "Sources": [(pd.Timestamp("2023-01-01"), ["source1"])],
            "Docs_Count": [5],
            "Paragraphs_Count": [10],
            "Source_Diversity": [1],
            "Representations": ["word1_word2"],
        }
    }
    topic_last_popularity = {"Topic A": 10}
    topic_last_update = {"Topic A": pd.Timestamp("2023-01-01")}
    updated_topics = set()
    topics_updated_next = set()
    current_timestamp = pd.Timestamp("2023-02-01")
    granularity = pd.Timedelta(days=30)
    decay_factor = 0.01
    decay_power = 2

    _apply_decay_to_inactive_topics(
        topic_sizes,
        topic_last_popularity,
        topic_last_update,
        updated_topics,
        topics_updated_next,
        current_timestamp,
        granularity,
        decay_factor,
        decay_power,
    )

    # Check that the topic_sizes was updated correctly
    assert len(topic_sizes["Topic A"]["Timestamps"]) == 2
    assert len(topic_sizes["Topic A"]["Popularity"]) == 2
    assert len(topic_sizes["Topic A"]["Docs_Count"]) == 2
    assert len(topic_sizes["Topic A"]["Paragraphs_Count"]) == 2
    assert len(topic_sizes["Topic A"]["Source_Diversity"]) == 2
    assert len(topic_sizes["Topic A"]["Representations"]) == 2

    # Check that the topic_last_popularity was updated
    periods_since_last_update = 1  # 1 month
    expected_decayed_popularity = 10 * np.exp(
        -decay_factor * (periods_since_last_update**decay_power)
    )
    assert topic_last_popularity["Topic A"] == expected_decayed_popularity


@patch("bertrend.trend_analysis.weak_signals.OpenAI_Client")
def test_analyze_signal(mock_openai_client, mock_bertrend):
    """Test that analyze_signal correctly analyzes a signal."""
    # Setup mock OpenAI_Client
    mock_client_instance = MagicMock()
    mock_openai_client.return_value = mock_client_instance

    # Setup mock responses
    mock_topic_summary_list = MagicMock(spec=TopicSummaryList)
    mock_topic_summary_list.model_dump_json.return_value = (
        '{"topic_summary_by_time_period": []}'
    )

    mock_signal_analysis = MagicMock(spec=SignalAnalysis)

    mock_client_instance.parse.side_effect = [
        mock_topic_summary_list,
        mock_signal_analysis,
    ]

    # Call the function
    topic_number = 0
    current_date = pd.Timestamp("2023-03-01")

    summaries, analysis = analyze_signal(
        bertrend=mock_bertrend,
        topic_number=topic_number,
        current_date=current_date,
    )

    # Check that the OpenAI_Client was created correctly
    mock_openai_client.assert_called_once()

    # Check that parse was called twice (once for each prompt)
    assert mock_client_instance.parse.call_count == 2

    # Check that the results are correct
    assert summaries == mock_topic_summary_list
    assert analysis == mock_signal_analysis

    # Test with no data available
    # Create an empty DataFrame with the required columns
    empty_df = pd.DataFrame(columns=["Timestamp", "Topic1", "Topic2"])
    mock_bertrend.all_merge_histories_df = empty_df

    summaries_empty, analysis_empty = analyze_signal(
        bertrend=mock_bertrend,
        topic_number=topic_number,
        current_date=current_date,
    )

    assert summaries_empty is None
    assert analysis_empty is None
