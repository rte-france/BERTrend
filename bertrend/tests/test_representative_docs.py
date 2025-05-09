#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
import pandas as pd
from unittest.mock import MagicMock
from bertopic import BERTopic

from bertrend.topic_analysis.representative_docs import get_most_representative_docs


@pytest.fixture
def mock_bertopic():
    """Create a mock BERTopic model for testing."""
    mock_model = MagicMock(spec=BERTopic)

    # Mock the get_document_info method
    mock_model.get_document_info.return_value = pd.DataFrame(
        {
            "Document": ["doc1", "doc2", "doc3", "doc4", "doc5"],
            "Topic": [0, 0, 1, 1, 2],
            "Probability": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )

    # Mock the get_representative_docs method
    mock_model.get_representative_docs.return_value = ["doc1", "doc3", "doc5"]

    return mock_model


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "title": ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"],
            "text": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        }
    )


@pytest.fixture
def sample_df_split():
    """Create a sample split DataFrame for testing."""
    return pd.DataFrame(
        {
            "title": ["Title 1", "Title 1", "Title 2", "Title 3", "Title 4"],
            "text": ["doc1 part1", "doc1 part2", "doc2", "doc3", "doc4"],
        }
    )


def test_get_most_representative_docs_with_df_split(
    mock_bertopic, sample_df, sample_df_split
):
    """Test get_most_representative_docs with df_split provided."""
    topics = [0, 0, 1, 1, 2]

    # Call the function with df_split
    result = get_most_representative_docs(
        topic_model=mock_bertopic,
        df=sample_df,
        topics=topics,
        df_split=sample_df_split,
        topic_number=0,
        top_n_docs=2,
    )

    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that the result contains the expected titles
    assert "Title 1" in result["title"].values

    # Check that we got the right number of documents
    assert len(result) <= 2


def test_get_most_representative_docs_cluster_probability(mock_bertopic, sample_df):
    """Test get_most_representative_docs with cluster_probability mode."""
    topics = [0, 0, 1, 1, 2]

    # Call the function with cluster_probability mode
    result = get_most_representative_docs(
        topic_model=mock_bertopic,
        df=sample_df,
        topics=topics,
        mode="cluster_probability",
        topic_number=0,
        top_n_docs=2,
    )

    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that we got the right number of documents
    assert len(result) <= 2

    # Check that the documents are sorted by probability
    if len(result) > 1:
        assert result["Probability"].iloc[0] >= result["Probability"].iloc[1]


def test_get_most_representative_docs_ctfidf_representation(mock_bertopic, sample_df):
    """Test get_most_representative_docs with ctfidf_representation mode."""
    topics = [0, 0, 1, 1, 2]

    # Call the function with ctfidf_representation mode
    result = get_most_representative_docs(
        topic_model=mock_bertopic,
        df=sample_df,
        topics=topics,
        mode="ctfidf_representation",
        topic_number=0,
        top_n_docs=2,
    )

    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that we got the right number of documents
    assert len(result) <= 2
