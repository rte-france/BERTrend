#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from bertopic import BERTopic

from bertrend.topic_analysis.data_structure import TopicDescription
from bertrend.topic_analysis.topic_description import (
    get_topic_description,
    generate_topic_description,
)


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing."""
    with patch(
        "bertrend.topic_analysis.topic_description.OpenAI_Client"
    ) as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the parse method to return a TopicDescription
        mock_client.parse.return_value = TopicDescription(
            title="Test Topic",
            description="This is a test topic description generated by the mock OpenAI client.",
        )

        yield mock_client


@pytest.fixture
def mock_bertopic():
    """Create a mock BERTopic model for testing."""
    mock_model = MagicMock(spec=BERTopic)

    # Mock the get_topic method
    mock_model.get_topic.return_value = [
        ("word1", 0.9),
        ("word2", 0.8),
        ("word3", 0.7),
        ("word4", 0.6),
        ("word5", 0.5),
        ("word6", 0.4),
        ("word7", 0.3),
        ("word8", 0.2),
        ("word9", 0.1),
        ("word10", 0.05),
        ("word11", 0.01),
    ]

    return mock_model


@pytest.fixture
def sample_docs():
    """Create a sample DataFrame with documents for testing."""
    return pd.DataFrame(
        {
            "title": ["Doc 1", "Doc 2", "Doc 3"],
            "text": [
                "This is the text of document 1.",
                "This is the text of document 2.",
                "This is the text of document 3.",
            ],
        }
    )


def test_get_topic_description(mock_openai_client):
    """Test the get_topic_description function."""
    # Call the function with test data
    with patch(
        "bertrend.topic_analysis.topic_description.LLM_CONFIG",
        {"api_key": "test_key", "endpoint": "test_endpoint", "model": "test_model"},
    ):
        result = get_topic_description(
            topic_representation="word1, word2, word3",
            docs_text="Doc 1: This is a test document.",
            language_code="en",
        )

    # Check that the result is a TopicDescription
    assert isinstance(result, TopicDescription)
    assert result.title == "Test Topic"
    assert (
        result.description
        == "This is a test topic description generated by the mock OpenAI client."
    )

    # Check that the OpenAI client was called with the correct parameters
    mock_openai_client.parse.assert_called_once()


def test_get_topic_description_error_handling(mock_openai_client):
    """Test error handling in the get_topic_description function."""
    # Make the OpenAI client raise an exception
    mock_openai_client.parse.side_effect = Exception("Test error")

    # Call the function with test data
    with patch(
        "bertrend.topic_analysis.topic_description.LLM_CONFIG",
        {"api_key": "test_key", "endpoint": "test_endpoint", "model": "test_model"},
    ):
        result = get_topic_description(
            topic_representation="word1, word2, word3",
            docs_text="Doc 1: This is a test document.",
            language_code="en",
        )

    # Check that the function returns None when an error occurs
    assert result is None


def test_generate_topic_description(mock_bertopic, sample_docs, mock_openai_client):
    """Test the generate_topic_description function."""
    # Call the function with test data
    with patch(
        "bertrend.topic_analysis.topic_description.LLM_CONFIG",
        {"api_key": "test_key", "endpoint": "test_endpoint", "model": "test_model"},
    ):
        result = generate_topic_description(
            topic_model=mock_bertopic,
            topic_number=1,
            filtered_docs=sample_docs,
            language_code="en",
        )

    # Check that the result is a TopicDescription
    assert isinstance(result, TopicDescription)
    assert result.title == "Test Topic"
    assert (
        result.description
        == "This is a test topic description generated by the mock OpenAI client."
    )

    # Check that the BERTopic model's get_topic method was called
    mock_bertopic.get_topic.assert_called_once_with(1)


def test_generate_topic_description_no_words(mock_bertopic, sample_docs):
    """Test the generate_topic_description function when no words are found for the topic."""
    # Make the BERTopic model return an empty list
    mock_bertopic.get_topic.return_value = []

    # Call the function with test data
    result = generate_topic_description(
        topic_model=mock_bertopic,
        topic_number=1,
        filtered_docs=sample_docs,
        language_code="en",
    )

    # Check that the function returns None when no words are found
    assert result is None
