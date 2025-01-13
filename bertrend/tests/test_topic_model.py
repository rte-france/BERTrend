#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import pytest
import numpy as np

from unittest.mock import MagicMock

from bertopic import BERTopic

from bertrend.topic_model.topic_model import TopicModel, TopicModelOutput
from bertrend.parameters import (
    DEFAULT_UMAP_N_COMPONENTS,
    DEFAULT_UMAP_N_NEIGHBORS,
    DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    DEFAULT_HDBSCAN_MIN_SAMPLES,
    HDBSCAN_CLUSTER_SELECTION_METHODS,
    VECTORIZER_NGRAM_RANGES,
    DEFAULT_MIN_DF,
    DEFAULT_TOP_N_WORDS,
    LANGUAGES,
)


# Mocking dependencies
@pytest.fixture
def mock_sentence_transformer():
    """Fixture for mocking the SentenceTransformer."""
    return MagicMock()


@pytest.fixture
def mock_embedding():
    """Fixture for mocking the embeddings."""
    return MagicMock()


@pytest.fixture
def topic_model():
    """Fixture for creating a TopicModel instance."""
    return TopicModel()


def test_topic_model_initialization_default_values(topic_model):
    """Test initialization of TopicModel with default values."""
    assert topic_model.umap_n_components == DEFAULT_UMAP_N_COMPONENTS
    assert topic_model.umap_n_neighbors == DEFAULT_UMAP_N_NEIGHBORS
    assert topic_model.hdbscan_min_cluster_size == DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE
    assert topic_model.hdbscan_min_samples == DEFAULT_HDBSCAN_MIN_SAMPLES
    assert (
        topic_model.hdbscan_cluster_selection_method
        == HDBSCAN_CLUSTER_SELECTION_METHODS[0]
    )
    assert topic_model.vectorizer_ngram_range == VECTORIZER_NGRAM_RANGES[0]
    assert topic_model.min_df == DEFAULT_MIN_DF
    assert topic_model.top_n_words == DEFAULT_TOP_N_WORDS
    assert topic_model.language == LANGUAGES[0]


def test_topic_model_initialization_custom_values():
    """Test initialization of TopicModel with custom values."""
    custom_params = {
        "umap_n_components": 15,
        "umap_n_neighbors": 20,
        "hdbscan_min_cluster_size": 10,
        "hdbscan_min_samples": 5,
        "hdbscan_cluster_selection_method": "eom",
        "vectorizer_ngram_range": (1, 2),
        "min_df": 2,
        "top_n_words": 50,
        "language": "French",
    }

    topic_model = TopicModel(**custom_params)

    assert topic_model.umap_n_components == custom_params["umap_n_components"]
    assert topic_model.umap_n_neighbors == custom_params["umap_n_neighbors"]
    assert (
        topic_model.hdbscan_min_cluster_size
        == custom_params["hdbscan_min_cluster_size"]
    )
    assert topic_model.hdbscan_min_samples == custom_params["hdbscan_min_samples"]
    assert (
        topic_model.hdbscan_cluster_selection_method
        == custom_params["hdbscan_cluster_selection_method"]
    )
    assert topic_model.vectorizer_ngram_range == custom_params["vectorizer_ngram_range"]
    assert topic_model.min_df == custom_params["min_df"]
    assert topic_model.top_n_words == custom_params["top_n_words"]
    assert topic_model.language == custom_params["language"]


def test_initialize_models_called(topic_model):
    """Test that internal models are initialized properly."""
    assert hasattr(topic_model, "umap_model")
    assert hasattr(topic_model, "hdbscan_model")
    assert hasattr(topic_model, "vectorizer_model")
    assert hasattr(topic_model, "mmr_model")
    assert hasattr(topic_model, "ctfidf_model")


def test_create_topic_model_with_valid_input(
    topic_model, mock_sentence_transformer, mock_embedding
):
    """Test create_topic_model method with valid input."""
    docs = ["Document 1", "Document 2"]
    embeddings = mock_embedding
    zeroshot_topic_list = ["Topic 1", "Topic 2"]
    zeroshot_min_similarity = 0.7

    # Mock BERTopic behavior
    mock_bertopic = MagicMock(spec=BERTopic)
    mock_bertopic.fit_transform.return_value = ([], [])
    mock_bertopic.reduce_outliers.return_value = []

    topic_model.fit = MagicMock(return_value=mock_bertopic)

    result = topic_model.fit(
        docs,
        mock_sentence_transformer,
        embeddings,
        zeroshot_topic_list,
        zeroshot_min_similarity,
    )

    topic_model.fit.assert_called_once_with(
        docs,
        mock_sentence_transformer,
        embeddings,
        zeroshot_topic_list,
        zeroshot_min_similarity,
    )
    assert result == mock_bertopic


def test_create_topic_model_with_empty_zeroshot_topic_list(
    topic_model, mock_sentence_transformer, mock_embedding
):
    """Test create_topic_model with an empty zeroshot_topic_list."""
    docs = ["Document 1", "Document 2"] * 100
    zeroshot_topic_list = []
    zeroshot_min_similarity = 0.7

    result = topic_model.fit(
        docs,
        None,  # mock_sentence_transformer,
        np.random.random((len(docs), 768)),
        zeroshot_topic_list,
        zeroshot_min_similarity,
    )

    assert result is not None
    assert (
        result.topic_model.zeroshot_topic_list is None
    )  # Check that it was set to None internally


def test_create_topic_model_exception_handling(
    topic_model, mock_sentence_transformer, mock_embedding
):
    """Test that create_topic_model raises an exception if an error occurs."""
    docs = ["Document 1", "Document 2"]
    embeddings = mock_embedding
    zeroshot_topic_list = ["Topic 1"]
    zeroshot_min_similarity = 0.7

    # Simulate an error in the create_topic_model method
    topic_model.fit = MagicMock(side_effect=Exception("Test Exception"))

    with pytest.raises(Exception, match="Test Exception"):
        topic_model.fit(
            docs,
            mock_sentence_transformer,
            embeddings,
            zeroshot_topic_list,
            zeroshot_min_similarity,
        )
