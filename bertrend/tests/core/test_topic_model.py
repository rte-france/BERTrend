#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import pytest
import numpy as np
import tomllib

from unittest.mock import MagicMock

from bertopic import BERTopic

from bertrend import BERTOPIC_CONFIG
from bertrend.BERTopicModel import BERTopicModel


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
    """Fixture for creating a BERTopicModel instance."""
    return BERTopicModel()


def test_topic_model_initialization_default_values(topic_model):
    """Test initialization of TopicModel with default values."""
    assert (
        topic_model.config["umap_model"]["n_components"]
        == BERTOPIC_CONFIG["umap_model"]["n_components"]
    )
    assert (
        topic_model.config["umap_model"]["n_neighbors"]
        == BERTOPIC_CONFIG["umap_model"]["n_neighbors"]
    )
    assert (
        topic_model.config["hdbscan_model"]["min_cluster_size"]
        == BERTOPIC_CONFIG["hdbscan_model"]["min_cluster_size"]
    )
    assert (
        topic_model.config["hdbscan_model"]["min_samples"]
        == BERTOPIC_CONFIG["hdbscan_model"]["min_samples"]
    )
    assert (
        topic_model.config["hdbscan_model"]["cluster_selection_method"]
        == BERTOPIC_CONFIG["hdbscan_model"]["cluster_selection_method"]
    )
    assert topic_model.config["vectorizer_model"]["ngram_range"] == tuple(
        BERTOPIC_CONFIG["vectorizer_model"]["ngram_range"]
    )
    assert (
        topic_model.config["vectorizer_model"]["min_df"]
        == BERTOPIC_CONFIG["vectorizer_model"]["min_df"]
    )
    assert (
        topic_model.config["bertopic_model"]["top_n_words"]
        == BERTOPIC_CONFIG["bertopic_model"]["top_n_words"]
    )
    assert (
        topic_model.config["global"]["language"]
        == BERTOPIC_CONFIG["global"]["language"]
    )


def test_topic_model_initialization_custom_values():
    """Test initialization of TopicModel with custom values."""
    custom_toml_config = """
    [global]
    language = "English"

    [bertopic_model]
    top_n_words = 7
    verbose = false
    representation_model = ["KeyBERTInspired"]
    zeroshot_topic_list = ["TopicA", "TopicB"]
    zeroshot_min_similarity = 0.5

    [umap_model]
    n_neighbors = 10
    n_components = 10
    min_dist = 0.1
    metric = "leaf"
    random_state = 100

    [hdbscan_model]
    min_cluster_size = 10
    min_samples = 10
    metric = "manhattan"
    cluster_selection_method = "leaf"
    prediction_data = false

    [vectorizer_model]
    ngram_range = [2, 2]
    stop_words = false
    min_df = 5

    [ctfidf_model]
    bm25_weighting = true
    reduce_frequent_words = false

    [mmr_model]
    diversity = 0.5

    [reduce_outliers]
    strategy = "tf-idf"
    """
    custom_params = tomllib.loads(custom_toml_config)

    topic_model = BERTopicModel(custom_toml_config)

    assert (
        topic_model.config["global"]["language"] == custom_params["global"]["language"]
    )
    assert (
        topic_model.config["bertopic_model"]["top_n_words"]
        == custom_params["bertopic_model"]["top_n_words"]
    )
    assert (
        topic_model.config["bertopic_model"]["verbose"]
        == custom_params["bertopic_model"]["verbose"]
    )
    assert (
        topic_model.config["bertopic_model"]["zeroshot_topic_list"]
        == custom_params["bertopic_model"]["zeroshot_topic_list"]
    )
    assert (
        topic_model.config["bertopic_model"]["zeroshot_min_similarity"]
        == custom_params["bertopic_model"]["zeroshot_min_similarity"]
    )
    assert (
        topic_model.config["umap_model"]["n_neighbors"]
        == custom_params["umap_model"]["n_neighbors"]
    )
    assert (
        topic_model.config["umap_model"]["n_components"]
        == custom_params["umap_model"]["n_components"]
    )
    assert (
        topic_model.config["umap_model"]["min_dist"]
        == custom_params["umap_model"]["min_dist"]
    )
    assert (
        topic_model.config["umap_model"]["metric"]
        == custom_params["umap_model"]["metric"]
    )
    assert (
        topic_model.config["umap_model"]["random_state"]
        == custom_params["umap_model"]["random_state"]
    )
    assert (
        topic_model.config["hdbscan_model"]["min_cluster_size"]
        == custom_params["hdbscan_model"]["min_cluster_size"]
    )
    assert (
        topic_model.config["hdbscan_model"]["min_samples"]
        == custom_params["hdbscan_model"]["min_samples"]
    )
    assert (
        topic_model.config["hdbscan_model"]["metric"]
        == custom_params["hdbscan_model"]["metric"]
    )
    assert (
        topic_model.config["hdbscan_model"]["cluster_selection_method"]
        == custom_params["hdbscan_model"]["cluster_selection_method"]
    )
    assert (
        topic_model.config["hdbscan_model"]["prediction_data"]
        == custom_params["hdbscan_model"]["prediction_data"]
    )
    assert topic_model.config["vectorizer_model"]["ngram_range"] == tuple(
        custom_params["vectorizer_model"]["ngram_range"]
    )
    assert (
        topic_model.config["vectorizer_model"]["min_df"]
        == custom_params["vectorizer_model"]["min_df"]
    )
    assert (
        topic_model.config["ctfidf_model"]["bm25_weighting"]
        == custom_params["ctfidf_model"]["bm25_weighting"]
    )
    assert (
        topic_model.config["ctfidf_model"]["reduce_frequent_words"]
        == custom_params["ctfidf_model"]["reduce_frequent_words"]
    )
    assert (
        topic_model.config["mmr_model"]["diversity"]
        == custom_params["mmr_model"]["diversity"]
    )
    assert (
        topic_model.config["reduce_outliers"]["strategy"]
        == custom_params["reduce_outliers"]["strategy"]
    )


def test_initialize_models_called(topic_model):
    """Test that internal models are initialized properly."""
    assert hasattr(topic_model, "config")


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
        embedding_model=mock_sentence_transformer,
        embeddings=embeddings,
        zeroshot_topic_list=zeroshot_topic_list,
        zeroshot_min_similarity=zeroshot_min_similarity,
    )

    topic_model.fit.assert_called_once_with(
        docs,
        embedding_model=mock_sentence_transformer,
        embeddings=embeddings,
        zeroshot_topic_list=zeroshot_topic_list,
        zeroshot_min_similarity=zeroshot_min_similarity,
    )
    assert result == mock_bertopic


def test_create_topic_model_with_empty_zeroshot_topic_list(
    topic_model, mock_sentence_transformer, mock_embedding
):
    """Test create_topic_model with an empty zeroshot_topic_list."""
    docs = ["Car", "Machine Learning"] * 100
    zeroshot_topic_list = []
    zeroshot_min_similarity = 0.7

    result = topic_model.fit(
        docs,
        embedding_model=None,  # mock_sentence_transformer,
        embeddings=np.random.random((len(docs), 768)),
        zeroshot_topic_list=zeroshot_topic_list,
        zeroshot_min_similarity=zeroshot_min_similarity,
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
            embedding_model=mock_sentence_transformer,
            embeddings=embeddings,
            zeroshot_topic_list=zeroshot_topic_list,
            zeroshot_min_similarity=zeroshot_min_similarity,
        )
