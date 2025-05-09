#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
from unittest.mock import patch, MagicMock, Mock
import torch
import numpy as np

from bertrend.services.summary.extractive_summarizer import (
    ExtractiveSummarizer,
    EnhancedExtractiveSummarizer,
    _summarize_based_on_cos_scores,
    summarize_embeddings,
)
from bertrend.services.summarizer import (
    DEFAULT_MAX_SENTENCES,
    DEFAULT_SUMMARIZATION_RATIO,
)


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer for testing."""
    mock = MagicMock()
    # Mock the encode method to return embeddings
    mock.encode.return_value = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    )
    return mock


@pytest.fixture
def mock_extractive_summarizer(mock_sentence_transformer):
    """Create a mock ExtractiveSummarizer with mocked dependencies."""
    with patch(
        "bertrend.services.summary.extractive_summarizer.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ):
        summarizer = ExtractiveSummarizer()
        # Set the mock model directly to ensure it's used
        summarizer.model = mock_sentence_transformer
        return summarizer


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing."""
    mock = MagicMock()
    mock.generate.return_value = "This is a mock enhanced summary."
    return mock


@pytest.fixture
def mock_enhanced_summarizer(mock_sentence_transformer, mock_openai_client):
    """Create a mock EnhancedExtractiveSummarizer with mocked dependencies."""
    with patch(
        "bertrend.services.summary.extractive_summarizer.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ), patch(
        "bertrend.services.summary.extractive_summarizer.OpenAI_Client",
        return_value=mock_openai_client,
    ):
        summarizer = EnhancedExtractiveSummarizer(
            api_key="test_key", endpoint="test_endpoint"
        )
        # Set the mocks directly to ensure they're used
        summarizer.model = mock_sentence_transformer
        summarizer.api = mock_openai_client
        return summarizer


def test_extractive_summarizer_initialization():
    """Test that the ExtractiveSummarizer initializes correctly."""
    with patch(
        "bertrend.services.summary.extractive_summarizer.SentenceTransformer"
    ) as mock_transformer:
        summarizer = ExtractiveSummarizer()
        mock_transformer.assert_called_once()


def test_enhanced_summarizer_initialization():
    """Test that the EnhancedExtractiveSummarizer initializes correctly."""
    with patch(
        "bertrend.services.summary.extractive_summarizer.SentenceTransformer"
    ) as mock_transformer, patch(
        "bertrend.services.summary.extractive_summarizer.OpenAI_Client"
    ) as mock_client:
        summarizer = EnhancedExtractiveSummarizer(
            api_key="test_key", endpoint="test_endpoint"
        )
        mock_transformer.assert_called_once()
        mock_client.assert_called_once()


def test_get_sentences(mock_extractive_summarizer):
    """Test the get_sentences method."""
    # Mock nltk.sent_tokenize to return a list of sentences
    with patch(
        "bertrend.services.summary.extractive_summarizer.nltk.sent_tokenize",
        return_value=["Sentence 1.", "Sentence 2.", "Sentence 3."],
    ):
        sentences = mock_extractive_summarizer.get_sentences(
            "This is a test text with multiple sentences."
        )
        assert len(sentences) == 3
        assert sentences[0] == "Sentence 1."


def test_get_sentences_embeddings(
    mock_extractive_summarizer, mock_sentence_transformer
):
    """Test the get_sentences_embeddings method."""
    sentences = ["Sentence 1", "Sentence 2", "Sentence 3"]
    embeddings = mock_extractive_summarizer.get_sentences_embeddings(sentences)

    # Check that the model's encode method was called with the correct parameters
    mock_sentence_transformer.encode.assert_called_once_with(
        sentences, convert_to_tensor=True
    )

    # Check that the result is a tensor with the correct shape
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (3, 3)  # 3 sentences, 3 dimensions per embedding


def test_summarize_text(mock_extractive_summarizer):
    """Test the summarize_text method."""
    # Mock the necessary methods
    with patch.object(
        mock_extractive_summarizer,
        "get_sentences",
        return_value=["Sentence 1.", "Sentence 2.", "Sentence 3."],
    ), patch.object(mock_extractive_summarizer, "get_sentences_embeddings"), patch(
        "bertrend.services.summary.extractive_summarizer.summarize_embeddings",
        return_value=[0, 2],
    ):

        summary = mock_extractive_summarizer.summarize_text("This is a test text.")

        # Check that the result is a list of the selected sentences
        assert summary == ["Sentence 1.", "Sentence 3."]


def test_generate_summary(mock_extractive_summarizer):
    """Test the generate_summary method."""
    # Mock the summarize_text method
    with patch.object(
        mock_extractive_summarizer,
        "summarize_text",
        return_value=["Sentence 1.", "Sentence 3."],
    ):
        summary = mock_extractive_summarizer.generate_summary("This is a test text.")

        # Check that the result is a string joining the selected sentences
        assert summary == "Sentence 1. Sentence 3."


def test_enhanced_generate_summary(mock_enhanced_summarizer, mock_openai_client):
    """Test the generate_summary method of EnhancedExtractiveSummarizer."""
    # Mock the parent class's generate_summary method
    with patch.object(
        ExtractiveSummarizer, "generate_summary", return_value="Extractive summary."
    ):
        summary = mock_enhanced_summarizer.generate_summary("This is a test text.")

        # Check that the OpenAI client was called
        mock_openai_client.generate.assert_called_once()

        # Check that the result is the mock enhanced summary
        assert summary == "This is a mock enhanced summary."


def test_summarize_based_on_cos_scores():
    """Test the _summarize_based_on_cos_scores function."""
    # Create mock cosine similarity scores
    cos_scores = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])

    # Get the summary indices for 2 sentences
    summary_indices = _summarize_based_on_cos_scores(cos_scores, 2)

    # Check that the result is a numpy array with the correct length
    assert isinstance(summary_indices, np.ndarray)
    assert len(summary_indices) == 2

    # Check that all indices are within the valid range
    assert all(0 <= idx < 3 for idx in summary_indices)


def test_summarize_embeddings():
    """Test the summarize_embeddings function."""
    # Create mock embeddings
    embeddings = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

    # Get the summary indices for 2 sentences
    summary_indices = summarize_embeddings(embeddings, 2)

    # Check that the result is a numpy array with the correct length
    assert isinstance(summary_indices, np.ndarray)
    assert len(summary_indices) == 2

    # Check that all indices are within the valid range
    assert all(0 <= idx < 3 for idx in summary_indices)


def test_summarize_chunks(mock_extractive_summarizer):
    """Test the summarize_chunks method."""
    # Mock the necessary methods
    with patch.object(mock_extractive_summarizer, "get_chunks_embeddings"), patch(
        "bertrend.services.summary.extractive_summarizer.summarize_embeddings",
        return_value=np.array([0, 2]),
    ):

        chunks = ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4", "Chunk 5"]
        # Explicitly set max_nb_chunks to 2 to avoid the early return condition
        summary = mock_extractive_summarizer.summarize_chunks(chunks, max_nb_chunks=2)

        # Check that the result is a list of the selected chunks
        assert summary == ["Chunk 1", "Chunk 3"]


def test_check_paraphrase(mock_extractive_summarizer):
    """Test the check_paraphrase method."""
    # Create sentences with some similarity
    sentences = [
        "This is the first sentence.",
        "This is a very similar first sentence.",
        "This is a completely different sentence.",
    ]

    # Mock the util.paraphrase_mining function to return a list of triplets
    expected_paraphrases = [
        [0.9, 0, 1],  # High similarity between sentences 0 and 1
        [0.3, 0, 2],  # Low similarity between sentences 0 and 2
        [0.2, 1, 2],  # Low similarity between sentences 1 and 2
    ]

    with patch(
        "bertrend.services.summary.extractive_summarizer.util.paraphrase_mining",
        return_value=expected_paraphrases,
    ):
        # Check that the method returns the expected paraphrases
        paraphrases = mock_extractive_summarizer.check_paraphrase(sentences)
        assert paraphrases == expected_paraphrases

        # Check that the first pair has high similarity (> 0.8)
        assert paraphrases[0][0] > 0.8
        assert paraphrases[0][1] == 0
        assert paraphrases[0][2] == 1
