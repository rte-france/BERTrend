#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bertrend.services.summary_server.main import app
from bertrend.services.summary_server.routers import summarize as summarize_module


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_summarizer_cache():
    """Clear the summarizer cache before each test."""
    summarize_module._summarizer_cache.clear()
    yield
    summarize_module._summarizer_cache.clear()


# --- Health & Info Endpoints ---


def test_health(client):
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_summarizers_none_loaded(client):
    """Test /summarizers returns all backends with loaded=false."""
    response = client.get("/summarizers")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    names = {s["name"] for s in data}
    assert names == {"llm", "extractive", "enhanced"}
    for s in data:
        assert s["loaded"] is False
        assert "description" in s
        assert "requires_api_key" in s
        assert "requires_gpu" in s


def test_list_summarizers_shows_loaded_status(client):
    """Test /summarizers reflects loaded status after a summarizer is cached."""
    mock_summarizer = MagicMock()
    summarize_module._summarizer_cache["extractive"] = mock_summarizer

    response = client.get("/summarizers")
    data = response.json()
    extractive = next(s for s in data if s["name"] == "extractive")
    llm = next(s for s in data if s["name"] == "llm")
    assert extractive["loaded"] is True
    assert llm["loaded"] is False


# --- Summarize Endpoint ---


def test_summarize_invalid_type(client):
    """Test /summarize returns 400 for unknown summarizer_type."""
    response = client.post(
        "/summarize",
        json={"text": "test", "summarizer_type": "nonexistent"},
    )
    assert response.status_code == 400
    assert "nonexistent" in response.json()["detail"]
    assert "Available" in response.json()["detail"]


def test_summarize_single_text(client):
    """Test /summarize with a single text string."""
    mock_summarizer = MagicMock()
    mock_summarizer.summarize_batch.return_value = ["Summary of text."]
    summarize_module._summarizer_cache["llm"] = mock_summarizer

    response = client.post(
        "/summarize",
        json={"text": "A long article text.", "summarizer_type": "llm"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["summaries"] == ["Summary of text."]
    assert data["summarizer_type"] == "llm"
    assert data["language"] == "fr"
    assert "processing_time_ms" in data

    # Verify summarize_batch was called with a list
    mock_summarizer.summarize_batch.assert_called_once()
    call_kwargs = mock_summarizer.summarize_batch.call_args
    assert call_kwargs.kwargs["article_texts"] == ["A long article text."]


def test_summarize_batch(client):
    """Test /summarize with a list of texts."""
    mock_summarizer = MagicMock()
    mock_summarizer.summarize_batch.return_value = ["Sum 1.", "Sum 2."]
    summarize_module._summarizer_cache["extractive"] = mock_summarizer

    response = client.post(
        "/summarize",
        json={
            "text": ["Article one.", "Article two."],
            "summarizer_type": "extractive",
            "max_sentences": 1,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["summaries"] == ["Sum 1.", "Sum 2."]
    assert data["summarizer_type"] == "extractive"

    call_kwargs = mock_summarizer.summarize_batch.call_args
    assert call_kwargs.kwargs["article_texts"] == ["Article one.", "Article two."]
    assert call_kwargs.kwargs["max_sentences"] == 1


def test_summarize_passes_language(client):
    """Test that language parameter is forwarded as prompt_language."""
    mock_summarizer = MagicMock()
    mock_summarizer.summarize_batch.return_value = ["English summary."]
    summarize_module._summarizer_cache["llm"] = mock_summarizer

    response = client.post(
        "/summarize",
        json={"text": "Some text.", "summarizer_type": "llm", "language": "en"},
    )
    assert response.status_code == 200
    assert response.json()["language"] == "en"
    call_kwargs = mock_summarizer.summarize_batch.call_args
    assert call_kwargs.kwargs["prompt_language"] == "en"


def test_summarize_optional_max_words(client):
    """Test that max_words is only passed when provided."""
    mock_summarizer = MagicMock()
    mock_summarizer.summarize_batch.return_value = ["Summary."]
    summarize_module._summarizer_cache["llm"] = mock_summarizer

    # Without max_words
    client.post(
        "/summarize",
        json={"text": "Text.", "summarizer_type": "llm"},
    )
    call_kwargs = mock_summarizer.summarize_batch.call_args.kwargs
    assert "max_words" not in call_kwargs

    # With max_words
    mock_summarizer.reset_mock()
    client.post(
        "/summarize",
        json={"text": "Text.", "summarizer_type": "llm", "max_words": 50},
    )
    call_kwargs = mock_summarizer.summarize_batch.call_args.kwargs
    assert call_kwargs["max_words"] == 50


def test_summarize_default_parameters(client):
    """Test that default request parameters are correct."""
    mock_summarizer = MagicMock()
    mock_summarizer.summarize_batch.return_value = ["Summary."]
    summarize_module._summarizer_cache["llm"] = mock_summarizer

    response = client.post(
        "/summarize",
        json={"text": "Text."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["summarizer_type"] == "llm"
    assert data["language"] == "fr"

    call_kwargs = mock_summarizer.summarize_batch.call_args.kwargs
    assert call_kwargs["max_sentences"] == 3
    assert call_kwargs["max_length_ratio"] == 0.2


# --- Lazy Loading ---


def test_lazy_loading_caches_summarizer(client):
    """Test that summarizers are lazily loaded and cached."""
    mock_instance = MagicMock()
    mock_instance.summarize_batch.return_value = ["Summary."]
    mock_class = MagicMock(return_value=mock_instance)

    with patch("importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_module.GPTSummarizer = mock_class
        mock_import.return_value = mock_module

        # First call: loads the summarizer
        client.post(
            "/summarize",
            json={"text": "Text.", "summarizer_type": "llm"},
        )
        assert mock_class.call_count == 1

        # Second call: uses cached instance
        client.post(
            "/summarize",
            json={"text": "More text.", "summarizer_type": "llm"},
        )
        assert mock_class.call_count == 1  # Not called again


# --- Registry ---


def test_registry_contains_expected_entries():
    """Test the summarizer registry has all expected entries."""
    registry = summarize_module.SUMMARIZER_REGISTRY
    assert "llm" in registry
    assert "extractive" in registry
    assert "enhanced" in registry
    for name, meta in registry.items():
        assert "class_path" in meta
        assert "description" in meta
        assert "requires_api_key" in meta
        assert "requires_gpu" in meta
