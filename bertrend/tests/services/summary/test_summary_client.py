#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from unittest.mock import MagicMock, patch

import pytest
import requests

from bertrend.services.summary_client import SummaryAPIClient


class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)


@pytest.fixture
def client():
    return SummaryAPIClient("http://localhost:6465")


# --- Initialization ---


def test_client_default_url():
    """Test default URL."""
    client = SummaryAPIClient()
    assert client.url == "http://localhost:6465"


def test_client_custom_url():
    """Test custom URL with trailing slash stripped."""
    client = SummaryAPIClient("http://example.com:9999/")
    assert client.url == "http://example.com:9999"


# --- Health ---


@patch("bertrend.services.summary_client.requests.get")
def test_health(mock_get, client):
    """Test health check."""
    mock_get.return_value = MockResponse({"status": "ok"})
    result = client.health()
    assert result == {"status": "ok"}
    mock_get.assert_called_once_with("http://localhost:6465/health")


@patch("bertrend.services.summary_client.requests.get")
def test_health_error(mock_get, client):
    """Test health check with server error."""
    mock_get.return_value = MockResponse(None, 500)
    with pytest.raises(requests.HTTPError):
        client.health()


# --- List Summarizers ---


@patch("bertrend.services.summary_client.requests.get")
def test_list_summarizers(mock_get, client):
    """Test listing summarizers."""
    expected = [
        {"name": "llm", "loaded": False},
        {"name": "extractive", "loaded": True},
    ]
    mock_get.return_value = MockResponse(expected)
    result = client.list_summarizers()
    assert result == expected
    mock_get.assert_called_once_with("http://localhost:6465/summarizers")


# --- Summarize ---


@patch("bertrend.services.summary_client.requests.post")
def test_summarize_single_text(mock_post, client):
    """Test summarizing a single text."""
    expected = {
        "summaries": ["Summary."],
        "summarizer_type": "llm",
        "language": "fr",
        "processing_time_ms": 42.0,
    }
    mock_post.return_value = MockResponse(expected)

    result = client.summarize("Some long text.")
    assert result == expected
    mock_post.assert_called_once_with(
        "http://localhost:6465/summarize",
        json={
            "text": "Some long text.",
            "summarizer_type": "llm",
            "language": "fr",
            "max_sentences": 3,
            "max_length_ratio": 0.2,
        },
    )


@patch("bertrend.services.summary_client.requests.post")
def test_summarize_batch(mock_post, client):
    """Test summarizing multiple texts."""
    expected = {
        "summaries": ["Sum 1.", "Sum 2."],
        "summarizer_type": "extractive",
        "language": "fr",
        "processing_time_ms": 15.0,
    }
    mock_post.return_value = MockResponse(expected)

    result = client.summarize(
        ["Text 1.", "Text 2."],
        summarizer_type="extractive",
    )
    assert result == expected


@patch("bertrend.services.summary_client.requests.post")
def test_summarize_with_all_params(mock_post, client):
    """Test summarize with all optional parameters."""
    mock_post.return_value = MockResponse(
        {
            "summaries": ["S."],
            "summarizer_type": "llm",
            "language": "en",
            "processing_time_ms": 1.0,
        }
    )

    client.summarize(
        "Text.",
        summarizer_type="enhanced",
        language="en",
        max_sentences=5,
        max_words=100,
        max_length_ratio=0.3,
    )

    call_json = mock_post.call_args.kwargs["json"]
    assert call_json["summarizer_type"] == "enhanced"
    assert call_json["language"] == "en"
    assert call_json["max_sentences"] == 5
    assert call_json["max_words"] == 100
    assert call_json["max_length_ratio"] == 0.3


@patch("bertrend.services.summary_client.requests.post")
def test_summarize_without_max_words(mock_post, client):
    """Test that max_words is omitted from payload when None."""
    mock_post.return_value = MockResponse(
        {
            "summaries": ["S."],
            "summarizer_type": "llm",
            "language": "fr",
            "processing_time_ms": 1.0,
        }
    )

    client.summarize("Text.")
    call_json = mock_post.call_args.kwargs["json"]
    assert "max_words" not in call_json


@patch("bertrend.services.summary_client.requests.post")
def test_summarize_server_error(mock_post, client):
    """Test summarize with server error."""
    mock_post.return_value = MockResponse(None, 500)
    with pytest.raises(requests.HTTPError):
        client.summarize("Text.")
