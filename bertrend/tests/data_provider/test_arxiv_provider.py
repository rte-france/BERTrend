#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from bertrend.bertrend_apps.data_provider.arxiv_provider import ArxivProvider


@pytest.fixture
def arxiv_provider():
    with patch("arxiv.Client"):
        return ArxivProvider()


def test_arxiv_parse_entry(arxiv_provider):
    mock_entry = MagicMock()
    mock_entry.entry_id = "http://arxiv.org/abs/1234.5678"
    mock_entry.title = "Test Paper"
    mock_entry.summary = "Test Summary"
    mock_entry.published = datetime(2023, 1, 1, 12, 0, 0)

    parsed = arxiv_provider._parse_entry(mock_entry)

    assert parsed["id"] == "http://arxiv.org/abs/1234.5678"
    assert parsed["title"] == "Test Paper"
    assert parsed["timestamp"] == "2023-01-01 12:00:00"


@patch.object(ArxivProvider, "process_entries")
def test_arxiv_get_articles(mock_process, arxiv_provider):
    arxiv_provider.client.results.return_value = [MagicMock()]
    mock_process.return_value = [
        {"id": "1", "title": "T", "timestamp": "2023-01-01 12:00:00"}
    ]

    results = arxiv_provider.get_articles("query", "2023-01-01", "2023-01-02", 10)

    assert len(results) == 1
    assert results[0]["id"] == "1"


@patch("requests.post")
def test_arxiv_add_citations_count(mock_post, arxiv_provider):
    entries = [{"id": "url1", "title": "Title 1", "text": "Content 1"}]

    mock_response = MagicMock()
    mock_response.json.return_value = [{"title": "Title 1", "citationCount": 10}]
    mock_post.return_value.__enter__.return_value = mock_response

    with patch.dict("os.environ", {"SEMANTIC_SCHOLAR_API_KEY": "test_key"}):
        results = arxiv_provider.add_citations_count(entries)

    assert len(results) == 1
    assert results[0]["citationCount"] == 10
