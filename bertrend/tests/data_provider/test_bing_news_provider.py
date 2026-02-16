#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from unittest.mock import MagicMock, patch

import pytest

from bertrend.bertrend_apps.data_provider.bing_news_provider import BingNewsProvider


@pytest.fixture
def bing_provider():
    return BingNewsProvider()

@patch.object(BingNewsProvider, "process_entries")
@patch("feedparser.parse")
def test_bing_get_articles(mock_parse, mock_process, bing_provider):
    mock_parse.return_value = {"entries": [{"link": "http://bing-news.com"}]}
    mock_process.return_value = [{"title": "Processed Bing"}]
    
    results = bing_provider.get_articles("query", "2023-01-01", "2023-01-02", 10)
    
    assert len(results) == 1
    assert results[0]["title"] == "Processed Bing"
    mock_parse.assert_called_once()

def test_bing_parse_entry(bing_provider):
    entry = {
        "link": "http://bing-news.com/url=https%3a%2f%2fexample.com",
        "summary": "Short description",
        "published": "2023-10-27T10:00:00.0000000Z"
    }
    
    with patch.object(bing_provider, "_get_text") as mock_get_text:
        mock_get_text.return_value = ("Bing Content", "Fetched Title")
        parsed = bing_provider._parse_entry(entry)
        
    assert parsed is not None
    assert parsed["url"] == "https://example.com"
    assert parsed["title"] == "Fetched Title"
    assert parsed["summary"] == "Short description"
    assert "2023-10-27" in parsed["timestamp"]
