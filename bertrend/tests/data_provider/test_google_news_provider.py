#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from unittest.mock import MagicMock, patch

import pytest

from bertrend.bertrend_apps.data_provider.google_news_provider import GoogleNewsProvider


@pytest.fixture
def google_provider():
    with patch("pygooglenews.GoogleNews"):
        return GoogleNewsProvider()


@patch(
    "bertrend.bertrend_apps.data_provider.google_news_provider.decode_google_news_url"
)
def test_google_parse_entry(mock_decode, google_provider):
    mock_decode.return_value = "http://decoded-url.com"
    entry = {
        "link": "http://encoded-url.com",
        "summary": "Google News Summary",
        "published": "Fri, 27 Oct 2023 10:00:00 GMT",
    }

    with patch.object(google_provider, "_get_text") as mock_get_text:
        mock_get_text.return_value = ("Article Content", "Article Title")
        parsed = google_provider._parse_entry(entry)

    assert parsed is not None
    assert parsed["title"] == "Article Title"
    assert parsed["url"] == "http://decoded-url.com"
    assert "2023-10-27" in parsed["timestamp"]
    assert parsed["text"] == "Article Content"


@patch.object(GoogleNewsProvider, "process_entries")
def test_google_get_articles(mock_process, google_provider):
    google_provider.gn.search = MagicMock(
        return_value={"entries": [{"title": "Entry 1"}]}
    )
    mock_process.return_value = [{"title": "Processed 1"}]

    results = google_provider.get_articles("query", "2023-01-01", "2023-01-02", 10)

    assert len(results) == 1
    assert results[0]["title"] == "Processed 1"
    google_provider.gn.search.assert_called_once()
