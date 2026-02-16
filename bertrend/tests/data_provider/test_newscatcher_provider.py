#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from unittest.mock import MagicMock, patch

import pytest

from bertrend.bertrend_apps.data_provider.newscatcher_provider import (
    NewsCatcherProvider,
)

newscatcherapi = pytest.importorskip("newscatcherapi")


@pytest.fixture
def newscatcher_provider():
    with patch("newscatcherapi.NewsCatcherApiClient"):
        return NewsCatcherProvider()


@patch.object(NewsCatcherProvider, "process_entries")
def test_newscatcher_get_articles(mock_process, newscatcher_provider):
    newscatcher_provider.newscatcher_client.get_search.return_value = {
        "articles": [{"title": "NC Article"}]
    }
    mock_process.return_value = [{"title": "Processed NC"}]

    with patch.dict("os.environ", {"NEWSCATCHER_API_KEY": "test_key"}):
        results = newscatcher_provider.get_articles(
            "query", "2023-01-01", "2023-01-02", 10
        )

    assert len(results) == 1
    assert results[0]["title"] == "Processed NC"
    newscatcher_provider.newscatcher_client.get_search.assert_called_once()


def test_newscatcher_parse_entry(newscatcher_provider):
    entry = {
        "link": "http://nc-news.com",
        "title": "NC Title",
        "summary": "NC Summary",
        "published_date": "2023-10-27 10:00:00",
    }

    with patch.object(newscatcher_provider, "_get_text") as mock_get_text:
        mock_get_text.return_value = ("NC Content", "NC Fetched Title")
        parsed = newscatcher_provider._parse_entry(entry)

    assert parsed["url"] == "http://nc-news.com"
    assert parsed["title"] == "NC Fetched Title"
    assert "2023-10-27" in parsed["timestamp"]
