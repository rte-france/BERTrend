#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from unittest.mock import MagicMock, patch

import pytest

from bertrend.bertrend_apps.data_provider.atom_feed_provider import ATOMFeedProvider
from bertrend.bertrend_apps.data_provider.rss_feed_provider import RSSFeedProvider


@pytest.fixture
def rss_provider():
    return RSSFeedProvider(feed_url="http://example.com/rss")


@pytest.fixture
def atom_provider():
    return ATOMFeedProvider(feed_url="http://example.com/atom")


@patch("feedparser.parse")
def test_rss_parse_feed(mock_parse, rss_provider):
    mock_parse.return_value = MagicMock(
        entries=[{"link": "http://1", "summary": "S1", "published": "2023-01-01"}]
    )
    with patch.object(rss_provider, "_get_text") as mock_get_text:
        mock_get_text.return_value = ("Content 1", "Title 1")
        results = rss_provider.parse_RSS_feed()

    assert len(results) == 1
    assert results[0]["title"] == "Title 1"
    assert results[0]["text"] == "Content 1"


@patch("feedparser.parse")
def test_atom_parse_feed(mock_parse, atom_provider):
    mock_parse.return_value = MagicMock(
        entries=[{"link": "http://2", "summary": "S2", "published": "2023-01-02"}]
    )
    with patch.object(atom_provider, "_get_text") as mock_get_text:
        mock_get_text.return_value = ("Content 2", "Title 2")
        results = atom_provider.parse_ATOM_feed()

    assert len(results) == 1
    assert results[0]["title"] == "Title 2"
    assert results[0]["text"] == "Content 2"


def test_rss_get_articles_with_query_url(rss_provider):
    with patch.object(rss_provider, "parse_RSS_feed") as mock_parse_rss:
        mock_parse_rss.return_value = [{"title": "T"}]
        # URL pattern matches
        results = rss_provider.get_articles(query="http://newfeed.com/rss")
        assert rss_provider.feed_url == "http://newfeed.com/rss"
        assert len(results) == 1


@patch.object(RSSFeedProvider, "_get_text")
def test_rss_parse_entry_curebot(mock_get_text, rss_provider):
    entry = {
        "URL de la ressource": "http://curebot",
        "Contenu de la ressource": "Curebot Summary",
        "Date de trouvaille": "2023-01-01",
    }
    mock_get_text.return_value = ("Curebot Text", "Curebot Title")

    parsed = rss_provider._parse_entry(entry)

    assert parsed["title"] == "Curebot Title"
    assert parsed["summary"] == "Curebot Summary"
