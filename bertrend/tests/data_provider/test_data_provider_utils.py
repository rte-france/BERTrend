#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bertrend.article_scoring.article_scoring import QualityLevel
from bertrend.bertrend_apps.data_provider.data_provider_utils import (
    auto_scrape,
    count_articles,
    generate_query_file,
    scrape,
    scrape_feed_from_config,
)


@pytest.fixture
def mock_provider():
    with patch(
        "bertrend.bertrend_apps.data_provider.data_provider_utils.PROVIDERS"
    ) as mock_providers:
        mock_class = MagicMock()
        mock_instance = mock_class.return_value
        mock_providers.get.return_value = mock_class
        yield mock_instance


def test_scrape(mock_provider, tmp_path):
    save_path = tmp_path / "test.jsonl"
    mock_provider.get_articles.return_value = [{"title": "test"}]

    results = scrape("query", "arxiv", "2023-01-01", "2023-01-02", 10, save_path, "en")

    assert results == [{"title": "test"}]
    mock_provider.get_articles.assert_called_once()
    mock_provider.store_articles.assert_called_once()


def test_auto_scrape(mock_provider, tmp_path):
    save_path = tmp_path / "test.jsonl"
    requests_file = tmp_path / "requests.txt"
    requests_file.write_text("query;2023-01-01;2023-01-02\n")

    mock_provider.get_articles_batch.return_value = [{"title": "test"}]

    results = auto_scrape(
        str(requests_file), 10, "google", save_path, "en", False, QualityLevel.AVERAGE
    )

    assert results == [{"title": "test"}]
    mock_provider.get_articles_batch.assert_called_once()
    mock_provider.store_articles.assert_called_once()


def test_generate_query_file(tmp_path):
    save_path = tmp_path / "queries.txt"
    line_count = generate_query_file("test", "2023-01-01", "2023-01-03", 1, save_path)

    assert line_count == 2
    content = save_path.read_text()
    assert "test;2023-01-01;2023-01-02" in content
    assert "test;2023-01-02;2023-01-03" in content


def test_count_articles(tmp_path):
    file_path = tmp_path / "test.jsonl"
    file_path.write_text("line1\nline2\nline3\n")
    assert count_articles(file_path) == 3
    assert count_articles(Path("non_existent")) == 0


@patch("bertrend.bertrend_apps.data_provider.data_provider_utils.load_toml_config")
@patch("bertrend.bertrend_apps.data_provider.data_provider_utils.scrape")
@patch(
    "bertrend.bertrend_apps.data_provider.data_provider_utils.FEED_BASE_PATH",
    Path("/tmp/bertrend/feeds"),
)
def test_scrape_feed_from_config_batch_provider(mock_scrape, mock_load_toml, tmp_path):
    cfg_path = tmp_path / "config.toml"
    mock_load_toml.return_value = {
        "data-feed": {
            "id": "test_feed",
            "provider": "arxiv",
            "query": "test query",
            "number_of_days": 7,
            "max_results": 100,
            "language": "en",
            "feed_dir_path": "test_dir",
        }
    }

    result_path = scrape_feed_from_config(cfg_path)

    assert "test_feed.jsonl" in result_path.name
    mock_scrape.assert_called_once()
