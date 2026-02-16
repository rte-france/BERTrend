#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bertrend.article_scoring.article_scoring import QualityLevel
from bertrend.bertrend_apps.data_provider.data_provider import DataProvider


class ConcreteDataProvider(DataProvider):
    def get_articles(self, query, after, before, max_results, language=None):
        return [
            {"title": "test", "text": "content", "timestamp": "2023-01-01 00:00:00"}
        ]

    def _parse_entry(self, entry):
        return entry


@pytest.fixture
def provider():
    return ConcreteDataProvider()


def test_store_and_load_articles(provider, tmp_path):
    data = [{"title": "A", "text": "Content A", "timestamp": "2023-01-01 00:00:00"}]
    file_path = tmp_path / "articles.jsonl"

    provider.store_articles(data, file_path)
    assert file_path.exists()

    # store_articles logs info, we check file content
    import jsonlines

    with jsonlines.open(file_path) as reader:
        loaded_data = list(reader)

    assert len(loaded_data) == 1
    assert loaded_data[0]["title"] == "A"


def test_parse_date(provider):
    date_str = "2023-10-27"
    assert provider.parse_date(date_str) == "2023-10-27 00:00:00"

    date_str_complex = "Fri, 27 Oct 2023 10:00:00 +0000"
    parsed = provider.parse_date(date_str_complex)
    assert "2023-10-27" in parsed


def test_filter_out_bad_text(provider):
    assert provider._filter_out_bad_text("") == ""
    assert provider._filter_out_bad_text("Some valid text") == "Some valid text"

    # It filters out strings containing specific words like 'cookie'
    bad_text = "Wait... This contains cookie consent message."
    assert provider._filter_out_bad_text(bad_text) is None
    assert provider._filter_out_bad_text("This is valid text.") == "This is valid text."


@patch("bertrend.bertrend_apps.data_provider.data_provider.Goose")
def test_get_text(mock_goose, provider):
    url = "http://example.com"

    # Mock Goose instance that is created in provider.__init__
    mock_g = MagicMock()
    provider.article_parser = mock_g

    mock_extract = MagicMock()
    mock_extract.cleaned_text = "Goose text"
    mock_extract.title = "Goose title"
    mock_g.extract.return_value = mock_extract

    text, title = provider._get_text(url)
    assert text == "Goose text"
    assert title == "Goose title"


@patch("bertrend.bertrend_apps.data_provider.data_provider.asyncio.run")
def test_evaluate_quality(mock_asyncio_run, provider):
    articles = [
        {"text": "high quality content", "id": 1},
        {"text": "low quality", "id": 2},
    ]

    # Mock score_articles result
    mock_result_high = MagicMock()
    mock_result_high.output.quality_level = QualityLevel.EXCELLENT
    mock_result_high.output.model_dump.return_value = {"score": 0.9}

    mock_result_low = MagicMock()
    mock_result_low.output.quality_level = QualityLevel.POOR
    mock_result_low.output.model_dump.return_value = {"score": 0.1}

    mock_asyncio_run.return_value = [mock_result_high, mock_result_low]

    filtered = provider.evaluate_quality(articles, QualityLevel.AVERAGE)

    assert len(filtered) == 1
    assert filtered[0]["id"] == 1
    assert filtered[0]["overall_quality"] == "EXCELLENT"


def test_process_entries(provider):
    entries = [{"title": "Entry 1"}, {"title": "Entry 2"}]

    with patch.object(provider, "_parse_entry") as mock_parse:
        mock_parse.side_effect = [
            {"title": "Parsed 1", "text": "Content 1", "url": "http://1.com"},
            {"title": "Parsed 2", "text": "Content 2", "url": "http://2.com"},
        ]

        results = provider.process_entries(entries)
        assert len(results) == 2
        titles = {res["title"] for res in results}
        assert titles == {"Parsed 1", "Parsed 2"}
