#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from unittest.mock import patch

import pytest

from bertrend.bertrend_apps.prospective_demo.feed_query_generator import (
    generate_google_news_query,
)


@patch("bertrend.bertrend_apps.prospective_demo.feed_query_generator.OpenAI_Client")
def test_generate_google_news_query(mock_client):
    mock_client.return_value.generate.return_value = (
        '("offshore wind" OR "floating wind") AND France'
    )

    query = generate_google_news_query(
        "Follow offshore and floating wind projects in France.", "English"
    )

    assert query == '("offshore wind" OR "floating wind") AND France'
    user_prompt = mock_client.return_value.generate.call_args.args[0]
    assert "English" in user_prompt
    assert "offshore and floating wind projects" in user_prompt


@patch("bertrend.bertrend_apps.prospective_demo.feed_query_generator.OpenAI_Client")
def test_generate_google_news_query_removes_markdown_fence(mock_client):
    mock_client.return_value.generate.return_value = (
        '```\n"hydrogène vert" AND France\n```'
    )

    query = generate_google_news_query(
        "Suivre les projets d'hydrogène vert en France.", "French"
    )

    assert query == '"hydrogène vert" AND France'


def test_generate_google_news_query_rejects_empty_brief():
    with pytest.raises(ValueError, match="cannot be empty"):
        generate_google_news_query("  ", "English")


@patch("bertrend.bertrend_apps.prospective_demo.feed_query_generator.OpenAI_Client")
@pytest.mark.parametrize("response", ["", "OpenAI API fatal error: unavailable"])
def test_generate_google_news_query_rejects_failed_response(mock_client, response):
    mock_client.return_value.generate.return_value = response

    with pytest.raises(RuntimeError, match="generate|empty"):
        generate_google_news_query("Follow grid-scale battery projects.", "English")
