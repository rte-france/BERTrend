#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import builtins
import importlib
import sys
from unittest.mock import patch

import pytest

from bertrend.bertrend_apps.data_provider import utils
from bertrend.bertrend_apps.data_provider.utils import decode_google_news_url

ENCODED_URL = "https://news.google.com/rss/articles/CBMiABCDEF?oc=5"


@patch("bertrend.bertrend_apps.data_provider.utils.time.sleep")
def test_non_encoded_url_is_returned_unchanged(mock_sleep):
    url = "https://www.example.com/article"
    assert decode_google_news_url(url) == url
    mock_sleep.assert_not_called()


@patch("bertrend.bertrend_apps.data_provider.utils.time.sleep")
@patch("googlenewsdecoder.new_decoderv1")
def test_encoded_url_is_decoded(mock_decoder, mock_sleep):
    mock_decoder.return_value = {
        "status": True,
        "decoded_url": "https://www.example.com/article",
    }
    assert decode_google_news_url(ENCODED_URL) == "https://www.example.com/article"
    mock_decoder.assert_called_once_with(ENCODED_URL)


@patch("bertrend.bertrend_apps.data_provider.utils.time.sleep")
@patch("googlenewsdecoder.new_decoderv1")
def test_decoding_failure_raises(mock_decoder, mock_sleep):
    mock_decoder.return_value = {"status": False, "message": "error"}
    with pytest.raises(Exception, match="Error decoding Google News URL"):
        decode_google_news_url(ENCODED_URL)


def test_module_imports_without_googlenewsdecoder():
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "googlenewsdecoder":
            raise ImportError("cannot import name 'new_decoderv1'")
        return real_import(name, *args, **kwargs)

    with (
        patch.dict(sys.modules, {"googlenewsdecoder": None}),
        patch("builtins.__import__", side_effect=failing_import),
    ):
        reloaded = importlib.reload(utils)
        assert reloaded.decode_google_news_url("https://www.example.com/a") == (
            "https://www.example.com/a"
        )
    importlib.reload(utils)
