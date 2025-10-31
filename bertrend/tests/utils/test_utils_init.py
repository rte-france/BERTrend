#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import ssl
from unittest.mock import patch

from bertrend.utils import ensure_stopwords


def test_ensure_stopwords_already_downloaded():
    """Test ensure_stopwords when stopwords are already downloaded."""
    # Mock stopwords.words to not raise an exception (simulating stopwords already downloaded)
    with patch("nltk.corpus.stopwords.words") as mock_words:
        # Call the function
        ensure_stopwords()

        # Verify stopwords.words was called with 'english'
        mock_words.assert_called_once_with("english")


def test_ensure_stopwords_not_downloaded():
    """Test ensure_stopwords when stopwords are not downloaded."""
    # Mock stopwords.words to raise LookupError (simulating stopwords not downloaded)
    with patch("nltk.corpus.stopwords.words", side_effect=LookupError):
        # Mock nltk.download
        with patch("nltk.download") as mock_download:
            # Call the function
            ensure_stopwords()

            # Verify nltk.download was called with 'stopwords'
            mock_download.assert_called_once_with("stopwords")


def test_ssl_workaround():
    """Test the SSL workaround in __init__.py."""
    # This is a bit tricky to test directly since it runs at import time
    # We can test the behavior by re-importing with different mock conditions

    # Case 1: _create_unverified_context exists
    with patch(
        "ssl._create_unverified_context", create=True
    ) as mock_unverified_context:
        # Store the original value
        original_default_context = ssl._create_default_https_context

        try:
            # Re-import to trigger the code
            import importlib
            import bertrend.utils

            importlib.reload(bertrend.utils)

            # Verify _create_default_https_context was set to _create_unverified_context
            assert ssl._create_default_https_context is mock_unverified_context
        finally:
            # Restore the original value to avoid affecting other tests
            ssl._create_default_https_context = original_default_context

    # Case 2: _create_unverified_context doesn't exist
    with patch(
        "ssl._create_unverified_context", create=False, side_effect=AttributeError
    ):
        # This should not raise an exception
        import importlib
        import bertrend.utils

        importlib.reload(bertrend.utils)
        # No assertion needed, we're just verifying it doesn't crash
