#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from unittest.mock import MagicMock, patch

import pytest

from bertrend.demos.summarization.summarizer_app import (
    DEFAULT_TEXT,
    SUMMARIZER_OPTIONS_MAPPER,
    app,
    get_summarizer,
)
from bertrend.services.summary.abstractive_summarizer import AbstractiveSummarizer
from bertrend.services.summary.chatgpt_summarizer import GPTSummarizer
from bertrend.services.summary.extractive_summarizer import (
    EnhancedExtractiveSummarizer,
    ExtractiveSummarizer,
)


# ---------------------------------------------------------------------------
# Tests for SUMMARIZER_OPTIONS_MAPPER
# ---------------------------------------------------------------------------


class TestSummarizerOptionsMapper:
    def test_contains_all_expected_keys(self):
        expected_keys = {
            "GPTSummarizer",
            "EnhancedExtractiveSummarizer",
            "ExtractiveSummarizer",
            "AbstractiveSummarizer",
        }
        assert set(SUMMARIZER_OPTIONS_MAPPER.keys()) == expected_keys

    def test_maps_to_correct_classes(self):
        assert SUMMARIZER_OPTIONS_MAPPER["GPTSummarizer"] is GPTSummarizer
        assert (
            SUMMARIZER_OPTIONS_MAPPER["EnhancedExtractiveSummarizer"]
            is EnhancedExtractiveSummarizer
        )
        assert SUMMARIZER_OPTIONS_MAPPER["ExtractiveSummarizer"] is ExtractiveSummarizer
        assert (
            SUMMARIZER_OPTIONS_MAPPER["AbstractiveSummarizer"] is AbstractiveSummarizer
        )

    def test_mapper_has_four_entries(self):
        assert len(SUMMARIZER_OPTIONS_MAPPER) == 4


# ---------------------------------------------------------------------------
# Tests for DEFAULT_TEXT
# ---------------------------------------------------------------------------


class TestDefaultText:
    def test_default_text_is_non_empty_string(self):
        assert isinstance(DEFAULT_TEXT, str)
        assert len(DEFAULT_TEXT) > 0

    def test_default_text_contains_rte(self):
        assert "RTE" in DEFAULT_TEXT


# ---------------------------------------------------------------------------
# Tests for get_summarizer
# ---------------------------------------------------------------------------


class TestGetSummarizer:
    @patch(
        "bertrend.demos.summarization.summarizer_app.SUMMARIZER_OPTIONS_MAPPER",
    )
    def test_get_summarizer_non_gpt(self, mock_mapper):
        """Non-GPT summarizers are instantiated without extra kwargs."""
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_mapper.__getitem__ = MagicMock(return_value=mock_cls)

        # Clear Streamlit cache so our patched version is used
        get_summarizer.clear()
        result = get_summarizer("ExtractiveSummarizer")

        mock_cls.assert_called_once_with()
        assert result is mock_instance

    @patch(
        "bertrend.demos.summarization.summarizer_app.SUMMARIZER_OPTIONS_MAPPER",
    )
    @patch("bertrend.demos.summarization.summarizer_app.st")
    def test_get_summarizer_gpt(self, mock_st, mock_mapper):
        """GPTSummarizer receives api_key and base_url from session_state."""
        mock_st.session_state.openai_api_key = "test-key"
        mock_st.session_state.openai_base_url = "http://test-url"

        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_mapper.__getitem__ = MagicMock(return_value=mock_cls)

        get_summarizer.clear()
        result = get_summarizer("GPTSummarizer")

        mock_cls.assert_called_once_with(api_key="test-key", base_url="http://test-url")
        assert result is mock_instance

    @patch(
        "bertrend.demos.summarization.summarizer_app.SUMMARIZER_OPTIONS_MAPPER",
    )
    def test_get_summarizer_abstractive(self, mock_mapper):
        """AbstractiveSummarizer is instantiated without extra kwargs."""
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_mapper.__getitem__ = MagicMock(return_value=mock_cls)

        get_summarizer.clear()
        result = get_summarizer("AbstractiveSummarizer")

        mock_cls.assert_called_once_with()
        assert result is mock_instance

    @patch(
        "bertrend.demos.summarization.summarizer_app.SUMMARIZER_OPTIONS_MAPPER",
    )
    def test_get_summarizer_enhanced_extractive(self, mock_mapper):
        """EnhancedExtractiveSummarizer is instantiated without extra kwargs."""
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_mapper.__getitem__ = MagicMock(return_value=mock_cls)

        get_summarizer.clear()
        result = get_summarizer("EnhancedExtractiveSummarizer")

        mock_cls.assert_called_once_with()
        assert result is mock_instance


# ---------------------------------------------------------------------------
# Tests for app()
# ---------------------------------------------------------------------------


class TestApp:
    @patch("bertrend.demos.summarization.summarizer_app.st")
    def test_app_renders_title(self, mock_st):
        """app() calls st.title with the expected text."""
        mock_st.columns.return_value = (MagicMock(), MagicMock())
        mock_st.selectbox.return_value = "ExtractiveSummarizer"
        mock_st.number_input.return_value = 20

        app()

        mock_st.title.assert_called_once_with("Comparison of summarizers")

    @patch("bertrend.demos.summarization.summarizer_app.st")
    def test_app_renders_warning(self, mock_st):
        """app() displays a warning about GPT summarizer usage."""
        mock_st.columns.return_value = (MagicMock(), MagicMock())
        mock_st.selectbox.return_value = "ExtractiveSummarizer"
        mock_st.number_input.return_value = 20

        app()

        mock_st.warning.assert_called_once()
        warning_text = mock_st.warning.call_args[0][0]
        assert "GPT summarizer" in warning_text

    @patch("bertrend.demos.summarization.summarizer_app.st")
    def test_app_creates_selectbox_with_all_options(self, mock_st):
        """app() creates a selectbox with all summarizer options."""
        mock_st.columns.return_value = (MagicMock(), MagicMock())
        mock_st.selectbox.return_value = "ExtractiveSummarizer"
        mock_st.number_input.return_value = 20

        app()

        mock_st.selectbox.assert_called_once()
        call_args = mock_st.selectbox.call_args
        assert call_args[0][0] == "summary model"

    @patch("bertrend.demos.summarization.summarizer_app.st")
    def test_app_creates_number_input_for_ratio(self, mock_st):
        """app() creates a number_input for summary ratio."""
        mock_st.columns.return_value = (MagicMock(), MagicMock())
        mock_st.selectbox.return_value = "ExtractiveSummarizer"
        mock_st.number_input.return_value = 20

        app()

        mock_st.number_input.assert_called_once()
        call_args = mock_st.number_input.call_args
        assert call_args[0][0] == "summary ratio"
        assert call_args[1]["min_value"] == 1
        assert call_args[1]["max_value"] == 100
        assert call_args[1]["value"] == 20

    @patch("bertrend.demos.summarization.summarizer_app.st")
    def test_app_creates_text_inputs_for_openai(self, mock_st):
        """app() creates text_input fields for OpenAI configuration."""
        mock_st.columns.return_value = (MagicMock(), MagicMock())
        mock_st.selectbox.return_value = "ExtractiveSummarizer"
        mock_st.number_input.return_value = 20

        app()

        # Should have 3 text_input calls: API key, base URL, model name
        assert mock_st.text_input.call_count == 3
        labels = [call[0][0] for call in mock_st.text_input.call_args_list]
        assert "Openai API key" in labels
        assert "Openai base URL" in labels
        assert "Openai model name" in labels

    @patch("bertrend.demos.summarization.summarizer_app.st")
    def test_app_creates_two_columns(self, mock_st):
        """app() creates two columns for input and output."""
        mock_st.columns.return_value = (MagicMock(), MagicMock())
        mock_st.selectbox.return_value = "ExtractiveSummarizer"
        mock_st.number_input.return_value = 20

        app()

        mock_st.columns.assert_called_once_with(2)

    @patch("bertrend.demos.summarization.summarizer_app.st")
    def test_app_creates_summarize_button(self, mock_st):
        """app() creates a Summarize button."""
        mock_st.columns.return_value = (MagicMock(), MagicMock())
        mock_st.selectbox.return_value = "ExtractiveSummarizer"
        mock_st.number_input.return_value = 20

        app()

        mock_st.button.assert_called_once()
        assert mock_st.button.call_args[0][0] == "Summarize"

    @patch("bertrend.demos.summarization.summarizer_app.get_summarizer")
    @patch("bertrend.demos.summarization.summarizer_app.st")
    def test_app_on_click_calls_summarizer(self, mock_st, mock_get_summarizer):
        """The on_click callback calls the summarizer and sets session_state.summary."""
        mock_st.columns.return_value = (MagicMock(), MagicMock())
        mock_st.selectbox.return_value = "ExtractiveSummarizer"
        mock_st.number_input.return_value = 20
        mock_st.session_state.text = "Some input text"

        mock_summarizer = MagicMock()
        mock_summarizer.generate_summary.return_value = "Summary result"
        mock_get_summarizer.return_value = mock_summarizer

        app()

        # Extract the on_click callback from st.button call
        on_click = mock_st.button.call_args[1]["on_click"]
        on_click()

        mock_get_summarizer.assert_called_once_with("ExtractiveSummarizer")
        mock_summarizer.generate_summary.assert_called_once_with(
            "Some input text", max_length_ratio=0.2
        )
        assert mock_st.session_state.summary == "Summary result"

    @patch("bertrend.demos.summarization.summarizer_app.get_summarizer")
    @patch("bertrend.demos.summarization.summarizer_app.st")
    def test_app_on_click_uses_correct_ratio(self, mock_st, mock_get_summarizer):
        """The on_click callback passes the correct summary ratio."""
        mock_st.columns.return_value = (MagicMock(), MagicMock())
        mock_st.selectbox.return_value = "ExtractiveSummarizer"
        mock_st.number_input.return_value = 50  # 50%
        mock_st.session_state.text = "Text"

        mock_summarizer = MagicMock()
        mock_summarizer.generate_summary.return_value = "Result"
        mock_get_summarizer.return_value = mock_summarizer

        app()

        on_click = mock_st.button.call_args[1]["on_click"]
        on_click()

        mock_summarizer.generate_summary.assert_called_once_with(
            "Text", max_length_ratio=0.5
        )

    @patch("bertrend.demos.summarization.summarizer_app.st")
    def test_app_api_key_input_is_password_type(self, mock_st):
        """The OpenAI API key input uses password type."""
        mock_st.columns.return_value = (MagicMock(), MagicMock())
        mock_st.selectbox.return_value = "ExtractiveSummarizer"
        mock_st.number_input.return_value = 20

        app()

        # Find the API key text_input call
        api_key_call = None
        for call in mock_st.text_input.call_args_list:
            if call[0][0] == "Openai API key":
                api_key_call = call
                break

        assert api_key_call is not None
        assert api_key_call[1]["type"] == "password"

    @patch("bertrend.demos.summarization.summarizer_app.st")
    def test_app_text_areas_created_in_columns(self, mock_st):
        """app() creates text_area widgets inside columns."""
        col1 = MagicMock()
        col2 = MagicMock()
        mock_st.columns.return_value = (col1, col2)
        mock_st.selectbox.return_value = "ExtractiveSummarizer"
        mock_st.number_input.return_value = 20

        app()

        # Verify columns are used as context managers
        col1.__enter__.assert_called()
        col2.__enter__.assert_called()
