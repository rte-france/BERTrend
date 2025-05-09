#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from bertrend.utils.data_loading import (
    find_compatible_files,
    load_data,
    split_data,
    group_by_days,
    _file_to_pd,
    clean_dataset,
    enhanced_split_df_by_paragraphs,
    preprocess_french_text,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "text": [
                "This is a test document.",
                "Another test document with more content.",
            ],
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "id": [1, 2],
            "url": ["https://example.com/page1", "https://example.com/page2"],
            "title": ["Title 1", "Title 2"],
            "source": ["example.com", "example.com"],
        }
    )


@pytest.fixture
def sample_path():
    """Create a sample Path object for testing."""
    return Path("/test/path")


def test_find_compatible_files(sample_path):
    """Test finding compatible files with specific extensions."""
    # Mock the Path.rglob method to return specific files
    with patch.object(
        Path,
        "rglob",
        return_value=[
            Path("/test/path/file1.txt"),
            Path("/test/path/file2.csv"),
            Path("/test/path/file3.json"),
        ],
    ):
        # Test with txt and csv extensions (without the dot)
        result = find_compatible_files(sample_path, ["txt", "csv"])
        assert len(result) == 2
        # The function returns relative paths
        assert Path("file1.txt") in result
        assert Path("file2.csv") in result


def test_load_data():
    """Test loading data from a file."""
    # Mock the _file_to_pd function
    mock_df = pd.DataFrame(
        {
            "text": ["This is a test document."],
            "timestamp": pd.to_datetime(["2023-01-01"]),
            "id": [1],
        }
    )

    with patch("bertrend.utils.data_loading._file_to_pd", return_value=mock_df):
        result = load_data(Path("test_file.csv"))

        # Verify the result is a DataFrame with expected columns
        assert isinstance(result, pd.DataFrame)
        assert "text" in result.columns
        assert "timestamp" in result.columns
        assert "id" in result.columns
        assert "document_id" in result.columns
        assert "source" in result.columns
        assert "url" in result.columns
        assert "title" in result.columns


def test_split_data_with_split(sample_dataframe):
    """Test split_data with paragraph splitting."""
    # Test with split_by_paragraph="yes"
    result_yes = split_data(sample_dataframe, min_chars=5, split_by_paragraph="yes")

    # Verify the result has at least the same number of rows as the input
    assert len(result_yes) >= len(sample_dataframe)

    # Test with split_by_paragraph="no"
    result_no = split_data(sample_dataframe, min_chars=5, split_by_paragraph="no")

    # Verify the result is the same as the input (no splitting)
    assert len(result_no) == len(sample_dataframe)

    # Test with split_by_paragraph="enhanced"
    # Mock the enhanced_split_df_by_paragraphs function
    mock_split_df = pd.DataFrame(
        {
            "text": [
                "This is a test.",
                "document.",
                "Another test.",
                "document with more content.",
            ],
            "timestamp": pd.to_datetime(
                ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"]
            ),
            "id": [1, 1, 2, 2],
            "url": [
                "https://example.com/page1",
                "https://example.com/page1",
                "https://example.com/page2",
                "https://example.com/page2",
            ],
            "title": ["Title 1", "Title 1", "Title 2", "Title 2"],
            "source": ["example.com", "example.com", "example.com", "example.com"],
        }
    )

    with patch(
        "bertrend.utils.data_loading.enhanced_split_df_by_paragraphs",
        return_value=mock_split_df,
    ) as mock_enhanced_split:
        embedding_model_name = "bert-base-uncased"
        result_enhanced = split_data(
            sample_dataframe,
            min_chars=5,
            split_by_paragraph="enhanced",
            embedding_model_name=embedding_model_name,
        )

        # Verify enhanced_split_df_by_paragraphs was called with the correct parameters
        mock_enhanced_split.assert_called_once()
        args, kwargs = mock_enhanced_split.call_args
        assert kwargs["dataset"].equals(sample_dataframe)
        assert kwargs["embedding_model_name"] == embedding_model_name

        # Verify the result has more rows than the input (due to splitting)
        assert len(result_enhanced) > len(sample_dataframe)
        assert len(result_enhanced) == len(mock_split_df)


def test_group_by_days(sample_dataframe):
    """Test grouping data by days."""
    # Add a row with the same date as the first row
    df = sample_dataframe.copy()
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "text": ["A third document."],
                    "timestamp": pd.to_datetime(["2023-01-01"]),
                    "id": [3],
                    "url": ["https://example.com/page3"],
                    "title": ["Title 3"],
                }
            ),
        ]
    )

    result_dict = group_by_days(df, day_granularity=1)

    # Verify the result is a dictionary with the expected keys
    assert isinstance(result_dict, dict)
    assert len(result_dict) == 2  # Two unique dates

    # Get the keys (timestamps) from the dictionary
    timestamps = list(result_dict.keys())

    # Check that the first timestamp is 2023-01-01
    first_timestamp = pd.Timestamp("2023-01-01")
    assert first_timestamp in timestamps

    # Check that the DataFrame for the first timestamp contains the expected texts
    first_day_df = result_dict[first_timestamp]
    assert len(first_day_df) == 2  # Two rows for 2023-01-01
    assert "This is a test document." in first_day_df["text"].values
    assert "A third document." in first_day_df["text"].values


def test_file_to_pd_csv():
    """Test converting a CSV file to a DataFrame."""
    # Mock CSV content
    csv_content = "id,text,timestamp\n1,Test text,2023-01-01\n2,More text,2023-01-02"

    with patch("bertrend.utils.data_loading.DATA_PATH", Path("/mock/data/path")):
        with patch(
            "pandas.read_csv",
            return_value=pd.DataFrame(
                {
                    "id": [1, 2],
                    "text": ["Test text", "More text"],
                    "timestamp": ["2023-01-01", "2023-01-02"],
                }
            ),
        ) as mock_read_csv:
            result = _file_to_pd(Path("test.csv"))

            # Verify read_csv was called with the correct path
            mock_read_csv.assert_called_once_with(Path("/mock/data/path/test.csv"))

            # Verify the result is a DataFrame with expected columns
            assert isinstance(result, pd.DataFrame)
            assert "text" in result.columns
            assert "timestamp" in result.columns
            assert "id" in result.columns


def test_clean_dataset(sample_dataframe):
    """Test cleaning a dataset based on length criteria."""
    # Add a row with short text that should be filtered out
    df = sample_dataframe.copy()
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "text": ["Short"],
                    "timestamp": pd.to_datetime(["2023-01-03"]),
                    "id": [3],
                    "url": ["https://example.com/page3"],
                    "title": ["Title 3"],
                }
            ),
        ]
    )

    result = clean_dataset(df, length_criteria=10)

    # Verify short text is filtered out
    assert len(result) == 2  # Only the two original rows should remain
    assert "Short" not in result["text"].values


def test_enhanced_split_df_by_paragraphs(sample_dataframe):
    """Test enhanced splitting of DataFrame by paragraphs."""
    # Mock the tokenizer
    mock_tokenizer = MagicMock()
    # Mock the encode method to return a list of token IDs
    mock_tokenizer.encode.side_effect = (
        lambda text, padding=None, truncation=None, add_special_tokens=None: [
            1,
            2,
            3,
            4,
            5,
        ]
    )

    # Create a patch for SentenceTransformer that directly returns the max_seq_length
    # This avoids the complex mocking of internal attributes
    with patch(
        "bertrend.utils.data_loading.SentenceTransformer"
    ) as mock_sentence_transformer_class:
        # Configure the mock to return a fixed max_seq_length
        mock_instance = MagicMock()
        mock_instance.get_max_seq_length.return_value = 512
        mock_sentence_transformer_class.return_value = mock_instance

        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
        ):
            # Provide an embedding_model_name parameter
            result = enhanced_split_df_by_paragraphs(
                sample_dataframe, embedding_model_name="bert-base-uncased"
            )

            # Verify the result has at least the same number of rows as the input
            assert len(result) >= len(sample_dataframe)

            # Verify the result has the expected columns
            expected_columns = set(sample_dataframe.columns).union(
                {"is_split", "num_tokens"}
            )
            assert set(result.columns) == expected_columns

            # Verify the tokenizer's encode method was called
            assert mock_tokenizer.encode.called

            # Verify the SentenceTransformer's get_max_seq_length method was called
            assert mock_instance.get_max_seq_length.called


def test_preprocess_french_text():
    """Test preprocessing French text."""
    # Test with various French text patterns
    test_cases = [
        ("l'exemple", "l'exemple"),  # Apostrophes are normalized but not replaced
        ("aujourd'hui", "aujourd'hui"),  # Apostrophes in common words
        (
            "C'est-à-dire",
            "C'est à dire",
        ),  # Hyphens at word boundaries are replaced with spaces
        ("anti-virus", "anti virus"),  # Hyphens at word boundaries
        ("M. Dupont", "M. Dupont"),  # Periods in abbreviations are preserved
        ("10.5%", "10.5 "),  # The % sign is replaced with a space
        ("www.example.com", "www.example.com"),  # URLs with periods are preserved
        ("word1/word2", "word1 word2"),  # Forward slashes at word boundaries
        ("word1;word2", "word1 word2"),  # Semicolons at word boundaries
        ("word1:word2", "word1 word2"),  # Colons at word boundaries
        ("camelCase", "camel Case"),  # Words with capitals in the middle are split
        ("ALL_CAPS", "ALL_CAPS"),  # Fully capitalized words are not split
        ("super²", "super2"),  # Superscripts are normalized
        ("sub₂", "sub2"),  # Subscripts are normalized
        (
            "multiple  spaces",
            "multiple spaces",
        ),  # Multiple spaces are replaced with a single space
    ]

    for input_text, expected_output in test_cases:
        result = preprocess_french_text(input_text)
        assert result == expected_output
