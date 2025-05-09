#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.


from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from bertrend.utils.cache_utils import (
    load_embeddings,
    save_embeddings,
    get_hash,
)


def test_load_embeddings():
    """Test loading embeddings from a pickle file."""
    # Mock data
    mock_embeddings = [1, 2, 3]
    mock_path = Path("test_embeddings.pkl")

    # Mock open and pickle.load
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("pickle.load", return_value=mock_embeddings) as mock_pickle_load:
            result = load_embeddings(mock_path)

            # Verify file was opened correctly
            mock_file.assert_called_once_with(mock_path, "rb")
            # Verify pickle.load was called with the file handle
            mock_pickle_load.assert_called_once()
            # Verify the result is correct
            assert result == mock_embeddings


def test_save_embeddings():
    """Test saving embeddings to a pickle file."""
    # Mock data
    mock_embeddings = [1, 2, 3]
    mock_path = Path("test_embeddings.pkl")

    # Mock open and pickle.dump
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("pickle.dump") as mock_pickle_dump:
            save_embeddings(mock_embeddings, mock_path)

            # Verify file was opened correctly
            mock_file.assert_called_once_with(mock_path, "wb")
            # Verify pickle.dump was called with the embeddings and file handle
            mock_pickle_dump.assert_called_once()
            args, _ = mock_pickle_dump.call_args
            assert args[0] == mock_embeddings


def test_get_hash_string():
    """Test get_hash with a string input."""
    # Test with a string
    test_string = "test_string"
    hash_result = get_hash(test_string)

    # Verify the hash is a string of the expected length (MD5 hash is 32 characters)
    assert isinstance(hash_result, str)
    assert len(hash_result) == 32

    # Verify the hash is deterministic (same input produces same hash)
    assert hash_result == get_hash(test_string)


def test_get_hash_list():
    """Test get_hash with a list input."""
    # Test with a list
    test_list = [1, 2, 3]
    hash_result = get_hash(test_list)

    # Verify the hash is a string of the expected length
    assert isinstance(hash_result, str)
    assert len(hash_result) == 32

    # Verify the hash is deterministic
    assert hash_result == get_hash(test_list)


def test_get_hash_dict():
    """Test get_hash with a dictionary input."""
    # Test with a dictionary
    test_dict = {"a": 1, "b": 2}
    hash_result = get_hash(test_dict)

    # Verify the hash is a string of the expected length
    assert isinstance(hash_result, str)
    assert len(hash_result) == 32

    # Verify the hash is deterministic
    assert hash_result == get_hash(test_dict)


def test_get_hash_different_inputs():
    """Test that different inputs produce different hashes."""
    # Test with different inputs
    hash1 = get_hash("test1")
    hash2 = get_hash("test2")
    hash3 = get_hash([1, 2, 3])
    hash4 = get_hash({"a": 1})

    # Verify different inputs produce different hashes
    assert hash1 != hash2
    assert hash1 != hash3
    assert hash1 != hash4
    assert hash2 != hash3
    assert hash2 != hash4
    assert hash3 != hash4
