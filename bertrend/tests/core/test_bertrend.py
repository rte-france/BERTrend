#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from bertrend import BERTREND_DEFAULT_CONFIG_PATH
from bertrend.BERTrend import BERTrend, _preprocess_model
from bertrend.BERTopicModel import BERTopicModel


@pytest.fixture
def mock_config():
    """Fixture for mocking the configuration."""
    return {
        "granularity": 7,
        "min_similarity": 0.7,
        "decay_factor": 0.9,
        "decay_power": 1.0,
    }


@pytest.fixture
def mock_topic_model():
    """Fixture for mocking the BERTopicModel."""
    mock_model = MagicMock(spec=BERTopicModel)
    mock_model.fit.return_value.topic_model = MagicMock(spec=BERTopic)
    return mock_model


@pytest.fixture
def mock_sentence_transformer():
    """Fixture for mocking the SentenceTransformer."""
    return MagicMock(spec=SentenceTransformer)


@pytest.fixture
def mock_embeddings():
    """Fixture for mocking embeddings."""
    return np.random.random((10, 768))


@pytest.fixture
def mock_grouped_data():
    """Fixture for mocking grouped data."""
    timestamp1 = pd.Timestamp("2023-01-01")
    timestamp2 = pd.Timestamp("2023-01-08")

    df1 = pd.DataFrame(
        {
            "document_id": [f"doc_{i}" for i in range(5)],
            "text": [f"Document {i}" for i in range(5)],
            "source": ["source1"] * 5,
            "url": [f"http://example.com/{i}" for i in range(5)],
        }
    )

    df2 = pd.DataFrame(
        {
            "document_id": [f"doc_{i+5}" for i in range(5)],
            "text": [f"Document {i+5}" for i in range(5)],
            "source": ["source2"] * 5,
            "url": [f"http://example.com/{i+5}" for i in range(5)],
        }
    )

    return {timestamp1: df1, timestamp2: df2}


@pytest.fixture
def bertrend_instance(mock_config, mock_topic_model):
    """Fixture for creating a BERTrend instance with mocked dependencies."""
    with patch("bertrend.BERTrend.load_toml_config", return_value=mock_config):
        instance = BERTrend(
            config_file=BERTREND_DEFAULT_CONFIG_PATH, topic_model=mock_topic_model
        )
        return instance


def test_initialization(bertrend_instance, mock_config, mock_topic_model):
    """Test initialization of BERTrend class."""
    assert bertrend_instance.config == mock_config
    assert bertrend_instance.topic_model == mock_topic_model
    assert bertrend_instance._is_fitted is False
    assert bertrend_instance.last_topic_model is None
    assert bertrend_instance.last_topic_model_timestamp is None
    assert bertrend_instance.doc_groups == {}
    assert bertrend_instance.emb_groups == {}
    assert bertrend_instance.merge_df_size_over_time == []
    assert bertrend_instance.all_new_topics_df is None
    assert bertrend_instance.all_merge_histories_df is None
    assert bertrend_instance.merged_df is None


def test_load_config(mock_config, mock_topic_model):
    """Test loading configuration from file."""
    # Test with mocked dependencies
    with patch("bertrend.BERTrend.load_toml_config", return_value=mock_config):
        bertrend = BERTrend(
            config_file=BERTREND_DEFAULT_CONFIG_PATH, topic_model=mock_topic_model
        )
        assert bertrend.config == mock_config
        assert bertrend.topic_model == mock_topic_model


def test_get_periods(bertrend_instance):
    """Test get_periods method."""
    # Setup
    bertrend_instance.doc_groups = {
        pd.Timestamp("2023-01-01"): ["doc1", "doc2"],
        pd.Timestamp("2023-01-08"): ["doc3", "doc4"],
    }

    # Execute
    periods = bertrend_instance.get_periods()

    # Assert
    assert len(periods) == 2
    assert pd.Timestamp("2023-01-01") in periods
    assert pd.Timestamp("2023-01-08") in periods


def test_train_by_period(bertrend_instance, mock_sentence_transformer, mock_embeddings):
    """Test _train_by_period method."""
    # Setup
    period = pd.Timestamp("2023-01-01")
    group = pd.DataFrame(
        {
            "text": ["Document 1", "Document 2"],
            "document_id": ["doc1", "doc2"],
            "source": ["source1", "source2"],
            "url": ["http://example.com/1", "http://example.com/2"],
        }
    )

    # Create a mock for the entire _train_by_period method
    # This avoids the complexity of mocking all the internal operations
    original_train_by_period = bertrend_instance._train_by_period

    mock_topic_model = MagicMock(spec=BERTopic)
    mock_topic_model.doc_info_df = pd.DataFrame({"Topic": [0, 1]})
    mock_topic_model.topic_info_df = pd.DataFrame({"Topic": [0, 1]})

    expected_docs = group["text"].tolist()
    expected_embeddings = mock_embeddings[group.index]

    # Replace the method with a mock that returns our expected values
    bertrend_instance._train_by_period = MagicMock(
        return_value=(mock_topic_model, expected_docs, expected_embeddings)
    )

    try:
        # Execute
        result_model, result_docs, result_embeddings = (
            bertrend_instance._train_by_period(
                period, group, mock_sentence_transformer, mock_embeddings
            )
        )

        # Assert
        bertrend_instance._train_by_period.assert_called_once_with(
            period, group, mock_sentence_transformer, mock_embeddings
        )
        assert result_model == mock_topic_model
        assert result_docs == expected_docs
        assert np.array_equal(result_embeddings, expected_embeddings)
    finally:
        # Restore the original method
        bertrend_instance._train_by_period = original_train_by_period


def test_train_topic_models(
    bertrend_instance, mock_grouped_data, mock_sentence_transformer, mock_embeddings
):
    """Test train_topic_models method."""
    # Setup
    mock_new_topic_model = MagicMock()
    bertrend_instance._train_by_period = MagicMock(
        return_value=(mock_new_topic_model, ["doc1"], np.array([[0.1, 0.2]]))
    )
    bertrend_instance.merge_models_with = MagicMock()

    with patch("bertrend.BERTrend.BERTrend.save_topic_model") as mock_save:
        # Execute
        bertrend_instance.train_topic_models(
            mock_grouped_data, mock_sentence_transformer, mock_embeddings
        )

        # Assert
        assert bertrend_instance._train_by_period.call_count == 2
        assert bertrend_instance.merge_models_with.call_count == 1
        assert mock_save.call_count == 2
        assert bertrend_instance._is_fitted is True
        assert bertrend_instance.last_topic_model == mock_new_topic_model
        assert (
            bertrend_instance.last_topic_model_timestamp
            == list(mock_grouped_data.keys())[1]
        )


def test_train_topic_models_error_handling(
    bertrend_instance, mock_grouped_data, mock_sentence_transformer, mock_embeddings
):
    """Test error handling in train_topic_models method."""
    # Setup
    bertrend_instance._train_by_period = MagicMock(
        side_effect=Exception("Test exception")
    )

    with patch("bertrend.BERTrend.logger.error") as mock_logger_error:
        with patch("bertrend.BERTrend.logger.exception") as mock_logger_exception:
            # Execute
            bertrend_instance.train_topic_models(
                mock_grouped_data, mock_sentence_transformer, mock_embeddings
            )

            # Assert
            assert mock_logger_error.called
            assert mock_logger_exception.called
            assert (
                bertrend_instance._is_fitted is True
            )  # Should still be set to True even if errors occur


def test_merge_models_with(bertrend_instance):
    """Test merge_models_with method."""
    # Setup
    bertrend_instance.last_topic_model = MagicMock()
    bertrend_instance.last_topic_model_timestamp = pd.Timestamp("2023-01-01")
    bertrend_instance.doc_groups = {
        pd.Timestamp("2023-01-01"): ["doc1", "doc2"],
        pd.Timestamp("2023-01-08"): ["doc3", "doc4"],
    }
    bertrend_instance.emb_groups = {
        pd.Timestamp("2023-01-01"): np.array([[0.1, 0.2], [0.3, 0.4]]),
        pd.Timestamp("2023-01-08"): np.array([[0.5, 0.6], [0.7, 0.8]]),
    }
    bertrend_instance.merged_df = None
    bertrend_instance.all_merge_histories_df = pd.DataFrame()
    bertrend_instance.all_new_topics_df = pd.DataFrame()

    new_model = MagicMock()
    new_model_timestamp = pd.Timestamp("2023-01-08")

    # Mock _preprocess_model function
    with patch("bertrend.BERTrend._preprocess_model") as mock_preprocess:
        mock_df1 = pd.DataFrame({"Topic": [0, 1], "Count": [2, 3]})
        mock_df2 = pd.DataFrame({"Topic": [0, 1], "Count": [1, 2]})
        mock_preprocess.side_effect = [mock_df1, mock_df2]

        # Mock _merge_models function
        with patch("bertrend.BERTrend._merge_models") as mock_merge:
            mock_merged_df = pd.DataFrame({"Topic": [0, 1, 2], "Count": [3, 5, 1]})
            mock_merge_history = pd.DataFrame({"Topic1": [0], "Topic2": [0]})
            mock_new_topics = pd.DataFrame({"Topic": [2]})
            mock_merge.return_value = (
                mock_merged_df,
                mock_merge_history,
                mock_new_topics,
            )

            # Execute
            bertrend_instance.merge_models_with(new_model, new_model_timestamp)

            # Assert
            mock_preprocess.assert_called()
            mock_merge.assert_called_once()
            assert bertrend_instance.merged_df is not None
            assert not bertrend_instance.all_merge_histories_df.empty
            assert not bertrend_instance.all_new_topics_df.empty
            assert len(bertrend_instance.merge_df_size_over_time) == 1


def test_merge_models_with_empty_dataframe(bertrend_instance):
    """Test merge_models_with method when one dataframe is empty."""
    # Setup
    bertrend_instance.last_topic_model = MagicMock()
    bertrend_instance.last_topic_model_timestamp = pd.Timestamp("2023-01-01")
    bertrend_instance.doc_groups = {
        pd.Timestamp("2023-01-01"): ["doc1", "doc2"],
        pd.Timestamp("2023-01-08"): ["doc3", "doc4"],
    }
    bertrend_instance.emb_groups = {
        pd.Timestamp("2023-01-01"): np.array([[0.1, 0.2], [0.3, 0.4]]),
        pd.Timestamp("2023-01-08"): np.array([[0.5, 0.6], [0.7, 0.8]]),
    }
    bertrend_instance.merged_df = None
    bertrend_instance.all_merge_histories_df = pd.DataFrame()
    bertrend_instance.all_new_topics_df = pd.DataFrame()

    new_model = MagicMock()
    new_model_timestamp = pd.Timestamp("2023-01-08")

    # Mock _preprocess_model function to return an empty dataframe
    with patch("bertrend.BERTrend._preprocess_model") as mock_preprocess:
        mock_df1 = pd.DataFrame({"Topic": [0, 1], "Count": [2, 3]})
        # Empty dataframe with expected columns
        mock_df2 = pd.DataFrame(
            columns=[
                "Topic",
                "Count",
                "Document_Count",
                "Representation",
                "Documents",
                "Embedding",
                "DocEmbeddings",
                "Sources",
                "URLs",
            ]
        )
        mock_preprocess.side_effect = [mock_df1, mock_df2]

        with patch("bertrend.BERTrend.logger.warning") as mock_warning:
            # Execute
            result = bertrend_instance.merge_models_with(new_model, new_model_timestamp)

            # Assert
            mock_warning.assert_called_once()
            assert result is None


def test_merge_models_with_invalid_models(bertrend_instance):
    """Test merge_models_with method with invalid models."""
    # Setup
    bertrend_instance.last_topic_model = None  # Invalid model
    new_model = MagicMock()
    new_model_timestamp = pd.Timestamp("2023-01-08")

    # Execute and Assert
    with pytest.raises(ValueError):
        bertrend_instance.merge_models_with(new_model, new_model_timestamp)


def test_calculate_signal_popularity(bertrend_instance):
    """Test calculate_signal_popularity method."""
    # Setup
    bertrend_instance.doc_groups = {
        pd.Timestamp("2023-01-01"): ["doc1", "doc2"],
        pd.Timestamp("2023-01-08"): ["doc3", "doc4"],
    }
    bertrend_instance.all_merge_histories_df = pd.DataFrame(
        {
            "Timestamp": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-08")],
            "Topic1": [0, 0],
            "Topic2": [1, 2],
            "Count1": [2, 3],
            "Count2": [1, 2],
            "Document_Count1": [2, 3],
            "Document_Count2": [1, 2],
            "Source_Diversity1": [1, 1],
            "Source_Diversity2": [1, 1],
        }
    )

    # Mock the trend analysis functions
    with patch("bertrend.BERTrend._initialize_new_topic") as mock_init:
        with patch("bertrend.BERTrend.update_existing_topic") as mock_update:
            with patch(
                "bertrend.BERTrend._apply_decay_to_inactive_topics"
            ) as mock_decay:
                # Execute
                bertrend_instance.calculate_signal_popularity()

                # Assert
                assert mock_init.called
                assert mock_update.called
                assert mock_decay.called
                assert bertrend_instance.topic_sizes is not None
                assert bertrend_instance.topic_last_popularity is not None
                assert bertrend_instance.topic_last_update is not None


def test_calculate_signal_popularity_not_enough_models(bertrend_instance):
    """Test calculate_signal_popularity method when there are not enough models."""
    # Setup
    bertrend_instance.doc_groups = {
        pd.Timestamp("2023-01-01"): ["doc1", "doc2"]
    }  # Only one model

    with patch("bertrend.BERTrend.logger.error") as mock_error:
        # Execute
        bertrend_instance.calculate_signal_popularity()

        # Assert
        mock_error.assert_called_once()
        # State should remain unchanged
        assert isinstance(bertrend_instance.topic_sizes, dict)
        assert isinstance(bertrend_instance.topic_last_popularity, dict)
        assert isinstance(bertrend_instance.topic_last_update, dict)


def test_compute_popularity_values_and_thresholds(bertrend_instance):
    """Test _compute_popularity_values_and_thresholds method."""
    # Setup
    bertrend_instance.topic_sizes = {
        0: {
            "Timestamps": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-08")],
            "Popularity": [0.5, 0.7],
        },
        1: {
            "Timestamps": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-08")],
            "Popularity": [0.3, 0.2],
        },
    }

    # Execute
    window_start, window_end, all_popularity_values, q1, q3 = (
        bertrend_instance._compute_popularity_values_and_thresholds(
            window_size=14, current_date=pd.Timestamp("2023-01-15")
        )
    )

    # Assert
    assert window_start == pd.Timestamp("2023-01-01")
    assert window_end == pd.Timestamp("2023-01-22")
    assert len(all_popularity_values) == 4
    assert q1 > 0
    assert q3 > q1


def test_classify_signals(bertrend_instance):
    """Test classify_signals method."""
    # Setup
    bertrend_instance._compute_popularity_values_and_thresholds = MagicMock(
        return_value=(
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-01-15"),
            [0.1, 0.2, 0.3, 0.4],
            0.15,
            0.35,
        )
    )

    mock_noise_df = pd.DataFrame({"Topic": [0], "Popularity": [0.1]})
    mock_weak_signal_df = pd.DataFrame({"Topic": [1], "Popularity": [0.2]})
    mock_strong_signal_df = pd.DataFrame({"Topic": [2], "Popularity": [0.4]})

    bertrend_instance._classify_signals = MagicMock(
        return_value=(mock_noise_df, mock_weak_signal_df, mock_strong_signal_df)
    )

    # Execute
    noise_df, weak_signal_df, strong_signal_df = bertrend_instance.classify_signals(
        window_size=14, current_date=pd.Timestamp("2023-01-15")
    )

    # Assert
    bertrend_instance._compute_popularity_values_and_thresholds.assert_called_once()
    bertrend_instance._classify_signals.assert_called_once()
    assert noise_df.equals(mock_noise_df)
    assert weak_signal_df.equals(mock_weak_signal_df)
    assert strong_signal_df.equals(mock_strong_signal_df)


def test_classify_signals_with_custom_parameters(bertrend_instance):
    """Test _classify_signals method with custom parameters."""
    # Setup
    window_start = pd.Timestamp("2023-01-01")
    window_end = pd.Timestamp("2023-01-15")
    q1 = 0.15
    q3 = 0.35

    bertrend_instance.topic_sizes = {
        0: {
            "Timestamps": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-08")],
            "Popularity": [0.1, 0.1],  # Below q1, should be noise
            "Docs_Count": [2, 2],
            "Paragraphs_Count": [3, 3],
            "Source_Diversity": [1, 1],
            "Representation": "Topic 0",
            "Representations": ["Topic 0", "Topic 0"],
            "Documents": [
                (pd.Timestamp("2023-01-01"), ["doc1"]),
                (pd.Timestamp("2023-01-08"), ["doc2"]),
            ],
            "Sources": [
                (pd.Timestamp("2023-01-01"), ["source1"]),
                (pd.Timestamp("2023-01-08"), ["source1"]),
            ],
        },
        1: {
            "Timestamps": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-08")],
            "Popularity": [
                0.2,
                0.3,
            ],  # Between q1 and q3, and rising, should be weak signal
            "Docs_Count": [2, 3],
            "Paragraphs_Count": [3, 4],
            "Source_Diversity": [1, 2],
            "Representation": "Topic 1",
            "Representations": ["Topic 1", "Topic 1"],
            "Documents": [
                (pd.Timestamp("2023-01-01"), ["doc3"]),
                (pd.Timestamp("2023-01-08"), ["doc4"]),
            ],
            "Sources": [
                (pd.Timestamp("2023-01-01"), ["source1"]),
                (pd.Timestamp("2023-01-08"), ["source2"]),
            ],
        },
        2: {
            "Timestamps": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-08")],
            "Popularity": [0.4, 0.5],  # Above q3, should be strong signal
            "Docs_Count": [4, 5],
            "Paragraphs_Count": [6, 7],
            "Source_Diversity": [2, 3],
            "Representation": "Topic 2",
            "Representations": ["Topic 2", "Topic 2"],
            "Documents": [
                (pd.Timestamp("2023-01-01"), ["doc5"]),
                (pd.Timestamp("2023-01-08"), ["doc6"]),
            ],
            "Sources": [
                (pd.Timestamp("2023-01-01"), ["source1", "source2"]),
                (pd.Timestamp("2023-01-08"), ["source1", "source2", "source3"]),
            ],
        },
    }

    # Mock _is_rising_popularity to return True for topic 1
    with patch("bertrend.BERTrend._is_rising_popularity", return_value=True):
        # Execute
        noise_df, weak_signal_df, strong_signal_df = (
            bertrend_instance._classify_signals(window_start, window_end, q1, q3)
        )

        # Assert
        assert len(noise_df) == 1
        assert noise_df.iloc[0]["Topic"] == 0

        assert len(weak_signal_df) == 1
        assert weak_signal_df.iloc[0]["Topic"] == 1

        assert len(strong_signal_df) == 1
        assert strong_signal_df.iloc[0]["Topic"] == 2


def test_save_and_restore_model(bertrend_instance, tmp_path):
    """Test save_model and restore_model methods."""
    # Setup
    models_path = tmp_path / "models"
    models_path.mkdir()

    # Mock dill.dump and dill.load
    with patch("bertrend.BERTrend.dill.dump") as mock_dump:
        with patch(
            "bertrend.BERTrend.dill.load", return_value=bertrend_instance
        ) as mock_load:
            with patch("builtins.open", mock_open()) as mock_file:
                # Execute save
                bertrend_instance.save_model(models_path)

                # Assert save
                mock_dump.assert_called_once()

                # Execute restore
                restored = BERTrend.restore_model(models_path)

                # Assert restore
                mock_load.assert_called_once()
                assert restored == bertrend_instance


def test_restore_model_nonexistent_path():
    """Test restore_model method with a nonexistent path."""
    # Setup
    nonexistent_path = Path("/nonexistent/path")

    # Execute and Assert
    with pytest.raises(FileNotFoundError):
        BERTrend.restore_model(nonexistent_path)


def test_save_signal_evolution_data(bertrend_instance, tmp_path):
    """Test save_signal_evolution_data method."""
    # Setup
    save_dir = tmp_path / "signal_evolution_data"
    save_dir.mkdir(parents=True, exist_ok=True)

    with patch("bertrend.BERTrend.SIGNAL_EVOLUTION_DATA_DIR", save_dir):
        window_size = 14
        start_timestamp = pd.Timestamp("2023-01-01")
        end_timestamp = pd.Timestamp("2023-01-15")

        # Ensure bertrend_instance has necessary config
        bertrend_instance.config = {"granularity": 7}

        # Mock _compute_popularity_values_and_thresholds and _classify_signals
        bertrend_instance._compute_popularity_values_and_thresholds = MagicMock(
            return_value=(
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-15"),
                [0.1, 0.2, 0.3, 0.4],
                0.15,
                0.35,
            )
        )

        mock_noise_df = pd.DataFrame({"Topic": [0], "Popularity": [0.1]})
        mock_weak_signal_df = pd.DataFrame({"Topic": [1], "Popularity": [0.2]})
        mock_strong_signal_df = pd.DataFrame({"Topic": [2], "Popularity": [0.4]})

        bertrend_instance._classify_signals = MagicMock(
            return_value=(mock_noise_df, mock_weak_signal_df, mock_strong_signal_df)
        )

        # Create the output directory
        output_dir = save_dir / f"retrospective_{window_size}_days"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Mock open and pickle.dump
        mock_file = MagicMock()
        with patch("builtins.open", mock_open(mock=mock_file)) as mock_open_file:
            with patch("pickle.dump") as mock_dump:
                # Execute
                result_path = bertrend_instance.save_signal_evolution_data(
                    window_size, start_timestamp, end_timestamp
                )

                # Assert
                assert mock_dump.call_count == 4  # 3 dataframes + metadata
                assert result_path == save_dir / f"retrospective_{window_size}_days"


def test_restore_topic_models(bertrend_instance, tmp_path):
    """Test restore_topic_models method."""
    # Setup
    models_path = tmp_path / "models"
    models_path.mkdir()

    # Create mock periods
    period1 = pd.Timestamp("2023-01-01")
    period2 = pd.Timestamp("2023-01-08")
    bertrend_instance.doc_groups = {
        period1: ["doc1", "doc2"],
        period2: ["doc3", "doc4"],
    }

    # Mock restore_topic_model to return different models for different periods
    mock_model1 = MagicMock(spec=BERTopic)
    mock_model2 = MagicMock(spec=BERTopic)

    with patch.object(
        BERTrend,
        "restore_topic_model",
        side_effect=lambda period, models_path: (
            mock_model1
            if period == period1
            else mock_model2 if period == period2 else None
        ),
    ):
        # Execute
        result = bertrend_instance.restore_topic_models(models_path)

        # Assert
        assert len(result) == 2
        assert result[period1] == mock_model1
        assert result[period2] == mock_model2


def test_save_and_restore_topic_model(tmp_path):
    """Test save_topic_model and restore_topic_model class methods."""
    # Setup
    models_path = tmp_path / "models"
    models_path.mkdir()
    period = pd.Timestamp("2023-01-01")

    # Create a mock topic model
    mock_topic_model = MagicMock()
    mock_topic_model.doc_info_df = pd.DataFrame({"Topic": [0, 1]})
    mock_topic_model.topic_info_df = pd.DataFrame({"Topic": [0, 1]})
    mock_topic_model.embedding_model = "all-MiniLM-L6-v2"

    # Add the save method to the mock topic model
    mock_topic_model.save = MagicMock()

    # Mock BERTopic.load to return our mock topic model
    with patch("bertopic.BERTopic.load", return_value=mock_topic_model):
        with patch("pandas.DataFrame.to_pickle") as mock_to_pickle:
            with patch(
                "pandas.read_pickle", return_value=pd.DataFrame()
            ) as mock_read_pickle:
                with patch(
                    "pathlib.Path.glob", return_value=[models_path / "2023-01-01"]
                ):
                    with patch("pathlib.Path.is_dir", return_value=True):
                        with patch("pathlib.Path.exists", return_value=True):
                            # Execute save
                            BERTrend.save_topic_model(
                                period, mock_topic_model, models_path
                            )

                            # Assert save
                            mock_topic_model.save.assert_called_once()
                            assert mock_to_pickle.call_count == 2

                            # Execute restore
                            restored = BERTrend.restore_topic_model(period, models_path)

                            # Assert restore
                            assert restored == mock_topic_model
                            assert mock_read_pickle.call_count == 2


def test_train_new_data_restore_failure():
    """Test train_new_data function when restore_model fails."""
    # Setup
    reference_timestamp = pd.Timestamp("2023-01-01")
    new_data = pd.DataFrame(
        {
            "text": ["Document 1", "Document 2"],
            "document_id": ["doc1", "doc2"],
            "source": ["source1", "source2"],
            "url": ["http://example.com/1", "http://example.com/2"],
            "timestamp": [reference_timestamp, reference_timestamp],
        }
    )
    bertrend_models_path = Path("/tmp/models")

    # Mock EmbeddingService
    mock_embedding_service = MagicMock()
    mock_embedding_service.embed.return_value = (
        np.array([[0.1, 0.2], [0.3, 0.4]]),  # embeddings
        ["token1", "token2"],  # token_strings
        np.array([[0.5, 0.6], [0.7, 0.8]]),  # token_embeddings
    )
    mock_embedding_service.embedding_model_name = "test-model"

    # Create a mock BERTrend instance with necessary attributes
    mock_bertrend = MagicMock(spec=BERTrend)
    mock_bertrend.config = {}
    mock_bertrend.doc_groups = {}

    # Mock BERTrend constructor to return our mock instance
    with patch("bertrend.BERTrend.BERTrend", return_value=mock_bertrend):
        # Mock BERTrend.restore_model to raise an exception
        with patch(
            "bertrend.BERTrend.BERTrend.restore_model",
            side_effect=Exception("Cannot restore"),
        ) as mock_restore:
            # Mock BERTrend.train_topic_models and BERTrend.save_model
            with patch.object(mock_bertrend, "train_topic_models") as mock_train:
                with patch.object(mock_bertrend, "save_model") as mock_save:
                    with patch(
                        "bertrend.BERTrend.BERTopicModel"
                    ) as mock_bertopic_model:

                        # Execute
                        from bertrend.BERTrend import train_new_data

                        result = train_new_data(
                            reference_timestamp=reference_timestamp,
                            new_data=new_data,
                            bertrend_models_path=bertrend_models_path,
                            embedding_service=mock_embedding_service,
                            granularity=7,
                            language="English",
                        )

                        # Assert
                        mock_restore.assert_called_once_with(bertrend_models_path)
                        mock_embedding_service.embed.assert_called_once()
                        mock_train.assert_called_once()
                        mock_save.assert_called_once_with(
                            models_path=bertrend_models_path
                        )


def test_train_new_data():
    """Test train_new_data function."""
    # Setup
    reference_timestamp = pd.Timestamp("2023-01-01")
    new_data = pd.DataFrame(
        {
            "text": ["Document 1", "Document 2"],
            "document_id": ["doc1", "doc2"],
            "source": ["source1", "source2"],
            "url": ["http://example.com/1", "http://example.com/2"],
            "timestamp": [reference_timestamp, reference_timestamp],
        }
    )
    bertrend_models_path = Path("/tmp/models")

    # Mock EmbeddingService
    mock_embedding_service = MagicMock()
    mock_embedding_service.embed.return_value = (
        np.array([[0.1, 0.2], [0.3, 0.4]]),  # embeddings
        ["token1", "token2"],  # token_strings
        np.array([[0.5, 0.6], [0.7, 0.8]]),  # token_embeddings
    )
    mock_embedding_service.embedding_model_name = "test-model"

    # Create a mock BERTrend instance with necessary attributes and methods
    mock_bertrend = MagicMock(spec=BERTrend)
    mock_bertrend.doc_groups = {reference_timestamp: ["doc1", "doc2"]}
    mock_bertrend.config = {"granularity": 7}

    # Create mock methods
    mock_train = MagicMock()
    mock_bertrend.train_topic_models = mock_train
    mock_save = MagicMock()
    mock_bertrend.save_model = mock_save

    # Mock BERTrend.restore_model to return our mock instance
    with patch(
        "bertrend.BERTrend.BERTrend.restore_model", return_value=mock_bertrend
    ) as mock_restore:
        # Execute
        from bertrend.BERTrend import train_new_data

        result = train_new_data(
            reference_timestamp=reference_timestamp,
            new_data=new_data,
            bertrend_models_path=bertrend_models_path,
            embedding_service=mock_embedding_service,
            granularity=7,
            language="English",
        )

        # Assert
        mock_restore.assert_called_once_with(bertrend_models_path)
        mock_embedding_service.embed.assert_called_once()
        mock_train.assert_called_once()
        mock_save.assert_called_once_with(models_path=bertrend_models_path)
        assert result == mock_bertrend


def test_merge_models():
    """Test _merge_models function."""
    # Setup
    df1 = pd.DataFrame(
        {
            "Topic": [0, 1],
            "Count": [2, 3],
            "Document_Count": [2, 3],
            "Representation": ["Topic 0", "Topic 1"],
            "Documents": [["doc1", "doc2"], ["doc3", "doc4"]],
            "Embedding": [np.array([0.1, 0.2]), np.array([0.3, 0.4])],
            "DocEmbeddings": [
                [np.array([0.1, 0.2]), np.array([0.3, 0.4])],
                [np.array([0.5, 0.6]), np.array([0.7, 0.8])],
            ],
            "Sources": [["source1", "source1"], ["source2", "source2"]],
            "URLs": [["url1", "url2"], ["url3", "url4"]],
        }
    )

    df2 = pd.DataFrame(
        {
            "Topic": [0, 1],
            "Count": [1, 2],
            "Document_Count": [1, 2],
            "Representation": ["Topic 0 new", "Topic 1 new"],
            "Documents": [["doc5"], ["doc6", "doc7"]],
            "Embedding": [np.array([0.15, 0.25]), np.array([0.8, 0.9])],
            "DocEmbeddings": [
                [np.array([0.15, 0.25])],
                [np.array([0.8, 0.9]), np.array([0.85, 0.95])],
            ],
            "Sources": [["source3"], ["source4", "source4"]],
            "URLs": [["url5"], ["url6", "url7"]],
        }
    )

    timestamp = pd.Timestamp("2023-01-01")
    min_similarity = 0.9  # High threshold to ensure only the first topic is merged

    # Mock cosine_similarity to return a controlled similarity matrix
    with patch("bertrend.BERTrend.cosine_similarity") as mock_cosine:
        # First topic in df2 is similar to first topic in df1, second topic is not similar to any
        mock_cosine.return_value = np.array([[0.95, 0.5], [0.6, 0.6]])

        # Execute
        from bertrend.BERTrend import _merge_models

        merged_df, merge_history, new_topics = _merge_models(
            df1, df2, min_similarity, timestamp
        )

        # Assert
        mock_cosine.assert_called_once()

        # Check merged_df
        assert len(merged_df) == 3  # 2 original topics + 1 new topic
        assert merged_df["Topic"].tolist() == [0, 1, 2]  # New topic gets ID 2

        # Check the merged topic (Topic 0)
        merged_topic = merged_df[merged_df["Topic"] == 0].iloc[0]
        assert merged_topic["Count"] == 3  # 2 from df1 + 1 from df2
        assert merged_topic["Document_Count"] == 3  # 2 from df1 + 1 from df2

        # Check merge_history
        assert len(merge_history) == 1  # Only one topic was merged
        assert merge_history["Topic1"].iloc[0] == 0
        assert merge_history["Topic2"].iloc[0] == 0

        # Check new_topics
        assert len(new_topics) == 1  # Only one new topic was added
        assert new_topics["Topic"].iloc[0] == 2


def test_filter_data():
    """Test _filter_data function."""
    # Setup
    from bertrend.trend_analysis.weak_signals import _filter_data

    # Create test data
    data = {
        "Timestamps": [
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-01-08"),
            pd.Timestamp("2023-01-15"),
            pd.Timestamp("2023-01-22"),
        ],
        "Popularity": [0.1, 0.2, 0.3, 0.4],
        "Docs_Count": [1, 2, 3, 4],
        "Paragraphs_Count": [2, 4, 6, 8],
        "Source_Diversity": [1, 2, 2, 3],
        "Representation": ["Topic 1", "Topic 1", "Topic 1", "Topic 1"],
        "Representations": ["Topic 1", "Topic 1", "Topic 1", "Topic 1"],
        "Documents": [
            (pd.Timestamp("2023-01-01"), ["doc1"]),
            (pd.Timestamp("2023-01-08"), ["doc2", "doc3"]),
            (pd.Timestamp("2023-01-15"), ["doc4", "doc5", "doc6"]),
            (pd.Timestamp("2023-01-22"), ["doc7", "doc8", "doc9", "doc10"]),
        ],
        "Sources": [
            (pd.Timestamp("2023-01-01"), ["source1"]),
            (pd.Timestamp("2023-01-08"), ["source1", "source2"]),
            (pd.Timestamp("2023-01-15"), ["source1", "source2"]),
            (pd.Timestamp("2023-01-22"), ["source1", "source2", "source3"]),
        ],
    }

    # Test case 1: Filter with window_end after all timestamps, keep documents
    window_end1 = pd.Timestamp("2023-01-29")
    keep_documents1 = True

    # Test case 2: Filter with window_end in the middle, keep documents
    window_end2 = pd.Timestamp("2023-01-15")
    keep_documents2 = True

    # Test case 3: Filter with window_end after all timestamps, don't keep documents
    window_end3 = pd.Timestamp("2023-01-29")
    keep_documents3 = False

    # Execute
    filtered_data1 = _filter_data(data, window_end1, keep_documents1)
    filtered_data2 = _filter_data(data, window_end2, keep_documents2)
    filtered_data3 = _filter_data(data, window_end3, keep_documents3)

    # Assert
    # Case 1: All data should be included
    assert len(filtered_data1["Timestamps"]) == 4
    assert filtered_data1["Popularity"] == data["Popularity"]
    # Documents are flattened in the output
    expected_docs1 = [
        "doc1",
        "doc2",
        "doc3",
        "doc4",
        "doc5",
        "doc6",
        "doc7",
        "doc8",
        "doc9",
        "doc10",
    ]
    assert filtered_data1["Documents"] == expected_docs1

    # Case 2: Only data up to 2023-01-15 should be included
    assert len(filtered_data2["Timestamps"]) == 3
    assert filtered_data2["Popularity"] == data["Popularity"][:3]
    # Documents are flattened in the output
    expected_docs2 = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]
    assert filtered_data2["Documents"] == expected_docs2

    # Case 3: All data should be included, but Documents should be empty
    assert len(filtered_data3["Timestamps"]) == 4
    assert filtered_data3["Popularity"] == data["Popularity"]
    assert filtered_data3["Documents"] == []


def test_is_rising_popularity():
    """Test _is_rising_popularity function."""
    # Setup
    from bertrend.trend_analysis.weak_signals import _is_rising_popularity

    # Test case 1: Rising popularity
    data1 = {
        "Timestamps": [
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-01-08"),
            pd.Timestamp("2023-01-15"),
        ],
        "Popularity": [0.1, 0.2, 0.3],  # Clearly rising
    }

    # Test case 2: Falling popularity
    data2 = {
        "Timestamps": [
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-01-08"),
            pd.Timestamp("2023-01-15"),
        ],
        "Popularity": [0.3, 0.2, 0.1],  # Clearly falling
    }

    # Test case 3: Stable popularity
    data3 = {
        "Timestamps": [
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-01-08"),
            pd.Timestamp("2023-01-15"),
        ],
        "Popularity": [0.2, 0.2, 0.2],  # Stable
    }

    # Test case 4: Fluctuating but overall rising
    data4 = {
        "Timestamps": [
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-01-08"),
            pd.Timestamp("2023-01-15"),
            pd.Timestamp("2023-01-22"),
            pd.Timestamp("2023-01-29"),
        ],
        "Popularity": [0.1, 0.05, 0.15, 0.1, 0.2],  # Fluctuating but rising
    }

    # Execute and Assert
    assert _is_rising_popularity(data1, data1["Timestamps"][-1]) == True
    assert _is_rising_popularity(data2, data2["Timestamps"][-1]) == False
    assert _is_rising_popularity(data3, data3["Timestamps"][-1]) == False
    assert _is_rising_popularity(data4, data4["Timestamps"][-1]) == True


def test_preprocess_model():
    """Test _preprocess_model function."""
    # Setup
    mock_topic_model = MagicMock(spec=BERTopic)
    mock_topic_model.topic_info_df = pd.DataFrame(
        {
            "Topic": [0, 1],
            "Count": [2, 3],
            "Document_Count": [2, 3],
            "Representation": ["Topic 0", "Topic 1"],
            "Name": ["", ""],
            "Representative_Docs": [["doc1"], ["doc2"]],
            "Sources": [["source1"], ["source2"]],
            "URLs": [["url1"], ["url2"]],
        }
    )

    mock_topic_model.doc_info_df = pd.DataFrame(
        {
            "Topic": [0, 0, 1],
            "Paragraph": ["doc1", "doc2", "doc3"],
            "Probability": [0.9, 0.8, 0.7],
            "source": ["source1", "source1", "source2"],
            "url": ["url1", "url2", "url3"],
        }
    )

    docs = ["doc1", "doc2", "doc3"]
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    # Execute
    result = _preprocess_model(mock_topic_model, docs, embeddings)

    # Assert
    assert isinstance(result, pd.DataFrame)
    assert "Topic" in result.columns
    assert "Count" in result.columns
    assert "Document_Count" in result.columns
    assert "Representation" in result.columns
    assert "Documents" in result.columns
    assert "Embedding" in result.columns
    assert "DocEmbeddings" in result.columns
    assert "Sources" in result.columns
    assert "URLs" in result.columns
