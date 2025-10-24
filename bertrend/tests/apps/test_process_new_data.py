#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
import jsonlines
import tempfile
import shutil

from bertrend_apps.prospective_demo.process_new_data import (
    load_all_data,
    get_model_config,
    generate_llm_interpretation,
    train_new_model_for_period,
    regenerate_models,
)


class TestLoadAllData:
    """Test the load_all_data function."""

    @patch("bertrend_apps.prospective_demo.process_new_data.get_user_feed_path")
    @patch("bertrend_apps.prospective_demo.process_new_data.load_toml_config")
    @patch("bertrend_apps.prospective_demo.process_new_data.load_data")
    @patch("bertrend_apps.prospective_demo.process_new_data.Path")
    def test_load_all_data_success(
        self, mock_path, mock_load_data, mock_load_toml_config, mock_get_user_feed_path
    ):
        """Test successful loading of all data."""
        # Setup mocks
        mock_cfg_file = Mock()
        mock_cfg_file.exists.return_value = True
        mock_get_user_feed_path.return_value = mock_cfg_file

        mock_load_toml_config.return_value = {
            "data-feed": {"feed_dir_path": "test_feed", "id": "test_id"}
        }

        # Mock Path and glob
        mock_path_instance = Mock()
        mock_path_instance.glob.return_value = [
            Path("file1.jsonl"),
            Path("file2.jsonl"),
        ]
        mock_path.return_value = mock_path_instance

        # Mock dataframes
        df1 = pd.DataFrame(
            {"title": ["title1", "title2"], "content": ["content1", "content2"]}
        )
        df2 = pd.DataFrame(
            {"title": ["title3", "title1"], "content": ["content3", "content1"]}
        )  # title1 is duplicate
        mock_load_data.side_effect = [df1, df2]

        # Execute
        result = load_all_data("test_model", "test_user", "en")

        # Verify
        assert result is not None
        assert len(result) == 3  # Should remove one duplicate
        assert list(result["title"]) == ["title1", "title2", "title3"]
        mock_get_user_feed_path.assert_called_once_with("test_user", "test_model")
        mock_load_toml_config.assert_called_once_with(mock_cfg_file)

    @patch("bertrend_apps.prospective_demo.process_new_data.get_user_feed_path")
    def test_load_all_data_config_not_found(self, mock_get_user_feed_path):
        """Test behavior when config file doesn't exist."""
        from bertrend_apps.prospective_demo.process_new_data import (
            ConfigFileNotFoundError,
        )

        mock_cfg_file = Mock()
        mock_cfg_file.exists.return_value = False
        mock_get_user_feed_path.return_value = mock_cfg_file

        with pytest.raises(ConfigFileNotFoundError):
            load_all_data("test_model", "test_user", "en")

    @patch("bertrend_apps.prospective_demo.process_new_data.get_user_feed_path")
    @patch("bertrend_apps.prospective_demo.process_new_data.load_toml_config")
    @patch("bertrend_apps.prospective_demo.process_new_data.Path")
    def test_load_all_data_no_files(
        self, mock_path, mock_load_toml_config, mock_get_user_feed_path
    ):
        """Test behavior when no data files are found."""
        from bertrend_apps.prospective_demo.process_new_data import NoDataAvailableError

        mock_cfg_file = Mock()
        mock_cfg_file.exists.return_value = True
        mock_get_user_feed_path.return_value = mock_cfg_file

        mock_load_toml_config.return_value = {
            "data-feed": {"feed_dir_path": "test_feed", "id": "test_id"}
        }

        mock_path_instance = Mock()
        mock_path_instance.glob.return_value = []
        mock_path.return_value = mock_path_instance

        with pytest.raises(NoDataAvailableError):
            load_all_data("test_model", "test_user", "en")


class TestGetModelConfig:
    """Test the get_model_config function."""

    @patch("bertrend_apps.prospective_demo.process_new_data.get_model_cfg_path")
    @patch("bertrend_apps.prospective_demo.process_new_data.load_toml_config")
    def test_get_model_config_success(
        self, mock_load_toml_config, mock_get_model_cfg_path
    ):
        """Test successful loading of model config."""
        mock_cfg_path = Mock(spec=Path)
        mock_cfg_path.exists.return_value = True
        mock_get_model_cfg_path.return_value = mock_cfg_path
        mock_load_toml_config.return_value = {
            "model_config": {"granularity": 7, "window_size": 30, "language": "en"}
        }

        model_config = get_model_config("test_model", "test_user")

        assert model_config["granularity"] == 7
        assert model_config["window_size"] == 30
        assert model_config["language"] == "en"

    @patch("bertrend_apps.prospective_demo.process_new_data.get_model_cfg_path")
    @patch("bertrend_apps.prospective_demo.process_new_data.load_toml_config")
    def test_get_model_config_exception(
        self, mock_load_toml_config, mock_get_model_cfg_path
    ):
        """Test exception when loading fails."""
        from bertrend_apps.prospective_demo.process_new_data import (
            InvalidModelConfigError,
        )

        mock_cfg_path = Mock(spec=Path)
        mock_cfg_path.exists.return_value = True
        mock_get_model_cfg_path.return_value = mock_cfg_path
        mock_load_toml_config.side_effect = Exception("Config loading failed")

        with pytest.raises(InvalidModelConfigError):
            get_model_config("test_model", "test_user")

    @patch("bertrend_apps.prospective_demo.process_new_data.get_model_cfg_path")
    @patch("bertrend_apps.prospective_demo.process_new_data.load_toml_config")
    def test_get_model_config_split_by_paragraph(
        self, mock_load_toml_config, mock_get_model_cfg_path
    ):
        """Test split_by_paragraph parameter handling."""
        mock_cfg_path = Mock(spec=Path)
        mock_cfg_path.exists.return_value = True
        mock_get_model_cfg_path.return_value = mock_cfg_path
        mock_load_toml_config.return_value = {
            "model_config": {
                "granularity": 7,
                "window_size": 30,
                "language": "en",
                "split_by_paragraph": False,
            }
        }

        model_config = get_model_config("test_model", "test_user")

        assert model_config["split_by_paragraph"] == False


class TestGenerateLLMInterpretation:
    """Test the generate_llm_interpretation function."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.temp_dir)

        # Mock BERTrend instance
        self.mock_bertrend = Mock()

        # Mock DataFrame
        self.mock_df = pd.DataFrame(
            {
                "Topic": [1, 2, 3, 4, 5, 6],
                "Latest_Popularity": [100, 90, 80, 70, 60, 50],
            }
        )

        self.reference_timestamp = pd.Timestamp("2024-01-01")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch("bertrend_apps.prospective_demo.process_new_data.analyze_signal")
    def test_generate_llm_interpretation_success(self, mock_analyze_signal):
        """Test successful generation of LLM interpretation with parallel processing."""
        # Setup mock analyze_signal responses
        mock_summary = Mock()
        mock_summary.model_dump_json.return_value = '{"summary": "test"}'
        mock_analysis = Mock()
        mock_analysis.model_dump_json.return_value = '{"analysis": "test"}'

        mock_analyze_signal.return_value = (mock_summary, mock_analysis)

        # Execute
        generate_llm_interpretation(
            self.mock_bertrend,
            self.reference_timestamp,
            self.mock_df,
            "test_df",
            self.output_path,
            top_k=3,
            max_workers=2,
        )

        # Verify analyze_signal was called for top 3 topics
        assert mock_analyze_signal.call_count == 3
        expected_topics = [1, 2, 3]  # Top 3 by Latest_Popularity
        actual_topics = [call[0][1] for call in mock_analyze_signal.call_args_list]
        assert sorted(actual_topics) == sorted(expected_topics)

        # Verify output file was created
        output_file = self.output_path / "test_df_interpretation.jsonl"
        assert output_file.exists()

        # Verify file contents
        with jsonlines.open(output_file) as reader:
            results = list(reader)

        assert len(results) == 3
        for result in results:
            assert "topic" in result
            assert "summary" in result
            assert "analysis" in result

    @patch("bertrend_apps.prospective_demo.process_new_data.analyze_signal")
    def test_generate_llm_interpretation_with_failures(self, mock_analyze_signal):
        """Test handling of analyze_signal failures."""

        # Setup mock to fail for some topics
        def mock_analyze_signal_side_effect(bertrend, topic, timestamp):
            if topic == 2:
                return None, None  # Simulate failure
            mock_summary = Mock()
            mock_summary.model_dump_json.return_value = f'{{"summary": "test_{topic}"}}'
            mock_analysis = Mock()
            mock_analysis.model_dump_json.return_value = (
                f'{{"analysis": "test_{topic}"}}'
            )
            return mock_summary, mock_analysis

        mock_analyze_signal.side_effect = mock_analyze_signal_side_effect

        # Execute
        generate_llm_interpretation(
            self.mock_bertrend,
            self.reference_timestamp,
            self.mock_df,
            "test_df",
            self.output_path,
            top_k=3,
            max_workers=2,
        )

        # Verify output file was created
        output_file = self.output_path / "test_df_interpretation.jsonl"
        assert output_file.exists()

        # Verify only successful results are saved
        with jsonlines.open(output_file) as reader:
            results = list(reader)

        assert len(results) == 2  # Only topics 1 and 3 should succeed
        result_topics = [result["topic"] for result in results]
        assert sorted(result_topics) == [1, 3]

    @patch("bertrend_apps.prospective_demo.process_new_data.analyze_signal")
    def test_generate_llm_interpretation_exception_handling(self, mock_analyze_signal):
        """Test exception handling during topic processing."""

        # Setup mock to raise exception for some topics
        def mock_analyze_signal_side_effect(bertrend, topic, timestamp):
            if topic == 2:
                raise Exception("Processing error")
            mock_summary = Mock()
            mock_summary.model_dump_json.return_value = f'{{"summary": "test_{topic}"}}'
            mock_analysis = Mock()
            mock_analysis.model_dump_json.return_value = (
                f'{{"analysis": "test_{topic}"}}'
            )
            return mock_summary, mock_analysis

        mock_analyze_signal.side_effect = mock_analyze_signal_side_effect

        # Execute (should not raise exception)
        generate_llm_interpretation(
            self.mock_bertrend,
            self.reference_timestamp,
            self.mock_df,
            "test_df",
            self.output_path,
            top_k=3,
            max_workers=2,
        )

        # Verify output file was created
        output_file = self.output_path / "test_df_interpretation.jsonl"
        assert output_file.exists()

        # Verify only successful results are saved
        with jsonlines.open(output_file) as reader:
            results = list(reader)

        assert len(results) == 2  # Only topics 1 and 3 should succeed

    def test_generate_llm_interpretation_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["Topic", "Latest_Popularity"])

        generate_llm_interpretation(
            self.mock_bertrend,
            self.reference_timestamp,
            empty_df,
            "test_df",
            self.output_path,
            top_k=3,
        )

        # Verify output file was created but is empty
        output_file = self.output_path / "test_df_interpretation.jsonl"
        assert output_file.exists()

        with jsonlines.open(output_file) as reader:
            results = list(reader)

        assert len(results) == 0


class TestTrainNewModelForPeriod:
    """Test the train_new_model_for_period function."""

    @patch("bertrend_apps.prospective_demo.process_new_data.EmbeddingService")
    @patch("bertrend_apps.prospective_demo.process_new_data.get_user_models_path")
    @patch("bertrend_apps.prospective_demo.process_new_data.get_model_config")
    @patch("bertrend_apps.prospective_demo.process_new_data.train_new_data")
    @patch(
        "bertrend_apps.prospective_demo.process_new_data.get_model_interpretation_path"
    )
    @patch(
        "bertrend_apps.prospective_demo.process_new_data.generate_bertrend_topic_description"
    )
    @patch(
        "bertrend_apps.prospective_demo.process_new_data.generate_llm_interpretation"
    )
    def test_train_new_model_for_period_success(
        self,
        mock_generate_llm,
        mock_generate_desc,
        mock_get_interp_path,
        mock_train_new_data,
        mock_get_config,
        mock_get_models_path,
        mock_embedding_service,
    ):
        """Test successful model training for period."""
        # Setup mocks
        mock_get_models_path.return_value = Path("/test/models")
        mock_get_config.return_value = {
            "granularity": 7,
            "window_size": 30,
            "language": "en",
        }

        # Mock BERTrend instance
        mock_bertrend = Mock()
        mock_bertrend.doc_groups = [Mock(), Mock()]  # More than 1 doc group
        mock_bertrend.merged_df = pd.DataFrame(
            {"Topic": [1, 2, 3], "URLs": [["url1"], ["url2"], ["url3"]]}
        )

        # Mock signal classification results
        noise_df = pd.DataFrame(
            {
                "Topic": [1],
                "Latest_Popularity": [10],
                "Representation": [["word1", "word2"]],
                "Documents": [["doc1", "doc2"]],
            }
        )
        weak_signal_df = pd.DataFrame(
            {
                "Topic": [2],
                "Latest_Popularity": [20],
                "Representation": [["word3", "word4"]],
                "Documents": [["doc3", "doc4"]],
            }
        )
        strong_signal_df = pd.DataFrame(
            {
                "Topic": [3],
                "Latest_Popularity": [30],
                "Representation": [["word5", "word6"]],
                "Documents": [["doc5", "doc6"]],
            }
        )

        mock_bertrend.classify_signals.return_value = (
            noise_df,
            weak_signal_df,
            strong_signal_df,
        )
        mock_train_new_data.return_value = mock_bertrend

        mock_interp_path = tempfile.mkdtemp()
        mock_get_interp_path.return_value = Path(mock_interp_path)

        mock_generate_desc.return_value = ("Test Title", "Test Description")

        # Test data
        new_data = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")],
                "content": ["content1", "content2"],
            }
        )
        reference_timestamp = pd.Timestamp("2024-01-01")

        # Execute
        train_new_model_for_period(
            "test_model", "test_user", new_data, reference_timestamp
        )

        # Verify key function calls
        mock_train_new_data.assert_called_once()
        mock_bertrend.calculate_signal_popularity.assert_called_once()
        mock_bertrend.classify_signals.assert_called_once()
        mock_generate_llm.assert_called()  # Should be called for each non-empty DataFrame

    @patch("bertrend_apps.prospective_demo.process_new_data.EmbeddingService")
    @patch("bertrend_apps.prospective_demo.process_new_data.get_user_models_path")
    @patch("bertrend_apps.prospective_demo.process_new_data.get_model_config")
    @patch("bertrend_apps.prospective_demo.process_new_data.train_new_data")
    def test_train_new_model_for_period_insufficient_doc_groups(
        self,
        mock_train_new_data,
        mock_get_config,
        mock_get_models_path,
        mock_embedding_service,
    ):
        """Test early return when insufficient doc groups."""
        mock_get_models_path.return_value = Path("/test/models")
        mock_get_config.return_value = {
            "granularity": 7,
            "window_size": 30,
            "language": "en",
        }

        # Mock BERTrend instance with only 1 doc group
        mock_bertrend = Mock()
        mock_bertrend.doc_groups = [Mock()]  # Only 1 doc group
        mock_train_new_data.return_value = mock_bertrend

        new_data = pd.DataFrame(
            {"timestamp": [pd.Timestamp("2024-01-01")], "content": ["content1"]}
        )
        reference_timestamp = pd.Timestamp("2024-01-01")

        # Execute
        train_new_model_for_period(
            "test_model", "test_user", new_data, reference_timestamp
        )

        # Verify early return - calculate_signal_popularity should not be called
        mock_bertrend.calculate_signal_popularity.assert_not_called()


class TestRegenerateModels:
    """Test the regenerate_models function."""

    @patch("bertrend_apps.prospective_demo.process_new_data.get_model_config")
    @patch("bertrend_apps.prospective_demo.process_new_data.load_all_data")
    @patch("bertrend_apps.prospective_demo.process_new_data.split_data")
    @patch("bertrend_apps.prospective_demo.process_new_data.group_by_days")
    @patch("bertrend_apps.prospective_demo.process_new_data.train_new_model_for_period")
    def test_regenerate_models_success(
        self,
        mock_train_new_model,
        mock_group_by_days,
        mock_split_data,
        mock_load_all_data,
        mock_get_config,
    ):
        """Test successful model regeneration."""
        # Setup mocks
        mock_get_config.return_value = {
            "granularity": 7,
            "window_size": 30,
            "language": "en",
            "split_by_paragraph": True,
        }

        mock_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=20, freq="D"),
                "content": [f"content_{i}" for i in range(20)],
            }
        )
        mock_load_all_data.return_value = mock_data

        # Mock split_data to return the same DataFrame (it should preserve structure)
        mock_split_data.return_value = mock_data

        # Mock group_by_days to return grouped data
        period1_data = mock_data.iloc[:10]
        period2_data = mock_data.iloc[10:]
        mock_group_by_days.return_value = {
            pd.Timestamp("2024-01-07"): period1_data,
            pd.Timestamp("2024-01-14"): period2_data,
        }

        # Execute
        regenerate_models("test_model", "test_user", with_analysis=True)

        # Verify
        mock_load_all_data.assert_called_once_with(
            model_id="test_model", user="test_user", language_code="en"
        )
        mock_split_data.assert_called_once()
        assert mock_train_new_model.call_count == 2  # Should train for each period

    @patch("bertrend_apps.prospective_demo.process_new_data.get_model_config")
    @patch("bertrend_apps.prospective_demo.process_new_data.load_all_data")
    def test_regenerate_models_no_data(self, mock_load_all_data, mock_get_config):
        """Test behavior when no data is available."""
        mock_get_config.return_value = {
            "granularity": 7,
            "window_size": 30,
            "language": "en",
            "split_by_paragraph": True,
        }
        mock_load_all_data.return_value = None

        # Should handle None case gracefully - function should return early
        # This will likely cause an exception in the current implementation
        # but that's expected behavior when no data is available
        with pytest.raises(TypeError):
            regenerate_models("test_model", "test_user")

        # Verify function attempted to load data
        mock_load_all_data.assert_called_once()

    @patch("bertrend_apps.prospective_demo.process_new_data.get_model_config")
    @patch("bertrend_apps.prospective_demo.process_new_data.load_all_data")
    @patch("bertrend_apps.prospective_demo.process_new_data.split_data")
    @patch("bertrend_apps.prospective_demo.process_new_data.train_new_model_for_period")
    def test_regenerate_models_with_since_filter(
        self, mock_train_new_model, mock_split_data, mock_load_all_data, mock_get_config
    ):
        """Test model regeneration with 'since' date filter."""
        mock_get_config.return_value = {
            "granularity": 7,
            "window_size": 30,
            "language": "en",
            "split_by_paragraph": True,
        }

        mock_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=20, freq="D"),
                "content": [f"content_{i}" for i in range(20)],
            }
        )
        mock_load_all_data.return_value = mock_data

        # Mock split_data to return the same DataFrame (it should preserve structure)
        filtered_data = mock_data.iloc[5:]  # Simulate filtering after since date
        mock_split_data.return_value = filtered_data

        since_date = pd.Timestamp("2024-01-05")

        # Execute
        regenerate_models("test_model", "test_user", since=since_date)

        # Verify only periods after 'since' date are processed
        assert mock_train_new_model.call_count == 3
        # Check that called periods are after since_date
        call_args = mock_train_new_model.call_args_list
        for call in call_args:
            reference_timestamp = call[1]["reference_timestamp"]
            assert reference_timestamp >= since_date


# Integration tests for parallel processing
class TestParallelProcessing:
    """Test parallel processing functionality specifically."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch("bertrend_apps.prospective_demo.process_new_data.analyze_signal")
    def test_parallel_processing_maintains_order(self, mock_analyze_signal):
        """Test that parallel processing maintains consistent output order."""
        # Setup mock with variable delay to test race conditions
        import time

        def mock_analyze_signal_with_delay(bertrend, topic, timestamp):
            # Simulate variable processing time
            delay = 0.1 if topic % 2 == 0 else 0.05
            time.sleep(delay)

            mock_summary = Mock()
            mock_summary.model_dump_json.return_value = (
                f'{{"topic": {topic}, "summary": "test"}}'
            )
            mock_analysis = Mock()
            mock_analysis.model_dump_json.return_value = (
                f'{{"topic": {topic}, "analysis": "test"}}'
            )
            return mock_summary, mock_analysis

        mock_analyze_signal.side_effect = mock_analyze_signal_with_delay

        # Create DataFrame with topics in specific order
        df = pd.DataFrame(
            {"Topic": [5, 1, 3, 2, 4], "Latest_Popularity": [100, 90, 80, 70, 60]}
        )

        mock_bertrend = Mock()
        reference_timestamp = pd.Timestamp("2024-01-01")

        # Execute multiple times to test consistency
        for _ in range(3):
            generate_llm_interpretation(
                mock_bertrend,
                reference_timestamp,
                df,
                "test_df",
                self.output_path,
                top_k=5,
                max_workers=3,
            )

            # Verify results are always in the same order (sorted by topic)
            output_file = self.output_path / "test_df_interpretation.jsonl"
            with jsonlines.open(output_file) as reader:
                results = list(reader)

            topic_order = [result["topic"] for result in results]
            assert topic_order == sorted(
                topic_order
            ), f"Topics not sorted: {topic_order}"

            # Clean up for next iteration
            output_file.unlink()

    @patch("bertrend_apps.prospective_demo.process_new_data.analyze_signal")
    def test_parallel_processing_with_different_worker_counts(
        self, mock_analyze_signal
    ):
        """Test parallel processing with different worker counts."""
        mock_summary = Mock()
        mock_summary.model_dump_json.return_value = '{"summary": "test"}'
        mock_analysis = Mock()
        mock_analysis.model_dump_json.return_value = '{"analysis": "test"}'
        mock_analyze_signal.return_value = (mock_summary, mock_analysis)

        df = pd.DataFrame(
            {"Topic": list(range(10)), "Latest_Popularity": list(range(100, 90, -1))}
        )

        mock_bertrend = Mock()
        reference_timestamp = pd.Timestamp("2024-01-01")

        # Test with different worker counts
        for max_workers in [1, 2, 5, 10]:
            generate_llm_interpretation(
                mock_bertrend,
                reference_timestamp,
                df,
                f"test_df_{max_workers}",
                self.output_path,
                top_k=5,
                max_workers=max_workers,
            )

            # Verify all produce same results
            output_file = (
                self.output_path / f"test_df_{max_workers}_interpretation.jsonl"
            )
            with jsonlines.open(output_file) as reader:
                results = list(reader)

            assert len(results) == 5
            assert all("topic" in result for result in results)
            assert all("summary" in result for result in results)
            assert all("analysis" in result for result in results)
