#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
import asyncio
import os
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import dataclass

from bertrend.llm_utils.agent_utils import (
    BaseAgentFactory,
    ProcessingResult,
    AsyncAgentConcurrentProcessor,
    progress_reporter,
)


class TestProcessingResult:
    """Test cases for ProcessingResult dataclass."""

    def test_processing_result_creation(self):
        """Test creating ProcessingResult with default values."""
        result = ProcessingResult(input_index=1)
        assert result.input_index == 1
        assert result.output is None
        assert result.error is None
        assert result.processing_time == 0.0
        assert result.timeout is False

    def test_processing_result_with_output(self):
        """Test creating ProcessingResult with output."""
        result = ProcessingResult(
            input_index=0,
            output="test output",
            processing_time=1.5
        )
        assert result.input_index == 0
        assert result.output == "test output"
        assert result.error is None
        assert result.processing_time == 1.5
        assert result.timeout is False

    def test_processing_result_with_error(self):
        """Test creating ProcessingResult with error."""
        result = ProcessingResult(
            input_index=2,
            error="test error",
            timeout=True
        )
        assert result.input_index == 2
        assert result.output is None
        assert result.error == "test error"
        assert result.processing_time == 0.0
        assert result.timeout is True


class TestBaseAgentFactory:
    """Test cases for BaseAgentFactory class."""

    def test_base_agent_factory_creation_default(self):
        """Test creating BaseAgentFactory with default parameters."""
        # Use explicit parameters to avoid environment dependency
        factory = BaseAgentFactory(api_key="test-api-key", base_url=None)
        assert factory.model_name == "gpt-4.1-mini"
        assert factory.api_key == "test-api-key"
        assert factory.base_url is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}, clear=True)
    def test_base_agent_factory_creation_custom(self):
        """Test creating BaseAgentFactory with custom parameters."""
        factory = BaseAgentFactory(
            model_name="gpt-4",
            api_key="custom-key",
            base_url="https://api.openai.com"
        )
        assert factory.model_name == "gpt-4"
        assert factory.api_key == "custom-key"
        assert factory.base_url == "https://api.openai.com"

    def test_base_agent_factory_no_api_key(self):
        """Test that BaseAgentFactory raises error when no API key is provided."""
        with pytest.raises(EnvironmentError) as exc_info:
            BaseAgentFactory(api_key=None)
        assert "OPENAI_API_KEY environment variable not found" in str(exc_info.value)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_DEFAULT_MODEL_NAME": "custom-model"}, clear=True)
    def test_base_agent_factory_env_model_name(self):
        """Test that BaseAgentFactory uses environment variable for model name."""
        factory = BaseAgentFactory(model_name=None)
        assert factory.model_name == "custom-model"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    @patch("bertrend.llm_utils.agent_utils.OpenAIChatCompletionsModel")
    @patch("bertrend.llm_utils.agent_utils.AsyncAzureOpenAI")
    def test_base_agent_factory_azure_model(self, mock_azure_client, mock_openai_model):
        """Test creating BaseAgentFactory with Azure URL."""
        factory = BaseAgentFactory(base_url="https://test.openai.azure.com")
        assert "azure.com" in factory.base_url
        mock_azure_client.assert_called_once()
        mock_openai_model.assert_called_once()

    @patch("bertrend.llm_utils.agent_utils.LitellmModel")
    def test_base_agent_factory_litellm_model(self, mock_litellm_model):
        """Test creating BaseAgentFactory with custom base URL."""
        factory = BaseAgentFactory(api_key="test-key", base_url="https://custom-api.com")
        mock_litellm_model.assert_called_once_with(
            model="gpt-4.1-mini",
            api_key="test-key",
            base_url="https://custom-api.com"
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("bertrend.llm_utils.agent_utils.Agent")
    def test_create_agent(self, mock_agent):
        """Test creating an agent."""
        factory = BaseAgentFactory()
        agent = factory.create_agent(system="test system prompt")
        mock_agent.assert_called_once_with(model=factory.model, system="test system prompt")

    def test_base_agent_factory_empty_base_url(self):
        """Test that empty base URL environment variable is handled correctly."""
        factory = BaseAgentFactory(api_key="test-key", base_url="")
        assert factory.base_url is None


class TestAsyncAgentConcurrentProcessor:
    """Test cases for AsyncAgentConcurrentProcessor class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        return Mock()

    @pytest.fixture
    def processor(self, mock_agent):
        """Create a processor instance for testing."""
        return AsyncAgentConcurrentProcessor(
            agent=mock_agent,
            max_concurrent=2,
            timeout=5.0
        )

    def test_processor_initialization(self, mock_agent):
        """Test AsyncAgentConcurrentProcessor initialization."""
        processor = AsyncAgentConcurrentProcessor(
            agent=mock_agent,
            max_concurrent=3,
            timeout=10.0
        )
        assert processor.agent == mock_agent
        assert processor.max_concurrent == 3
        assert processor.timeout == 10.0
        assert processor.semaphore._value == 3

    def test_processor_default_values(self, mock_agent):
        """Test AsyncAgentConcurrentProcessor with default values."""
        processor = AsyncAgentConcurrentProcessor(agent=mock_agent)
        assert processor.max_concurrent == 5
        assert processor.timeout == 300.0

    @pytest.mark.asyncio
    @patch("bertrend.llm_utils.agent_utils.Runner.run")
    async def test_process_single_item_success(self, mock_runner, processor):
        """Test successful processing of a single item."""
        # Mock the response
        mock_response = Mock()
        mock_response.final_output = "test output"
        mock_runner.return_value = mock_response

        result = await processor.process_single_item("test input", 0)

        assert result.input_index == 0
        assert result.output == "test output"
        assert result.error is None
        assert result.processing_time > 0
        assert result.timeout is False

    @pytest.mark.asyncio
    @patch("bertrend.llm_utils.agent_utils.Runner.run")
    async def test_process_single_item_response_without_final_output(self, mock_runner, processor):
        """Test processing when response doesn't have final_output attribute."""
        # Mock the response without final_output
        mock_response = "direct response"
        mock_runner.return_value = mock_response

        result = await processor.process_single_item("test input", 1)

        assert result.input_index == 1
        assert result.output == "direct response"
        assert result.error is None

    @pytest.mark.asyncio
    @patch("bertrend.llm_utils.agent_utils.Runner.run")
    async def test_process_single_item_agent_error(self, mock_runner, processor):
        """Test processing when agent raises an error."""
        mock_runner.side_effect = Exception("Agent error")

        result = await processor.process_single_item("test input", 0)

        assert result.input_index == 0
        assert result.output is None
        assert "Agent execution error: Agent error" in result.error
        assert result.timeout is False

    @pytest.mark.asyncio
    async def test_process_single_item_timeout(self, processor):
        """Test processing with timeout."""
        processor.timeout = 0.1  # Very short timeout

        with patch("bertrend.llm_utils.agent_utils.Runner.run") as mock_runner:
            # Make the runner sleep longer than timeout
            async def slow_runner(*args, **kwargs):
                await asyncio.sleep(0.2)
                return Mock(final_output="should not reach here")

            mock_runner.side_effect = slow_runner

            result = await processor.process_single_item("test input", 0)

            assert result.input_index == 0
            assert result.output is None
            assert "timed out after" in result.error
            assert result.timeout is True

    @pytest.mark.asyncio
    async def test_process_list_concurrent_empty_list(self, processor):
        """Test processing an empty list."""
        results = await processor.process_list_concurrent([])
        assert results == []

    @pytest.mark.asyncio
    @patch("bertrend.llm_utils.agent_utils.Runner.run")
    async def test_process_list_concurrent_success(self, mock_runner, processor):
        """Test successful concurrent processing of multiple items."""
        # Mock successful responses
        mock_responses = [Mock(final_output=f"output {i}") for i in range(3)]
        mock_runner.side_effect = mock_responses

        input_list = ["item1", "item2", "item3"]
        results = await processor.process_list_concurrent(input_list)

        assert len(results) == 3
        # Results should be sorted by input_index
        for i, result in enumerate(results):
            assert result.input_index == i
            assert result.output == f"output {i}"
            assert result.error is None

    @pytest.mark.asyncio
    async def test_process_list_concurrent_with_progress_callback(self, processor):
        """Test processing with progress callback."""
        callback_calls = []

        def progress_callback(current, total, result):
            callback_calls.append((current, total, result))

        with patch("bertrend.llm_utils.agent_utils.Runner.run") as mock_runner:
            mock_runner.return_value = Mock(final_output="test output")

            input_list = ["item1", "item2"]
            await processor.process_list_concurrent(
                input_list, 
                progress_callback=progress_callback
            )

            assert len(callback_calls) == 2
            assert callback_calls[0][1] == 2  # total items
            assert callback_calls[1][1] == 2  # total items

    @pytest.mark.asyncio
    async def test_process_list_concurrent_with_chunking(self, processor):
        """Test processing with chunking."""
        with patch.object(processor, '_process_in_chunks') as mock_chunks:
            mock_chunks.return_value = [ProcessingResult(input_index=i) for i in range(5)]

            input_list = ["item"] * 5
            results = await processor.process_list_concurrent(
                input_list, 
                chunk_size=2
            )

            mock_chunks.assert_called_once_with(input_list, 2, None, None)
            assert len(results) == 5

    @pytest.mark.asyncio
    async def test_process_list_concurrent_overall_timeout(self, processor):
        """Test processing with overall timeout."""
        processor.timeout = 10.0  # Individual timeout

        with patch("bertrend.llm_utils.agent_utils.Runner.run") as mock_runner:
            # Make some tasks take longer
            async def slow_runner(*args, **kwargs):
                await asyncio.sleep(0.1)
                return Mock(final_output="output")

            mock_runner.side_effect = slow_runner

            input_list = ["item1", "item2", "item3"]
            results = await processor.process_list_concurrent(
                input_list,
                overall_timeout=0.05  # Very short overall timeout
            )

            # Should get results with timeout errors
            assert len(results) == 3
            # At least some should have timeout errors
            timeout_errors = [r for r in results if "timeout" in (r.error or "").lower()]
            assert len(timeout_errors) > 0

    @pytest.mark.asyncio
    @patch("bertrend.llm_utils.agent_utils.Runner.run")
    async def test_collect_results_with_progress_success(self, mock_runner, processor):
        """Test _collect_results_with_progress method."""
        mock_runner.return_value = Mock(final_output="test output")

        # Create mock tasks with names
        tasks = []
        for i in range(2):
            task = asyncio.create_task(
                processor.process_single_item(f"item{i}", i),
                name=f"item_{i}"
            )
            tasks.append(task)

        progress_calls = []

        def progress_callback(current, total, result):
            progress_calls.append((current, total))

        results = await processor._collect_results_with_progress(
            tasks, 2, progress_callback
        )

        assert len(results) == 2
        assert len(progress_calls) == 2

    @pytest.mark.asyncio
    async def test_process_in_chunks_success(self, processor):
        """Test _process_in_chunks method."""
        with patch.object(processor, 'process_list_concurrent') as mock_process:
            mock_process.return_value = [
                ProcessingResult(input_index=0, output="output0"),
                ProcessingResult(input_index=1, output="output1")
            ]

            input_list = ["item0", "item1", "item2", "item3"]
            results = await processor._process_in_chunks(
                input_list,
                chunk_size=2
            )

            # Should be called twice (2 chunks of size 2)
            assert mock_process.call_count == 2
            assert len(results) == 4  # 2 results per chunk * 2 chunks

    @pytest.mark.asyncio
    async def test_process_in_chunks_with_overall_timeout(self, processor):
        """Test _process_in_chunks with overall timeout."""
        with patch.object(processor, 'process_list_concurrent') as mock_process:
            # Make the first chunk take some time
            async def slow_process(*args, **kwargs):
                await asyncio.sleep(0.1)
                return [ProcessingResult(input_index=0, output="output")]

            mock_process.side_effect = slow_process

            input_list = ["item0", "item1", "item2", "item3"]
            results = await processor._process_in_chunks(
                input_list,
                chunk_size=2,
                overall_timeout=0.05  # Very short timeout
            )

            # Should get timeout results for remaining items
            timeout_results = [r for r in results if r.timeout]
            assert len(timeout_results) > 0

    @pytest.mark.asyncio
    async def test_process_in_chunks_with_error(self, processor):
        """Test _process_in_chunks when chunk processing fails."""
        with patch.object(processor, 'process_list_concurrent') as mock_process:
            mock_process.side_effect = Exception("Chunk processing error")

            input_list = ["item0", "item1"]
            results = await processor._process_in_chunks(
                input_list,
                chunk_size=1
            )

            # Should get error results
            assert len(results) == 2
            for result in results:
                assert "Chunk processing failed" in result.error


class TestProgressReporter:
    """Test cases for progress_reporter function."""

    @patch("bertrend.llm_utils.agent_utils.logger")
    def test_progress_reporter_success(self, mock_logger):
        """Test progress_reporter with successful result."""
        result = ProcessingResult(
            input_index=0,
            output="test output"
        )
        
        progress_reporter(5, 10, result)
        
        # Check that logger.info was called with success status
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "5/10" in call_args
        assert "✓" in call_args

    @patch("bertrend.llm_utils.agent_utils.logger")
    def test_progress_reporter_timeout(self, mock_logger):
        """Test progress_reporter with timeout result."""
        result = ProcessingResult(
            input_index=0,
            timeout=True
        )
        
        progress_reporter(3, 10, result)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "3/10" in call_args
        assert "⏱" in call_args

    @patch("bertrend.llm_utils.agent_utils.logger")
    def test_progress_reporter_error(self, mock_logger):
        """Test progress_reporter with error result."""
        result = ProcessingResult(
            input_index=0,
            error="test error message that is quite long and should be truncated"
        )
        
        progress_reporter(1, 5, result)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "1/5" in call_args
        assert "✗" in call_args
        
        # Should also log the error details
        mock_logger.debug.assert_called_once()
        debug_call_args = mock_logger.debug.call_args[0][0]
        assert "Last error:" in debug_call_args

    @patch("bertrend.llm_utils.agent_utils.logger")
    def test_progress_reporter_percentage_calculation(self, mock_logger):
        """Test that progress_reporter calculates percentage correctly."""
        result = ProcessingResult(input_index=0, output="test")
        
        progress_reporter(7, 10, result)
        
        call_args = mock_logger.info.call_args[0][0]
        assert "7.0%" in call_args  # (7/10) * 10 = 7.0%