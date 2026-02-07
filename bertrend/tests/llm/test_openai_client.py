#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from openai import OpenAI, Stream
from pydantic import BaseModel

from bertrend.llm_utils.agent_utils import run_config_no_tracing
from bertrend.llm_utils.openai_client import APIType, OpenAI_Client


@pytest.fixture
def mock_api_key():
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    yield
    del os.environ["OPENAI_API_KEY"]


def test_initialization_with_base_url(mock_api_key):
    """Test client initialization when using a custom base URL"""
    with patch("bertrend.llm_utils.openai_client.OpenAI") as mock_openai:
        client = OpenAI_Client(
            api_key="test_api_key", base_url="https://custom-base.example.com"
        )
        assert client.llm_client == mock_openai.return_value
        _, kwargs = mock_openai.call_args
        assert kwargs["api_key"] == "test_api_key"
        assert kwargs["base_url"] == "https://custom-base.example.com"


def test_initialization_without_base_url(mock_api_key):
    """Test client initialization when using default OpenAI configuration"""
    with patch("bertrend.llm_utils.openai_client.OpenAI") as mock_openai:
        client = OpenAI_Client(api_key="test_api_key")
        assert client.llm_client == mock_openai.return_value
        _, kwargs = mock_openai.call_args
        assert kwargs["api_key"] == "test_api_key"
        assert kwargs["base_url"] is None


def test_generate_user_prompt(mock_api_key):
    """Test generation with a simple user prompt"""
    client = OpenAI_Client(api_key="test_api_key", api_type=APIType.RESPONSES)

    # Mocking the llm_client's responses create method
    with patch.object(
        client.llm_client.responses,
        "create",
        return_value=MagicMock(output_text="This is a mock response"),
    ):
        result = client.generate("What is the weather today?")
        assert result == "This is a mock response"


def test_generate_from_history(mock_api_key):
    """Test generation using history of messages"""
    client = OpenAI_Client(api_key="test_api_key", api_type=APIType.RESPONSES)

    # Mocking the llm_client's responses create method
    with patch.object(
        client.llm_client.responses,
        "create",
        return_value=MagicMock(output_text="This is a mock response"),
    ):
        messages = [{"role": "user", "content": "What is the weather today?"}]
        result = client.generate_from_history(messages)
        assert result == "This is a mock response"


def test_api_error_handling(mock_api_key):
    """Test if an API error is properly handled"""
    client = OpenAI_Client(api_key="test_api_key", api_type=APIType.RESPONSES)

    # Simulate an error during API call
    with patch.object(
        client.llm_client.responses, "create", side_effect=Exception("API Error")
    ):
        result = client.generate("What is the weather today?")
        assert result == "OpenAI API fatal error: API Error"


def test_generate_with_streaming(mock_api_key):
    """Test if streaming works when 'stream' is True"""
    client = OpenAI_Client(api_key="test_api_key", api_type=APIType.RESPONSES)

    # Mock streaming response
    mock_stream = MagicMock(spec=Stream)
    with patch.object(client.llm_client.responses, "create", return_value=mock_stream):
        result = client.generate("What is the weather today?", stream=True)
        assert result == mock_stream


# Define a test Pydantic model for parse tests
# Using underscore prefix to prevent pytest from collecting it as a test class
class _TestResponseModel(BaseModel):
    answer: str
    confidence: float


def test_parse_basic_functionality(mock_api_key):
    """Test parse method with a simple user prompt"""
    client = OpenAI_Client(api_key="test_api_key")

    mock_agent = Mock()
    mock_parsed = _TestResponseModel(answer="This is a test answer", confidence=0.95)
    mock_result = Mock(final_output=mock_parsed)

    with (
        patch(
            "bertrend.llm_utils.openai_client.BaseAgentFactory.create_agent",
            return_value=mock_agent,
        ) as mock_create_agent,
        patch(
            "bertrend.llm_utils.openai_client.Runner.run_sync",
            return_value=mock_result,
        ) as mock_run_sync,
    ):
        result = client.parse(
            "What is the weather today?", response_format=_TestResponseModel
        )
        assert result == mock_parsed
        mock_create_agent.assert_called_once_with(
            name="parsing_agent",
            model_name=client.model_name,
            instructions=None,
            output_type=_TestResponseModel,
            model_settings=None,
        )
        mock_run_sync.assert_called_once_with(
            input="What is the weather today?",
            starting_agent=mock_agent,
            run_config=run_config_no_tracing,
        )


def test_parse_with_system_prompt(mock_api_key):
    """Test parse method with both user and system prompts"""
    client = OpenAI_Client(api_key="test_api_key")

    mock_agent = Mock()
    mock_parsed = _TestResponseModel(answer="System prompt response", confidence=0.9)
    mock_result = Mock(final_output=mock_parsed)

    with (
        patch(
            "bertrend.llm_utils.openai_client.BaseAgentFactory.create_agent",
            return_value=mock_agent,
        ) as mock_create_agent,
        patch(
            "bertrend.llm_utils.openai_client.Runner.run_sync",
            return_value=mock_result,
        ) as mock_run_sync,
    ):
        result = client.parse(
            "What is the weather today?",
            system_prompt="You are a weather assistant",
            response_format=_TestResponseModel,
        )
        assert result == mock_parsed
        mock_create_agent.assert_called_once_with(
            name="parsing_agent",
            model_name=client.model_name,
            instructions="You are a weather assistant",
            output_type=_TestResponseModel,
            model_settings=None,
        )
        mock_run_sync.assert_called_once_with(
            input="What is the weather today?",
            starting_agent=mock_agent,
            run_config=run_config_no_tracing,
        )


def test_parse_error_handling(mock_api_key):
    """Test if parse propagates errors from the runner"""
    client = OpenAI_Client(api_key="test_api_key")

    mock_agent = Mock()
    with (
        patch(
            "bertrend.llm_utils.openai_client.BaseAgentFactory.create_agent",
            return_value=mock_agent,
        ),
        patch(
            "bertrend.llm_utils.openai_client.Runner.run_sync",
            side_effect=Exception("API Parse Error"),
        ),
        pytest.raises(Exception, match="API Parse Error"),
    ):
        client.parse("What is the weather today?", response_format=_TestResponseModel)


def test_parse_with_none_response_format(mock_api_key):
    """Test parse method with response_format=None"""
    client = OpenAI_Client(api_key="test_api_key")

    mock_agent = Mock()
    mock_parsed = {"answer": "Default response", "confidence": 0.8}
    mock_result = Mock(final_output=mock_parsed)

    with (
        patch(
            "bertrend.llm_utils.openai_client.BaseAgentFactory.create_agent",
            return_value=mock_agent,
        ) as mock_create_agent,
        patch(
            "bertrend.llm_utils.openai_client.Runner.run_sync",
            return_value=mock_result,
        ) as mock_run_sync,
    ):
        result = client.parse("What is the weather today?", response_format=None)

        mock_create_agent.assert_called_once_with(
            name="parsing_agent",
            model_name=client.model_name,
            instructions=None,
            output_type=None,
            model_settings=None,
        )
        mock_run_sync.assert_called_once_with(
            input="What is the weather today?",
            starting_agent=mock_agent,
            run_config=run_config_no_tracing,
        )
        assert result == mock_parsed
