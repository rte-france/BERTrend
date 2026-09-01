#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from openai import Stream
from pydantic import BaseModel

from bertrend.llm_utils.openai_client import APIType, OpenAI_Client, PARSE_TIMEOUT


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
    """Test parse uses the synchronous responses.parse and returns output_parsed."""
    # Use a non-GPT-5 model so this test covers the branch without reasoning.
    client = OpenAI_Client(api_key="test_api_key", model="gpt-4.1-mini")

    mock_parsed = _TestResponseModel(answer="This is a test answer", confidence=0.95)
    mock_response = Mock(output_parsed=mock_parsed)

    with patch.object(
        client.llm_client.responses, "parse", return_value=mock_response
    ) as mock_parse:
        result = client.parse(
            "What is the weather today?", response_format=_TestResponseModel
        )
        assert result == mock_parsed
        mock_parse.assert_called_once()
        _, kwargs = mock_parse.call_args
        assert kwargs["input"] == [
            {"role": "user", "content": "What is the weather today?"}
        ]
        assert kwargs["text_format"] == _TestResponseModel
        # A bounded per-request timeout is always set.
        assert kwargs["timeout"] == PARSE_TIMEOUT
        # A non-GPT-5 model must not receive a reasoning parameter.
        assert "reasoning" not in kwargs


def test_parse_with_system_prompt(mock_api_key):
    """Test parse builds a system+user message list from both prompts."""
    # Use a non-GPT-5 model so this test covers the branch without reasoning.
    client = OpenAI_Client(api_key="test_api_key", model="gpt-4.1-mini")

    mock_parsed = _TestResponseModel(answer="System prompt response", confidence=0.9)
    mock_response = Mock(output_parsed=mock_parsed)

    with patch.object(
        client.llm_client.responses, "parse", return_value=mock_response
    ) as mock_parse:
        result = client.parse(
            "What is the weather today?",
            system_prompt="You are a weather assistant",
            response_format=_TestResponseModel,
        )
        assert result == mock_parsed
        _, kwargs = mock_parse.call_args
        assert kwargs["input"] == [
            {"role": "system", "content": "You are a weather assistant"},
            {"role": "user", "content": "What is the weather today?"},
        ]
        assert kwargs["text_format"] == _TestResponseModel


def test_parse_error_handling(mock_api_key):
    """Test if parse propagates errors from the underlying client."""
    client = OpenAI_Client(api_key="test_api_key")

    with (
        patch.object(
            client.llm_client.responses,
            "parse",
            side_effect=Exception("API Parse Error"),
        ),
        pytest.raises(Exception, match="API Parse Error"),
    ):
        client.parse("What is the weather today?", response_format=_TestResponseModel)


def test_parse_with_none_response_format(mock_api_key):
    """With response_format=None, parse falls back to a plain text generation."""
    # Use a non-GPT-5 model so this test covers the branch without reasoning.
    client = OpenAI_Client(api_key="test_api_key", model="gpt-4.1-mini")

    with patch.object(
        client.llm_client.responses,
        "create",
        return_value=MagicMock(output_text="Default response"),
    ) as mock_create:
        result = client.parse("What is the weather today?", response_format=None)

        assert result == "Default response"
        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        assert kwargs["input"] == [
            {"role": "user", "content": "What is the weather today?"}
        ]


def test_parse_completions_api(mock_api_key):
    """With the COMPLETIONS API, parse uses chat.completions.parse."""
    client = OpenAI_Client(
        api_key="test_api_key", model="gpt-4.1-mini", api_type=APIType.COMPLETIONS
    )

    mock_parsed = _TestResponseModel(answer="Completions answer", confidence=0.7)
    mock_completion = Mock(
        choices=[Mock(message=Mock(parsed=mock_parsed))],
    )

    with patch.object(
        client.llm_client.chat.completions, "parse", return_value=mock_completion
    ) as mock_parse:
        result = client.parse(
            "What is the weather today?", response_format=_TestResponseModel
        )
        assert result == mock_parsed
        _, kwargs = mock_parse.call_args
        assert kwargs["response_format"] == _TestResponseModel
        assert kwargs["messages"] == [
            {"role": "user", "content": "What is the weather today?"}
        ]


def test_parse_includes_reasoning_for_gpt5(mock_api_key):
    """Test parse passes a reasoning effort when using a GPT-5 model."""
    client = OpenAI_Client(api_key="test_api_key", model="gpt-5")

    mock_response = Mock(
        output_parsed=_TestResponseModel(answer="Ok", confidence=0.5)
    )

    with patch.object(
        client.llm_client.responses, "parse", return_value=mock_response
    ) as mock_parse:
        client.parse("What is the weather today?", response_format=_TestResponseModel)

        _, kwargs = mock_parse.call_args
        assert kwargs["reasoning"] == {"effort": "low"}
        # Reasoning models require temperature == 1.
        assert kwargs["temperature"] == 1


def test_parse_custom_reasoning_effort_for_gpt5(mock_api_key):
    """A per-task reasoning_effort overrides the default for GPT-5 models."""
    client = OpenAI_Client(api_key="test_api_key", model="gpt-5")

    mock_response = Mock(
        output_parsed=_TestResponseModel(answer="Ok", confidence=0.5)
    )

    with patch.object(
        client.llm_client.responses, "parse", return_value=mock_response
    ) as mock_parse:
        client.parse(
            "What is the weather today?",
            response_format=_TestResponseModel,
            reasoning_effort="high",
        )

        _, kwargs = mock_parse.call_args
        assert kwargs["reasoning"] == {"effort": "high"}


def test_parse_invalid_reasoning_effort_falls_back(mock_api_key):
    """An invalid reasoning_effort falls back to the default effort."""
    from bertrend.llm_utils.openai_client import DEFAULT_REASONING_EFFORT

    client = OpenAI_Client(api_key="test_api_key", model="gpt-5")

    mock_response = Mock(
        output_parsed=_TestResponseModel(answer="Ok", confidence=0.5)
    )

    with patch.object(
        client.llm_client.responses, "parse", return_value=mock_response
    ) as mock_parse:
        client.parse(
            "What is the weather today?",
            response_format=_TestResponseModel,
            reasoning_effort="ultra",  # not a valid effort
        )

        _, kwargs = mock_parse.call_args
        assert kwargs["reasoning"] == {"effort": DEFAULT_REASONING_EFFORT}


def test_close_closes_underlying_client(mock_api_key):
    """close() releases the underlying HTTP client's pooled connections."""
    client = OpenAI_Client(api_key="test_api_key")

    with patch.object(client.llm_client, "close") as mock_close:
        client.close()
        mock_close.assert_called_once()


def test_context_manager_closes_client(mock_api_key):
    """Using the client as a context manager closes it on exit."""
    with patch.object(OpenAI_Client, "close") as mock_close:
        with OpenAI_Client(api_key="test_api_key") as client:
            assert isinstance(client, OpenAI_Client)
        mock_close.assert_called_once()


def test_resolve_reasoning_effort_override_wins(monkeypatch):
    """An explicit override takes precedence over env vars."""
    from bertrend.llm_utils.openai_client import resolve_reasoning_effort

    monkeypatch.setenv("OPENAI_REASONING_EFFORT_SIGNAL_ANALYSIS", "high")
    assert (
        resolve_reasoning_effort(task="signal_analysis", override="medium") == "medium"
    )


def test_resolve_reasoning_effort_per_task_env(monkeypatch):
    """A per-task env var is used when no override is given."""
    from bertrend.llm_utils.openai_client import resolve_reasoning_effort

    monkeypatch.setenv("OPENAI_REASONING_EFFORT_SIGNAL_ANALYSIS", "high")
    assert resolve_reasoning_effort(task="signal_analysis") == "high"


def test_resolve_reasoning_effort_falls_back_to_default(monkeypatch):
    """Without override or task env var, the global default is returned."""
    from bertrend.llm_utils.openai_client import (
        resolve_reasoning_effort,
        DEFAULT_REASONING_EFFORT,
    )

    monkeypatch.delenv("OPENAI_REASONING_EFFORT_SIGNAL_ANALYSIS", raising=False)
    assert resolve_reasoning_effort(task="signal_analysis") == DEFAULT_REASONING_EFFORT
