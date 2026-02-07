#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
import re
from enum import Enum
from typing import Type

from agents import ModelSettings, RunConfig, Runner
from loguru import logger
from openai import OpenAI, Stream, Timeout
from openai.types import Reasoning
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

from bertrend.llm_utils.agent_utils import BaseAgentFactory, run_config_no_tracing

# Note: .env is loaded in bertrend/__init__.py which is imported before this module

MAX_ATTEMPTS = 3
TIMEOUT = 60.0
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MODEL = "gpt-4.1-mini"


class APIType(Enum):
    """Allow choosing between completions and responses API from OpenAI"""

    # NB. Preferred: RESPONSES
    COMPLETIONS = "completions"
    RESPONSES = "responses"


class OpenAI_Client:
    """
    Generic client for OpenAI API (either direct API or via Azure).

    This class provides a unified interface for interacting with OpenAI models,
    supporting both direct API access and Azure-hosted deployments. It handles
    authentication, request formatting, and error handling.

    Notes
    -----
    The API key and the BASE_URL must be set using environment variables OPENAI_API_KEY and
    OPENAI_BASE_URL respectively. The base_url should only be set for Azure or local deployments (such as LiteLLM)..
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        temperature: float = DEFAULT_TEMPERATURE,
        api_type: APIType = APIType.RESPONSES,
    ):
        """
        Initialize the OpenAI client.

        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. If None, will try to get from OPENAI_API_KEY environment variable.
        base_url : str, optional
            API base_url URL (LiteLLM and openAI compatible deployments). If None, will try to get from OPENAI_BASE_URL environment variable.
        model : str, optional
            Name of the model to use. If None, will try to get from OPENAI_DEFAULT_MODEL environment variable.
        temperature : float, default=DEFAULT_TEMPERATURE
            Temperature parameter for controlling randomness in generation.

        Raises
        ------
        EnvironmentError
            If api_key is None and OPENAI_API_KEY environment variable is not set.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error(
                "WARNING: OPENAI_API_KEY environment variable not found. Please set it before using OpenAI services."
            )
            raise EnvironmentError("OPENAI_API_KEY environment variable not found.")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL")) or None

        openai_params = {
            "base_url": base_url,
            "api_key": api_key,
            "timeout": Timeout(TIMEOUT, connect=10.0),
            "max_retries": MAX_ATTEMPTS,
        }
        self.llm_client = OpenAI(**openai_params)
        self.model_name = model or os.getenv("OPENAI_DEFAULT_MODEL") or DEFAULT_MODEL
        self.temperature = temperature if not test_gpt5_version(self.model_name) else 1
        self.api_type = api_type

    def generate(
        self,
        user_prompt,
        system_prompt=None,
        **kwargs,
    ) -> ChatCompletion | Stream[ChatCompletionChunk] | str:
        """
        Call OpenAI model for text generation.

        Parameters
        ----------
        user_prompt : str
            Prompt to send to the model with role=user.
        system_prompt : str, optional
            Prompt to send to the model with role=system.
        **kwargs : dict
            Additional arguments to pass to the OpenAI API.

        Returns
        -------
        str or Stream[ChatCompletionChunk]
            Model response as text, or a stream of response chunks if stream=True is passed in kwargs.
        """
        # Transform messages into OpenAI API compatible format
        messages = [{"role": "user", "content": user_prompt}]
        # Add a system prompt if one is provided
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        return self.generate_from_history(messages, **kwargs)

    def generate_from_history(
        self,
        messages: list[dict],
        **kwargs,
    ) -> ChatCompletion | Stream[ChatCompletionChunk] | str:
        """
        Call OpenAI model for text generation using a conversation history.

        Parameters
        ----------
        messages : list[dict]
            List of message dictionaries to pass to the API in OpenAI format.
            Each message should have 'role' and 'content' keys.
        **kwargs : dict
            Additional arguments to pass to the OpenAI API.

        Returns
        -------
        str or Stream[ChatCompletionChunk]
            Model response as text, or a stream of response chunks if stream=True is passed in kwargs.
        """
        # For important parameters, set a default value if not given
        if not kwargs.get("model"):
            kwargs["model"] = self.model_name

        kwargs["temperature"] = kwargs.get("temperature", self.temperature)
        if test_gpt5_version(kwargs["model"]):
            kwargs["temperature"] = 1

        if self.api_type == APIType.COMPLETIONS:
            try:
                answer = self.llm_client.chat.completions.create(
                    messages=messages,
                    **kwargs,
                )
                logger.debug(f"API returned: {answer}")
                if kwargs.get("stream", False):
                    return answer
                else:
                    return answer.choices[0].message.content
                # Details of errors available here: https://platform.openai.com/docs/guides/error-codes/api-errors
            except Exception as e:
                msg = f"OpenAI API fatal error: {e}"
                logger.error(msg)
                return msg

        elif self.api_type == APIType.RESPONSES:
            try:
                response = self.llm_client.responses.create(input=messages, **kwargs)
                logger.debug(f"API returned: {response}")
                if kwargs.get("stream", False):
                    return response
                else:
                    return response.output_text
                # Details of errors available here: https://platform.openai.com/docs/guides/error-codes/api-errors
            except Exception as e:
                msg = f"OpenAI API fatal error: {e}"
                logger.error(msg)
                return msg
        return ""

    def parse(
        self,
        user_prompt: str,
        system_prompt: str = None,
        response_format: Type[BaseModel] = None,
        **kwargs,
    ) -> BaseModel | None:
        """Call OpenAI model for generation with structured output"""
        # Due to recurrent problems with the parse function of openai with Litellm, use of the agents sdk for that
        if not kwargs.get("model"):
            kwargs["model"] = self.model_name
        model_name = kwargs["model"]
        model_settings = (
            ModelSettings(
                reasoning=Reasoning(effort="low"),
                verbosity="low",
            )
            if test_gpt5_version(model_name)
            else None
        )

        parsing_agent = BaseAgentFactory().create_agent(
            name="parsing_agent",
            model_name=model_name,
            instructions=system_prompt,
            output_type=response_format,
            model_settings=model_settings,
        )

        # invoke agent
        result = Runner.run_sync(
            input=user_prompt,
            starting_agent=parsing_agent,
            run_config=run_config_no_tracing,
        )
        response = (
            result.final_output if hasattr(result, "final_output") else str(result)
        )
        return response


def test_gpt5_version(version_string):
    # Regular expression to match "gpt-" followed by a number (integer or float)
    pattern = r"^gpt-(\d+(\.\d+)?).*$"  # Matches numbers like 4, 4.1, 5, 10.0
    match = re.match(pattern, version_string)

    if match:
        # Extract the version number as a float
        version_number = float(match.group(1))
        if version_number >= 5:
            return True
    return False
