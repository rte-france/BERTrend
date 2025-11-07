#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
import re
from enum import Enum
from typing import Type

from openai import OpenAI, AzureOpenAI, Timeout, Stream
from loguru import logger
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

# Note: .env is loaded in bertrend/__init__.py which is imported before this module

MAX_ATTEMPTS = 3
TIMEOUT = 60.0
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_MODEL = "gpt-4.1-mini"
AZURE_API_VERSION = "2025-03-01-preview"


class APIType(Enum):
    """Allow choosing between completions and responses API from OpenAI"""

    # NB. LiteLLM does not support responses API for <=gpt4* models
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
        api_version: str = AZURE_API_VERSION,
        api_type: APIType = APIType.COMPLETIONS,
    ):
        """
        Initialize the OpenAI client.

        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. If None, will try to get from OPENAI_API_KEY environment variable.
        base_url : str, optional
            API endpoint (Azure) or base_url URL (LiteLLM and openAI compatible deployments). If None, will try to get from OPENAI_BASE_URL environment variable.
            Should be set for Azure or local deployments.
        model : str, optional
            Name of the model to use. If None, will try to get from OPENAI_DEFAULT_MODEL environment variable.
        temperature : float, default=DEFAULT_TEMPERATURE
            Temperature parameter for controlling randomness in generation.
        api_version : str, default=AZURE_API_VERSION
            API version to use for Azure OpenAI service.

        Raises
        ------
        EnvironmentError
            If api_key is None and OPENAI_API_KEY environment variable is not set.
        """
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error(
                "WARNING: OPENAI_API_KEY environment variable not found. Please set it before using OpenAI services."
            )
            raise EnvironmentError(f"OPENAI_API_KEY environment variable not found.")

        base_url = base_url or os.getenv("OPENAI_BASE_URL", None)
        if base_url == "":  # check empty env var
            base_url = None

        run_on_azure = "azure.com" in base_url if base_url else False

        common_params = {
            "api_key": api_key,
            "timeout": Timeout(TIMEOUT, connect=10.0),
            "max_retries": MAX_ATTEMPTS,
        }
        openai_params = {
            "base_url": base_url,
        }
        azure_params = {
            "azure_endpoint": base_url,
            "api_version": api_version or AZURE_API_VERSION,
        }

        if not run_on_azure:
            self.llm_client = OpenAI(
                **common_params,
                **openai_params,
            )
        else:
            self.llm_client = AzureOpenAI(
                **common_params,
                **azure_params,
            )
        self.model_name = model or os.getenv("OPENAI_DEFAULT_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self.max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS

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

        if not kwargs.get("temperature"):
            kwargs["temperature"] = self.temperature
        if test_gpt_version(kwargs["model"]):  # gpt >=5
            kwargs["temperature"] = 1

        if self.api_type == APIType.COMPLETIONS:
            if not kwargs.get("max_tokens"):
                kwargs["max_tokens"] = self.max_output_tokens
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
            if not kwargs.get("max_output_tokens"):
                kwargs["max_output_tokens"] = self.max_output_tokens
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
        """
        Call OpenAI model for generation with structured output.

        Parameters
        ----------
        user_prompt : str
            Prompt to send to the model with role=user.
        system_prompt : str, optional
            Prompt to send to the model with role=system.
        response_format : Type[BaseModel], optional
            Pydantic model class defining the expected output structure.
        **kwargs : dict
            Additional arguments to pass to the OpenAI API.

        Returns
        -------
        BaseModel or None
            A pydantic object instance of the specified response_format type,
            or None if an error occurs.

        Notes
        -----
        This method uses the chat.completions.parse API which supports
        structured outputs in the format defined by the response_format parameter.
        """
        # Transform messages into OpenAI API compatible format
        messages = [{"role": "user", "content": user_prompt}]
        # Add a system prompt if one is provided
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        # For important parameters, set a default value if not given
        if not kwargs.get("model"):
            kwargs["model"] = self.model_name
        if not kwargs.get("temperature"):
            kwargs["temperature"] = self.temperature

        try:
            # NB. here use beta.chat...parse to support structured outputs
            answer = self.llm_client.chat.completions.parse(
                messages=messages,
                response_format=response_format,
                **kwargs,
            )
            logger.debug(f"API returned: {answer}")
            return answer.choices[0].message.parsed
            # Details of errors available here: https://platform.openai.com/docs/guides/error-codes/api-errors
        except Exception as e:
            msg = f"OpenAI API fatal error: {e}"
            logger.error(msg)


def test_gpt_version(version_string):
    # Regular expression to match "gpt-" followed by a number (integer or float)
    pattern = r"^gpt-(\d+(\.\d+)?).*$"  # Matches numbers like 4, 4.1, 5, 10.0
    match = re.match(pattern, version_string)

    if match:
        # Extract the version number as a float
        version_number = float(match.group(1))
        if version_number >= 5:
            return True
    return False
