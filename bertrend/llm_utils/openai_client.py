#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
from typing import Type

from openai import OpenAI, AzureOpenAI, Timeout, Stream
from loguru import logger
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

MAX_ATTEMPTS = 3
TIMEOUT = 60.0
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_OUTPUT_TOKENS = 512

AZURE_API_VERSION = "2025-03-01-preview"


class OpenAI_Client:
    """
    Generic client for OpenAI API (either direct API or via Azure).

    This class provides a unified interface for interacting with OpenAI models,
    supporting both direct API access and Azure-hosted deployments. It handles
    authentication, request formatting, and error handling.

    Notes
    -----
    The API key and the ENDPOINT must be set using environment variables OPENAI_API_KEY and
    OPENAI_ENDPOINT respectively. The endpoint should only be set for Azure or local deployments.
    """

    def __init__(
        self,
        api_key: str = None,
        endpoint: str = None,
        model: str = None,
        temperature: float = DEFAULT_TEMPERATURE,
        api_version: str = AZURE_API_VERSION,
    ):
        """
        Initialize the OpenAI client.

        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. If None, will try to get from OPENAI_API_KEY environment variable.
        endpoint : str, optional
            API endpoint URL. If None, will try to get from OPENAI_ENDPOINT environment variable.
            Should be set for Azure or local deployments.
        model : str, optional
            Name of the model to use. If None, will try to get from OPENAI_DEFAULT_MODEL_NAME environment variable.
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

        endpoint = endpoint or os.getenv("OPENAI_ENDPOINT", None)
        if endpoint == "":  # check empty env var
            endpoint = None

        run_on_azure = "azure.com" in endpoint if endpoint else False

        common_params = {
            "api_key": api_key,
            "timeout": Timeout(TIMEOUT, connect=10.0),
            "max_retries": MAX_ATTEMPTS,
        }
        openai_params = {
            "base_url": endpoint,
        }
        azure_params = {
            "azure_endpoint": endpoint,
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
        self.model_name = model or os.getenv("OPENAI_DEFAULT_MODEL_NAME")
        self.temperature = temperature
        self.max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS

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
        This method uses the beta.chat.completions.parse API which supports
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
            answer = self.llm_client.beta.chat.completions.parse(
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
