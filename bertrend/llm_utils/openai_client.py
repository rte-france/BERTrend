#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
import re
from enum import Enum
from typing import Type

import httpx
from loguru import logger
from openai import OpenAI, Stream, Timeout
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

# Note: .env is loaded in bertrend/__init__.py which is imported before this module

MAX_ATTEMPTS = 3
TIMEOUT = 60.0
# Per-request timeout (seconds) for a single structured-output (parse) call.
# Generous enough for reasoning models, but bounded so that a stalled
# connection can never hang the worker (and therefore the whole queue).
PARSE_TIMEOUT = float(os.getenv("OPENAI_PARSE_TIMEOUT", 180.0))

# HTTP connection-pool bounds for the OpenAI client. A bounded pool with a short
# keep-alive expiry ensures idle keep-alive sockets are actively reaped instead
# of piling up (previously the agents-SDK path leaked connections into
# CLOSE-WAIT — see the worker connection-leak fix).
OPENAI_MAX_CONNECTIONS = int(os.getenv("OPENAI_MAX_CONNECTIONS", 20))
OPENAI_MAX_KEEPALIVE_CONNECTIONS = int(
    os.getenv("OPENAI_MAX_KEEPALIVE_CONNECTIONS", 10)
)
OPENAI_KEEPALIVE_EXPIRY = float(os.getenv("OPENAI_KEEPALIVE_EXPIRY", 30.0))

DEFAULT_TEMPERATURE = 0.1
DEFAULT_MODEL = "gpt-5.6-luna"
# Reasoning effort applied to GPT-5-family models (low keeps latency/cost down;
# bump to "medium"/"high" via OPENAI_REASONING_EFFORT if deeper reasoning is needed).
# Individual LLM tasks can override this per call via parse(reasoning_effort=...),
# or via a per-task env var OPENAI_REASONING_EFFORT_<TASK> (see resolve_reasoning_effort).
VALID_REASONING_EFFORTS = {"minimal", "low", "medium", "high"}
DEFAULT_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")

# LLM task identifiers used to resolve a per-task reasoning effort from the
# environment variable OPENAI_REASONING_EFFORT_<TASK> (task name upper-cased).
REASONING_TASK_TOPIC_DESCRIPTION = "topic_description"
REASONING_TASK_SIGNAL_ANALYSIS = "signal_analysis"


def resolve_reasoning_effort(
    task: str | None = None, override: str | None = None
) -> str | None:
    """Resolve the GPT-5 reasoning effort for a given LLM task.

    Resolution order (first match wins):
      1. an explicit ``override`` passed by the caller;
      2. a per-task env var ``OPENAI_REASONING_EFFORT_<TASK>`` (task upper-cased,
         e.g. ``OPENAI_REASONING_EFFORT_SIGNAL_ANALYSIS``);
      3. the global default ``DEFAULT_REASONING_EFFORT`` (env
         ``OPENAI_REASONING_EFFORT``, "low").

    The returned value is validated downstream by ``OpenAI_Client.parse``.
    """
    if override:
        return override
    if task:
        value = os.getenv(f"OPENAI_REASONING_EFFORT_{task.upper()}")
        if value:
            return value
    return DEFAULT_REASONING_EFFORT


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
            # Bounded connection pool with a short keep-alive expiry so idle
            # sockets are reaped rather than accumulating (avoids the CLOSE-WAIT
            # leak). A single client is reused across calls; remember to close()
            # it (or use it as a context manager) when done.
            "http_client": httpx.Client(
                limits=httpx.Limits(
                    max_connections=OPENAI_MAX_CONNECTIONS,
                    max_keepalive_connections=OPENAI_MAX_KEEPALIVE_CONNECTIONS,
                    keepalive_expiry=OPENAI_KEEPALIVE_EXPIRY,
                ),
            ),
        }
        self.llm_client = OpenAI(**openai_params)
        self.model = model or os.getenv("OPENAI_DEFAULT_MODEL") or DEFAULT_MODEL
        self.temperature = temperature if not test_gpt5_version(self.model) else 1
        self.api_type = api_type

    def close(self) -> None:
        """Close the underlying HTTP client and release pooled connections."""
        try:
            self.llm_client.close()
        except Exception as e:  # pragma: no cover - best-effort cleanup
            logger.debug(f"Error closing OpenAI client: {e}")

    def __enter__(self) -> "OpenAI_Client":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

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
            kwargs["model"] = self.model

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
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> BaseModel | None:
        """Call OpenAI model for generation with structured output.

        Uses the shared, synchronous ``self.llm_client`` (native OpenAI structured
        outputs) rather than the openai-agents SDK. This reuses a single bounded
        HTTP connection pool instead of spinning up a throwaway event loop and a
        new async client per call — which previously leaked connections into
        CLOSE-WAIT in the queue workers.

        Parameters
        ----------
        reasoning_effort : str, optional
            Reasoning effort for GPT-5-family models: one of "minimal", "low",
            "medium", "high". Lets each LLM task pick its own level (e.g. a light
            "low" for topic descriptions, a higher level for in-depth analysis).
            Defaults to DEFAULT_REASONING_EFFORT (env OPENAI_REASONING_EFFORT,
            "low"). Ignored for non-GPT-5 models.
        """
        kwargs.setdefault("model", self.model)
        kwargs.setdefault("timeout", PARSE_TIMEOUT)
        model = kwargs["model"]
        is_gpt5 = test_gpt5_version(model)

        # Reasoning models (GPT-5 family) require temperature == 1.
        kwargs["temperature"] = (
            1 if is_gpt5 else kwargs.get("temperature", self.temperature)
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # No schema requested: fall back to a plain text generation.
        if response_format is None:
            return self.generate_from_history(messages, **kwargs)

        effort = reasoning_effort or DEFAULT_REASONING_EFFORT
        if effort not in VALID_REASONING_EFFORTS:
            logger.warning(
                f"Invalid reasoning_effort '{effort}'; falling back to "
                f"'{DEFAULT_REASONING_EFFORT}'. Valid values: {sorted(VALID_REASONING_EFFORTS)}"
            )
            effort = DEFAULT_REASONING_EFFORT

        if self.api_type == APIType.COMPLETIONS:
            if is_gpt5:
                kwargs["reasoning_effort"] = effort
            completion = self.llm_client.chat.completions.parse(
                messages=messages,
                response_format=response_format,
                **kwargs,
            )
            return completion.choices[0].message.parsed

        # RESPONSES API (default)
        if is_gpt5:
            kwargs["reasoning"] = {"effort": effort}
        response = self.llm_client.responses.parse(
            input=messages,
            text_format=response_format,
            **kwargs,
        )
        return response.output_parsed


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
